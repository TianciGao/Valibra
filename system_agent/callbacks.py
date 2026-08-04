"""ADK callbacks for a-interact mode: budget management and turn limiting.
专门负责a-interact模式的 ADK 回调函数，确保在评测过程中遵循预算管理和交互轮次限制。"""

import json
import logging
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from shared.audit import (
    count_tokens,
    json_text,
    normalize_usage,
    summarize_usage,
    to_jsonable,
    utc_now,
)
from shared.config import settings

logger = logging.getLogger(__name__)

MAX_MODEL_TURNS = 60 # 最大交互轮次限制，超过此轮次后，智能体将被强制停止

TOOL_COSTS = {
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
    "ask_user": 2.0,
    "submit_sql": 3.0,
}


def _preview(value: Any, limit: int = 2000) -> Any: # 预览工具的返回值，限制输出长度为 limit 个字符
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text[:limit] + "...<truncated>" if len(text) > limit else text


async def before_model_callback( # 在Agent调用模型之前的回调函数，用于限制交互轮次和预算
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Cap LLM invocations at MAX_MODEL_TURNS."""
    calls = callback_context.state.get("system_agent_llm_calls", [])
    request_data = to_jsonable(llm_request)
    calls.append({
        "call_index": len(calls) + 1,
        "timestamp": utc_now(),
        "model": settings.system_agent_model,
        "user_simulator": settings.user_sim_model,
        "generation_parameters": {
            "thinking": {
                "type": settings.system_agent_thinking,
                "clear_thinking": settings.system_agent_clear_thinking,
            },
            "reasoning_effort": settings.system_agent_reasoning_effort,
            "max_tokens": settings.system_agent_max_tokens,
            "temperature": settings.system_agent_temperature,
            "tool_choice": settings.system_agent_tool_choice,
        },
        "request": request_data,
        "prompt": json_text(request_data),
        "response": None,
        "raw_response": None,
        "usage": {},
        "actions": [],
    })
    callback_context.state["system_agent_llm_calls"] = calls
    callback_context.state["_active_llm_call_index"] = len(calls) - 1

    turns = callback_context.state.get("model_turns", 0) + 1
    callback_context.state["model_turns"] = turns
    if turns > MAX_MODEL_TURNS:
        logger.warning("Max model turns (%d) reached, forcing stop.", MAX_MODEL_TURNS)
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(
                    text="Maximum interaction turns reached. Task ended."
                )],
            ),
        )

    if callback_context.state.get("task_done", False): # 如果任务已经完成，则返回一个 LlmResponse，提示任务已完成
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Task completed.")],
            ),
        )

    budget = callback_context.state.get("budget_remaining", None) # 剩余预算
    if budget is not None and budget < 0:
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Budget exhausted. Task ended.")],
            ),
        )

    return None


async def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Retain the raw model response and provider-reported token usage."""
    calls = callback_context.state.get("system_agent_llm_calls", [])
    index = callback_context.state.get("_active_llm_call_index")
    if not isinstance(index, int) or index < 0 or index >= len(calls):
        calls.append({
            "call_index": len(calls) + 1,
            "timestamp": utc_now(),
            "model": settings.system_agent_model,
            "user_simulator": settings.user_sim_model,
            "request": None,
            "prompt": "",
            "actions": [],
        })
        index = len(calls) - 1

    response_data = to_jsonable(llm_response)
    usage = normalize_usage(getattr(llm_response, "usage_metadata", None))
    calls[index]["response"] = json_text(response_data)
    calls[index]["raw_response"] = response_data
    calls[index]["usage"] = usage
    calls[index]["completed_at"] = utc_now()
    callback_context.state["system_agent_llm_calls"] = calls
    callback_context.state["system_agent_token_usage"] = summarize_usage(calls)
    return None


async def before_tool_callback( # 在Agent调用工具之前的回调函数，用于检查预算是否足够
    tool, args: dict, tool_context: ToolContext
) -> dict | None:
    """Deduct budget. Free submit exit when exhausted."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    cost = TOOL_COSTS.get(tool_name)
    if cost is None:
        return None

    budget = tool_context.state.get("budget_remaining", 0)
    tool_context.state["_tool_phase_before"] = tool_context.state.get("current_phase", 1)

    if budget < cost:
        tool_context.state["_budget_before"] = budget
        if tool_name == "submit_sql":
            tool_context.state["budget_remaining"] = -1
            return None  # free exit, -1 signals stop after this
        return {
            "error": f"Budget exhausted ({budget:.1f} remaining). "
            "You MUST call submit_sql now with your best SQL."
        }

    tool_context.state["_budget_before"] = budget
    remaining = budget - cost
    # After submit drains budget to 0, signal stop with -1
    if tool_name == "submit_sql" and remaining <= 0:
        remaining = -1
    tool_context.state["budget_remaining"] = remaining
    return None


async def after_tool_callback( # 在Agent调用工具之后的回调函数，用于记录工具调用事件和预算使用情况
    tool, args: dict, tool_context: ToolContext, tool_response
) -> dict | None:
    """Record tool event in trajectory and append budget note to response."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    cost = TOOL_COSTS.get(tool_name, 0)
    budget_before = tool_context.state.get("_budget_before")
    budget_after = tool_context.state.get("budget_remaining")
    initial = tool_context.state.get("initial_budget", 0)

    trajectory = tool_context.state.get("tool_trajectory", [])
    phase = tool_context.state.get("_tool_phase_before", tool_context.state.get("current_phase", 1))
    args_data = to_jsonable(args)
    response_data = to_jsonable(tool_response)
    action_input_tokens = count_tokens(args_data)
    action_output_tokens = count_tokens(response_data)
    event = {
        "type": "tool",
        "tool": tool_name,
        "phase": phase,
        "args": args_data,
        "result": response_data,
        "cost": cost,
        "budget_before": budget_before,
        "budget_after": budget_after,
        "action_input_tokens": action_input_tokens,
        "action_output_tokens": action_output_tokens,
        "timestamp": utc_now(),
    }
    trajectory.append(event)
    tool_context.state["tool_trajectory"] = trajectory

    calls = tool_context.state.get("system_agent_llm_calls", [])
    index = tool_context.state.get("_active_llm_call_index")
    if isinstance(index, int) and 0 <= index < len(calls):
        calls[index].setdefault("actions", []).append(event)
        calls[index]["action"] = tool_name
        calls[index]["remaining_budget"] = budget_after
        calls[index]["action_input_tokens"] = action_input_tokens
        calls[index]["action_output_tokens"] = action_output_tokens
        calls[index]["action_cost"] = cost
        tool_context.state["system_agent_llm_calls"] = calls

    # Append budget note to agent-visible response (matches reference implementation)
    if budget_after is not None and budget_after >= 0:
        budget_note = f"\n\n[SYSTEM NOTE: Remaining budget: {budget_after:.1f}/{initial:.1f}]"
        return str(tool_response) + budget_note
    return None
