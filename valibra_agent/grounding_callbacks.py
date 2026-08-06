"""V0 callback shell with grounding deliberately disabled.

Each callback delegates exactly once to the frozen Baseline callback and
returns its result unchanged.  Grounding state and behavior begin in later
phases, not in V0.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.tool_context import ToolContext
else:
    CallbackContext = Any
    LlmRequest = Any
    LlmResponse = Any
    ToolContext = Any


async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    from system_agent import callbacks as baseline_callbacks

    return await baseline_callbacks.before_model_callback(
        callback_context,
        llm_request,
    )


async def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    from system_agent import callbacks as baseline_callbacks

    return await baseline_callbacks.after_model_callback(
        callback_context,
        llm_response,
    )


async def before_tool_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
) -> dict | None:
    from system_agent import callbacks as baseline_callbacks

    return await baseline_callbacks.before_tool_callback(
        tool,
        args,
        tool_context,
    )


async def after_tool_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    tool_response: Any,
) -> Any:
    from system_agent import callbacks as baseline_callbacks

    return await baseline_callbacks.after_tool_callback(
        tool,
        args,
        tool_context,
        tool_response,
    )
