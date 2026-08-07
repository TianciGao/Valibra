"""把规则 Shadow 包在 Baseline 回调外层。

Shadow 只旁路记录有限的 Observation 和临时 Frame，不修改模型请求、工具协议
或 Baseline 返回值；完整工具结果仍由 Baseline 轨迹保存。
"""

from __future__ import annotations

import hashlib
import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from valibra_agent.requirement_grounding.models import (
    PendingToolCall,
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.observations import (
    build_observation,
    canonical_json,
)
from valibra_agent.requirement_grounding.reducer import validate_runtime
from valibra_agent.requirement_grounding.service import (
    add_pending_tool_call,
    process_observation,
    process_phase_transition,
    remove_pending_tool_call,
)
from valibra_agent.requirement_grounding.telemetry import record_failure
from valibra_agent.requirement_grounding.updater import RuleUpdater

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


GROUNDING_RUNTIME_KEY = "valibra:grounding_runtime"
GROUNDING_SEQUENCE_KEY = "valibra:grounding_sequence"
SHADOW_AUDIT_KEY = "valibra_shadow"
SHADOW_MODE = "rule"

_MAX_ARGS_SUMMARY_CHARS = 768
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

_OBSERVATION_TYPES = {
    "execute_sql": "sql_execution",
    "get_schema": "schema",
    "get_all_column_meanings": "metadata",
    "get_column_meaning": "metadata",
    "get_all_external_knowledge_names": "knowledge",
    "get_knowledge_definition": "knowledge",
    "get_all_knowledge_definitions": "knowledge",
    "ask_user": "user_answer",
    "submit_sql": "submission",
}
_RULE_UPDATER = RuleUpdater()


@dataclass(slots=True)
class _BoundTurnMessage:
    """当前异步任务绑定的用户消息，只允许消费一次。"""

    task_id: str
    mode: str
    message: str
    consumed: bool = False


_ACTIVE_TURN_MESSAGE: ContextVar[_BoundTurnMessage | None] = ContextVar(
    "valibra_active_turn_message",
    default=None,
)


def _bind_turn_message(task_id: str, mode: str, message: str) -> Any:
    """把 run_turn 原始消息绑定到当前异步上下文。"""

    return _ACTIVE_TURN_MESSAGE.set(
        _BoundTurnMessage(task_id=task_id, mode=mode, message=message)
    )


def _reset_turn_message(token: Any) -> None:
    """恢复绑定前的上下文，防止并发任务互相污染。"""

    _ACTIVE_TURN_MESSAGE.reset(token)


async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """尽力初始化 Shadow，再原样交给 Baseline 的模型前回调。"""

    state = getattr(callback_context, "state", None)
    if state is not None:
        try:
            _consume_bound_user_message(state)
            _ensure_runtime(state)
        except Exception as exc:
            # Grounding 初始化失败只能记日志，不能打断 Baseline。
            _record_callback_failure(
                state,
                stage="observation",
                exception=exc,
                sequence=_current_sequence(state),
            )
    from system_agent import callbacks as baseline_callbacks

    return await baseline_callbacks.before_model_callback(
        callback_context,
        llm_request,
    )


async def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """模型响应只交给 Baseline 处理一次，Valibra 不改内容。"""

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
    """先让 Baseline 判断预算；获准后再登记待完成工具调用。"""

    from system_agent import callbacks as baseline_callbacks

    baseline_result = await baseline_callbacks.before_tool_callback(
        tool,
        args,
        tool_context,
    )
    if baseline_result is not None:
        # 非 None 表示 Baseline 已拒绝或接管，本层不能再登记调用。
        return baseline_result

    state = getattr(tool_context, "state", None)
    if state is None:
        return baseline_result
    sequence = _current_sequence(state)
    try:
        function_call_id = _require_function_call_id(tool_context)
        tool_name = _tool_name(tool)
        phase_before = _phase(state.get("current_phase", 1))
        args_json = canonical_json(args)
        sequence = _next_sequence(state)
        pending = PendingToolCall(
            function_call_id=function_call_id,
            tool_name=tool_name,
            args_summary=_bounded_args_summary(args_json),
            args_digest=hashlib.sha256(args_json.encode("utf-8")).hexdigest(),
            phase_before=phase_before,
            sequence=sequence,
            started_at=None,
        )
        runtime = add_pending_tool_call(_ensure_runtime(state), pending)
        _store_runtime(state, runtime)
    except Exception as exc:
        _record_callback_failure(
            state,
            stage="service",
            exception=exc,
            sequence=sequence,
            function_call_id=_valid_context_identifier(tool_context),
        )
    return baseline_result


async def after_tool_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    tool_response: Any,
) -> Any:
    """先执行 Baseline 回调，再用原始工具结果更新 Shadow。"""

    from system_agent import callbacks as baseline_callbacks

    state = getattr(tool_context, "state", None)
    trajectory_before = (
        _trajectory_length(state) if state is not None else None
    )
    try:
        baseline_override = await baseline_callbacks.after_tool_callback(
            tool,
            args,
            tool_context,
            tool_response,
        )
    except BaseException:
        if state is not None:
            _cleanup_pending_best_effort(state, tool_context)
        raise

    if state is None:
        return baseline_override

    audit_index = _new_trajectory_index(state, trajectory_before)
    raw_log_ref = (
        f"session://tool_trajectory/{audit_index}"
        if audit_index is not None
        else None
    )
    sequence = _current_sequence(state)
    audit_metadata: dict[str, Any] | None = None
    try:
        function_call_id = _require_function_call_id(tool_context)
        runtime = _ensure_runtime(state)
        runtime, pending = remove_pending_tool_call(runtime, function_call_id)
        # 先删除 Pending 并落盘；后续失败也不会遗留“正在调用”的假状态。
        _store_runtime(state, runtime)
        if pending is None:
            return baseline_override

        tool_name = _tool_name(tool)
        if pending.tool_name != tool_name:
            raise ValueError(
                "pending tool name does not match the exact function call"
            )
        task_id = _task_id(state)
        sequence = _next_sequence(state)
        observation = build_observation(
            task_id=task_id,
            observation_type=_observation_type(tool_name),
            phase=pending.phase_before,
            sequence=sequence,
            source="adk_tool_result",
            raw=tool_response,
            raw_log_ref=raw_log_ref,
            function_call_id=function_call_id,
            invocation_id=_valid_invocation_id(tool_context),
            tool_name=tool_name,
        )
        result = process_observation(
            runtime,
            observation,
            updater=_RULE_UPDATER,
        )
        runtime = result.runtime

        transition_observation_id: str | None = None
        phase_after = _phase(state.get("current_phase", pending.phase_before))
        if (
            tool_name == "submit_sql"
            and pending.phase_before == 1
            and phase_after == 2
        ):
            # submit_sql 让 Baseline 进入 Phase 2 时，显式记录阶段切换。
            transition_sequence = _next_sequence(state)
            transition_observation = build_observation(
                task_id=task_id,
                observation_type="phase_transition",
                phase=2,
                sequence=transition_sequence,
                source="adk_lifecycle",
                raw={
                    "submission_observation_id": observation.observation_id,
                    "phase_before": 1,
                    "phase_after": 2,
                },
                summary="legal submit_sql transitioned Baseline phase 1 to phase 2",
                raw_log_ref=raw_log_ref,
                function_call_id=function_call_id,
                invocation_id=_valid_invocation_id(tool_context),
                tool_name=tool_name,
            )
            transition_result = process_phase_transition(
                runtime,
                transition_observation,
            )
            runtime = transition_result.runtime
            transition_observation_id = transition_observation.observation_id

        _store_runtime(state, runtime)
        audit_metadata = {
            "mode": SHADOW_MODE,
            "status": result.status,
            "function_call_id": function_call_id,
            "tool_name": tool_name,
            "phase_before": pending.phase_before,
            "phase_after": phase_after,
            "observation_id": observation.observation_id,
            "transition_observation_id": transition_observation_id,
            "raw_digest": observation.raw_digest,
            "raw_log_ref": raw_log_ref,
            "summary": observation.summary,
            "grounding_revision": runtime.grounding_revision,
            "runtime_bytes": _runtime_bytes(runtime),
        }
    except Exception as exc:
        function_call_id = _valid_context_identifier(tool_context)
        _cleanup_pending_best_effort(state, tool_context)
        runtime = _record_callback_failure(
            state,
            stage="observation",
            exception=exc,
            sequence=sequence,
            function_call_id=function_call_id,
        )
        audit_metadata = {
            "mode": SHADOW_MODE,
            "status": "failed",
            "function_call_id": function_call_id,
            "tool_name": _safe_tool_name(tool),
            "raw_log_ref": raw_log_ref,
            "error_type": type(exc).__name__[:128],
            "grounding_revision": runtime.grounding_revision,
            "runtime_bytes": _runtime_bytes(runtime),
        }
    finally:
        _cleanup_pending_best_effort(state, tool_context)
        if audit_index is not None and audit_metadata is not None:
            _attach_shadow_audit(state, audit_index, audit_metadata)

    return baseline_override


async def on_tool_error_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    error: Exception,
) -> None:
    """工具报错时只清理对应 Pending，记有限错误信息后交还 ADK。"""

    del tool, args
    state = tool_context.state
    sequence = _current_sequence(state)
    function_call_id = _valid_context_identifier(tool_context)
    try:
        exact_id = _require_function_call_id(tool_context)
        runtime = _ensure_runtime(state)
        runtime, _ = remove_pending_tool_call(runtime, exact_id)
        sequence = _next_sequence(state)
        runtime = record_failure(
            runtime,
            stage="service",
            exception=error,
            sequence=sequence,
            function_call_id=exact_id,
        )
        _store_runtime(state, runtime)
    except Exception as callback_error:
        _record_callback_failure(
            state,
            stage="service",
            exception=callback_error,
            sequence=sequence,
            function_call_id=function_call_id,
        )
    return None


def _ensure_runtime(state: Any) -> RequirementGroundingRuntime:
    """读取并校验 Runtime；坏数据回退为空状态，始终不阻断主流程。"""

    try:
        raw = state.get(GROUNDING_RUNTIME_KEY)
        runtime = (
            RequirementGroundingRuntime()
            if raw is None
            else RequirementGroundingRuntime.model_validate(raw)
        )
        validate_runtime(runtime)
    except Exception as exc:
        runtime = RequirementGroundingRuntime()
        try:
            runtime = record_failure(
                runtime,
                stage="service",
                exception=exc,
                sequence=_current_sequence(state),
            )
        except Exception:
            runtime = RequirementGroundingRuntime()
    _store_runtime(state, runtime)
    if state.get(GROUNDING_SEQUENCE_KEY) is None:
        state[GROUNDING_SEQUENCE_KEY] = 0
    return runtime


def _consume_bound_user_message(state: Any) -> None:
    """把本轮用户消息转成一次 Observation，并交给规则 Updater。"""

    bound = _ACTIVE_TURN_MESSAGE.get()
    if bound is None or bound.consumed or bound.mode != "a-interact":
        return
    # 先标记已消费：即使处理失败，同一轮多次 before_model 也不会重复记账。
    bound.consumed = True
    if _task_id(state) != bound.task_id:
        raise ValueError("run_turn task_id does not match ADK session state")
    sequence = _next_sequence(state)
    observation = build_observation(
        task_id=bound.task_id,
        observation_type="user_query",
        phase=_phase(state.get("current_phase", 1)),
        sequence=sequence,
        source="adk_run_turn_message",
        raw=bound.message,
        summary=_user_message_summary(bound.message),
    )
    result = process_observation(
        _ensure_runtime(state),
        observation,
        updater=_RULE_UPDATER,
    )
    _store_runtime(state, result.runtime)


def _user_message_summary(message: str) -> str:
    """从 Baseline 拼装的提示中取出真正的用户问题。"""

    if not isinstance(message, str):
        raise TypeError("run_turn message must be a string")
    marker = "User Query:\n"
    suffix = "\n\nYou have a budget"
    start = message.find(marker)
    if start < 0:
        return message
    start += len(marker)
    end = message.find(suffix, start)
    if end < 0:
        return message[start:]
    return message[start:end]


def _store_runtime(state: Any, runtime: RequirementGroundingRuntime) -> None:
    """校验后以纯 JSON 数据写回 ADK Session State。"""

    validate_runtime(runtime)
    state[GROUNDING_RUNTIME_KEY] = runtime.model_dump(mode="json")


def _record_callback_failure(
    state: Any,
    *,
    stage: str,
    exception: BaseException,
    sequence: int,
    function_call_id: str | None = None,
) -> RequirementGroundingRuntime:
    """尽力记录回调错误；记录失败也不能覆盖 Baseline 结果。"""

    runtime = _ensure_runtime(state)
    try:
        runtime = record_failure(
            runtime,
            stage=stage,
            exception=exception,
            sequence=max(0, sequence),
            function_call_id=function_call_id,
        )
        _store_runtime(state, runtime)
    except Exception:
        # 连诊断本身失败也直接忽略，Baseline 的结果优先。
        pass
    return runtime


def _cleanup_pending_best_effort(state: Any, tool_context: Any) -> None:
    """尽力清理一个准确匹配的 Pending，不向外抛错。"""

    function_call_id = _valid_context_identifier(tool_context)
    if function_call_id is None:
        return
    try:
        runtime = _ensure_runtime(state)
        runtime, _ = remove_pending_tool_call(runtime, function_call_id)
        _store_runtime(state, runtime)
    except Exception:
        return


def _require_function_call_id(tool_context: Any) -> str:
    """强制使用 ADK 的真实调用 ID，不允许自造兜底 ID。"""

    function_call_id = getattr(tool_context, "function_call_id", None)
    if not isinstance(function_call_id, str) or not _IDENTIFIER_RE.fullmatch(
        function_call_id
    ):
        raise ValueError(
            "google-adk ToolContext.function_call_id is required; no fallback ID"
        )
    return function_call_id


def _valid_context_identifier(tool_context: Any) -> str | None:
    value = getattr(tool_context, "function_call_id", None)
    return value if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) else None


def _valid_invocation_id(tool_context: Any) -> str | None:
    value = getattr(tool_context, "invocation_id", None)
    return value if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) else None


def _tool_name(tool: Any) -> str:
    value = getattr(tool, "name", None)
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError("tool has no bounded official name")
    return value


def _safe_tool_name(tool: Any) -> str:
    value = getattr(tool, "name", None)
    if isinstance(value, str):
        return value[:64]
    return type(tool).__name__[:64]


def _observation_type(tool_name: str) -> str:
    try:
        return _OBSERVATION_TYPES[tool_name]
    except KeyError as exc:
        raise ValueError(f"unsupported a-interact tool: {tool_name}") from exc


def _task_id(state: Any) -> str:
    task_id = state.get("task_id")
    if not isinstance(task_id, str) or not _IDENTIFIER_RE.fullmatch(task_id):
        raise ValueError("bounded task_id is required for Shadow Observation")
    return task_id


def _phase(value: Any) -> int:
    if isinstance(value, bool) or value not in (1, 2):
        raise ValueError("phase must be exactly 1 or 2")
    return int(value)


def _current_sequence(state: Any) -> int:
    value = state.get(GROUNDING_SEQUENCE_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if 0 <= value <= _MAX_SEQUENCE else 0


def _next_sequence(state: Any) -> int:
    """生成严格递增的 Shadow 事件序号。"""

    current = state.get(GROUNDING_SEQUENCE_KEY, 0)
    if (
        isinstance(current, bool)
        or not isinstance(current, int)
        or current < 0
        or current >= _MAX_SEQUENCE
    ):
        raise ValueError("invalid or exhausted Shadow sequence")
    next_value = current + 1
    state[GROUNDING_SEQUENCE_KEY] = next_value
    return next_value


def _bounded_args_summary(args_json: str) -> str:
    if len(args_json) <= _MAX_ARGS_SUMMARY_CHARS:
        return args_json
    return args_json[: _MAX_ARGS_SUMMARY_CHARS - 1] + "…"


def _trajectory_length(state: Any) -> int | None:
    trajectory = state.get("tool_trajectory", [])
    return len(trajectory) if isinstance(trajectory, list) else None


def _new_trajectory_index(state: Any, before: int | None) -> int | None:
    trajectory = state.get("tool_trajectory", [])
    if before is None or not isinstance(trajectory, list):
        return None
    if len(trajectory) != before + 1:
        return None
    return before


def _attach_shadow_audit(
    state: Any,
    index: int,
    metadata: dict[str, Any],
) -> None:
    """把有限 Shadow 摘要挂到对应 Baseline 工具轨迹。"""

    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list) or not 0 <= index < len(trajectory):
        return
    event = trajectory[index]
    if not isinstance(event, dict):
        return
    # 摘要一旦意外膨胀就不写；这里严禁原始响应、Prompt、凭据和模型正文。
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 4096:
        return
    updated = list(trajectory)
    updated_event = dict(event)
    updated_event[SHADOW_AUDIT_KEY] = metadata
    updated[index] = updated_event
    state["tool_trajectory"] = updated


def _runtime_bytes(runtime: RequirementGroundingRuntime) -> int:
    """计算 Runtime 的规范 JSON 大小，供审计查看。"""

    return len(
        json.dumps(
            runtime.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
