"""把可选的 Rule/LLM Shadow 包在 Baseline 回调外层。

Shadow 只旁路记录有限的 Observation 和临时 Frame，不修改模型请求、工具协议
或 Baseline 返回值；完整工具结果仍由 Baseline 轨迹保存。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shared.config import PROJECT_ROOT
from valibra_agent.requirement_grounding.evidence_bridge import (
    EVIDENCE_OBSERVATION_TYPES,
    EvidenceBridgeUpdater,
)
from valibra_agent.requirement_grounding.models import (
    Observation,
    PendingToolCall,
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.observations import (
    ObservationNormalizationError,
    build_observation,
    canonical_json,
    classify_tool_observation_type,
    extract_submit_follow_up,
)
from valibra_agent.requirement_grounding.prompt_view import (
    count_prompt_view_tokens,
    prompt_view_sha256,
    render_prompt_view,
)
from valibra_agent.requirement_grounding.reducer import validate_runtime
from valibra_agent.requirement_grounding.service import (
    add_pending_tool_call,
    process_observation,
    process_observation_with_llm,
    process_phase_transition,
    remove_pending_tool_call,
)
from valibra_agent.requirement_grounding.telemetry import record_failure
from valibra_agent.requirement_grounding.updater import (
    GROUNDING_LLM_OBSERVATION_TYPES,
    LLMUpdater,
    LiteLLMGroundingClient,
    NoOpUpdater,
    RuleUpdater,
    load_grounding_llm_config,
    load_grounding_provider_config,
)

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
REQUIREMENT_VIEW_AUDIT_KEY = "valibra_requirement_view"
GROUNDING_UPDATER_MODE_ENV = "GROUNDING_UPDATER_MODE"

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
_NOOP_UPDATER = NoOpUpdater()
_EVIDENCE_UPDATER = EvidenceBridgeUpdater()


def _requested_updater_mode(
    environment: Mapping[str, str] | None = None,
) -> str:
    """解析唯一模式开关；不做 trim、大小写或隐式回退。"""

    source = environment if environment is not None else os.environ
    raw_mode = source.get(GROUNDING_UPDATER_MODE_ENV)
    if raw_mode is None or raw_mode == "":
        return "rule"
    if raw_mode == "llm":
        return "llm"
    return "invalid"


def _grounding_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """只复制显式 GROUNDING_* 变量，隔离主 Agent 的连接配置。"""

    source = environment if environment is not None else os.environ
    return {
        key: str(value)
        for key, value in source.items()
        if key.startswith("GROUNDING_")
    }


def _build_llm_updater(
    environment: Mapping[str, str] | None = None,
) -> LLMUpdater:
    """按调用即时构造 LLM Updater；对象和凭据不会进入 Session State。"""

    grounding_environment = _grounding_environment(environment)
    llm_config = load_grounding_llm_config(
        PROJECT_ROOT,
        grounding_environment,
    )
    provider_config = load_grounding_provider_config(
        PROJECT_ROOT,
        llm_config,
        grounding_environment,
    )
    client = LiteLLMGroundingClient(
        provider_config,
        environment=grounding_environment,
    )
    return LLMUpdater(client, llm_config)


def grounding_updater_status(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """返回 health 可公开的有效模式，不暴露配置内容或凭据。"""

    requested_mode = _requested_updater_mode(environment)
    if requested_mode == "rule":
        return {
            "requested_mode": "rule",
            "effective_mode": "rule",
            "configuration_valid": True,
            "error_type": None,
        }
    if requested_mode == "invalid":
        return {
            "requested_mode": "invalid",
            "effective_mode": "invalid",
            "configuration_valid": False,
            "error_type": "InvalidUpdaterMode",
        }
    try:
        # 这里只校验冻结配置并构造对象；不会读取密钥内容或发送请求。
        _build_llm_updater(environment)
    except Exception as exc:
        return {
            "requested_mode": "llm",
            "effective_mode": "invalid",
            "configuration_valid": False,
            "error_type": type(exc).__name__[:128],
        }
    return {
        "requested_mode": "llm",
        "effective_mode": "llm",
        "configuration_valid": True,
        "error_type": None,
    }


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
    """生成审计用 Requirement View，再原样交给 Baseline 回调。"""

    state = getattr(callback_context, "state", None)
    model_call_count: int | None = None
    view_audit: dict[str, Any] | None = None
    if state is not None:
        try:
            model_call_count = _model_call_count(state)
            await _consume_bound_user_message(state)
            runtime = _ensure_runtime(state)
            view = render_prompt_view(runtime.grounding_state)
            view_audit = {
                "mode": "shadow",
                "view": view,
                "view_sha256": prompt_view_sha256(view),
                "chars": len(view),
                "tokens_cl100k": count_prompt_view_tokens(view),
                "injected": False,
            }
        except Exception as exc:
            # Grounding 初始化失败只能记日志，不能打断 Baseline。
            _record_callback_failure(
                state,
                stage="observation",
                exception=exc,
                sequence=_current_sequence(state),
            )
            view_audit = {
                "mode": "shadow",
                "injected": False,
                "error_type": type(exc).__name__[:128],
            }
    from system_agent import callbacks as baseline_callbacks

    baseline_result = await baseline_callbacks.before_model_callback(
        callback_context,
        llm_request,
    )
    if state is not None and view_audit is not None:
        try:
            model_call_index = _new_model_call_index(state, model_call_count)
            if model_call_index is not None:
                _attach_requirement_view_audit(
                    state,
                    model_call_index,
                    view_audit,
                )
        except Exception as exc:
            _record_callback_failure(
                state,
                stage="service",
                exception=exc,
                sequence=_current_sequence(state),
            )
    return baseline_result


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
    shadow_mode = _requested_updater_mode()
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
        success_type = _observation_type(tool_name)
        observation_type = classify_tool_observation_type(
            tool_name=tool_name,
            tool_response=tool_response,
            success_type=success_type,
        )
        observation = build_observation(
            task_id=task_id,
            observation_type=observation_type,
            phase=pending.phase_before,
            sequence=sequence,
            source="adk_tool_result",
            raw=tool_response,
            raw_log_ref=raw_log_ref,
            function_call_id=function_call_id,
            invocation_id=_valid_invocation_id(tool_context),
            tool_name=tool_name,
        )
        runtime, result_status, llm_audit = await _process_shadow_observation(
            runtime,
            observation,
        )
        if observation_type == "tool_error":
            runtime = record_failure(
                runtime,
                stage="observation",
                exception=RuntimeError(
                    f"official {tool_name} returned its frozen error form"
                ),
                sequence=observation.sequence,
                observation_id=observation.observation_id,
                function_call_id=function_call_id,
            )

        transition_observation_id: str | None = None
        follow_up_observation_id: str | None = None
        follow_up_status: str | None = None
        follow_up_llm_audit: dict[str, Any] | None = None
        phase_after = _phase(state.get("current_phase", pending.phase_before))
        if (
            tool_name == "submit_sql"
            and observation_type == "submission"
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
            if transition_result.status == "processed" and runtime.phase == 2:
                try:
                    follow_up = extract_submit_follow_up(tool_response)
                    follow_up_sequence = _next_sequence(state)
                    follow_up_observation = build_observation(
                        task_id=task_id,
                        observation_type="user_query",
                        phase=2,
                        sequence=follow_up_sequence,
                        source="submit_sql_follow_up",
                        raw=follow_up,
                        summary=follow_up,
                        raw_log_ref=raw_log_ref,
                        function_call_id=function_call_id,
                        invocation_id=_valid_invocation_id(tool_context),
                        tool_name=tool_name,
                    )
                    runtime, follow_up_status, follow_up_llm_audit = (
                        await _process_shadow_observation(
                            runtime,
                            follow_up_observation,
                        )
                    )
                    follow_up_observation_id = follow_up_observation.observation_id
                except ObservationNormalizationError as exc:
                    follow_up_status = "failed"
                    runtime = record_failure(
                        runtime,
                        stage="observation",
                        exception=exc,
                        sequence=_current_sequence(state),
                        observation_id=transition_observation.observation_id,
                        function_call_id=function_call_id,
                    )

        _store_runtime(state, runtime)
        audit_metadata = {
            "mode": shadow_mode,
            "status": result_status,
            "function_call_id": function_call_id,
            "tool_name": tool_name,
            "phase_before": pending.phase_before,
            "phase_after": phase_after,
            "observation_id": observation.observation_id,
            "observation_type": observation.observation_type,
            "transition_observation_id": transition_observation_id,
            "follow_up_observation_id": follow_up_observation_id,
            "follow_up_status": follow_up_status,
            "raw_digest": observation.raw_digest,
            "raw_log_ref": raw_log_ref,
            "summary": observation.summary,
            "grounding_revision": runtime.grounding_revision,
            "runtime_bytes": _runtime_bytes(runtime),
        }
        if llm_audit is not None:
            audit_metadata["llm"] = llm_audit
        if follow_up_llm_audit is not None:
            audit_metadata["follow_up_llm"] = follow_up_llm_audit
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
            "mode": shadow_mode,
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
    """真实工具异常形成有界 ToolErrorObservation，并精确清理 Pending。"""

    del args
    state = tool_context.state
    sequence = _current_sequence(state)
    function_call_id = _valid_context_identifier(tool_context)
    try:
        exact_id = _require_function_call_id(tool_context)
        runtime = _ensure_runtime(state)
        runtime, pending = remove_pending_tool_call(runtime, exact_id)
        _store_runtime(state, runtime)
        if pending is None:
            return None
        tool_name = _tool_name(tool)
        if pending.tool_name != tool_name:
            raise ValueError(
                "pending tool name does not match the exact function call"
            )
        sequence = _next_sequence(state)
        observation = build_observation(
            task_id=_task_id(state),
            observation_type="tool_error",
            phase=pending.phase_before,
            sequence=sequence,
            source="adk_tool_error",
            raw={"error_type": type(error).__name__[:128]},
            summary=f"{tool_name} raised {type(error).__name__[:128]}",
            function_call_id=exact_id,
            invocation_id=_valid_invocation_id(tool_context),
            tool_name=tool_name,
        )
        runtime, _, _ = await _process_shadow_observation(runtime, observation)
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


async def _consume_bound_user_message(state: Any) -> None:
    """把本轮用户消息转成一次 Observation，并交给所选 Shadow。"""

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
    runtime, _, _ = await _process_shadow_observation(
        _ensure_runtime(state),
        observation,
    )
    _store_runtime(state, runtime)


async def _process_shadow_observation(
    runtime: RequirementGroundingRuntime,
    observation: Observation,
) -> tuple[RequirementGroundingRuntime, str, dict[str, Any] | None]:
    """按精确模式处理 Observation；错误只写有限 telemetry，不降级到 Rule。"""

    if observation.observation_type in EVIDENCE_OBSERVATION_TYPES:
        result = process_observation(
            runtime,
            observation,
            updater=_EVIDENCE_UPDATER,
        )
        return result.runtime, result.status, None

    if observation.observation_type == "tool_error":
        result = process_observation(
            runtime,
            observation,
            updater=_NOOP_UPDATER,
        )
        return result.runtime, result.status, None

    mode = _requested_updater_mode()
    if mode == "rule":
        result = process_observation(
            runtime,
            observation,
            updater=_RULE_UPDATER,
        )
        return result.runtime, result.status, None

    if mode == "invalid":
        failed = record_failure(
            runtime,
            stage="updater",
            exception=ValueError("invalid GROUNDING_UPDATER_MODE"),
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
            metric_namespace="llm",
        )
        return failed, "failed", None

    if observation.observation_type not in GROUNDING_LLM_OBSERVATION_TYPES:
        # 非用户 Observation 只记生命周期，不调用模型也不消耗 LLM call budget。
        result = process_observation(
            runtime,
            observation,
            updater=_NOOP_UPDATER,
        )
        return result.runtime, result.status, None

    try:
        updater = _build_llm_updater()
    except Exception as exc:
        failed = record_failure(
            runtime,
            stage="updater",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
            metric_namespace="llm",
        )
        return failed, "failed", None

    result = await process_observation_with_llm(
        runtime,
        observation,
        updater=updater,
    )
    return (
        result.runtime,
        result.status,
        _bounded_llm_audit(result.llm_audit),
    )


def _bounded_llm_audit(audit: Any) -> dict[str, Any]:
    """保留有限、无正文的单次 LLM 审计摘要。"""

    return {
        "status": audit.status,
        "attempted": audit.attempted,
        "configuration_sha256": audit.configuration_sha256,
        "prompt_sha256": audit.prompt_sha256,
        "input_tokens": audit.input_tokens,
        "output_tokens": audit.output_tokens,
        "reasoning_tokens": audit.reasoning_tokens,
        "total_tokens": audit.total_tokens,
        "latency_ms": audit.latency_ms,
        "cost": audit.cost,
        "model": audit.model,
        "provider": audit.provider,
        "credential_source": audit.credential_source,
        "request_sha256": audit.request_sha256,
        "response_sha256": audit.response_sha256,
        "raw_audit_ref": audit.raw_audit_ref,
        "timed_out": audit.timed_out,
        "provider_may_continue_after_cancel": (
            audit.provider_may_continue_after_cancel
        ),
        "provider_may_bill_after_cancel": audit.provider_may_bill_after_cancel,
        "error_type": audit.error_type,
    }


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


def _model_call_count(state: Any) -> int | None:
    if state is None:
        return None
    calls = state.get("system_agent_llm_calls", [])
    return len(calls) if isinstance(calls, list) else None


def _new_model_call_index(state: Any, before: int | None) -> int | None:
    calls = state.get("system_agent_llm_calls", [])
    active = state.get("_active_llm_call_index")
    if before is None or not isinstance(calls, list):
        return None
    if len(calls) != before + 1 or active != before:
        return None
    return before


def _attach_requirement_view_audit(
    state: Any,
    index: int,
    metadata: dict[str, Any],
) -> None:
    """Attach one bounded Shadow View to its exact Baseline model call."""

    calls = state.get("system_agent_llm_calls", [])
    if not isinstance(calls, list) or not 0 <= index < len(calls):
        return
    call = calls[index]
    if not isinstance(call, dict):
        return
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 4096:
        raise ValueError("Requirement View audit exceeds bounded size")
    updated = list(calls)
    updated_call = dict(call)
    updated_call[REQUIREMENT_VIEW_AUDIT_KEY] = metadata
    updated[index] = updated_call
    state["system_agent_llm_calls"] = updated


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
