"""串起 Observation、Updater 和 Reducer；Grounding 失败时保留旧状态。"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import Field

from valibra_agent.requirement_grounding.models import (
    FAILED_FRAME_INITIALIZATION_REASONS,
    FrameInitializationReason,
    FrameInitializationStatus,
    KernelModel,
    Observation,
    PendingToolCall,
    PhaseTransition,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    RequirementGroundingState,
)
from valibra_agent.requirement_grounding.reducer import apply_patch, validate_runtime
from valibra_agent.requirement_grounding.telemetry import (
    LLMCallAudit,
    LLMCallTelemetryRecorder,
    increment_metrics,
    increment_llm_call_metrics,
    record_failure,
    set_last_error,
)
from valibra_agent.requirement_grounding.updater import (
    GROUNDING_LLM_OBSERVATION_TYPES,
    GroundingProviderError,
    LLMFrameUpdateError,
    LLMUpdater,
    NoOpUpdater,
)


class Updater(Protocol):
    """同步 Updater 只负责根据 Observation 提出 Patch。"""

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch: ...


Reducer = Callable[
    [RequirementGroundingRuntime, RequirementGroundingPatch],
    RequirementGroundingRuntime,
]


class GroundingServiceResult(KernelModel):
    """同步处理结果。"""

    status: Literal["processed", "duplicate", "failed"]
    runtime: RequirementGroundingRuntime
    patch_id: str | None = Field(default=None, max_length=128)
    failure_reason: FrameInitializationReason | None = None


class LLMGroundingServiceResult(KernelModel):
    """异步 LLM 处理结果；业务状态仍只认 Runtime。"""

    status: Literal["processed", "duplicate", "failed", "skipped"]
    runtime: RequirementGroundingRuntime
    patch_id: str | None = Field(default=None, max_length=128)
    llm_audit: LLMCallAudit
    failure_reason: FrameInitializationReason | None = None


class GroundingControlError(ValueError):
    """Pending 等运行控制状态的修改不合法。"""


def set_frame_initialization_result(
    runtime: RequirementGroundingRuntime,
    *,
    status: FrameInitializationStatus,
    reason: FrameInitializationReason | None,
    observation: Observation,
) -> RequirementGroundingRuntime:
    """原子记录第一次 Phase-1 user_query 的终态，不推动两个 revision。"""

    if observation.observation_type != "user_query" or observation.phase != 1:
        raise GroundingControlError(
            "frame initialization requires a Phase-1 user_query Observation"
        )
    observation_id = observation.observation_id

    current = (
        runtime.frame_initialization_status,
        runtime.frame_initialization_reason,
        runtime.frame_initialization_observation_id,
    )
    requested = (status, reason, observation_id)
    if current[0] != "not_attempted":
        if current == requested:
            return runtime
        raise GroundingControlError("frame initialization result is already terminal")
    if status == "not_attempted":
        raise GroundingControlError("frame initialization must enter a terminal state")

    frame = runtime.grounding_state.requirement_frame
    slot_count = (
        len(frame.value_slots)
        + len(frame.schema_slots)
        + len(frame.operation_slots)
    )
    if status == "ready" and slot_count == 0:
        raise GroundingControlError("ready initialization requires at least one Slot")
    if status in {"empty", "failed"} and slot_count != 0:
        raise GroundingControlError(
            f"{status} initialization cannot retain an initial Slot"
        )
    if status == "failed" and reason not in FAILED_FRAME_INITIALIZATION_REASONS:
        raise GroundingControlError("failed initialization requires a failure reason")

    try:
        result = _validated_runtime_copy(
            runtime,
            frame_initialization_status=status,
            frame_initialization_reason=reason,
            frame_initialization_observation_id=observation_id,
        )
    except Exception as exc:
        raise GroundingControlError("invalid frame initialization result") from exc
    _assert_revision_unchanged(runtime, result)
    validate_runtime(result)
    return result


def add_pending_tool_call(
    runtime: RequirementGroundingRuntime,
    pending: PendingToolCall,
) -> RequirementGroundingRuntime:
    """登记一个准确 ID 的 Pending 调用，不增加业务 revision。"""

    if pending.function_call_id in runtime.pending_tool_calls:
        raise GroundingControlError(
            f"pending tool call already exists: {pending.function_call_id}"
        )
    pending_calls = dict(runtime.pending_tool_calls)
    pending_calls[pending.function_call_id] = pending
    result = _validated_runtime_copy(
        runtime,
        pending_tool_calls=pending_calls,
    )
    _assert_revision_unchanged(runtime, result)
    validate_runtime(result)
    return result


def remove_pending_tool_call(
    runtime: RequirementGroundingRuntime,
    function_call_id: str,
) -> tuple[RequirementGroundingRuntime, PendingToolCall | None]:
    """只删除与 function_call_id 精确匹配的 Pending。"""

    pending = runtime.pending_tool_calls.get(function_call_id)
    if pending is None:
        return runtime, None
    pending_calls = dict(runtime.pending_tool_calls)
    del pending_calls[function_call_id]
    result = _validated_runtime_copy(
        runtime,
        pending_tool_calls=pending_calls,
    )
    _assert_revision_unchanged(runtime, result)
    validate_runtime(result)
    return result, pending


def process_phase_transition(
    runtime: RequirementGroundingRuntime,
    observation: Observation,
) -> GroundingServiceResult:
    """根据明确的阶段事件进入 Phase 2，不做语义猜测。"""

    if observation.observation_id in runtime.processed_observation_ids:
        return GroundingServiceResult(
            status="duplicate",
            runtime=runtime,
            patch_id=None,
        )
    if observation.observation_type != "phase_transition":
        failed = record_failure(
            runtime,
            stage="observation",
            exception=ValueError(
                "phase transition service requires phase_transition observation"
            ),
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
        )
        return GroundingServiceResult(status="failed", runtime=failed)

    # 阶段切换也走标准 Patch/Reducer，保证原子性和统一审计。
    patch = RequirementGroundingPatch(
        patch_id=f"phase-transition:{observation.observation_id}",
        base_revision=runtime.grounding_revision,
        source_observation_ids=(observation.observation_id,),
        phase_transition=PhaseTransition(
            target_phase=2,
            reason="legal submit_sql transition from phase 1 to phase 2",
            sequence=observation.sequence,
        ),
        diagnostics=("explicit ADK lifecycle transition; no semantic update",),
    )
    try:
        reduced = apply_patch(runtime, patch)
        reduced = increment_metrics(
            reduced,
            observations_seen=1,
            patches_applied=1,
        )
        reduced = set_last_error(reduced, None)
        return GroundingServiceResult(
            status="processed",
            runtime=reduced,
            patch_id=patch.patch_id,
        )
    except Exception as exc:
        failed = record_failure(
            runtime,
            stage="reducer",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
        )
        return GroundingServiceResult(
            status="failed",
            runtime=failed,
            patch_id=patch.patch_id,
            failure_reason="reducer_rejected",
        )


def process_observation(
    runtime: RequirementGroundingRuntime,
    observation: Observation,
    *,
    updater: Updater | None = None,
    reducer: Reducer = apply_patch,
) -> GroundingServiceResult:
    """同步处理一条 Observation；失败时返回仍然有效的旧业务状态。"""

    if observation.observation_id in runtime.processed_observation_ids:
        return GroundingServiceResult(
            status="duplicate",
            runtime=runtime,
            patch_id=None,
        )

    selected_updater = updater or NoOpUpdater()
    # Updater 和 Reducer 只拿到深拷贝；即使扩展代码先修改后报错，也污染不了旧状态。
    working = RequirementGroundingRuntime.model_validate_json(
        runtime.model_dump_json()
    )
    counted = increment_metrics(
        runtime,
        observations_seen=1,
        updater_calls=1,
    )
    try:
        # Updater 只提建议，是否真正落地由下一步 Reducer 决定。
        patch = selected_updater.propose(
            observation,
            working.grounding_state,
            base_revision=working.grounding_revision,
        )
    except Exception as exc:
        failed = record_failure(
            counted,
            stage="updater",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
        )
        return GroundingServiceResult(
            status="failed",
            runtime=failed,
            failure_reason="unexpected_error",
        )

    try:
        reduced = reducer(working, patch)
        reduced = increment_metrics(
            reduced,
            observations_seen=1,
            updater_calls=1,
            patches_applied=1,
        )
        reduced = set_last_error(reduced, None)
        return GroundingServiceResult(
            status="processed",
            runtime=reduced,
            patch_id=patch.patch_id,
        )
    except Exception as exc:
        failed = record_failure(
            counted,
            stage="reducer",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
        )
        return GroundingServiceResult(
            status="failed",
            runtime=failed,
            patch_id=patch.patch_id,
            failure_reason="reducer_rejected",
        )


async def process_observation_with_llm(
    runtime: RequirementGroundingRuntime,
    observation: Observation,
    *,
    updater: LLMUpdater,
    reducer: Reducer = apply_patch,
    monotonic: Callable[[], float] = time.monotonic,
) -> LLMGroundingServiceResult:
    """执行一次异步 LLM 更新；超时或异常都不破坏原业务状态。"""

    config = updater.config
    if observation.observation_id in runtime.processed_observation_ids:
        return LLMGroundingServiceResult(
            status="duplicate",
            runtime=runtime,
            llm_audit=_llm_audit(
                updater,
                status="duplicate",
                attempted=False,
            ),
        )

    if observation.observation_type not in GROUNDING_LLM_OBSERVATION_TYPES:
        # 当前只允许用户问题和用户回答触发 LLM，工具结果直接跳过。
        return LLMGroundingServiceResult(
            status="skipped",
            runtime=runtime,
            llm_audit=_llm_audit(
                updater,
                status="ineligible",
                attempted=False,
            ),
        )

    calls_so_far = runtime.metrics.root.get("llm_updater_calls", 0)
    if (
        isinstance(calls_so_far, bool)
        or not isinstance(calls_so_far, (int, float))
        or calls_so_far >= config.max_calls_per_task
    ):
        # 每题调用次数有硬上限，避免失控调用和额外费用。
        counted = increment_metrics(runtime, observations_seen=1)
        error = RuntimeError("Grounding LLM call limit reached")
        failed = record_failure(
            counted,
            stage="updater",
            exception=error,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
            metric_namespace="llm",
        )
        return LLMGroundingServiceResult(
            status="failed",
            runtime=failed,
            failure_reason="configuration_error",
            llm_audit=_llm_audit(
                updater,
                status="limit_rejected",
                attempted=False,
                error=error,
            ),
        )

    # 和同步路径一样，模型及 Reducer 都只处理与旧状态隔离的副本。
    working = RequirementGroundingRuntime.model_validate_json(
        runtime.model_dump_json()
    )
    recorder = LLMCallTelemetryRecorder()
    started = monotonic()
    counted = increment_metrics(runtime, observations_seen=1)
    try:
        patch = await asyncio.wait_for(
            updater.propose(
                observation,
                working.grounding_state,
                base_revision=working.grounding_revision,
                telemetry=recorder,
            ),
            timeout=config.timeout_seconds,
        )
    except TimeoutError as exc:
        # 本地取消不代表 Provider 一定停止执行，因此审计会标出潜在继续计费。
        latency_ms = _elapsed_ms(monotonic, started)
        failed = increment_llm_call_metrics(
            counted,
            recorder,
            calls=int(recorder.provider_attempted),
            latency_ms=(latency_ms if recorder.provider_attempted else 0.0),
            timeouts=1,
        )
        failed = record_failure(
            failed,
            stage="updater",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
            retryable=True,
            metric_namespace="llm",
        )
        return LLMGroundingServiceResult(
            status="failed",
            runtime=failed,
            failure_reason="timeout",
            llm_audit=_llm_audit(
                updater,
                status="timed_out",
                attempted=recorder.provider_attempted,
                recorder=recorder,
                latency_ms=latency_ms,
                timed_out=True,
                error=exc,
            ),
        )
    except Exception as exc:
        latency_ms = _elapsed_ms(monotonic, started)
        failure_reason = _llm_failure_reason(exc)
        failed = increment_llm_call_metrics(
            counted,
            recorder,
            calls=int(recorder.provider_attempted),
            latency_ms=(latency_ms if recorder.provider_attempted else 0.0),
        )
        failed = record_failure(
            failed,
            stage="updater",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
            metric_namespace="llm",
        )
        return LLMGroundingServiceResult(
            status="failed",
            runtime=failed,
            failure_reason=failure_reason,
            llm_audit=_llm_audit(
                updater,
                status="failed",
                attempted=recorder.provider_attempted,
                recorder=recorder,
                latency_ms=latency_ms,
                error=exc,
            ),
        )

    latency_ms = _elapsed_ms(monotonic, started)
    try:
        # 只有 Provider 返回、响应解析和 Reducer 全部成功，Patch 才会落地。
        reduced = reducer(working, patch)
        reduced = increment_metrics(
            reduced,
            observations_seen=1,
            patches_applied=1,
        )
        reduced = increment_llm_call_metrics(
            reduced,
            recorder,
            calls=int(recorder.provider_attempted),
            latency_ms=latency_ms,
        )
        reduced = set_last_error(reduced, None)
        return LLMGroundingServiceResult(
            status="processed",
            runtime=reduced,
            patch_id=patch.patch_id,
            llm_audit=_llm_audit(
                updater,
                status="succeeded",
                attempted=recorder.provider_attempted,
                recorder=recorder,
                latency_ms=latency_ms,
            ),
        )
    except Exception as exc:
        failed = increment_llm_call_metrics(
            counted,
            recorder,
            calls=int(recorder.provider_attempted),
            latency_ms=latency_ms,
        )
        failed = record_failure(
            failed,
            stage="reducer",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
            metric_namespace="llm",
        )
        return LLMGroundingServiceResult(
            status="failed",
            runtime=failed,
            patch_id=patch.patch_id,
            failure_reason="reducer_rejected",
            llm_audit=_llm_audit(
                updater,
                status="failed",
                attempted=recorder.provider_attempted,
                recorder=recorder,
                latency_ms=latency_ms,
                error=exc,
            ),
        )


def _llm_failure_reason(exc: BaseException) -> FrameInitializationReason:
    """按异常类型映射有限原因；禁止从错误消息文本猜测。"""

    if isinstance(exc, GroundingProviderError):
        return "provider_error"
    if isinstance(exc, LLMFrameUpdateError):
        return exc.reason
    return "unexpected_error"


def _elapsed_ms(monotonic: Callable[[], float], started: float) -> float:
    """安全计算非负毫秒耗时；异常时按 0 处理。"""

    try:
        elapsed = monotonic() - started
    except Exception:
        return 0.0
    if not math.isfinite(elapsed):
        return 0.0
    return max(0.0, elapsed * 1000.0)


def _llm_audit(
    updater: LLMUpdater,
    *,
    status: str,
    attempted: bool,
    recorder: LLMCallTelemetryRecorder | None = None,
    latency_ms: float = 0.0,
    timed_out: bool = False,
    error: BaseException | None = None,
) -> LLMCallAudit:
    """把单次调用记录器整理成有限、可序列化的审计结果。"""

    usage = recorder or LLMCallTelemetryRecorder()
    timeout_relevant = timed_out and attempted
    return LLMCallAudit(
        status=status,
        attempted=attempted,
        configuration_sha256=updater.config.configuration_sha256,
        prompt_sha256=updater.config.prompt_sha256,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        total_tokens=usage.total_tokens,
        latency_ms=latency_ms,
        cost=usage.cost,
        model=usage.model,
        provider=usage.provider,
        credential_source=usage.credential_source,
        request_sha256=usage.request_sha256,
        response_sha256=usage.response_sha256,
        raw_audit_ref=usage.raw_audit_ref,
        timed_out=timed_out,
        provider_may_continue_after_cancel=(
            _declared_client_bool(
                updater.client,
                "provider_may_continue_after_cancel",
            )
            if timeout_relevant
            else None
        ),
        provider_may_bill_after_cancel=(
            _declared_client_bool(
                updater.client,
                "provider_may_bill_after_cancel",
            )
            if timeout_relevant
            else None
        ),
        error_type=(type(error).__name__[:128] if error is not None else None),
    )


def _declared_client_bool(client: object, name: str) -> bool | None:
    """只接受客户端明确声明的布尔能力。"""

    value = getattr(client, name, None)
    return value if isinstance(value, bool) else None


def _validated_runtime_copy(
    runtime: RequirementGroundingRuntime,
    **updates: object,
) -> RequirementGroundingRuntime:
    """构造并校验 Runtime 副本。"""

    data = runtime.model_dump(mode="python")
    data.update(updates)
    return RequirementGroundingRuntime.model_validate(data)


def _assert_revision_unchanged(
    before: RequirementGroundingRuntime,
    after: RequirementGroundingRuntime,
) -> None:
    """确保控制状态和遥测不会偷偷增加业务 revision。"""

    if before.grounding_revision != after.grounding_revision:
        raise AssertionError("runtime-control changes cannot increment revision")
    if before.requirement_revision != after.requirement_revision:
        raise AssertionError(
            "runtime-control changes cannot increment requirement revision"
        )
