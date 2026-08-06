"""Pure diagnostic updates that never advance the grounding revision."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field

from valibra_agent.requirement_grounding.models import (
    KernelModel,
    MAX_ERROR_CHARS,
    RuntimeMetrics,
    ValibraError,
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.reducer import validate_runtime

_REDACTIONS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\b(api[_ -]?key|token|password)\s*[:=]\s*\S+"),
)


class LLMCallTelemetryRecorder:
    """Per-call mutable usage sink; it is never stored in business State."""

    __slots__ = (
        "provider_attempted",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cost",
    )

    def __init__(self) -> None:
        self.provider_attempted = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cost = 0.0

    def mark_provider_attempted(self) -> None:
        self.provider_attempted = True

    def capture_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        cost: float,
    ) -> None:
        values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost": cost,
        }
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"LLM usage {key} must be numeric")
            if value < 0:
                raise ValueError(f"LLM usage {key} must be non-negative")
        if any(isinstance(value, float) and not _is_finite(value) for value in values.values()):
            raise ValueError("LLM usage values must be finite")
        self.input_tokens = int(input_tokens)
        self.output_tokens = int(output_tokens)
        self.reasoning_tokens = int(reasoning_tokens)
        self.cost = float(cost)


class LLMCallAudit(KernelModel):
    """Bounded per-call audit returned by the offline async service path."""

    status: Literal[
        "succeeded",
        "failed",
        "timed_out",
        "limit_rejected",
        "duplicate",
    ]
    attempted: bool
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost: float = Field(default=0.0, ge=0.0)
    timed_out: bool = False
    provider_may_continue_after_cancel: bool | None = None
    provider_may_bill_after_cancel: bool | None = None
    error_type: str | None = Field(default=None, max_length=128)


def increment_metrics(
    runtime: RequirementGroundingRuntime,
    **deltas: int | float,
) -> RequirementGroundingRuntime:
    """Return a validated runtime with diagnostic counters incremented."""

    values = dict(runtime.metrics.root)
    for key, delta in deltas.items():
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise ValueError(f"metric delta must be numeric: {key}")
        values[key] = values.get(key, 0) + delta
    metrics = RuntimeMetrics.model_validate(values)
    result = _validated_runtime_copy(runtime, metrics=metrics)
    _assert_revision_unchanged(runtime, result)
    validate_runtime(result)
    return result


def increment_llm_call_metrics(
    runtime: RequirementGroundingRuntime,
    recorder: LLMCallTelemetryRecorder,
    *,
    calls: int,
    latency_ms: float,
    timeouts: int = 0,
) -> RequirementGroundingRuntime:
    """Record LLM-Updater-only aggregate usage without changing revision."""

    return increment_metrics(
        runtime,
        llm_updater_calls=calls,
        llm_updater_input_tokens=recorder.input_tokens,
        llm_updater_output_tokens=recorder.output_tokens,
        llm_updater_reasoning_tokens=recorder.reasoning_tokens,
        llm_updater_latency_ms=latency_ms,
        llm_updater_timeouts=timeouts,
        llm_updater_cost=recorder.cost,
    )


def set_last_error(
    runtime: RequirementGroundingRuntime,
    error: ValibraError | None,
) -> RequirementGroundingRuntime:
    result = _validated_runtime_copy(runtime, last_error=error)
    _assert_revision_unchanged(runtime, result)
    validate_runtime(result)
    return result


def record_failure(
    runtime: RequirementGroundingRuntime,
    *,
    stage: str,
    exception: BaseException,
    sequence: int,
    observation_id: str | None = None,
    function_call_id: str | None = None,
    retryable: bool = False,
    timestamp: str | None = None,
    metric_namespace: Literal["generic", "llm"] = "generic",
) -> RequirementGroundingRuntime:
    error = ValibraError(
        stage=stage,
        error_type=type(exception).__name__[:128] or "Error",
        message_preview=_bounded_redacted_message(exception),
        retryable=retryable,
        observation_id=observation_id,
        function_call_id=function_call_id,
        sequence=sequence,
        timestamp=timestamp,
    )
    result = set_last_error(runtime, error)
    if stage == "updater":
        metric = (
            "llm_updater_errors"
            if metric_namespace == "llm"
            else "updater_errors"
        )
        result = increment_metrics(result, **{metric: 1})
    elif stage == "reducer":
        result = increment_metrics(result, patches_rejected=1)
        if metric_namespace == "llm":
            result = increment_metrics(result, llm_updater_errors=1)
    return result


def _bounded_redacted_message(exception: BaseException) -> str:
    message = re.sub(r"\s+", " ", str(exception)).strip()
    for pattern in _REDACTIONS:
        message = pattern.sub("<redacted>", message)
    if len(message) > MAX_ERROR_CHARS:
        return message[: MAX_ERROR_CHARS - 1] + "…"
    return message


def _validated_runtime_copy(
    runtime: RequirementGroundingRuntime,
    **updates: Any,
) -> RequirementGroundingRuntime:
    data = runtime.model_dump(mode="python")
    data.update(updates)
    return RequirementGroundingRuntime.model_validate(data)


def _assert_revision_unchanged(
    before: RequirementGroundingRuntime,
    after: RequirementGroundingRuntime,
) -> None:
    if before.grounding_revision != after.grounding_revision:
        raise AssertionError("telemetry cannot change grounding_revision")


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
