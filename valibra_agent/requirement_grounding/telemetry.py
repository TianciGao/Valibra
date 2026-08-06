"""Pure diagnostic updates that never advance the grounding revision."""

from __future__ import annotations

import re
from typing import Any

from valibra_agent.requirement_grounding.models import (
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
        result = increment_metrics(result, updater_errors=1)
    elif stage == "reducer":
        result = increment_metrics(result, patches_rejected=1)
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
