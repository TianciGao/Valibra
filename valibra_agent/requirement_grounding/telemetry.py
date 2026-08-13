"""只更新诊断和用量数据，绝不推进 Grounding 业务 revision。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

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
    """单次 LLM 调用的临时记录器，不属于业务 State。"""

    __slots__ = (
        "provider_attempted",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost",
        "model",
        "provider",
        "credential_source",
        "request_sha256",
        "response_sha256",
        "raw_audit_ref",
    )

    def __init__(self) -> None:
        self.provider_attempted = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.total_tokens = 0
        self.cost: float | None = None
        self.model = ""
        self.provider = ""
        self.credential_source = ""
        self.request_sha256 = ""
        self.response_sha256 = ""
        self.raw_audit_ref = ""

    def mark_provider_attempted(self) -> None:
        """标记请求已经越过本地边界，可能产生 Provider 费用。"""

        self.provider_attempted = True

    def capture_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
        cost: float | None,
    ) -> None:
        """校验并记录本次 Token 和费用。"""

        values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"LLM usage {key} must be numeric")
            if value < 0:
                raise ValueError(f"LLM usage {key} must be non-negative")
        if any(
            isinstance(value, float) and not _is_finite(value)
            for value in values.values()
        ):
            raise ValueError("LLM usage values must be finite")
        if cost is not None:
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                raise ValueError("LLM usage cost must be numeric or null")
            if cost < 0 or not _is_finite(float(cost)):
                raise ValueError("LLM usage cost must be finite and non-negative")
        self.input_tokens = int(input_tokens)
        self.output_tokens = int(output_tokens)
        self.reasoning_tokens = int(reasoning_tokens)
        self.total_tokens = int(total_tokens)
        self.cost = float(cost) if cost is not None else None

    def capture_provider_audit(
        self,
        *,
        model: str,
        provider: str,
        credential_source: str,
        request_sha256: str,
        response_sha256: str,
        raw_audit_ref: str,
    ) -> None:
        """记录不含密钥的 Provider 与原始审计引用。"""

        self.model = model
        self.provider = provider
        self.credential_source = credential_source
        self.request_sha256 = request_sha256
        self.response_sha256 = response_sha256
        self.raw_audit_ref = raw_audit_ref


class LLMCallAudit(KernelModel):
    """异步 LLM 路径返回的单次有限审计。"""

    status: Literal[
        "succeeded",
        "failed",
        "timed_out",
        "limit_rejected",
        "duplicate",
        "ineligible",
    ]
    attempted: bool
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost: float | None = Field(default=None, ge=0.0)
    model: str = Field(default="", max_length=256)
    provider: str = Field(default="", max_length=128)
    credential_source: Literal["", "direct", "file"] = ""
    request_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    response_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    raw_audit_ref: str = Field(default="", max_length=1024)
    timed_out: bool = False
    provider_may_continue_after_cancel: bool | None = None
    provider_may_bill_after_cancel: bool | None = None
    error_type: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_finite_values(self) -> "LLMCallAudit":
        if not _is_finite(self.latency_ms):
            raise ValueError("LLM audit latency must be finite")
        if self.cost is not None and not _is_finite(self.cost):
            raise ValueError("LLM audit cost must be finite")
        return self


class ModelUsageLedger(KernelModel):
    """一个模型角色的 Token/费用账本；不混入 bird-coin。"""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_finite_cost(self) -> "ModelUsageLedger":
        if self.cost is not None and not _is_finite(self.cost):
            raise ValueError("model usage cost must be finite")
        return self


class CombinedModelUsage(KernelModel):
    """只读汇总主 Agent 与 Grounding 两套模型用量。"""

    main_agent: ModelUsageLedger
    grounding: ModelUsageLedger
    total_model_tokens: int = Field(ge=0)
    total_model_cost: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_totals(self) -> "CombinedModelUsage":
        expected_tokens = self.main_agent.total_tokens + self.grounding.total_tokens
        if self.total_model_tokens != expected_tokens:
            raise ValueError("total_model_tokens does not match role ledgers")
        if self.total_model_cost is not None:
            if not _is_finite(self.total_model_cost):
                raise ValueError("total model cost must be finite")
            if self.main_agent.cost is None or self.grounding.cost is None:
                raise ValueError("total model cost requires both role costs")
            expected_cost = self.main_agent.cost + self.grounding.cost
            if self.total_model_cost != expected_cost:
                raise ValueError("total_model_cost does not match role ledgers")
        return self


def increment_metrics(
    runtime: RequirementGroundingRuntime,
    **deltas: int | float,
) -> RequirementGroundingRuntime:
    """累加诊断指标并返回已校验的新 Runtime。"""

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
    """汇总 LLM Updater 用量，不改变业务 revision。"""

    deltas: dict[str, int | float] = {
        "llm_updater_calls": calls,
        "llm_updater_input_tokens": recorder.input_tokens,
        "llm_updater_output_tokens": recorder.output_tokens,
        "llm_updater_reasoning_tokens": recorder.reasoning_tokens,
        "llm_updater_total_tokens": recorder.total_tokens,
        "llm_updater_latency_ms": latency_ms,
        "llm_updater_timeouts": timeouts,
    }
    if recorder.cost is not None:
        deltas["llm_updater_cost"] = recorder.cost
    return increment_metrics(runtime, **deltas)


def summarize_model_usage_totals(
    main_agent_usage: Mapping[str, Any],
    runtime: RequirementGroundingRuntime,
    *,
    main_agent_cost: float | None = None,
) -> CombinedModelUsage:
    """合并两套模型账本，但不改 Baseline 原有用量格式。"""

    main_input = _usage_int(main_agent_usage, "input_tokens")
    main_output = _usage_int(main_agent_usage, "output_tokens")
    main_reasoning = _usage_int(main_agent_usage, "reasoning_tokens")
    main_total = _usage_int(
        main_agent_usage,
        "total_tokens",
        default=main_input + main_output,
    )
    _validate_optional_cost(main_agent_cost, "main_agent_cost")
    main_calls = _usage_int(main_agent_usage, "model_calls")
    resolved_main_cost = (
        0.0
        if main_calls == 0 and main_agent_cost is None
        else main_agent_cost
    )

    metrics = runtime.metrics.root
    grounding_input = _metric_int(metrics, "llm_updater_input_tokens")
    grounding_output = _metric_int(metrics, "llm_updater_output_tokens")
    grounding_reasoning = _metric_int(metrics, "llm_updater_reasoning_tokens")
    grounding_total = _metric_int(
        metrics,
        "llm_updater_total_tokens",
        default=grounding_input + grounding_output,
    )
    grounding_calls = _metric_int(metrics, "llm_updater_calls")
    raw_grounding_cost = metrics.get("llm_updater_cost")
    if raw_grounding_cost is None:
        grounding_cost = 0.0 if grounding_calls == 0 else None
    else:
        _validate_optional_cost(raw_grounding_cost, "llm_updater_cost")
        grounding_cost = float(raw_grounding_cost)

    total_cost = (
        resolved_main_cost + grounding_cost
        if resolved_main_cost is not None and grounding_cost is not None
        else None
    )
    return CombinedModelUsage(
        main_agent=ModelUsageLedger(
            input_tokens=main_input,
            output_tokens=main_output,
            reasoning_tokens=main_reasoning,
            total_tokens=main_total,
            cost=resolved_main_cost,
        ),
        grounding=ModelUsageLedger(
            input_tokens=grounding_input,
            output_tokens=grounding_output,
            reasoning_tokens=grounding_reasoning,
            total_tokens=grounding_total,
            cost=grounding_cost,
        ),
        total_model_tokens=main_total + grounding_total,
        total_model_cost=total_cost,
    )


def _usage_int(
    usage: Mapping[str, Any],
    key: str,
    *,
    default: int = 0,
) -> int:
    value = usage.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"main model usage {key} must be numeric")
    if not _is_finite(float(value)) or value < 0 or int(value) != value:
        raise ValueError(f"main model usage {key} must be a non-negative integer")
    return int(value)


def _metric_int(
    metrics: Mapping[str, Any],
    key: str,
    *,
    default: int = 0,
) -> int:
    value = metrics.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Grounding metric {key} must be numeric")
    if not _is_finite(float(value)) or value < 0 or int(value) != value:
        raise ValueError(f"Grounding metric {key} must be a non-negative integer")
    return int(value)


def _validate_optional_cost(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric or null")
    if value < 0 or not _is_finite(float(value)):
        raise ValueError(f"{name} must be finite and non-negative")


def set_last_error(
    runtime: RequirementGroundingRuntime,
    error: ValibraError | None,
) -> RequirementGroundingRuntime:
    """设置或清空最近错误，不改变业务状态。"""

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
    """记录一条脱敏错误，并按阶段累加失败指标。"""

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
    """压平并截断错误信息，同时遮掉常见密钥格式。"""

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
    """构造并校验 Runtime 副本。"""

    data = runtime.model_dump(mode="python")
    data.update(updates)
    return RequirementGroundingRuntime.model_validate(data)


def _assert_revision_unchanged(
    before: RequirementGroundingRuntime,
    after: RequirementGroundingRuntime,
) -> None:
    """确保遥测更新没有改变业务 revision。"""

    if before.grounding_revision != after.grounding_revision:
        raise AssertionError("telemetry cannot change grounding_revision")
    if before.requirement_revision != after.requirement_revision:
        raise AssertionError("telemetry cannot change requirement_revision")


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
