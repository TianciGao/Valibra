"""Bounded SQL Grounding telemetry kept outside persisted Runtime."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from valibra_agent.sql_grounding.models import (
    ContractModel,
    FocusDimension,
    GroundingDimension,
    GroundingStage,
)


class GroundingTokenUsage(ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> "GroundingTokenUsage":
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens are a subdivision of output tokens")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total tokens must equal input plus output tokens")
        return self


class GroundingProviderAttemptTelemetry(ContractModel):
    """One real Provider execution inside one logical Grounding call."""

    attempt_number: int = Field(ge=1, le=2)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    finish_reason: str | None = Field(default=None, max_length=128)
    usage: GroundingTokenUsage = Field(default_factory=GroundingTokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    content_empty: bool | None = None
    status: Literal["response", "invalid_response", "failed", "timed_out"]
    error_type: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_attempt(self) -> "GroundingProviderAttemptTelemetry":
        if not math.isfinite(self.latency_ms):
            raise ValueError("attempt latency must be finite")
        if self.status == "response":
            if self.content_empty is None or self.error_type is not None:
                raise ValueError("a Provider response requires content visibility")
        elif self.status == "invalid_response":
            if self.content_empty is not None or not self.error_type:
                raise ValueError("an invalid Provider response requires an error type")
        else:
            if self.content_empty is not None or self.finish_reason is not None:
                raise ValueError("a failed Provider attempt cannot invent a response")
            if not self.error_type:
                raise ValueError("a failed Provider attempt requires an error type")
        return self


class GroundingLLMTelemetry(ContractModel):
    attempted: bool
    status: Literal["succeeded", "failed", "timed_out", "rejected"]
    usage: GroundingTokenUsage = Field(default_factory=GroundingTokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    timed_out: bool = False
    error_type: str | None = Field(default=None, max_length=128)

    provider_reported_cost: float | None = Field(default=None, ge=0.0)
    model: str = Field(default="", max_length=256)
    provider: str = Field(default="", max_length=128)
    credential_source: Literal["", "direct", "file"] = ""
    raw_private_audit_ref: str = Field(default="", max_length=1024)
    provider_may_continue_after_cancel: bool | None = None
    provider_may_bill_after_cancel: bool | None = None

    configured_max_tokens: int = Field(default=0, ge=0, le=131_072)
    attempt_count: int = Field(default=0, ge=0, le=2)
    retry_triggered: bool = False
    retry_trigger_reason: str | None = Field(default=None, max_length=128)
    provider_attempts: tuple[GroundingProviderAttemptTelemetry, ...] = ()
    final_selected_attempt: int | None = Field(default=None, ge=1, le=2)

    request_sha256: str = Field(pattern=r"^(?:|[0-9a-f]{64})$")
    response_sha256: str = Field(pattern=r"^(?:|[0-9a-f]{64})$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    form_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_shape(self) -> "GroundingLLMTelemetry":
        if not math.isfinite(self.latency_ms):
            raise ValueError("latency must be finite")
        if self.provider_reported_cost is not None and not math.isfinite(
            self.provider_reported_cost
        ):
            raise ValueError("provider reported cost must be finite")
        if self.timed_out != (self.status == "timed_out"):
            raise ValueError("timed_out must match telemetry status")
        if self.status == "succeeded" and self.error_type is not None:
            raise ValueError("successful telemetry cannot carry an error")
        if self.provider_attempts:
            if self.attempt_count != len(self.provider_attempts):
                raise ValueError("attempt_count must match Provider attempts")
            expected_numbers = tuple(range(1, self.attempt_count + 1))
            if (
                tuple(item.attempt_number for item in self.provider_attempts)
                != expected_numbers
            ):
                raise ValueError("Provider attempts must be contiguous and ordered")
            request_shas = {item.request_sha256 for item in self.provider_attempts}
            if len(request_shas) != 1 or self.request_sha256 not in request_shas:
                raise ValueError("all Provider attempts must use the same request SHA")
            if self.configured_max_tokens <= 0:
                raise ValueError("Provider attempts require configured_max_tokens")
        elif self.attempt_count:
            raise ValueError("attempt_count requires Provider attempt telemetry")
        if self.attempt_count == 2 and not self.retry_triggered:
            raise ValueError("a second Provider attempt requires a retry trigger")
        if self.retry_triggered and self.attempt_count < 1:
            raise ValueError("a retry trigger requires a completed first attempt")
        if self.retry_triggered != (self.retry_trigger_reason is not None):
            raise ValueError("retry trigger reason must match retry_triggered")
        if self.final_selected_attempt is not None and (
            not self.provider_attempts
            or self.final_selected_attempt > self.attempt_count
        ):
            raise ValueError("final selected attempt must reference a real attempt")
        if not self.attempted and any(
            (
                self.usage.total_tokens,
                self.latency_ms,
                bool(self.response_sha256),
                self.provider_reported_cost is not None,
                bool(self.model),
                bool(self.provider),
                bool(self.credential_source),
                bool(self.raw_private_audit_ref),
                self.attempt_count,
                self.retry_triggered,
                bool(self.retry_trigger_reason),
                bool(self.provider_attempts),
                self.final_selected_attempt is not None,
            )
        ):
            raise ValueError("a non-attempted call cannot report provider results")
        return self


class StateUpdateTelemetry(ContractModel):
    observation_id: str = Field(min_length=1, max_length=256)
    stage: GroundingStage
    status: Literal["accepted", "rejected", "noop"]
    old_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    new_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_dimensions: tuple[GroundingDimension, ...] = ()
    revision_before: int = Field(ge=0)
    revision_after: int = Field(ge=0)
    focus_before: FocusDimension
    focus_after: FocusDimension
    error_type: str | None = Field(default=None, max_length=128)

    @field_validator("changed_dimensions")
    @classmethod
    def validate_changed_dimensions(
        cls,
        value: tuple[GroundingDimension, ...],
    ) -> tuple[GroundingDimension, ...]:
        if len(value) != len(set(value)):
            raise ValueError("changed_dimensions cannot contain duplicates")
        order = {
            "tables": 0,
            "join_keys": 1,
            "column_mapping": 2,
            "domain_knowledge": 3,
        }
        if tuple(sorted(value, key=order.__getitem__)) != value:
            raise ValueError("changed_dimensions must use canonical dimension order")
        return value

    @model_validator(mode="after")
    def validate_revision_accounting(self) -> "StateUpdateTelemetry":
        expected = self.revision_before + int(bool(self.changed_dimensions))
        if self.status != "rejected" and self.revision_after != expected:
            raise ValueError("accepted revision does not match the State diff")
        if self.status == "rejected" and (
            self.revision_after != self.revision_before
            or self.old_state_sha256 != self.new_state_sha256
            or self.changed_dimensions
            or self.focus_after != self.focus_before
        ):
            raise ValueError("rejected update must preserve the complete Runtime")
        if self.status == "noop" and (
            self.changed_dimensions or self.focus_after != self.focus_before
        ):
            raise ValueError("noop cannot change State or focus")
        if self.status == "accepted" and (
            not self.changed_dimensions and self.focus_after == self.focus_before
        ):
            raise ValueError("accepted update must change State or focus")
        if self.status == "rejected" and self.error_type is None:
            raise ValueError("rejected update requires a bounded error type")
        if self.status != "rejected" and self.error_type is not None:
            raise ValueError("non-rejected update cannot carry an error")
        return self
