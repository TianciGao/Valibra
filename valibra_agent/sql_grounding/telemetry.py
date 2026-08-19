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


class GroundingLLMTelemetry(ContractModel):
    attempted: bool
    status: Literal["succeeded", "failed", "timed_out", "rejected"]
    usage: GroundingTokenUsage = Field(default_factory=GroundingTokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    timed_out: bool = False
    error_type: str | None = Field(default=None, max_length=128)

    request_sha256: str = Field(pattern=r"^(?:|[0-9a-f]{64})$")
    response_sha256: str = Field(pattern=r"^(?:|[0-9a-f]{64})$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    form_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_shape(self) -> "GroundingLLMTelemetry":
        if not math.isfinite(self.latency_ms):
            raise ValueError("latency must be finite")
        if self.timed_out != (self.status == "timed_out"):
            raise ValueError("timed_out must match telemetry status")
        if self.status == "succeeded" and self.error_type is not None:
            raise ValueError("successful telemetry cannot carry an error")
        if not self.attempted and any(
            (
                self.usage.total_tokens,
                self.latency_ms,
                bool(self.response_sha256),
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
