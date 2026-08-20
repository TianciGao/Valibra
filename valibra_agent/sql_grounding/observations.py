"""Bounded, deterministic observations for SQL Grounding V1.

Observations describe only data already visible in the official BIRD session.
They are transient updater inputs, not an Evidence store and not part of the
persisted :class:`GroundingRuntime`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from valibra_agent.sql_grounding.models import ContractModel, canonical_json

ObservationType: TypeAlias = Literal[
    "user_query",
    "p2_follow_up",
    "schema",
    "metadata",
    "knowledge",
    "user_answer",
    "sql_execution",
    "submission",
    "tool_error",
]

MAX_OBSERVATION_CONTENT_BYTES = 65_536
MAX_OBSERVATION_SUMMARY_CHARS = 2_048
MAX_OBSERVATION_REF_CHARS = 1_024
MAX_OBSERVATION_IDENTITY_CHARS = 256

_TOOL_OBSERVATION_TYPES = frozenset(
    {
        "schema",
        "metadata",
        "knowledge",
        "user_answer",
        "sql_execution",
        "submission",
        "tool_error",
    }
)


class SQLGroundingObservation(ContractModel):
    """One normalized input that the SQL Grounding updater may inspect."""

    observation_id: str = Field(
        min_length=1,
        max_length=MAX_OBSERVATION_IDENTITY_CHARS,
        pattern=r"^sqlgobs_[0-9a-f]{32}$",
    )
    task_id: str = Field(min_length=1, max_length=MAX_OBSERVATION_IDENTITY_CHARS)
    phase: Literal[1, 2]
    sequence: int = Field(ge=1)
    observation_type: ObservationType
    tool_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_OBSERVATION_IDENTITY_CHARS,
    )
    function_call_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_OBSERVATION_IDENTITY_CHARS,
    )
    content: Any
    summary: str = Field(min_length=1, max_length=MAX_OBSERVATION_SUMMARY_CHARS)
    raw_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    private_raw_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_OBSERVATION_REF_CHARS,
    )

    @field_validator(
        "task_id",
        "tool_name",
        "function_call_id",
        "summary",
        "private_raw_ref",
    )
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("Observation text fields cannot have outer whitespace")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: Any) -> Any:
        try:
            encoded_text = canonical_json(value)
            encoded = encoded_text.encode("utf-8")
            normalized = json.loads(encoded_text)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("Observation content must be finite JSON data") from exc
        if len(encoded) > MAX_OBSERVATION_CONTENT_BYTES:
            raise ValueError("Observation content exceeds the bounded input limit")
        return normalized

    @model_validator(mode="after")
    def validate_identity_and_tool_pairing(self) -> "SQLGroundingObservation":
        is_tool_observation = self.observation_type in _TOOL_OBSERVATION_TYPES
        has_both_tool_fields = self.tool_name is not None and self.function_call_id is not None
        has_any_tool_field = self.tool_name is not None or self.function_call_id is not None
        if (is_tool_observation and not has_both_tool_fields) or (
            not is_tool_observation and has_any_tool_field
        ):
            raise ValueError(
                "tool observations require tool_name and function_call_id; "
                "user-query observations forbid them"
            )
        expected_digest = _content_digest(self.content)
        if self.raw_digest != expected_digest:
            raise ValueError("raw_digest does not match Observation content")
        if self.observation_id != _observation_id(
            task_id=self.task_id,
            phase=self.phase,
            sequence=self.sequence,
            observation_type=self.observation_type,
            tool_name=self.tool_name,
            function_call_id=self.function_call_id,
            raw_digest=self.raw_digest,
        ):
            raise ValueError("observation_id does not match normalized identity")
        return self


def build_sql_grounding_observation(
    *,
    task_id: str,
    phase: Literal[1, 2],
    sequence: int,
    observation_type: ObservationType,
    content: Any,
    summary: str,
    tool_name: str | None = None,
    function_call_id: str | None = None,
    private_raw_ref: str | None = None,
) -> SQLGroundingObservation:
    """Normalize caller-provided legal input into a stable Observation."""

    raw_digest = _content_digest(content)
    observation_id = _observation_id(
        task_id=task_id,
        phase=phase,
        sequence=sequence,
        observation_type=observation_type,
        tool_name=tool_name,
        function_call_id=function_call_id,
        raw_digest=raw_digest,
    )
    return SQLGroundingObservation(
        observation_id=observation_id,
        task_id=task_id,
        phase=phase,
        sequence=sequence,
        observation_type=observation_type,
        tool_name=tool_name,
        function_call_id=function_call_id,
        content=content,
        summary=summary,
        raw_digest=raw_digest,
        private_raw_ref=private_raw_ref,
    )


def observation_for_updater(observation: SQLGroundingObservation) -> dict[str, Any]:
    """Return the bounded model-facing projection, excluding private raw refs."""

    return {
        "content": observation.content,
        "function_call_id": observation.function_call_id,
        "observation_id": observation.observation_id,
        "observation_type": observation.observation_type,
        "phase": observation.phase,
        "raw_digest": observation.raw_digest,
        "sequence": observation.sequence,
        "summary": observation.summary,
        "tool_name": observation.tool_name,
    }


def _content_digest(content: Any) -> str:
    try:
        encoded = canonical_json(content).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Observation content must be finite JSON data") from exc
    if len(encoded) > MAX_OBSERVATION_CONTENT_BYTES:
        raise ValueError("Observation content exceeds the bounded input limit")
    return hashlib.sha256(encoded).hexdigest()


def _observation_id(
    *,
    task_id: str,
    phase: int,
    sequence: int,
    observation_type: str,
    tool_name: str | None,
    function_call_id: str | None,
    raw_digest: str,
) -> str:
    identity = json.dumps(
        {
            "function_call_id": function_call_id,
            "observation_type": observation_type,
            "phase": phase,
            "raw_digest": raw_digest,
            "sequence": sequence,
            "task_id": task_id,
            "tool_name": tool_name,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sqlgobs_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
