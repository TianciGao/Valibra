"""需求 Grounding 内核的数据结构。

这里只保存有大小上限、可转成 JSON 的状态；不依赖 ADK，不做 I/O，也不直接改 Runtime。
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0"

MAX_ID_CHARS = 128
MAX_MENTION_CHARS = 256
MAX_INTERPRETATION_CHARS = 512
MAX_SUMMARY_CHARS = 512
MAX_REASON_CHARS = 512
MAX_ERROR_CHARS = 512
MAX_ARGS_SUMMARY_BYTES = 4096

Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_ID_CHARS,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Phase = Literal[1, 2]
GroundingStatus = Literal[
    "missing",
    "hypothesized",
    "grounded",
    "confirmed",
    "unanswerable",
]
SlotLifecycle = Literal["active", "superseded"]
AmbiguityStatus = Literal[
    "unresolved",
    "resolved",
    "deferred",
    "unanswerable",
]
ObservationType = Literal[
    "user_query",
    "schema",
    "metadata",
    "knowledge",
    "user_answer",
    "sql_execution",
    "submission",
    "phase_transition",
    "tool_error",
]


def _unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique IDs")
    return values


class KernelModel(BaseModel):
    """所有内核模型的严格基类：禁止多余字段，创建后不可变。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class GroundingEvidence(KernelModel):
    """从一次 Observation 提炼出的有限证据引用。"""

    evidence_id: Identifier
    observation_id: Identifier
    source_type: Annotated[str, Field(min_length=1, max_length=64)]
    phase: Phase
    summary: Annotated[str, Field(max_length=MAX_SUMMARY_CHARS)]
    raw_digest: Digest
    raw_log_ref: Annotated[str, Field(max_length=256)] | None = None
    sequence: Annotated[int, Field(ge=0)]
    timestamp: Annotated[str, Field(max_length=64)] | None = None


class SQLImpact(KernelModel):
    """某种解释会怎样改变 SQL。"""

    effect_key: Identifier
    summary: Annotated[str, Field(min_length=1, max_length=MAX_SUMMARY_CHARS)]


class InterpretationCandidate(KernelModel):
    """一个歧义的候选解释及其 SQL 影响。"""

    candidate_id: Identifier
    interpretation: Annotated[
        str,
        Field(min_length=1, max_length=MAX_INTERPRETATION_CHARS),
    ]
    sql_impact: SQLImpact
    evidence_refs: tuple[Identifier, ...] = ()
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, "candidate evidence_refs")


class SlotBase(KernelModel):
    """需求槽位的公共字段：原词、当前解释、证据和生命周期。"""

    slot_id: Identifier
    slot_kind: str
    slot_role: Annotated[str, Field(min_length=1, max_length=64)]
    mention: Annotated[str, Field(max_length=MAX_MENTION_CHARS)] = ""
    current_interpretation: Annotated[
        str,
        Field(max_length=MAX_INTERPRETATION_CHARS),
    ] | None = None
    grounding_status: GroundingStatus = "missing"
    evidence_refs: tuple[Identifier, ...] = ()
    ambiguity_refs: tuple[Identifier, ...] = ()
    origin: Annotated[str, Field(min_length=1, max_length=64)]
    lifecycle: SlotLifecycle = "active"
    introduced_in_phase: Phase = 1
    last_updated_phase: Phase = 1
    sequence: Annotated[int, Field(ge=0)]

    @field_validator("evidence_refs", "ambiguity_refs")
    @classmethod
    def validate_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, "slot references")

    @model_validator(mode="after")
    def validate_phase_order(self) -> "SlotBase":
        if self.last_updated_phase < self.introduced_in_phase:
            raise ValueError("last_updated_phase cannot precede introduced_in_phase")
        return self


class ValueSlot(SlotBase):
    """值类约束，例如日期、阈值或范围。"""

    slot_kind: Literal["value"] = "value"
    value_type: Annotated[str, Field(max_length=64)] | None = None


class SchemaSlot(SlotBase):
    """可能对应表、列或指标的自然语言概念。"""

    slot_kind: Literal["schema"] = "schema"
    binding_type: Literal[
        "table",
        "column",
        "metric",
        "dimension",
        "join_path",
        "unknown",
    ] = "unknown"
    bound_identifier: Annotated[str, Field(max_length=256)] | None = None


class OperationSlot(SlotBase):
    """查询操作，例如过滤、聚合、排序或限制条数。"""

    slot_kind: Literal["operation"] = "operation"
    operation_type: Literal[
        "projection",
        "filter",
        "aggregation",
        "group",
        "order",
        "limit",
        "distinct",
        "other",
    ]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_bounded_parameters(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        if len(value) > 16:
            raise ValueError("operation parameters exceed item limit")
        for key, item in value.items():
            if len(key) > 64:
                raise ValueError("operation parameter key exceeds length limit")
            if isinstance(item, (dict, list)):
                raise ValueError("operation parameters must be JSON scalars")
            if isinstance(item, str) and len(item) > 256:
                raise ValueError("operation parameter value exceeds length limit")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("operation parameter must be finite")
        return value


GroundingSlot = Annotated[
    Union[ValueSlot, SchemaSlot, OperationSlot],
    Field(discriminator="slot_kind"),
]


class RequirementFrame(KernelModel):
    """当前需求的三类槽位集合。"""

    value_slots: tuple[ValueSlot, ...] = ()
    schema_slots: tuple[SchemaSlot, ...] = ()
    operation_slots: tuple[OperationSlot, ...] = ()


class GroundedAmbiguityHypothesis(KernelModel):
    """一个待确认的歧义，以及候选解释和依赖关系。"""

    ambiguity_id: Identifier
    pivot_term: Annotated[str, Field(min_length=1, max_length=MAX_MENTION_CHARS)]
    primary_slot_id: Identifier
    affected_slot_ids: tuple[Identifier, ...]
    ambiguity_family: Annotated[str, Field(min_length=1, max_length=64)]
    candidate_interpretations: tuple[InterpretationCandidate, ...] = ()
    source_grounding: tuple[Identifier, ...] = ()
    dependency_ids: tuple[Identifier, ...] = ()
    status: AmbiguityStatus = "deferred"
    resolution: Identifier | None = None
    reopen_reason: Annotated[str, Field(max_length=MAX_REASON_CHARS)] | None = None
    sequence: Annotated[int, Field(ge=0)]

    @field_validator("affected_slot_ids", "source_grounding", "dependency_ids")
    @classmethod
    def validate_id_lists(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, "ambiguity references")

    @model_validator(mode="after")
    def validate_local_shape(self) -> "GroundedAmbiguityHypothesis":
        if self.primary_slot_id not in self.affected_slot_ids:
            raise ValueError("primary_slot_id must appear in affected_slot_ids")
        candidate_ids = [item.candidate_id for item in self.candidate_interpretations]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within an ambiguity")
        return self


class RequirementGroundingState(KernelModel):
    """业务状态：需求 Frame、歧义索引和证据。"""

    requirement_frame: RequirementFrame = Field(default_factory=RequirementFrame)
    ambiguity_index: tuple[GroundedAmbiguityHypothesis, ...] = ()
    # 空 evidence 不序列化，以保持默认 Runtime 与冻结规格完全一致。
    evidence: tuple[GroundingEvidence, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )


class Observation(KernelModel):
    """Agent 已合法看到的一条用户消息或工具结果摘要。"""

    observation_id: Digest
    observation_type: ObservationType
    task_id: Identifier
    phase: Phase
    sequence: Annotated[int, Field(ge=0)]
    source: Annotated[str, Field(min_length=1, max_length=64)]
    summary: Annotated[str, Field(max_length=MAX_SUMMARY_CHARS)]
    raw_digest: Digest
    raw_log_ref: Annotated[str, Field(max_length=256)] | None = None
    function_call_id: Identifier | None = None
    invocation_id: Identifier | None = None
    tool_name: Annotated[str, Field(min_length=1, max_length=64)] | None = None

    @model_validator(mode="after")
    def validate_tool_identity(self) -> "Observation":
        tool_types = {
            "schema",
            "metadata",
            "knowledge",
            "user_answer",
            "sql_execution",
            "submission",
            "tool_error",
        }
        if self.observation_type in tool_types:
            if self.function_call_id is None or self.tool_name is None:
                raise ValueError(
                    "tool observations require function_call_id and tool_name"
                )
        if (self.function_call_id is None) != (self.tool_name is None):
            raise ValueError("function_call_id and tool_name must be supplied together")
        return self


class PendingToolCall(KernelModel):
    """before_tool 已放行、但 after_tool 尚未收尾的调用。"""

    function_call_id: Identifier
    tool_name: Annotated[str, Field(min_length=1, max_length=64)]
    args_summary: dict[str, JsonValue] | Annotated[str, Field(max_length=1024)]
    args_digest: Digest
    phase_before: Phase
    sequence: Annotated[int, Field(ge=0)]
    started_at: Annotated[str, Field(max_length=64)] | None = None

    @model_validator(mode="after")
    def validate_args_size(self) -> "PendingToolCall":
        encoded = json.dumps(
            self.args_summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_ARGS_SUMMARY_BYTES:
            raise ValueError("args_summary exceeds bounded storage limit")
        return self


MetricName = Literal[
    "observations_seen",
    "patches_applied",
    "patches_rejected",
    "updater_calls",
    "updater_input_tokens",
    "updater_output_tokens",
    "updater_reasoning_tokens",
    "updater_latency_ms",
    "updater_timeouts",
    "updater_errors",
    "llm_updater_calls",
    "llm_updater_input_tokens",
    "llm_updater_output_tokens",
    "llm_updater_reasoning_tokens",
    "llm_updater_total_tokens",
    "llm_updater_latency_ms",
    "llm_updater_timeouts",
    "llm_updater_errors",
    "llm_updater_cost",
    "prompt_view_chars",
    "prompt_view_tokens",
]


class RuntimeMetrics(RootModel[dict[MetricName, int | float]]):
    """只用于诊断的非负计数和耗时指标。"""

    root: dict[MetricName, int | float] = Field(default_factory=dict)

    @field_validator("root")
    @classmethod
    def validate_values(
        cls,
        value: dict[str, int | float],
    ) -> dict[str, int | float]:
        for key, item in value.items():
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"metric {key} must be numeric")
            if not math.isfinite(float(item)) or item < 0:
                raise ValueError(f"metric {key} must be finite and non-negative")
        return value


class ValibraError(KernelModel):
    """脱敏且有长度上限的最近一次错误。"""

    stage: Literal["observation", "updater", "reducer", "service", "telemetry"]
    error_type: Annotated[str, Field(min_length=1, max_length=128)]
    message_preview: Annotated[str, Field(max_length=MAX_ERROR_CHARS)]
    retryable: bool = False
    observation_id: Digest | None = None
    function_call_id: Identifier | None = None
    sequence: Annotated[int, Field(ge=0)]
    timestamp: Annotated[str, Field(max_length=64)] | None = None


class PhaseTransition(KernelModel):
    """从 Phase 1 进入 Phase 2 时要执行的显式状态变更。"""

    target_phase: Literal[2] = 2
    supersede_slot_ids: tuple[Identifier, ...] = ()
    reopen_ambiguity_ids: tuple[Identifier, ...] = ()
    reason: Annotated[str, Field(max_length=MAX_REASON_CHARS)] | None = None
    sequence: Annotated[int, Field(ge=0)]

    @field_validator("supersede_slot_ids", "reopen_ambiguity_ids")
    @classmethod
    def validate_transition_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, "phase transition IDs")

    @model_validator(mode="after")
    def validate_reopen_reason(self) -> "PhaseTransition":
        if self.reopen_ambiguity_ids and not self.reason:
            raise ValueError("reopening ambiguity requires a reason")
        return self


class RequirementGroundingPatch(KernelModel):
    """Updater 提出的原子变更包；Reducer 要么全收，要么全拒。"""

    patch_id: Identifier
    base_revision: Annotated[int, Field(ge=0)]
    source_observation_ids: tuple[Digest, ...]
    slot_additions: tuple[GroundingSlot, ...] = ()
    slot_updates: tuple[GroundingSlot, ...] = ()
    ambiguity_additions: tuple[GroundedAmbiguityHypothesis, ...] = ()
    ambiguity_updates: tuple[GroundedAmbiguityHypothesis, ...] = ()
    evidence_additions: tuple[GroundingEvidence, ...] = ()
    phase_transition: PhaseTransition | None = None
    diagnostics: tuple[Annotated[str, Field(max_length=256)], ...] = ()

    @field_validator("source_observation_ids")
    @classmethod
    def validate_source_observations(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError("source_observation_ids cannot be empty")
        return _unique(value, "source_observation_ids")

    @model_validator(mode="after")
    def validate_patch_local_ids(self) -> "RequirementGroundingPatch":
        groups: tuple[tuple[Any, ...], ...] = (
            self.slot_additions,
            self.slot_updates,
            self.ambiguity_additions,
            self.ambiguity_updates,
            self.evidence_additions,
        )
        attrs = (
            "slot_id",
            "slot_id",
            "ambiguity_id",
            "ambiguity_id",
            "evidence_id",
        )
        for values, attr in zip(groups, attrs):
            ids = [getattr(item, attr) for item in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {attr} in one patch operation")
        if {
            item.slot_id for item in self.slot_additions
        } & {item.slot_id for item in self.slot_updates}:
            raise ValueError("a slot cannot be both added and updated")
        if {
            item.ambiguity_id for item in self.ambiguity_additions
        } & {item.ambiguity_id for item in self.ambiguity_updates}:
            raise ValueError("an ambiguity cannot be both added and updated")
        return self


class RequirementGroundingRuntime(KernelModel):
    """一个任务的完整 Grounding 运行状态。"""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    grounding_revision: Annotated[int, Field(ge=0)] = 0
    phase: Phase = 1
    grounding_state: RequirementGroundingState = Field(
        default_factory=RequirementGroundingState
    )
    processed_observation_ids: tuple[Digest, ...] = ()
    pending_tool_calls: dict[Identifier, PendingToolCall] = Field(
        default_factory=dict
    )
    metrics: RuntimeMetrics = Field(default_factory=RuntimeMetrics)
    last_error: ValibraError | None = None

    @field_validator("processed_observation_ids")
    @classmethod
    def validate_processed_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(value, "processed_observation_ids")

    @model_validator(mode="after")
    def validate_pending_keys(self) -> "RequirementGroundingRuntime":
        for key, pending in self.pending_tool_calls.items():
            if key != pending.function_call_id:
                raise ValueError("pending_tool_calls key must match function_call_id")
        return self


# 短别名只方便调用；序列化仍以完整类名对应的结构为准。
Runtime = RequirementGroundingRuntime
State = RequirementGroundingState
Evidence = GroundingEvidence
Candidate = InterpretationCandidate
Ambiguity = GroundedAmbiguityHypothesis
Patch = RequirementGroundingPatch
Metrics = RuntimeMetrics
Error = ValibraError
