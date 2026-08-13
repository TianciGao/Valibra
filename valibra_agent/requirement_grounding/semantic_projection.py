"""Requirement 语义的确定性投影与摘要。

该投影只回答“需求含义是否变化”。Evidence、文本锚点、来源、顺序和
Phase 审计字段不参与比较；它们仍由完整 Grounding State 和
``grounding_revision`` 管理。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from valibra_agent.requirement_grounding.models import (
    GroundedAmbiguityHypothesis,
    OperationSlot,
    RequirementGroundingState,
    SchemaSlot,
    ValueSlot,
)


def requirement_semantic_projection(
    state: RequirementGroundingState,
) -> dict[str, Any]:
    """返回只包含冻结 Requirement 语义字段的稳定 JSON 投影。"""

    frame = state.requirement_frame
    return {
        "requirement_frame": {
            "value_slots": [
                {
                    **_slot_projection(slot),
                    "value_type": slot.value_type,
                }
                for slot in sorted(
                    frame.value_slots,
                    key=lambda item: item.slot_id,
                )
            ],
            "schema_slots": [
                {
                    **_slot_projection(slot),
                    "binding_type": slot.binding_type,
                    "bound_identifier": slot.bound_identifier,
                }
                for slot in sorted(
                    frame.schema_slots,
                    key=lambda item: item.slot_id,
                )
            ],
            "operation_slots": [
                {
                    **_slot_projection(slot),
                    "operation_type": slot.operation_type,
                    # Pydantic frozen models are shallow-frozen. Copy the
                    # bounded scalar map so callers cannot mutate Runtime
                    # state through the returned projection.
                    "parameters": dict(slot.parameters),
                }
                for slot in sorted(
                    frame.operation_slots,
                    key=lambda item: item.slot_id,
                )
            ],
        },
        "ambiguity_index": [
            _ambiguity_projection(ambiguity)
            for ambiguity in sorted(
                state.ambiguity_index,
                key=lambda item: item.ambiguity_id,
            )
        ],
    }


def requirement_semantic_sha256(state: RequirementGroundingState) -> str:
    """计算规范 JSON 编码的 Requirement 语义 SHA256。"""

    encoded = json.dumps(
        requirement_semantic_projection(state),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slot_projection(
    slot: ValueSlot | SchemaSlot | OperationSlot,
) -> dict[str, Any]:
    """投影 Slot 的冻结公共语义字段。"""

    return {
        "slot_id": slot.slot_id,
        "slot_kind": slot.slot_kind,
        "slot_role": slot.slot_role,
        "current_interpretation": slot.current_interpretation,
        "grounding_status": slot.grounding_status,
        "lifecycle": slot.lifecycle,
        "ambiguity_refs": sorted(slot.ambiguity_refs),
    }


def _ambiguity_projection(
    ambiguity: GroundedAmbiguityHypothesis,
) -> dict[str, Any]:
    """投影歧义的解释、SQL 影响、依赖和解决状态。

    ``pivot_term`` 与 Slot ``mention`` 一样只是原文锚点；证据引用、
    ``reopen_reason``、``sequence`` 和非真值 ``confidence`` 都属于
    来源/审计信息，因此排除。
    """

    return {
        "ambiguity_id": ambiguity.ambiguity_id,
        "primary_slot_id": ambiguity.primary_slot_id,
        "affected_slot_ids": sorted(ambiguity.affected_slot_ids),
        "ambiguity_family": ambiguity.ambiguity_family,
        "candidate_interpretations": [
            {
                "candidate_id": candidate.candidate_id,
                "interpretation": candidate.interpretation,
                "sql_impact": {
                    "effect_key": candidate.sql_impact.effect_key,
                    "summary": candidate.sql_impact.summary,
                },
            }
            for candidate in sorted(
                ambiguity.candidate_interpretations,
                key=lambda item: item.candidate_id,
            )
        ],
        "dependency_ids": sorted(ambiguity.dependency_ids),
        "status": ambiguity.status,
        "resolution": ambiguity.resolution,
    }
