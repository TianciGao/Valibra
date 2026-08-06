"""Atomic reducer and cross-object invariants for the K0 business state."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from valibra_agent.requirement_grounding.models import (
    GroundedAmbiguityHypothesis,
    GroundingEvidence,
    GroundingSlot,
    OperationSlot,
    RequirementFrame,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    RequirementGroundingState,
    SchemaSlot,
    ValueSlot,
)

MAX_SLOTS = 512
MAX_AMBIGUITIES = 256
MAX_EVIDENCE = 1024
MAX_CANDIDATES = 1024
MAX_PROCESSED_OBSERVATIONS = 4096
MAX_PENDING_TOOL_CALLS = 128
MAX_RUNTIME_JSON_BYTES = 524_288


class ReductionError(ValueError):
    """A patch failed CAS, shape, reference, invariant, or size validation."""


def apply_patch(
    runtime: RequirementGroundingRuntime,
    patch: RequirementGroundingPatch,
) -> RequirementGroundingRuntime:
    """Apply one complete patch or raise without changing ``runtime``.

    Observation replay is checked before revision CAS so a retry of an already
    committed observation remains idempotent even when its patch carries the
    former base revision.
    """

    processed = set(runtime.processed_observation_ids)
    sources = set(patch.source_observation_ids)
    already_processed = sources & processed
    if already_processed == sources:
        return runtime
    if already_processed:
        raise ReductionError("patch mixes processed and unprocessed observations")
    if patch.base_revision != runtime.grounding_revision:
        raise ReductionError(
            "base_revision mismatch: "
            f"expected {runtime.grounding_revision}, got {patch.base_revision}"
        )

    next_processed = runtime.processed_observation_ids + patch.source_observation_ids
    if len(next_processed) > MAX_PROCESSED_OBSERVATIONS:
        raise ReductionError("processed observation limit exceeded")

    candidate_state, candidate_phase = _apply_business_operations(
        runtime.grounding_state,
        runtime.phase,
        patch,
    )
    _validate_state(candidate_state, set(next_processed))

    business_changed = (
        candidate_state != runtime.grounding_state
        or candidate_phase != runtime.phase
    )
    next_revision = runtime.grounding_revision + int(business_changed)
    candidate_runtime = _validated_runtime_copy(
        runtime,
        grounding_state=candidate_state,
        phase=candidate_phase,
        grounding_revision=next_revision,
        processed_observation_ids=next_processed,
    )
    validate_runtime(candidate_runtime)
    return candidate_runtime


def validate_runtime(runtime: RequirementGroundingRuntime) -> None:
    """Validate the complete runtime without changing it."""

    if len(runtime.processed_observation_ids) > MAX_PROCESSED_OBSERVATIONS:
        raise ReductionError("processed observation limit exceeded")
    if len(runtime.pending_tool_calls) > MAX_PENDING_TOOL_CALLS:
        raise ReductionError("pending tool-call limit exceeded")
    _validate_state(runtime.grounding_state, set(runtime.processed_observation_ids))
    encoded = json.dumps(
        runtime.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_RUNTIME_JSON_BYTES:
        raise ReductionError("runtime exceeds total serialized-size limit")


def _apply_business_operations(
    state: RequirementGroundingState,
    phase: int,
    patch: RequirementGroundingPatch,
) -> tuple[RequirementGroundingState, int]:
    value_slots = list(state.requirement_frame.value_slots)
    schema_slots = list(state.requirement_frame.schema_slots)
    operation_slots = list(state.requirement_frame.operation_slots)
    ambiguities = list(state.ambiguity_index)
    evidence = list(state.evidence)

    slot_lists: dict[str, list[GroundingSlot]] = {
        "value": value_slots,
        "schema": schema_slots,
        "operation": operation_slots,
    }
    existing_slots = {
        slot.slot_id: slot
        for slot in _all_slots_from_lists(slot_lists)
    }
    for slot in patch.slot_additions:
        if slot.slot_id in existing_slots:
            raise ReductionError(f"slot already exists: {slot.slot_id}")
        slot_lists[slot.slot_kind].append(slot)
        existing_slots[slot.slot_id] = slot
    for slot in patch.slot_updates:
        old = existing_slots.get(slot.slot_id)
        if old is None:
            raise ReductionError(f"slot update is dangling: {slot.slot_id}")
        if old.slot_kind != slot.slot_kind:
            raise ReductionError(f"slot kind cannot change: {slot.slot_id}")
        _replace_by_id(slot_lists[slot.slot_kind], "slot_id", slot)
        existing_slots[slot.slot_id] = slot

    ambiguity_index = {item.ambiguity_id: item for item in ambiguities}
    for ambiguity in patch.ambiguity_additions:
        if ambiguity.ambiguity_id in ambiguity_index:
            raise ReductionError(
                f"ambiguity already exists: {ambiguity.ambiguity_id}"
            )
        ambiguities.append(ambiguity)
        ambiguity_index[ambiguity.ambiguity_id] = ambiguity
    for ambiguity in patch.ambiguity_updates:
        if ambiguity.ambiguity_id not in ambiguity_index:
            raise ReductionError(
                f"ambiguity update is dangling: {ambiguity.ambiguity_id}"
            )
        _replace_by_id(ambiguities, "ambiguity_id", ambiguity)
        ambiguity_index[ambiguity.ambiguity_id] = ambiguity

    evidence_index = {item.evidence_id: item for item in evidence}
    for item in patch.evidence_additions:
        if item.evidence_id in evidence_index:
            raise ReductionError(f"evidence already exists: {item.evidence_id}")
        evidence.append(item)
        evidence_index[item.evidence_id] = item

    next_phase = phase
    transition = patch.phase_transition
    if transition is not None:
        if phase not in (1, 2):
            raise ReductionError(f"invalid current phase: {phase}")
        next_phase = transition.target_phase
        for slot_id in transition.supersede_slot_ids:
            slot = existing_slots.get(slot_id)
            if slot is None:
                raise ReductionError(f"phase transition references missing slot: {slot_id}")
            replacement = type(slot).model_validate(
                {
                    **slot.model_dump(mode="python"),
                    "lifecycle": "superseded",
                    "last_updated_phase": 2,
                }
            )
            _replace_by_id(slot_lists[slot.slot_kind], "slot_id", replacement)
            existing_slots[slot_id] = replacement
        for ambiguity_id in transition.reopen_ambiguity_ids:
            ambiguity = ambiguity_index.get(ambiguity_id)
            if ambiguity is None:
                raise ReductionError(
                    "phase transition references missing ambiguity: "
                    f"{ambiguity_id}"
                )
            replacement = GroundedAmbiguityHypothesis.model_validate(
                {
                    **ambiguity.model_dump(mode="python"),
                    "status": "unresolved",
                    "resolution": None,
                    "reopen_reason": transition.reason,
                    "sequence": transition.sequence,
                }
            )
            _replace_by_id(ambiguities, "ambiguity_id", replacement)
            ambiguity_index[ambiguity_id] = replacement

    candidate_state = RequirementGroundingState(
        requirement_frame=RequirementFrame(
            value_slots=tuple(value_slots),
            schema_slots=tuple(schema_slots),
            operation_slots=tuple(operation_slots),
        ),
        ambiguity_index=tuple(ambiguities),
        evidence=tuple(evidence),
    )
    return candidate_state, next_phase


def _validate_state(
    state: RequirementGroundingState,
    processed_observation_ids: set[str],
) -> None:
    slots = list(_all_slots(state.requirement_frame))
    ambiguities = list(state.ambiguity_index)
    evidence = list(state.evidence)
    candidates = [
        candidate
        for ambiguity in ambiguities
        for candidate in ambiguity.candidate_interpretations
    ]
    if len(slots) > MAX_SLOTS:
        raise ReductionError("slot limit exceeded")
    if len(ambiguities) > MAX_AMBIGUITIES:
        raise ReductionError("ambiguity limit exceeded")
    if len(evidence) > MAX_EVIDENCE:
        raise ReductionError("evidence limit exceeded")
    if len(candidates) > MAX_CANDIDATES:
        raise ReductionError("candidate limit exceeded")

    all_ids = (
        [slot.slot_id for slot in slots]
        + [item.ambiguity_id for item in ambiguities]
        + [item.evidence_id for item in evidence]
        + [item.candidate_id for item in candidates]
    )
    if len(all_ids) != len(set(all_ids)):
        raise ReductionError("all state object IDs must be globally unique")

    slot_index = {slot.slot_id: slot for slot in slots}
    ambiguity_index = {item.ambiguity_id: item for item in ambiguities}
    evidence_ids = {item.evidence_id for item in evidence}

    for item in evidence:
        if item.observation_id not in processed_observation_ids:
            raise ReductionError(
                f"evidence references unprocessed observation: {item.evidence_id}"
            )

    for slot in slots:
        _require_refs(slot.evidence_refs, evidence_ids, "slot evidence")
        _require_refs(slot.ambiguity_refs, set(ambiguity_index), "slot ambiguity")
        for ambiguity_id in slot.ambiguity_refs:
            ambiguity = ambiguity_index[ambiguity_id]
            if slot.slot_id not in ambiguity.affected_slot_ids:
                raise ReductionError(
                    f"slot/ambiguity reverse reference missing: {slot.slot_id}"
                )

    for ambiguity in ambiguities:
        if ambiguity.primary_slot_id not in slot_index:
            raise ReductionError(
                f"ambiguity primary slot is dangling: {ambiguity.ambiguity_id}"
            )
        _require_refs(
            ambiguity.affected_slot_ids,
            set(slot_index),
            "ambiguity affected slot",
        )
        _require_refs(
            ambiguity.source_grounding,
            evidence_ids,
            "ambiguity source evidence",
        )
        if not ambiguity.source_grounding:
            raise ReductionError(
                f"ambiguity lacks source grounding: {ambiguity.ambiguity_id}"
            )
        for slot_id in ambiguity.affected_slot_ids:
            if ambiguity.ambiguity_id not in slot_index[slot_id].ambiguity_refs:
                raise ReductionError(
                    f"ambiguity/slot reverse reference missing: {ambiguity.ambiguity_id}"
                )
        candidate_ids = {
            item.candidate_id for item in ambiguity.candidate_interpretations
        }
        for candidate in ambiguity.candidate_interpretations:
            if not candidate.evidence_refs:
                raise ReductionError(
                    f"candidate lacks evidence: {candidate.candidate_id}"
                )
            _require_refs(
                candidate.evidence_refs,
                evidence_ids,
                "candidate evidence",
            )
        effect_keys = {
            item.sql_impact.effect_key
            for item in ambiguity.candidate_interpretations
        }
        if ambiguity.status == "unresolved":
            if len(ambiguity.candidate_interpretations) < 2:
                raise ReductionError(
                    "unresolved ambiguity requires at least two candidates"
                )
            if len(effect_keys) < 2:
                raise ReductionError(
                    "unresolved ambiguity requires distinct SQL effects"
                )
            if ambiguity.resolution is not None:
                raise ReductionError("unresolved ambiguity cannot have resolution")
        elif ambiguity.status == "resolved":
            if ambiguity.resolution not in candidate_ids:
                raise ReductionError(
                    "resolved ambiguity must reference a valid candidate"
                )
            selected = next(
                item
                for item in ambiguity.candidate_interpretations
                if item.candidate_id == ambiguity.resolution
            )
            primary = slot_index[ambiguity.primary_slot_id]
            if primary.current_interpretation != selected.interpretation:
                raise ReductionError(
                    "resolved candidate must match primary slot interpretation"
                )
        elif ambiguity.resolution is not None:
            raise ReductionError(
                f"{ambiguity.status} ambiguity cannot have resolution"
            )
    _validate_dependencies(ambiguity_index)


def _validate_dependencies(
    ambiguity_index: dict[str, GroundedAmbiguityHypothesis],
) -> None:
    known = set(ambiguity_index)
    graph: dict[str, tuple[str, ...]] = {}
    for ambiguity_id, ambiguity in ambiguity_index.items():
        if ambiguity_id in ambiguity.dependency_ids:
            raise ReductionError(f"ambiguity dependency is self-referential: {ambiguity_id}")
        _require_refs(ambiguity.dependency_ids, known, "ambiguity dependency")
        graph[ambiguity_id] = ambiguity.dependency_ids

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ReductionError("ambiguity dependency cycle detected")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)


def _require_refs(refs: Iterable[str], known: set[str], label: str) -> None:
    for ref in refs:
        if ref not in known:
            raise ReductionError(f"dangling {label} reference: {ref}")


def _all_slots(frame: RequirementFrame) -> Iterable[GroundingSlot]:
    yield from frame.value_slots
    yield from frame.schema_slots
    yield from frame.operation_slots


def _all_slots_from_lists(
    slot_lists: dict[str, list[GroundingSlot]],
) -> Iterable[GroundingSlot]:
    yield from slot_lists["value"]
    yield from slot_lists["schema"]
    yield from slot_lists["operation"]


def _replace_by_id(values: list[Any], attr: str, replacement: Any) -> None:
    target = getattr(replacement, attr)
    for index, item in enumerate(values):
        if getattr(item, attr) == target:
            values[index] = replacement
            return
    raise ReductionError(f"replacement target is missing: {target}")


def _validated_runtime_copy(
    runtime: RequirementGroundingRuntime,
    **updates: Any,
) -> RequirementGroundingRuntime:
    data = runtime.model_dump(mode="python")
    data.update(updates)
    return RequirementGroundingRuntime.model_validate(data)
