"""Deterministic updater contracts for K0 and P4.1 Rule Shadow."""

from __future__ import annotations

import json
from collections import Counter

from valibra_agent.requirement_grounding.linguistic_hints import (
    LinguisticHint,
    extract_linguistic_hints,
)
from valibra_agent.requirement_grounding.models import (
    GroundingEvidence,
    GroundingSlot,
    Observation,
    OperationSlot,
    RequirementGroundingPatch,
    RequirementGroundingState,
    SchemaSlot,
    ValueSlot,
)


class NoOpUpdater:
    """Produce a deterministic empty Patch without changing State."""

    mode = "noop"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        # Force contract validation of the supplied State without retaining or
        # mutating it.  K0 deliberately derives no semantic facts.
        RequirementGroundingState.model_validate(state.model_dump(mode="python"))
        return RequirementGroundingPatch(
            patch_id=f"noop:{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            diagnostics=("K0 NoOp updater: no business-state operations",),
        )


class RuleUpdater:
    """Build provisional Frame patches from bounded deterministic hints."""

    mode = "rule"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        detached_state = RequirementGroundingState.model_validate(
            state.model_dump(mode="python")
        )
        if observation.observation_type not in {"user_query", "user_answer"}:
            return _rule_noop_patch(observation, base_revision)

        hints = extract_linguistic_hints(_observation_text(observation))
        if not hints:
            return _rule_noop_patch(observation, base_revision)

        evidence_id = f"evidence.{observation.observation_id}"
        evidence = GroundingEvidence(
            evidence_id=evidence_id,
            observation_id=observation.observation_id,
            source_type=f"rule_{observation.observation_type}",
            phase=observation.phase,
            summary=observation.summary,
            raw_digest=observation.raw_digest,
            raw_log_ref=observation.raw_log_ref,
            sequence=observation.sequence,
            timestamp=None,
        )

        existing = _slot_index(detached_state)
        additions = []
        updates = []
        answer_role_counts = Counter(
            (hint.slot_kind, hint.slot_role) for hint in hints
        )
        for hint in hints:
            stable_id = _slot_id(hint)
            current = existing.get(stable_id)
            if current is not None and current.lifecycle != "active":
                continue
            if current is None and observation.observation_type == "user_answer":
                if answer_role_counts[(hint.slot_kind, hint.slot_role)] != 1:
                    continue
                candidates = [
                    slot
                    for slot in existing.values()
                    if slot.lifecycle == "active"
                    and slot.slot_kind == hint.slot_kind
                    and slot.slot_role == hint.slot_role
                ]
                if len(candidates) > 1:
                    continue
                if len(candidates) == 1:
                    current = candidates[0]

            slot = _slot_from_hint(
                hint,
                slot_id=current.slot_id if current is not None else stable_id,
                evidence_id=evidence_id,
                observation=observation,
                current=current,
            )
            if current is None:
                additions.append(slot)
                existing[slot.slot_id] = slot
            else:
                updates.append(slot)
                existing[slot.slot_id] = slot

        if not additions and not updates:
            return _rule_noop_patch(observation, base_revision)
        return RequirementGroundingPatch(
            patch_id=f"rule.{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            slot_additions=tuple(additions),
            slot_updates=tuple(updates),
            ambiguity_additions=(),
            ambiguity_updates=(),
            evidence_additions=(evidence,),
            diagnostics=(
                "P4.1 deterministic provisional Frame; no ambiguity or schema binding",
            ),
        )


def build_noop_patch(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    return NoOpUpdater().propose(
        observation,
        state,
        base_revision=base_revision,
    )


def build_rule_patch(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    return RuleUpdater().propose(
        observation,
        state,
        base_revision=base_revision,
    )


def _rule_noop_patch(
    observation: Observation,
    base_revision: int,
) -> RequirementGroundingPatch:
    return RequirementGroundingPatch(
        patch_id=f"rule-noop.{observation.observation_id}",
        base_revision=base_revision,
        source_observation_ids=(observation.observation_id,),
        diagnostics=("P4.1 RuleUpdater found no safe provisional Frame change",),
    )


def _observation_text(observation: Observation) -> str:
    # Tool string responses were canonicalized as JSON in P3; decode only that
    # exact scalar shape. No history, hidden state, or audit log is consulted.
    text = observation.summary
    if observation.observation_type == "user_answer" and text.startswith('"'):
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            return text
        if isinstance(decoded, str):
            return decoded
    return text


def _slot_index(state: RequirementGroundingState) -> dict[str, GroundingSlot]:
    frame = state.requirement_frame
    return {
        slot.slot_id: slot
        for slot in (
            *frame.value_slots,
            *frame.schema_slots,
            *frame.operation_slots,
        )
    }


def _slot_id(hint: LinguisticHint) -> str:
    return f"slot.{hint.slot_kind}.{hint.hint_id.removeprefix('hint.')}"


def _slot_from_hint(
    hint: LinguisticHint,
    *,
    slot_id: str,
    evidence_id: str,
    observation: Observation,
    current: GroundingSlot | None,
) -> GroundingSlot:
    existing_evidence = tuple(getattr(current, "evidence_refs", ()))
    evidence_refs = tuple(dict.fromkeys((*existing_evidence, evidence_id)))
    introduced_in_phase = getattr(
        current,
        "introduced_in_phase",
        observation.phase,
    )
    common = {
        "slot_id": slot_id,
        "slot_role": hint.slot_role,
        "mention": hint.mention,
        "current_interpretation": hint.interpretation,
        "grounding_status": "hypothesized",
        "evidence_refs": evidence_refs,
        "ambiguity_refs": tuple(getattr(current, "ambiguity_refs", ())),
        "origin": "rule_provisional",
        "lifecycle": "active",
        "introduced_in_phase": introduced_in_phase,
        "last_updated_phase": observation.phase,
        "sequence": observation.sequence,
    }
    if hint.slot_kind == "value":
        return ValueSlot(**common, value_type=hint.value_type)
    if hint.slot_kind == "schema":
        return SchemaSlot(
            **common,
            binding_type="unknown",
            bound_identifier=None,
        )
    if hint.slot_kind == "operation":
        return OperationSlot(
            **common,
            operation_type=hint.operation_type or "other",
            parameters=dict(hint.parameters),
        )
    raise ValueError(f"unsupported deterministic hint kind: {hint.slot_kind}")
