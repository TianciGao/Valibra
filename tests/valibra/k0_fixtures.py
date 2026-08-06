"""Deterministic fixtures for K0 unit tests."""

from valibra_agent.requirement_grounding.models import (
    GroundedAmbiguityHypothesis,
    GroundingEvidence,
    InterpretationCandidate,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    SQLImpact,
    ValueSlot,
)
from valibra_agent.requirement_grounding.observations import build_observation


def make_observation(sequence=1, raw=None, task_id="task-1"):
    return build_observation(
        task_id=task_id,
        observation_type="user_query",
        phase=1,
        sequence=sequence,
        source="unit_test",
        raw={"query": "test"} if raw is None else raw,
    )


def make_evidence(observation, evidence_id="ev-1", summary="bounded evidence"):
    return GroundingEvidence(
        evidence_id=evidence_id,
        observation_id=observation.observation_id,
        source_type="user_query",
        phase=observation.phase,
        summary=summary,
        raw_digest=observation.raw_digest,
        raw_log_ref="audit://unit-test",
        sequence=observation.sequence,
        timestamp="2026-08-06T00:00:00Z",
    )


def make_candidate(
    candidate_id,
    interpretation,
    effect_key,
    evidence_id="ev-1",
):
    return InterpretationCandidate(
        candidate_id=candidate_id,
        interpretation=interpretation,
        sql_impact=SQLImpact(
            effect_key=effect_key,
            summary=f"SQL effect {effect_key}",
        ),
        evidence_refs=(evidence_id,),
        confidence=0.5,
    )


def make_slot(
    slot_id="slot-1",
    *,
    interpretation=None,
    evidence_refs=("ev-1",),
    ambiguity_refs=(),
    lifecycle="active",
    phase=1,
    sequence=1,
):
    return ValueSlot(
        slot_id=slot_id,
        slot_role="filter_value",
        mention="region",
        current_interpretation=interpretation,
        grounding_status="hypothesized" if interpretation else "missing",
        evidence_refs=evidence_refs,
        ambiguity_refs=ambiguity_refs,
        origin="user_query",
        lifecycle=lifecycle,
        introduced_in_phase=1,
        last_updated_phase=phase,
        sequence=sequence,
        value_type="category",
    )


def make_ambiguity(
    *,
    ambiguity_id="amb-1",
    slot_id="slot-1",
    candidates=(),
    status="deferred",
    resolution=None,
    dependency_ids=(),
    evidence_id="ev-1",
    sequence=1,
):
    return GroundedAmbiguityHypothesis(
        ambiguity_id=ambiguity_id,
        pivot_term="region",
        primary_slot_id=slot_id,
        affected_slot_ids=(slot_id,),
        ambiguity_family="value_scope",
        candidate_interpretations=tuple(candidates),
        source_grounding=(evidence_id,),
        dependency_ids=tuple(dependency_ids),
        status=status,
        resolution=resolution,
        sequence=sequence,
    )


def graph_patch(
    runtime: RequirementGroundingRuntime,
    observation,
    *,
    ambiguity,
    slot=None,
    evidence=None,
    extra_slots=(),
    extra_ambiguities=(),
):
    slot = slot or make_slot(ambiguity_refs=(ambiguity.ambiguity_id,))
    evidence = evidence or make_evidence(observation)
    return RequirementGroundingPatch(
        patch_id=f"patch:{observation.sequence}",
        base_revision=runtime.grounding_revision,
        source_observation_ids=(observation.observation_id,),
        slot_additions=(slot, *extra_slots),
        ambiguity_additions=(ambiguity, *extra_ambiguities),
        evidence_additions=(evidence,),
    )
