"""P7.1a tests for technical and Requirement-semantic revision separation."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from tests.valibra.k0_fixtures import (
    make_ambiguity,
    make_candidate,
    make_evidence,
    make_observation,
    make_slot,
)
from valibra_agent.requirement_grounding.models import (
    OperationSlot,
    PhaseTransition,
    RequirementFrame,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    RequirementGroundingState,
    SchemaSlot,
    ValibraError,
)
from valibra_agent.requirement_grounding.reducer import (
    ReductionError,
    apply_patch,
    validate_runtime,
)
from valibra_agent.requirement_grounding.semantic_projection import (
    requirement_semantic_projection,
    requirement_semantic_sha256,
)
from valibra_agent.requirement_grounding.telemetry import (
    increment_metrics,
    set_last_error,
)


def _validated_copy(model, **updates):
    payload = model.model_dump(mode="python")
    payload.update(updates)
    return type(model).model_validate(payload)


def _initial_frame_runtime():
    runtime = RequirementGroundingRuntime()
    observation = make_observation(sequence=1, raw={"query": "region north"})
    evidence = make_evidence(observation, evidence_id="ev-1")
    slot = make_slot(
        interpretation="north",
        evidence_refs=(evidence.evidence_id,),
    )
    patch = RequirementGroundingPatch(
        patch_id="initial-frame",
        base_revision=runtime.grounding_revision,
        source_observation_ids=(observation.observation_id,),
        slot_additions=(slot,),
        evidence_additions=(evidence,),
    )
    return runtime, observation, patch, apply_patch(runtime, patch)


def _initial_ambiguity_runtime():
    runtime = RequirementGroundingRuntime()
    observation = make_observation(sequence=1, raw={"query": "which region"})
    evidence = make_evidence(observation, evidence_id="ev-1")
    candidates = (
        make_candidate("candidate-a", "North", "north"),
        make_candidate("candidate-b", "South", "south"),
    )
    ambiguity = make_ambiguity(
        candidates=candidates,
        status="unresolved",
    )
    slot = make_slot(
        evidence_refs=(evidence.evidence_id,),
        ambiguity_refs=(ambiguity.ambiguity_id,),
    )
    patch = RequirementGroundingPatch(
        patch_id="initial-ambiguity",
        base_revision=runtime.grounding_revision,
        source_observation_ids=(observation.observation_id,),
        slot_additions=(slot,),
        ambiguity_additions=(ambiguity,),
        evidence_additions=(evidence,),
    )
    return apply_patch(runtime, patch)


class RevisionMatrixTests(unittest.TestCase):
    def test_initial_nonempty_frame_increments_both_revisions(self):
        before, _observation, _patch, after = _initial_frame_runtime()

        self.assertEqual((before.grounding_revision, before.requirement_revision), (0, 0))
        self.assertEqual((after.grounding_revision, after.requirement_revision), (1, 1))

    def test_evidence_only_increments_grounding_but_not_requirement(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()
        observation = make_observation(sequence=2, raw={"query": "audit only"})
        evidence = make_evidence(observation, evidence_id="ev-2")
        patch = RequirementGroundingPatch(
            patch_id="evidence-only",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            evidence_additions=(evidence,),
        )

        updated = apply_patch(runtime, patch)

        self.assertEqual(updated.grounding_revision, runtime.grounding_revision + 1)
        self.assertEqual(updated.requirement_revision, runtime.requirement_revision)

    def test_slot_evidence_refs_only_increment_grounding_revision(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()
        observation = make_observation(sequence=2, raw={"query": "more support"})
        evidence = make_evidence(observation, evidence_id="ev-2")
        old_slot = runtime.grounding_state.requirement_frame.value_slots[0]
        audited_slot = _validated_copy(
            old_slot,
            evidence_refs=old_slot.evidence_refs + (evidence.evidence_id,),
        )
        patch = RequirementGroundingPatch(
            patch_id="slot-evidence-ref-only",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            slot_updates=(audited_slot,),
            evidence_additions=(evidence,),
        )

        updated = apply_patch(runtime, patch)

        self.assertEqual(updated.grounding_revision, runtime.grounding_revision + 1)
        self.assertEqual(updated.requirement_revision, runtime.requirement_revision)

    def test_real_slot_semantic_change_increments_both_revisions(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()
        observation = make_observation(sequence=2, raw={"query": "region south"})
        old_slot = runtime.grounding_state.requirement_frame.value_slots[0]
        changed_slot = _validated_copy(
            old_slot,
            current_interpretation="south",
            grounding_status="confirmed",
        )
        patch = RequirementGroundingPatch(
            patch_id="semantic-slot-update",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            slot_updates=(changed_slot,),
        )

        updated = apply_patch(runtime, patch)

        self.assertEqual(updated.grounding_revision, runtime.grounding_revision + 1)
        self.assertEqual(updated.requirement_revision, runtime.requirement_revision + 1)

    def test_real_ambiguity_semantic_change_increments_both_revisions(self):
        runtime = _initial_ambiguity_runtime()
        observation = make_observation(sequence=2, raw={"query": "cannot answer"})
        ambiguity = runtime.grounding_state.ambiguity_index[0]
        changed_ambiguity = _validated_copy(
            ambiguity,
            status="unanswerable",
        )
        patch = RequirementGroundingPatch(
            patch_id="semantic-ambiguity-update",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            ambiguity_updates=(changed_ambiguity,),
        )

        updated = apply_patch(runtime, patch)

        self.assertEqual(updated.grounding_revision, runtime.grounding_revision + 1)
        self.assertEqual(updated.requirement_revision, runtime.requirement_revision + 1)

    def test_ambiguity_audit_only_change_increments_only_grounding_revision(self):
        runtime = _initial_ambiguity_runtime()
        observation = make_observation(sequence=2, raw={"query": "audit"})
        ambiguity = runtime.grounding_state.ambiguity_index[0]
        audit_changed = _validated_copy(
            ambiguity,
            reopen_reason="bounded audit reason",
            sequence=2,
        )
        patch = RequirementGroundingPatch(
            patch_id="ambiguity-audit-update",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            ambiguity_updates=(audit_changed,),
        )

        updated = apply_patch(runtime, patch)

        self.assertEqual(updated.grounding_revision, runtime.grounding_revision + 1)
        self.assertEqual(updated.requirement_revision, runtime.requirement_revision)

    def test_phase_only_increments_grounding_but_not_requirement(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()
        observation = make_observation(sequence=2, raw={"phase": 2})
        patch = RequirementGroundingPatch(
            patch_id="phase-only",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            phase_transition=PhaseTransition(sequence=2),
        )

        updated = apply_patch(runtime, patch)

        self.assertEqual(updated.phase, 2)
        self.assertEqual(updated.grounding_revision, runtime.grounding_revision + 1)
        self.assertEqual(updated.requirement_revision, runtime.requirement_revision)

    def test_phase_with_supersede_increments_both_revisions(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()
        observation = make_observation(sequence=2, raw={"phase": 2})
        patch = RequirementGroundingPatch(
            patch_id="phase-with-supersede",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            phase_transition=PhaseTransition(
                sequence=2,
                supersede_slot_ids=("slot-1",),
            ),
        )

        updated = apply_patch(runtime, patch)

        self.assertEqual(updated.phase, 2)
        self.assertEqual(updated.grounding_revision, runtime.grounding_revision + 1)
        self.assertEqual(updated.requirement_revision, runtime.requirement_revision + 1)

    def test_metrics_and_error_change_neither_revision(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()

        with_metrics = increment_metrics(runtime, observations_seen=1)
        with_error = set_last_error(
            with_metrics,
            ValibraError(
                stage="telemetry",
                error_type="SyntheticError",
                message_preview="bounded",
                sequence=2,
            ),
        )

        expected = (runtime.grounding_revision, runtime.requirement_revision)
        self.assertEqual(
            (with_metrics.grounding_revision, with_metrics.requirement_revision),
            expected,
        )
        self.assertEqual(
            (with_error.grounding_revision, with_error.requirement_revision),
            expected,
        )


class SemanticProjectionTests(unittest.TestCase):
    def test_projection_has_frozen_slot_semantic_fields_only(self):
        value_slot = make_slot(interpretation="north", evidence_refs=())
        schema_slot = SchemaSlot(
            slot_id="schema-1",
            slot_role="dimension",
            mention="customer",
            current_interpretation="customer dimension",
            grounding_status="grounded",
            origin="schema",
            sequence=2,
            binding_type="column",
            bound_identifier="customers.name",
        )
        operation_slot = OperationSlot(
            slot_id="operation-1",
            slot_role="sort",
            mention="descending",
            current_interpretation="sort descending",
            grounding_status="confirmed",
            origin="user_query",
            sequence=3,
            operation_type="order",
            parameters={"direction": "desc"},
        )
        projection = requirement_semantic_projection(
            RequirementGroundingState(
                requirement_frame=RequirementFrame(
                    value_slots=(value_slot,),
                    schema_slots=(schema_slot,),
                    operation_slots=(operation_slot,),
                )
            )
        )["requirement_frame"]

        common_fields = {
            "slot_id",
            "slot_kind",
            "slot_role",
            "current_interpretation",
            "grounding_status",
            "lifecycle",
            "ambiguity_refs",
        }
        self.assertEqual(set(projection["value_slots"][0]), common_fields | {"value_type"})
        self.assertEqual(
            set(projection["schema_slots"][0]),
            common_fields | {"binding_type", "bound_identifier"},
        )
        self.assertEqual(
            set(projection["operation_slots"][0]),
            common_fields | {"operation_type", "parameters"},
        )

    def test_slot_audit_fields_are_excluded_from_projection_and_sha(self):
        original = make_slot(
            interpretation="north",
            evidence_refs=("ev-old",),
            phase=1,
            sequence=1,
        )
        audit_changed = _validated_copy(
            original,
            mention="northern region",
            evidence_refs=("ev-new",),
            origin="user_answer",
            introduced_in_phase=2,
            last_updated_phase=2,
            sequence=99,
        )
        original_state = RequirementGroundingState(
            requirement_frame=RequirementFrame(value_slots=(original,))
        )
        changed_state = RequirementGroundingState(
            requirement_frame=RequirementFrame(value_slots=(audit_changed,))
        )

        self.assertEqual(
            requirement_semantic_projection(original_state),
            requirement_semantic_projection(changed_state),
        )
        self.assertEqual(
            requirement_semantic_sha256(original_state),
            requirement_semantic_sha256(changed_state),
        )

    def test_projection_and_sha_are_deterministic_across_container_order(self):
        first = make_slot(
            slot_id="slot-a",
            interpretation="A",
            evidence_refs=(),
            ambiguity_refs=("amb-b", "amb-a"),
        )
        second = make_slot(
            slot_id="slot-b",
            interpretation="B",
            evidence_refs=(),
        )
        operation_one = OperationSlot(
            slot_id="operation-1",
            slot_role="order",
            current_interpretation="descending",
            grounding_status="hypothesized",
            origin="user_query",
            sequence=3,
            operation_type="order",
            parameters={"z": 1, "a": "desc"},
        )
        operation_two = _validated_copy(
            operation_one,
            parameters={"a": "desc", "z": 1},
        )
        state_one = RequirementGroundingState(
            requirement_frame=RequirementFrame(
                value_slots=(second, first),
                operation_slots=(operation_one,),
            )
        )
        state_two = RequirementGroundingState(
            requirement_frame=RequirementFrame(
                value_slots=(first, second),
                operation_slots=(operation_two,),
            )
        )

        self.assertEqual(
            requirement_semantic_projection(state_one),
            requirement_semantic_projection(state_two),
        )
        self.assertEqual(
            requirement_semantic_sha256(state_one),
            requirement_semantic_sha256(state_two),
        )
        self.assertRegex(requirement_semantic_sha256(state_one), r"^[0-9a-f]{64}$")

    def test_projection_mutation_cannot_mutate_runtime_state_or_sha(self):
        operation = OperationSlot(
            slot_id="operation-1",
            slot_role="order",
            current_interpretation="descending",
            grounding_status="hypothesized",
            origin="user_query",
            sequence=1,
            operation_type="order",
            parameters={"direction": "desc"},
        )
        state = RequirementGroundingState(
            requirement_frame=RequirementFrame(operation_slots=(operation,))
        )
        before_sha = requirement_semantic_sha256(state)
        projection = requirement_semantic_projection(state)

        projection["requirement_frame"]["operation_slots"][0]["parameters"][
            "direction"
        ] = "asc"

        self.assertEqual(operation.parameters, {"direction": "desc"})
        self.assertEqual(requirement_semantic_sha256(state), before_sha)

    def test_candidate_confidence_is_not_requirement_semantics(self):
        candidates = (
            make_candidate("candidate-a", "North", "north"),
            make_candidate("candidate-b", "South", "south"),
        )
        ambiguity = make_ambiguity(candidates=candidates, status="unresolved")
        changed_candidates = tuple(
            _validated_copy(candidate, confidence=0.9)
            for candidate in candidates
        )
        changed = _validated_copy(
            ambiguity,
            candidate_interpretations=changed_candidates,
        )

        self.assertEqual(
            requirement_semantic_sha256(
                RequirementGroundingState(ambiguity_index=(ambiguity,))
            ),
            requirement_semantic_sha256(
                RequirementGroundingState(ambiguity_index=(changed,))
            ),
        )

    def test_evidence_and_ambiguity_audit_fields_do_not_change_semantic_sha(self):
        candidates = (
            make_candidate("candidate-a", "North", "north"),
            make_candidate("candidate-b", "South", "south"),
        )
        original = make_ambiguity(
            candidates=candidates,
            status="unresolved",
            evidence_id="ev-old",
            sequence=1,
        )
        audit_changed = _validated_copy(
            original,
            source_grounding=("ev-new",),
            sequence=99,
        )
        observation = make_observation()
        evidence = make_evidence(observation)
        state_one = RequirementGroundingState(
            ambiguity_index=(original,),
            evidence=(evidence,),
        )
        state_two = RequirementGroundingState(ambiguity_index=(audit_changed,))

        self.assertEqual(
            requirement_semantic_sha256(state_one),
            requirement_semantic_sha256(state_two),
        )

    def test_ambiguity_semantics_participate_in_projection(self):
        candidates = (
            make_candidate("candidate-a", "North", "north"),
            make_candidate("candidate-b", "South", "south"),
        )
        unresolved = make_ambiguity(candidates=candidates, status="unresolved")
        resolved = _validated_copy(
            unresolved,
            status="resolved",
            resolution="candidate-a",
        )
        unresolved_state = RequirementGroundingState(ambiguity_index=(unresolved,))
        resolved_state = RequirementGroundingState(ambiguity_index=(resolved,))

        self.assertNotEqual(
            requirement_semantic_sha256(unresolved_state),
            requirement_semantic_sha256(resolved_state),
        )


class RevisionSafetyTests(unittest.TestCase):
    def test_patch_cas_continues_to_use_grounding_revision(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()
        second_observation = make_observation(sequence=2, raw={"query": "audit"})
        second_evidence = make_evidence(second_observation, evidence_id="ev-2")
        evidence_runtime = apply_patch(
            runtime,
            RequirementGroundingPatch(
                patch_id="evidence-advance",
                base_revision=runtime.grounding_revision,
                source_observation_ids=(second_observation.observation_id,),
                evidence_additions=(second_evidence,),
            ),
        )
        self.assertEqual(
            evidence_runtime.requirement_revision,
            runtime.requirement_revision,
        )

        third_observation = make_observation(sequence=3, raw={"query": "stale"})
        stale_patch = RequirementGroundingPatch(
            patch_id="stale-uses-semantic-revision",
            base_revision=evidence_runtime.requirement_revision,
            source_observation_ids=(third_observation.observation_id,),
        )
        with self.assertRaisesRegex(ReductionError, "base_revision mismatch"):
            apply_patch(evidence_runtime, stale_patch)

    def test_replay_is_identity_and_changes_neither_revision(self):
        _empty, _observation, patch, runtime = _initial_frame_runtime()

        replayed = apply_patch(runtime, patch)

        self.assertIs(replayed, runtime)
        self.assertEqual(
            (replayed.grounding_revision, replayed.requirement_revision),
            (runtime.grounding_revision, runtime.requirement_revision),
        )

    def test_rejected_patch_is_atomic_for_both_revisions(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()
        observation = make_observation(sequence=2, raw={"query": "invalid"})
        old_slot = runtime.grounding_state.requirement_frame.value_slots[0]
        invalid_update = _validated_copy(
            old_slot,
            current_interpretation="changed but invalid",
            evidence_refs=("ev-missing",),
        )
        patch = RequirementGroundingPatch(
            patch_id="atomic-rejection",
            base_revision=runtime.grounding_revision,
            source_observation_ids=(observation.observation_id,),
            slot_updates=(invalid_update,),
        )
        before = runtime.model_dump(mode="json")

        with self.assertRaises(ReductionError):
            apply_patch(runtime, patch)

        self.assertEqual(runtime.model_dump(mode="json"), before)

    def test_runtime_json_round_trip_preserves_both_revisions(self):
        _empty, _first_observation, _first_patch, runtime = _initial_frame_runtime()

        restored = RequirementGroundingRuntime.model_validate_json(
            runtime.model_dump_json()
        )

        self.assertEqual(restored, runtime)
        self.assertEqual(restored.grounding_revision, 1)
        self.assertEqual(restored.requirement_revision, 1)
        validate_runtime(restored)

    def test_legacy_payload_without_history_fields_migrates_as_unknown(self):
        from valibra_agent.requirement_grounding.models import (
            migrate_requirement_grounding_runtime,
        )

        payload = RequirementGroundingRuntime(
            grounding_revision=3,
        ).model_dump(mode="json")
        payload["schema_version"] = "1.0"
        for field in (
            "requirement_revision",
            "frame_initialization_status",
            "frame_initialization_reason",
            "frame_initialization_observation_id",
        ):
            payload.pop(field)

        migrated = migrate_requirement_grounding_runtime(json.loads(json.dumps(payload)))
        restored = migrated.runtime

        self.assertTrue(migrated.legacy_unknown_history)
        self.assertEqual(restored.schema_version, "1.1")
        self.assertEqual(restored.grounding_revision, 3)
        self.assertEqual(restored.requirement_revision, 0)

    def test_requirement_revision_cannot_exceed_grounding_revision(self):
        with self.assertRaisesRegex(
            ValidationError,
            "requirement_revision cannot exceed grounding_revision",
        ):
            RequirementGroundingRuntime(
                grounding_revision=1,
                requirement_revision=2,
            )


if __name__ == "__main__":
    unittest.main()
