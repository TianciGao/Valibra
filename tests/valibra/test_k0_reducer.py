import unittest
from unittest.mock import patch as mock_patch

from tests.valibra.k0_fixtures import (
    graph_patch,
    make_ambiguity,
    make_candidate,
    make_evidence,
    make_observation,
    make_slot,
)
from valibra_agent.requirement_grounding.models import (
    PhaseTransition,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.reducer import (
    ReductionError,
    apply_patch,
    validate_runtime,
)


class AmbiguityInvariantTests(unittest.TestCase):
    def assert_rejected_ambiguity(self, ambiguity):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        candidate_patch = graph_patch(runtime, observation, ambiguity=ambiguity)
        before = runtime.model_dump(mode="json")
        with self.assertRaises(ReductionError):
            apply_patch(runtime, candidate_patch)
        self.assertEqual(runtime.model_dump(mode="json"), before)

    def test_zero_candidates_cannot_be_unresolved(self):
        self.assert_rejected_ambiguity(
            make_ambiguity(status="unresolved", candidates=())
        )

    def test_one_candidate_cannot_be_unresolved(self):
        self.assert_rejected_ambiguity(
            make_ambiguity(
                status="unresolved",
                candidates=(make_candidate("cand-a", "A", "effect-a"),),
            )
        )

    def test_different_text_with_same_sql_effect_cannot_be_unresolved(self):
        self.assert_rejected_ambiguity(
            make_ambiguity(
                status="unresolved",
                candidates=(
                    make_candidate("cand-a", "North America", "same-effect"),
                    make_candidate("cand-b", "NA region", "same-effect"),
                ),
            )
        )

    def test_two_distinct_sql_effects_are_valid(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        ambiguity = make_ambiguity(
            status="unresolved",
            candidates=(
                make_candidate("cand-a", "A", "effect-a"),
                make_candidate("cand-b", "B", "effect-b"),
            ),
        )
        updated = apply_patch(
            runtime,
            graph_patch(runtime, observation, ambiguity=ambiguity),
        )
        self.assertEqual(updated.grounding_revision, 1)
        self.assertEqual(updated.grounding_state.ambiguity_index[0], ambiguity)
        validate_runtime(updated)

    def test_resolved_without_legal_resolution_is_rejected(self):
        self.assert_rejected_ambiguity(
            make_ambiguity(
                status="resolved",
                resolution="cand-missing",
                candidates=(
                    make_candidate("cand-a", "A", "effect-a"),
                    make_candidate("cand-b", "B", "effect-b"),
                ),
            )
        )

    def test_resolved_candidate_must_match_primary_slot_interpretation(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        ambiguity = make_ambiguity(
            status="resolved",
            resolution="cand-a",
            candidates=(
                make_candidate("cand-a", "A", "effect-a"),
                make_candidate("cand-b", "B", "effect-b"),
            ),
        )
        slot = make_slot(
            interpretation="B",
            ambiguity_refs=("amb-1",),
        )
        with self.assertRaises(ReductionError):
            apply_patch(
                runtime,
                graph_patch(runtime, observation, ambiguity=ambiguity, slot=slot),
            )


class GraphInvariantTests(unittest.TestCase):
    def test_global_duplicate_id_is_rejected(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        ambiguity = make_ambiguity(status="deferred")
        slot = make_slot(slot_id="ev-1", ambiguity_refs=("amb-1",))
        with self.assertRaisesRegex(ReductionError, "globally unique"):
            apply_patch(
                runtime,
                graph_patch(runtime, observation, ambiguity=ambiguity, slot=slot),
            )

    def test_dangling_evidence_reference_is_rejected(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        ambiguity = make_ambiguity(status="deferred")
        slot = make_slot(
            evidence_refs=("ev-missing",),
            ambiguity_refs=("amb-1",),
        )
        with self.assertRaisesRegex(ReductionError, "dangling slot evidence"):
            apply_patch(
                runtime,
                graph_patch(runtime, observation, ambiguity=ambiguity, slot=slot),
            )

    def test_missing_reverse_slot_ambiguity_reference_is_rejected(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        ambiguity = make_ambiguity(status="deferred")
        slot = make_slot(ambiguity_refs=())
        with self.assertRaisesRegex(ReductionError, "reverse reference"):
            apply_patch(
                runtime,
                graph_patch(runtime, observation, ambiguity=ambiguity, slot=slot),
            )

    def test_self_dependency_is_rejected(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        ambiguity = make_ambiguity(
            status="deferred",
            dependency_ids=("amb-1",),
        )
        with self.assertRaisesRegex(ReductionError, "self-referential"):
            apply_patch(
                runtime,
                graph_patch(runtime, observation, ambiguity=ambiguity),
            )

    def test_dependency_cycle_is_rejected(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        first = make_ambiguity(
            ambiguity_id="amb-1",
            slot_id="slot-1",
            status="deferred",
            dependency_ids=("amb-2",),
        )
        second = make_ambiguity(
            ambiguity_id="amb-2",
            slot_id="slot-2",
            status="deferred",
            dependency_ids=("amb-1",),
        )
        second_slot = make_slot(
            slot_id="slot-2",
            ambiguity_refs=("amb-2",),
        )
        with self.assertRaisesRegex(ReductionError, "cycle"):
            apply_patch(
                runtime,
                graph_patch(
                    runtime,
                    observation,
                    ambiguity=first,
                    extra_slots=(second_slot,),
                    extra_ambiguities=(second,),
                ),
            )


class AtomicReducerTests(unittest.TestCase):
    def _valid_runtime_and_patch(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        ambiguity = make_ambiguity(
            status="unresolved",
            candidates=(
                make_candidate("cand-a", "A", "effect-a"),
                make_candidate("cand-b", "B", "effect-b"),
            ),
        )
        return runtime, graph_patch(runtime, observation, ambiguity=ambiguity)

    def test_old_revision_patch_is_rejected(self):
        runtime, first_patch = self._valid_runtime_and_patch()
        updated = apply_patch(runtime, first_patch)
        new_observation = make_observation(sequence=2, raw={"query": "next"})
        stale = RequirementGroundingPatch(
            patch_id="stale-patch",
            base_revision=0,
            source_observation_ids=(new_observation.observation_id,),
        )
        with self.assertRaisesRegex(ReductionError, "base_revision mismatch"):
            apply_patch(updated, stale)

    def test_replay_returns_identical_runtime_even_after_revision_advance(self):
        runtime, candidate_patch = self._valid_runtime_and_patch()
        updated = apply_patch(runtime, candidate_patch)
        replayed = apply_patch(updated, candidate_patch)
        self.assertIs(replayed, updated)

    def test_partially_valid_patch_rolls_back_everything(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        invalid_ambiguity = make_ambiguity(
            status="unresolved",
            candidates=(
                make_candidate("cand-a", "A", "same"),
                make_candidate("cand-b", "B", "same"),
            ),
        )
        candidate_patch = graph_patch(
            runtime,
            observation,
            ambiguity=invalid_ambiguity,
        )
        before = runtime.model_dump(mode="json")
        with self.assertRaises(ReductionError):
            apply_patch(runtime, candidate_patch)
        self.assertEqual(runtime.model_dump(mode="json"), before)
        self.assertEqual(runtime.processed_observation_ids, ())

    def test_processed_id_limit_rejects_without_dropping_or_marking(self):
        first = make_observation()
        runtime = RequirementGroundingRuntime(
            processed_observation_ids=(first.observation_id,)
        )
        second = make_observation(sequence=2, raw={"query": "second"})
        candidate_patch = RequirementGroundingPatch(
            patch_id="limit-patch",
            base_revision=0,
            source_observation_ids=(second.observation_id,),
        )
        with mock_patch(
            "valibra_agent.requirement_grounding.reducer.MAX_PROCESSED_OBSERVATIONS",
            1,
        ):
            with self.assertRaisesRegex(ReductionError, "limit exceeded"):
                apply_patch(runtime, candidate_patch)
        self.assertEqual(runtime.processed_observation_ids, (first.observation_id,))

    def test_total_runtime_size_limit_rejects_atomically(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        candidate_patch = RequirementGroundingPatch(
            patch_id="size-patch",
            base_revision=0,
            source_observation_ids=(observation.observation_id,),
        )
        with mock_patch(
            "valibra_agent.requirement_grounding.reducer.MAX_RUNTIME_JSON_BYTES",
            1,
        ):
            with self.assertRaisesRegex(ReductionError, "serialized-size"):
                apply_patch(runtime, candidate_patch)
        self.assertEqual(runtime, RequirementGroundingRuntime())

    def test_phase2_supersede_and_reopen_preserve_unrelated_state(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        candidates = (
            make_candidate("cand-a", "A", "effect-a"),
            make_candidate("cand-b", "B", "effect-b"),
        )
        ambiguity = make_ambiguity(
            status="resolved",
            resolution="cand-a",
            candidates=candidates,
        )
        primary = make_slot(
            interpretation="A",
            ambiguity_refs=("amb-1",),
        )
        unrelated = make_slot(slot_id="slot-2", interpretation="keep-me")
        initial = apply_patch(
            runtime,
            graph_patch(
                runtime,
                observation,
                ambiguity=ambiguity,
                slot=primary,
                extra_slots=(unrelated,),
            ),
        )
        prior_evidence = initial.grounding_state.evidence
        phase2_observation = make_observation(
            sequence=2,
            raw={"follow_up": "caller-authorized"},
        )
        transition_patch = RequirementGroundingPatch(
            patch_id="phase2-transition",
            base_revision=initial.grounding_revision,
            source_observation_ids=(phase2_observation.observation_id,),
            phase_transition=PhaseTransition(
                supersede_slot_ids=("slot-1",),
                reopen_ambiguity_ids=("amb-1",),
                reason="follow-up changes scope",
                sequence=2,
            ),
        )
        transitioned = apply_patch(initial, transition_patch)
        slots = {
            slot.slot_id: slot
            for slot in transitioned.grounding_state.requirement_frame.value_slots
        }
        reopened = transitioned.grounding_state.ambiguity_index[0]
        self.assertEqual(transitioned.phase, 2)
        self.assertEqual(transitioned.grounding_revision, 2)
        self.assertEqual(slots["slot-1"].lifecycle, "superseded")
        self.assertEqual(slots["slot-1"].last_updated_phase, 2)
        self.assertEqual(slots["slot-2"], unrelated)
        self.assertEqual(transitioned.grounding_state.evidence, prior_evidence)
        self.assertEqual(reopened.status, "unresolved")
        self.assertIsNone(reopened.resolution)
        self.assertEqual(reopened.reopen_reason, "follow-up changes scope")
        validate_runtime(transitioned)


if __name__ == "__main__":
    unittest.main()
