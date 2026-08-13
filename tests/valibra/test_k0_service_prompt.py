import unittest

from tests.valibra.k0_fixtures import (
    graph_patch,
    make_ambiguity,
    make_candidate,
    make_evidence,
    make_observation,
    make_slot,
)
from valibra_agent.requirement_grounding.models import (
    RequirementFrame,
    RequirementGroundingRuntime,
    RequirementGroundingState,
)
from valibra_agent.requirement_grounding.prompt_view import render_prompt_view
from valibra_agent.requirement_grounding.reducer import apply_patch
from valibra_agent.requirement_grounding.service import process_observation
from valibra_agent.requirement_grounding.telemetry import (
    increment_metrics,
    record_failure,
)
from valibra_agent.requirement_grounding.updater import NoOpUpdater


class NoOpAndTelemetryTests(unittest.TestCase):
    def test_noop_marks_observation_without_business_revision(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        state_before = runtime.grounding_state
        patch = NoOpUpdater().propose(
            observation,
            runtime.grounding_state,
            base_revision=runtime.grounding_revision,
        )
        updated = apply_patch(runtime, patch)
        self.assertEqual(updated.grounding_state, state_before)
        self.assertEqual(updated.grounding_revision, 0)
        self.assertEqual(updated.requirement_revision, 0)
        self.assertEqual(
            updated.processed_observation_ids,
            (observation.observation_id,),
        )

    def test_metrics_and_error_never_increment_business_revision(self):
        runtime = RequirementGroundingRuntime()
        with_metrics = increment_metrics(runtime, observations_seen=1)
        with_error = record_failure(
            with_metrics,
            stage="updater",
            exception=RuntimeError("Bearer top-secret-token"),
            sequence=4,
        )
        self.assertEqual(with_metrics.grounding_revision, 0)
        self.assertEqual(with_error.grounding_revision, 0)
        self.assertEqual(with_metrics.requirement_revision, 0)
        self.assertEqual(with_error.requirement_revision, 0)
        self.assertEqual(with_error.grounding_state, runtime.grounding_state)
        self.assertNotIn("top-secret-token", with_error.model_dump_json())
        self.assertIn("<redacted>", with_error.last_error.message_preview)

    def test_service_noop_success_and_duplicate_are_idempotent(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        first = process_observation(runtime, observation)
        second = process_observation(first.runtime, observation)
        self.assertEqual(first.status, "processed")
        self.assertEqual(first.runtime.grounding_revision, 0)
        self.assertEqual(second.status, "duplicate")
        self.assertIs(second.runtime, first.runtime)

    def test_updater_exception_is_fail_open(self):
        class BrokenUpdater:
            def propose(self, observation, state, **kwargs):
                object.__setattr__(
                    state,
                    "requirement_frame",
                    RequirementFrame(
                        value_slots=(make_slot(slot_id="pollution"),)
                    ),
                )
                raise RuntimeError("synthetic updater failure")

        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        result = process_observation(
            runtime,
            observation,
            updater=BrokenUpdater(),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.processed_observation_ids, ())
        self.assertEqual(result.runtime.last_error.stage, "updater")

    def test_reducer_exception_is_fail_open(self):
        def broken_reducer(candidate_runtime, candidate_patch):
            object.__setattr__(candidate_runtime, "grounding_revision", 999)
            raise RuntimeError("synthetic reducer failure")

        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        result = process_observation(
            runtime,
            observation,
            reducer=broken_reducer,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.processed_observation_ids, ())
        self.assertEqual(result.runtime.last_error.stage, "reducer")


class PromptViewTests(unittest.TestCase):
    def _state(self):
        observation = make_observation()
        evidence = make_evidence(
            observation,
            summary="RAW PRIVATE LARGE TEXT MUST NOT APPEAR " + "x" * 300,
        )
        first = make_slot(slot_id="slot-b", interpretation="B")
        second = make_slot(slot_id="slot-a", interpretation="A")
        return RequirementGroundingState(
            requirement_frame=RequirementFrame(value_slots=(first, second)),
            evidence=(evidence,),
        )

    def test_prompt_view_is_stable_and_sorted(self):
        state = self._state()
        first = render_prompt_view(state)
        second = render_prompt_view(state)
        self.assertEqual(first, second)
        self.assertLess(first.index('"A"'), first.index('"B"'))
        self.assertNotIn("slot-a", first)
        self.assertNotIn("slot-b", first)

    def test_prompt_view_is_strictly_bounded(self):
        state = self._state()
        for limit in (0, 1, 8, 32, 80, 200):
            with self.subTest(limit=limit):
                rendered = render_prompt_view(state, max_chars=limit, max_items=1)
                self.assertLessEqual(len(rendered), limit)

    def test_prompt_view_never_renders_raw_evidence_or_log_reference(self):
        rendered = render_prompt_view(self._state())
        self.assertNotIn("RAW PRIVATE LARGE TEXT", rendered)
        self.assertNotIn("audit://unit-test", rendered)
        self.assertNotIn("x" * 50, rendered)

    def test_item_limit_is_deterministic(self):
        state = self._state()
        rendered = render_prompt_view(state, max_chars=1000, max_items=1)
        self.assertIn('"A"', rendered)
        self.assertNotIn('"B"', rendered)
        self.assertIn("... omitted=1", rendered)


if __name__ == "__main__":
    unittest.main()
