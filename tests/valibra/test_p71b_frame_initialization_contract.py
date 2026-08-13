"""P7.1b tests for the one-shot initial Frame result contract.

These tests intentionally stop at Runtime metadata.  P7.1c owns the LLM
``proposal_outcome`` form change, so a legacy V1 empty form must not be
reinterpreted here as either of the two explicit ``empty`` outcomes.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from tests.valibra.k0_fixtures import (
    make_evidence,
    make_observation,
    make_slot,
)
from tests.valibra.test_p4d_llm_callback_shadow import (
    FakeClient,
    QUERY,
    QUERY_CONTENT,
    _content,
    _llm_config,
    _runtime,
    _state,
)
from valibra_agent import grounding_callbacks
from valibra_agent.requirement_grounding.models import (
    Observation,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    migrate_requirement_grounding_runtime,
)
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.reducer import apply_patch
from valibra_agent.requirement_grounding.service import (
    GroundingControlError,
    process_observation_with_llm,
    set_frame_initialization_result,
)
from valibra_agent.requirement_grounding.updater import (
    GroundingProviderError,
    LLMFrameUpdateError,
    LLMUpdater,
)


EMPTY_REASONS = (
    "insufficient_information",
    "no_extractable_requirement",
)
FAILED_REASONS = (
    "configuration_error",
    "provider_error",
    "timeout",
    "transport_format_invalid",
    "json_invalid",
    "duplicate_json_key",
    "form_validation_failed",
    "mention_validation_failed",
    "patch_rejected",
    "reducer_rejected",
    "unexpected_error",
)


def _runtime_with_initial_frame():
    runtime = RequirementGroundingRuntime()
    observation = make_observation(
        sequence=1,
        raw={"query": "show rows from 2024"},
    )
    evidence = make_evidence(observation, evidence_id="initial-evidence")
    slot = make_slot(
        slot_id="initial-slot",
        interpretation="calendar year 2024",
        evidence_refs=(evidence.evidence_id,),
    )
    patch = RequirementGroundingPatch(
        patch_id="initial-frame",
        base_revision=runtime.grounding_revision,
        source_observation_ids=(observation.observation_id,),
        slot_additions=(slot,),
        evidence_additions=(evidence,),
    )
    return observation, apply_patch(runtime, patch)


def _copy_runtime(
    runtime: RequirementGroundingRuntime,
    **updates: object,
) -> RequirementGroundingRuntime:
    payload = runtime.model_dump(mode="python")
    payload.update(updates)
    return RequirementGroundingRuntime.model_validate(payload)


class FrameInitializationModelContractTests(unittest.TestCase):
    def test_default_is_not_attempted_and_json_round_trips(self):
        runtime = RequirementGroundingRuntime()

        self.assertEqual(runtime.frame_initialization_status, "not_attempted")
        self.assertIsNone(runtime.frame_initialization_reason)
        self.assertIsNone(runtime.frame_initialization_observation_id)
        self.assertEqual(
            RequirementGroundingRuntime.model_validate_json(
                runtime.model_dump_json()
            ),
            runtime,
        )

    def test_legacy_v1_runtime_without_initialization_fields_is_compatible(self):
        payload = RequirementGroundingRuntime().model_dump(mode="json")
        payload["schema_version"] = "1.0"
        payload.pop("requirement_revision")
        payload.pop("frame_initialization_status")
        payload.pop("frame_initialization_reason")
        payload.pop("frame_initialization_observation_id")

        migration = migrate_requirement_grounding_runtime(payload)
        restored = migration.runtime

        # Compatibility defaults are unknown historical metadata, not proof
        # that a historical V1 session never attempted initialization.
        self.assertTrue(migration.legacy_unknown_history)
        self.assertEqual(restored.schema_version, "1.1")
        self.assertEqual(restored.requirement_revision, 0)
        self.assertEqual(restored.frame_initialization_status, "not_attempted")
        self.assertIsNone(restored.frame_initialization_reason)
        self.assertIsNone(restored.frame_initialization_observation_id)

    def test_all_frozen_terminal_reason_pairs_are_json_safe(self):
        observation = make_observation()
        ready_observation, ready_runtime = _runtime_with_initial_frame()
        legal = [
            set_frame_initialization_result(
                ready_runtime,
                status="ready",
                reason=None,
                observation=ready_observation,
            ),
            *(
                set_frame_initialization_result(
                    RequirementGroundingRuntime(),
                    status="empty",
                    reason=reason,
                    observation=observation,
                )
                for reason in EMPTY_REASONS
            ),
            *(
                set_frame_initialization_result(
                    RequirementGroundingRuntime(),
                    status="failed",
                    reason=reason,
                    observation=observation,
                )
                for reason in FAILED_REASONS
            ),
        ]

        for runtime in legal:
            with self.subTest(
                status=runtime.frame_initialization_status,
                reason=runtime.frame_initialization_reason,
            ):
                self.assertEqual(
                    RequirementGroundingRuntime.model_validate_json(
                        runtime.model_dump_json()
                    ),
                    runtime,
                )

    def test_status_reason_and_observation_invariants_reject_bad_shapes(self):
        observation_id = make_observation().observation_id
        invalid = (
            {
                "frame_initialization_status": "not_attempted",
                "frame_initialization_reason": "provider_error",
                "frame_initialization_observation_id": observation_id,
            },
            {
                "frame_initialization_status": "ready",
                "frame_initialization_reason": "provider_error",
                "frame_initialization_observation_id": observation_id,
            },
            {
                "frame_initialization_status": "ready",
                "frame_initialization_reason": None,
                "frame_initialization_observation_id": None,
            },
            {
                "frame_initialization_status": "empty",
                "frame_initialization_reason": None,
                "frame_initialization_observation_id": observation_id,
            },
            {
                "frame_initialization_status": "empty",
                "frame_initialization_reason": "provider_error",
                "frame_initialization_observation_id": observation_id,
            },
            {
                "frame_initialization_status": "failed",
                "frame_initialization_reason": None,
                "frame_initialization_observation_id": observation_id,
            },
            {
                "frame_initialization_status": "failed",
                "frame_initialization_reason": "insufficient_information",
                "frame_initialization_observation_id": observation_id,
            },
            {
                "frame_initialization_status": "unknown",
                "frame_initialization_reason": None,
                "frame_initialization_observation_id": observation_id,
            },
        )

        for updates in invalid:
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                _copy_runtime(RequirementGroundingRuntime(), **updates)


class FrameInitializationStateMachineTests(unittest.TestCase):
    def test_terminal_writer_requires_phase_one_user_query(self):
        invalid_observations = (
            build_observation(
                task_id="task-1",
                observation_type="schema",
                phase=1,
                sequence=1,
                source="unit_test",
                raw={"schema": "bounded"},
                function_call_id="call-1",
                tool_name="get_schema",
            ),
            build_observation(
                task_id="task-1",
                observation_type="user_query",
                phase=2,
                sequence=1,
                source="unit_test",
                raw={"query": "test"},
            ),
        )
        for observation in invalid_observations:
            with self.subTest(
                observation_type=observation.observation_type,
                phase=observation.phase,
            ), self.assertRaises(GroundingControlError):
                set_frame_initialization_result(
                    RequirementGroundingRuntime(),
                    status="failed",
                    reason="unexpected_error",
                    observation=observation,
                )

    def test_ready_requires_a_nonempty_initial_frame(self):
        observation = make_observation()

        with self.assertRaises(GroundingControlError):
            set_frame_initialization_result(
                RequirementGroundingRuntime(),
                status="ready",
                reason=None,
                observation=observation,
            )

    def test_empty_transition_rejects_an_already_populated_initial_frame(self):
        observation, runtime = _runtime_with_initial_frame()

        with self.assertRaises(GroundingControlError):
            set_frame_initialization_result(
                runtime,
                status="empty",
                reason="insufficient_information",
                observation=observation,
            )

    def test_terminal_transition_is_atomic_and_does_not_change_revisions(self):
        observation, runtime = _runtime_with_initial_frame()
        before_state = runtime.grounding_state
        before_processed = runtime.processed_observation_ids
        before_metrics = runtime.metrics
        before_error = runtime.last_error

        updated = set_frame_initialization_result(
            runtime,
            status="ready",
            reason=None,
            observation=observation,
        )

        self.assertEqual(
            (updated.grounding_revision, updated.requirement_revision),
            (runtime.grounding_revision, runtime.requirement_revision),
        )
        self.assertEqual(updated.grounding_state, before_state)
        self.assertEqual(updated.processed_observation_ids, before_processed)
        self.assertEqual(updated.metrics, before_metrics)
        self.assertEqual(updated.last_error, before_error)

    def test_exact_terminal_replay_is_idempotent(self):
        observation = make_observation()
        terminal = set_frame_initialization_result(
            RequirementGroundingRuntime(),
            status="failed",
            reason="provider_error",
            observation=observation,
        )

        replayed = set_frame_initialization_result(
            terminal,
            status="failed",
            reason="provider_error",
            observation=observation,
        )

        self.assertIs(replayed, terminal)
        self.assertEqual(
            replayed.model_dump(mode="json"),
            terminal.model_dump(mode="json"),
        )

    def test_terminal_result_cannot_be_overwritten(self):
        first = make_observation(sequence=1)
        second = make_observation(sequence=2)
        terminal = set_frame_initialization_result(
            RequirementGroundingRuntime(),
            status="failed",
            reason="timeout",
            observation=first,
        )
        conflicts = (
            ("failed", "provider_error", first),
            ("failed", "timeout", second),
            ("empty", "insufficient_information", first),
        )

        for status, reason, observation in conflicts:
            with self.subTest(
                status=status,
                reason=reason,
                observation_id=observation.observation_id,
            ), self.assertRaises(GroundingControlError):
                set_frame_initialization_result(
                    terminal,
                    status=status,
                    reason=reason,
                    observation=observation,
                )
        self.assertEqual(terminal.frame_initialization_status, "failed")
        self.assertEqual(terminal.frame_initialization_reason, "timeout")
        self.assertEqual(
            terminal.frame_initialization_observation_id,
            first.observation_id,
        )

    def test_phase_two_and_later_frame_changes_do_not_rewrite_terminal_history(self):
        observation = make_observation()
        terminal = set_frame_initialization_result(
            RequirementGroundingRuntime(),
            status="empty",
            reason="no_extractable_requirement",
            observation=observation,
        )
        # The three fields describe the historical first Phase-1 attempt, not
        # whether the current Frame is empty forever.  A later phase may add a
        # Frame while the initialization result remains immutable.
        later_observation, later_frame = _runtime_with_initial_frame()
        later = _copy_runtime(
            later_frame,
            phase=2,
            frame_initialization_status=terminal.frame_initialization_status,
            frame_initialization_reason=terminal.frame_initialization_reason,
            frame_initialization_observation_id=(
                terminal.frame_initialization_observation_id
            ),
        )

        replayed = set_frame_initialization_result(
            later,
            status="empty",
            reason="no_extractable_requirement",
            observation=observation,
        )

        self.assertIs(replayed, later)
        self.assertEqual(replayed.phase, 2)
        self.assertTrue(
            replayed.grounding_state.requirement_frame.value_slots
        )
        self.assertNotEqual(
            later_observation.observation_id,
            observation.observation_id,
        )

    def test_legacy_empty_llm_form_is_failed_not_guessed_as_empty(self):
        observation = make_observation(
            raw={
                "value_slots": [],
                "schema_slots": [],
                "operation_slots": [],
                "ambiguities": [],
            }
        )
        runtime = RequirementGroundingRuntime()

        result = set_frame_initialization_result(
            runtime,
            status="failed",
            reason="form_validation_failed",
            observation=observation,
        )

        self.assertEqual(result.frame_initialization_status, "failed")
        self.assertEqual(
            result.frame_initialization_reason,
            "form_validation_failed",
        )
        self.assertNotIn(
            result.frame_initialization_reason,
            EMPTY_REASONS,
        )
        self.assertEqual(result.grounding_state, runtime.grounding_state)
        self.assertEqual(
            (result.grounding_revision, result.requirement_revision),
            (0, 0),
        )


def _llm_observation(
    *,
    task_id: str = "task-p71b-typed",
    sequence: int = 1,
) -> Observation:
    return make_observation(
        task_id=task_id,
        sequence=sequence,
        raw=QUERY,
    )


class TypedLLMFailureClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def _result(
        self,
        *,
        content: str = QUERY_CONTENT,
        error: BaseException | None = None,
        block: bool = False,
        timeout: str = "0.05",
        reducer=apply_patch,
    ):
        client = FakeClient(content=content, error=error, block=block)
        updater = LLMUpdater(client, _llm_config(timeout=timeout))
        result = await process_observation_with_llm(
            RequirementGroundingRuntime(),
            _llm_observation(),
            updater=updater,
            reducer=reducer,
        )
        return result, client

    async def test_json_duplicate_form_and_mention_failures_are_typed(self):
        duplicate = (
            '{"proposal_outcome":"no_extractable_requirement",'
            '"value_slots":[],"value_slots":[],"schema_slots":[],'
            '"operation_slots":[],"ambiguities":[]}'
        )
        form_invalid = json.dumps(
            {
                "proposal_outcome": "populated",
                "value_slots": [],
                "schema_slots": [],
                "operation_slots": [
                    {
                        "slot_role": "ordering",
                        "mention": "sorted by total descending",
                        "interpretation": "descending order",
                        "operation_type": "sort",
                        "parameters": {"direction": "desc"},
                    }
                ],
                "ambiguities": [],
            }
        )
        scenarios = (
            ("json_invalid", "{"),
            ("duplicate_json_key", duplicate),
            ("form_validation_failed", form_invalid),
            ("mention_validation_failed", _content(value_mention="2099")),
        )

        for expected, content in scenarios:
            with self.subTest(reason=expected):
                result, client = await self._result(content=content)
                self.assertEqual(client.calls, 1)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.failure_reason, expected)
                self.assertEqual(result.runtime.grounding_revision, 0)
                self.assertEqual(result.runtime.requirement_revision, 0)
                self.assertEqual(
                    result.runtime.frame_initialization_status,
                    "not_attempted",
                )

    async def test_provider_timeout_and_reducer_failures_are_typed(self):
        def reject_reducer(_runtime, _patch):
            raise ValueError("synthetic reducer rejection")

        scenarios = (
            {
                "expected": "provider_error",
                "error": GroundingProviderError("offline provider error"),
            },
            {
                "expected": "timeout",
                "block": True,
                "timeout": "0.001",
            },
            {
                "expected": "reducer_rejected",
                "reducer": reject_reducer,
            },
        )

        for raw_scenario in scenarios:
            scenario = dict(raw_scenario)
            expected = scenario.pop("expected")
            with self.subTest(reason=expected):
                result, client = await self._result(**scenario)
                self.assertEqual(client.calls, 1)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.failure_reason, expected)
                self.assertEqual(result.runtime.grounding_revision, 0)
                self.assertEqual(result.runtime.requirement_revision, 0)

    async def test_patch_construction_failure_is_typed(self):
        with patch(
            "valibra_agent.requirement_grounding.updater._llm_patch_from_proposal",
            side_effect=ValueError("synthetic patch construction rejection"),
        ):
            result, client = await self._result()

        self.assertEqual(client.calls, 1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_reason, "patch_rejected")
        self.assertIsInstance(
            LLMFrameUpdateError("patch_rejected"),
            ValueError,
        )
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.requirement_revision, 0)


async def _consume_bound_message(state: dict, message: str = QUERY):
    token = grounding_callbacks._bind_turn_message(
        state["task_id"],
        "a-interact",
        message,
    )
    try:
        first = await grounding_callbacks._consume_bound_user_message(state)
        second = await grounding_callbacks._consume_bound_user_message(state)
        return first, second
    finally:
        grounding_callbacks._reset_turn_message(token)


class FrameInitializationCallbackLifecycleTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_phase_one_rule_ready_and_no_hint_are_terminal(self):
        ready_state = _state("task-p71b-rule-ready")
        with patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}):
            ready_audit, ready_replay = await _consume_bound_message(ready_state)

        ready_runtime = _runtime(ready_state)
        self.assertIsNone(ready_replay)
        self.assertEqual(ready_audit["status"], "processed")
        self.assertEqual(ready_runtime.frame_initialization_status, "ready")
        self.assertIsNone(ready_runtime.frame_initialization_reason)
        self.assertGreater(ready_runtime.requirement_revision, 0)

        no_hint_state = _state("task-p71b-rule-no-hint")
        with patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}):
            no_hint_audit, no_hint_replay = await _consume_bound_message(
                no_hint_state,
                "hello",
            )

        no_hint_runtime = _runtime(no_hint_state)
        self.assertIsNone(no_hint_replay)
        self.assertEqual(no_hint_audit["status"], "failed")
        self.assertEqual(no_hint_runtime.frame_initialization_status, "failed")
        self.assertEqual(
            no_hint_runtime.frame_initialization_reason,
            "unexpected_error",
        )
        self.assertEqual(no_hint_runtime.requirement_revision, 0)

    async def test_invalid_mode_is_configuration_failure_without_rule_fallback(self):
        state = _state("task-p71b-invalid-mode")
        with patch.dict(
            os.environ,
            {"GROUNDING_UPDATER_MODE": "LLM"},
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertIsNone(second)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(runtime.frame_initialization_status, "failed")
        self.assertEqual(
            runtime.frame_initialization_reason,
            "configuration_error",
        )
        self.assertEqual(runtime.requirement_revision, 0)
        self.assertFalse(
            runtime.grounding_state.requirement_frame.value_slots
            or runtime.grounding_state.requirement_frame.schema_slots
            or runtime.grounding_state.requirement_frame.operation_slots
        )

    async def test_llm_builder_configuration_failure_is_terminal(self):
        state = _state("task-p71b-invalid-llm-config")
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                side_effect=ValueError("synthetic configuration failure"),
            ),
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertIsNone(second)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(runtime.frame_initialization_status, "failed")
        self.assertEqual(
            runtime.frame_initialization_reason,
            "configuration_error",
        )
        self.assertEqual(runtime.requirement_revision, 0)

    async def test_phase_one_llm_ready_is_recorded_once(self):
        state = _state("task-p71b-ready")
        client = FakeClient(content=QUERY_CONTENT)
        updater = LLMUpdater(client, _llm_config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(first["status"], "processed")
        self.assertEqual(runtime.frame_initialization_status, "ready")
        self.assertIsNone(runtime.frame_initialization_reason)
        self.assertEqual(
            runtime.frame_initialization_observation_id,
            first["observation_id"],
        )
        self.assertEqual(first["frame_initialization_status"], "ready")
        self.assertGreater(runtime.requirement_revision, 0)

    async def test_phase_one_typed_failure_records_failed_once(self):
        state = _state("task-p71b-failed")
        client = FakeClient(content="{")
        updater = LLMUpdater(client, _llm_config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(second)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(runtime.frame_initialization_status, "failed")
        self.assertEqual(runtime.frame_initialization_reason, "json_invalid")
        self.assertEqual(
            runtime.frame_initialization_observation_id,
            first["observation_id"],
        )
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertEqual(runtime.requirement_revision, 0)

    async def test_legacy_empty_form_is_failed_without_guessing_empty_reason(self):
        state = _state("task-p71b-old-empty")
        old_empty_form = json.dumps(
            {
                "value_slots": [],
                "schema_slots": [],
                "operation_slots": [],
                "ambiguities": [],
            }
        )
        client = FakeClient(content=old_empty_form)
        updater = LLMUpdater(client, _llm_config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertEqual(client.calls, 1)
        self.assertIsNone(second)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(runtime.frame_initialization_status, "failed")
        self.assertEqual(
            runtime.frame_initialization_reason,
            "form_validation_failed",
        )
        self.assertNotIn(runtime.frame_initialization_reason, EMPTY_REASONS)
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertEqual(runtime.requirement_revision, 0)

    async def test_phase_two_user_query_does_not_initialize(self):
        state = _state("task-p71b-phase-two", phase=2)
        client = FakeClient(content=QUERY_CONTENT)
        updater = LLMUpdater(client, _llm_config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(runtime.frame_initialization_status, "not_attempted")
        self.assertIsNone(runtime.frame_initialization_reason)
        self.assertIsNone(runtime.frame_initialization_observation_id)

    async def test_later_phase_one_query_cannot_overwrite_terminal_result(self):
        state = _state("task-p71b-terminal")
        initial = make_observation(task_id=state["task_id"])
        terminal = set_frame_initialization_result(
            RequirementGroundingRuntime(),
            status="failed",
            reason="timeout",
            observation=initial,
        )
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY] = terminal.model_dump(
            mode="json"
        )
        client = FakeClient(content=QUERY_CONTENT)
        updater = LLMUpdater(client, _llm_config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(runtime.frame_initialization_status, "failed")
        self.assertEqual(runtime.frame_initialization_reason, "timeout")
        self.assertEqual(
            runtime.frame_initialization_observation_id,
            initial.observation_id,
        )

    async def test_legacy_v1_runtime_remains_unknown_without_backfill(self):
        state = _state("task-p71b-legacy-unknown")
        legacy = RequirementGroundingRuntime().model_dump(mode="json")
        legacy["schema_version"] = "1.0"
        legacy.pop("requirement_revision")
        legacy.pop("frame_initialization_status")
        legacy.pop("frame_initialization_reason")
        legacy.pop("frame_initialization_observation_id")
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY] = legacy
        client = FakeClient(content=QUERY_CONTENT)
        updater = LLMUpdater(client, _llm_config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
        ):
            first, second = await _consume_bound_message(state)

        runtime = _runtime(state)
        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        # The compatibility defaults stay an explicit unknown sentinel.  New
        # code must not infer historical initialization from the current Frame.
        self.assertEqual(runtime.frame_initialization_status, "not_attempted")
        self.assertIsNone(runtime.frame_initialization_reason)
        self.assertIsNone(runtime.frame_initialization_observation_id)

        # A later Phase-1 turn must remain unknown as well.  The Session-level
        # sentinel survives serialization of the default compatibility fields.
        second_client = FakeClient(content=QUERY_CONTENT)
        second_updater = LLMUpdater(second_client, _llm_config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=second_updater,
            ),
        ):
            later_first, later_second = await _consume_bound_message(
                state,
                "Show the top 3 orders from 2023.",
            )
        later_runtime = _runtime(state)
        self.assertEqual(second_client.calls, 1)
        self.assertIsNotNone(later_first)
        self.assertIsNone(later_second)
        self.assertEqual(
            later_runtime.frame_initialization_status,
            "not_attempted",
        )
        self.assertIsNone(later_runtime.frame_initialization_reason)
        self.assertIsNone(later_runtime.frame_initialization_observation_id)


if __name__ == "__main__":
    unittest.main()
