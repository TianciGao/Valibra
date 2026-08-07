import asyncio
import copy
import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from system_agent.adk_runtime import AdkRuntime as BaselineAdkRuntime
from valibra_agent import grounding_callbacks
from valibra_agent import adk_runtime as valibra_runtime_module
from valibra_agent.adk_runtime import AdkRuntime
from valibra_agent.requirement_grounding import linguistic_hints
from valibra_agent.requirement_grounding import updater as updater_module
from valibra_agent.requirement_grounding.linguistic_hints import (
    extract_linguistic_hints,
)
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.service import (
    process_observation,
    process_phase_transition,
)
from valibra_agent.requirement_grounding.updater import RuleUpdater


_RULE_MODE_PATCHER = None


def setUpModule():
    """P4.1 回归固定验证 Rule Shadow，不依赖本机 .env。"""

    global _RULE_MODE_PATCHER
    _RULE_MODE_PATCHER = patch.dict(
        os.environ,
        {"GROUNDING_UPDATER_MODE": ""},
    )
    _RULE_MODE_PATCHER.start()


def tearDownModule():
    if _RULE_MODE_PATCHER is not None:
        _RULE_MODE_PATCHER.stop()


def _observation(
    text,
    *,
    observation_type="user_query",
    phase=1,
    sequence=1,
):
    tool_fields = {}
    if observation_type == "user_answer":
        tool_fields = {
            "function_call_id": f"call-{sequence}",
            "tool_name": "ask_user",
        }
    return build_observation(
        task_id="task-rule",
        observation_type=observation_type,
        phase=phase,
        sequence=sequence,
        source="synthetic_unit_input",
        raw=text,
        summary=text,
        **tool_fields,
    )


def _all_slots(runtime):
    frame = runtime.grounding_state.requirement_frame
    return (*frame.value_slots, *frame.schema_slots, *frame.operation_slots)


class LinguisticHintRuleTests(unittest.TestCase):
    def assert_categories(self, text, expected):
        actual = {hint.category for hint in extract_linguistic_hints(text)}
        self.assertTrue(set(expected).issubset(actual), (expected, actual))

    def test_time_rule_positive_and_negative(self):
        self.assert_categories(
            "Show orders from 2024 and the past 30 days.",
            {"time"},
        )
        self.assertNotIn(
            "time",
            {hint.category for hint in extract_linguistic_hints("Show order labels.")},
        )

    def test_comparison_order_and_top_k_rules_have_safe_negatives(self):
        self.assert_categories(
            "Return top 5 products above 100 sorted descending.",
            {"comparison", "ordering", "top_k"},
        )
        categories = {
            hint.category
            for hint in extract_linguistic_hints("Product quality is important.")
        }
        self.assertTrue({"comparison", "ordering", "top_k"}.isdisjoint(categories))

    def test_aggregation_rule_positive_and_negative(self):
        self.assert_categories("Count orders and show average value.", {"aggregation"})
        self.assertNotIn(
            "aggregation",
            {hint.category for hint in extract_linguistic_hints("Describe order labels.")},
        )

    def test_negation_and_range_rules_have_safe_negatives(self):
        self.assert_categories(
            "Show orders excluding returns between 10 and 20.",
            {"negation", "range"},
        )
        categories = {
            hint.category
            for hint in extract_linguistic_hints("Show the standard customer category.")
        }
        self.assertTrue({"negation", "range"}.isdisjoint(categories))

    def test_conservative_schema_candidate_positive_and_negative(self):
        hints = extract_linguistic_hints(
            "Show customer names ordered by revenue descending."
        )
        candidates = [hint for hint in hints if hint.category == "schema_candidate"]
        self.assertEqual([hint.mention for hint in candidates], ["customer names"])
        self.assertNotIn(
            "schema_candidate",
            {
                hint.category
                for hint in extract_linguistic_hints(
                    "Can you help me understand this request?"
                )
            },
        )
        self.assertNotIn(
            "schema_candidate",
            {hint.category for hint in extract_linguistic_hints("Show data.")},
        )

    def test_same_input_has_identical_hints_and_ids(self):
        text = "Show customer names in 2024, top 5 by total descending."
        first = extract_linguistic_hints(text)
        second = extract_linguistic_hints(text)
        self.assertEqual(first, second)
        self.assertEqual(
            [hint.hint_id for hint in first],
            [hint.hint_id for hint in second],
        )

    def test_hint_module_is_standard_library_only_and_bounded(self):
        source = inspect.getsource(linguistic_hints)
        for forbidden in ("spacy", "stanza", "torch", "httpx", "requests", "litellm"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source.lower())
        with self.assertRaisesRegex(ValueError, "character limit"):
            extract_linguistic_hints("x" * 4097)
        many_years = " ".join(str(year) for year in range(1900, 1933))
        with self.assertRaisesRegex(ValueError, "item limit"):
            extract_linguistic_hints(many_years)

    def test_rule_sources_do_not_access_hidden_data_tools_or_prompt_view(self):
        source = "\n".join(
            (
                inspect.getsource(linguistic_hints),
                inspect.getsource(updater_module),
                inspect.getsource(valibra_runtime_module),
            )
        )
        for forbidden in (
            "task_data",
            "follow_up",
            "sol_sql",
            "test_cases",
            "render_prompt_view",
            "get_schema(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class RuleUpdaterTests(unittest.TestCase):
    def test_same_input_state_produces_identical_patch_and_ids(self):
        observation = _observation(
            "Show customer names in 2024 top 5 by total descending."
        )
        state = RequirementGroundingRuntime().grounding_state
        first = RuleUpdater().propose(observation, state, base_revision=0)
        second = RuleUpdater().propose(observation, state, base_revision=0)
        self.assertEqual(first, second)
        self.assertEqual(first.patch_id, second.patch_id)
        self.assertEqual(
            [slot.slot_id for slot in first.slot_additions],
            [slot.slot_id for slot in second.slot_additions],
        )

    def test_rule_creates_only_provisional_unbound_frame(self):
        observation = _observation(
            "Show customer names from 2024, top 5 by total descending, "
            "excluding returns above 100."
        )
        result = process_observation(
            RequirementGroundingRuntime(),
            observation,
            updater=RuleUpdater(),
        )
        self.assertEqual(result.status, "processed")
        self.assertGreater(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.grounding_state.ambiguity_index, ())
        slots = _all_slots(result.runtime)
        self.assertGreater(len(slots), 0)
        for slot in slots:
            self.assertEqual(slot.grounding_status, "hypothesized")
            self.assertEqual(slot.origin, "rule_provisional")
            self.assertEqual(slot.ambiguity_refs, ())
        for slot in result.runtime.grounding_state.requirement_frame.schema_slots:
            self.assertEqual(slot.binding_type, "unknown")
            self.assertIsNone(slot.bound_identifier)

    def test_uncertain_text_is_noop_instead_of_guessing(self):
        observation = _observation("Please help with this request.")
        result = process_observation(
            RequirementGroundingRuntime(),
            observation,
            updater=RuleUpdater(),
        )
        self.assertEqual(result.status, "processed")
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(_all_slots(result.runtime), ())
        self.assertEqual(result.runtime.grounding_state.ambiguity_index, ())

    def test_duplicate_observation_is_idempotent(self):
        observation = _observation("Show customer names in 2024.")
        first = process_observation(
            RequirementGroundingRuntime(), observation, updater=RuleUpdater()
        )
        replay = process_observation(first.runtime, observation, updater=RuleUpdater())
        self.assertEqual(replay.status, "duplicate")
        self.assertIs(replay.runtime, first.runtime)

    def test_user_answer_updates_existing_slot_without_duplicate_id(self):
        query = _observation("Show customer names from 2024.")
        first = process_observation(
            RequirementGroundingRuntime(), query, updater=RuleUpdater()
        )
        original = first.runtime.grounding_state.requirement_frame.value_slots[0]
        answer = _observation(
            "Use 2023 instead.",
            observation_type="user_answer",
            sequence=2,
        )
        second = process_observation(first.runtime, answer, updater=RuleUpdater())
        values = second.runtime.grounding_state.requirement_frame.value_slots
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].slot_id, original.slot_id)
        self.assertEqual(values[0].mention, "2023")
        self.assertEqual(len(values[0].evidence_refs), 2)
        self.assertEqual(second.runtime.grounding_state.ambiguity_index, ())

    def test_phase_two_addition_preserves_unrelated_phase_one_state_and_evidence(self):
        query = _observation("Show customer names from 2024.")
        first = process_observation(
            RequirementGroundingRuntime(), query, updater=RuleUpdater()
        )
        original_slots = _all_slots(first.runtime)
        original_evidence = first.runtime.grounding_state.evidence
        transition = build_observation(
            task_id="task-rule",
            observation_type="phase_transition",
            phase=2,
            sequence=2,
            source="synthetic_lifecycle",
            raw={"phase_before": 1, "phase_after": 2},
        )
        phase_two = process_phase_transition(first.runtime, transition).runtime
        follow_up = _observation(
            "Return total value descending.",
            phase=2,
            sequence=3,
        )
        updated = process_observation(
            phase_two,
            follow_up,
            updater=RuleUpdater(),
        ).runtime
        self.assertEqual(updated.phase, 2)
        for slot in original_slots:
            self.assertIn(slot, _all_slots(updated))
        for evidence in original_evidence:
            self.assertIn(evidence, updated.grounding_state.evidence)
        self.assertEqual(updated.grounding_state.ambiguity_index, ())

    def test_hint_exception_is_fail_open(self):
        observation = _observation("Show orders in 2024.")
        runtime = RequirementGroundingRuntime()
        with patch(
            "valibra_agent.requirement_grounding.updater.extract_linguistic_hints",
            side_effect=RuntimeError("synthetic hint failure"),
        ):
            result = process_observation(runtime, observation, updater=RuleUpdater())
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.processed_observation_ids, ())

    def test_rule_updater_exception_is_fail_open(self):
        class BrokenRuleUpdater(RuleUpdater):
            def propose(self, observation, state, **kwargs):
                raise RuntimeError("synthetic rule failure")

        observation = _observation("Show orders in 2024.")
        runtime = RequirementGroundingRuntime()
        result = process_observation(
            runtime,
            observation,
            updater=BrokenRuleUpdater(),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(result.runtime.grounding_revision, 0)


class RuleShadowRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_turn_uses_exact_message_and_never_mutates_llm_request(self):
        runtime = AdkRuntime()
        state = {"task_id": "task-runtime", "current_phase": 1}
        request = {"contents": [{"role": "user", "text": "unchanged"}]}
        request_before = copy.deepcopy(request)
        seen_messages = []
        delegate = AsyncMock(return_value=None)

        message = (
            "Database: archive_1999\n"
            "Task ID: task-2001\n\n"
            "User Query:\nShow customer names from 2024, top 5.\n\n"
            "You have a budget of 10.0 bird-coins."
        )

        async def fake_baseline_turn(
            runtime_self, task_id, mode, message, **kwargs
        ):
            seen_messages.append(message)
            await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=state),
                request,
            )
            # A tool loop may cause another model call in the same turn; the
            # bound user message must still be consumed exactly once.
            await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=state),
                request,
            )
            return {"state": state, "response": "baseline"}

        with (
            patch.object(
                BaselineAdkRuntime,
                "run_turn",
                new=fake_baseline_turn,
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            result = await runtime.run_turn(
                task_id="task-runtime",
                mode="a-interact",
                message=message,
            )

        self.assertEqual(seen_messages, [message])
        self.assertEqual(request, request_before)
        self.assertEqual(result["response"], "baseline")
        self.assertEqual(delegate.await_count, 2)
        grounding = RequirementGroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(len(grounding.processed_observation_ids), 1)
        self.assertEqual(
            [slot.mention for slot in grounding.grounding_state.requirement_frame.value_slots],
            ["2024"],
        )
        self.assertNotIn("1999", grounding.model_dump_json())
        self.assertNotIn("2001", grounding.model_dump_json())

    async def test_concurrent_run_turn_messages_do_not_cross_tasks(self):
        runtime = AdkRuntime()
        states = {
            "task-a": {"task_id": "task-a", "current_phase": 1},
            "task-b": {"task_id": "task-b", "current_phase": 1},
        }

        async def fake_baseline_turn(
            runtime_self, task_id, mode, message, **kwargs
        ):
            await asyncio.sleep(0)
            await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=states[task_id]),
                {"message": message},
            )
            return {"state": states[task_id], "response": task_id}

        with (
            patch.object(
                BaselineAdkRuntime,
                "run_turn",
                new=fake_baseline_turn,
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                AsyncMock(return_value=None),
            ),
        ):
            await asyncio.gather(
                runtime.run_turn("task-a", "a-interact", "Show orders in 2022."),
                runtime.run_turn("task-b", "a-interact", "Show orders in 2023."),
            )

        first = RequirementGroundingRuntime.model_validate(
            states["task-a"][grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        second = RequirementGroundingRuntime.model_validate(
            states["task-b"][grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(
            first.grounding_state.requirement_frame.value_slots[0].mention,
            "2022",
        )
        self.assertEqual(
            second.grounding_state.requirement_frame.value_slots[0].mention,
            "2023",
        )

    async def test_user_message_rule_failure_is_fail_open_to_baseline(self):
        runtime = AdkRuntime()
        state = {"task_id": "task-fail", "current_phase": 1}
        request = {"prompt": "unchanged"}
        request_before = copy.deepcopy(request)
        delegate = AsyncMock(return_value="baseline-result")

        async def fake_baseline_turn(
            runtime_self, task_id, mode, message, **kwargs
        ):
            value = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=state), request
            )
            return {"response": value, "state": state}

        with (
            patch.object(
                BaselineAdkRuntime,
                "run_turn",
                new=fake_baseline_turn,
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
            patch(
                "valibra_agent.grounding_callbacks.process_observation",
                side_effect=RuntimeError("synthetic user rule failure"),
            ),
        ):
            result = await runtime.run_turn(
                "task-fail",
                "a-interact",
                "Show orders in 2024.",
            )

        self.assertEqual(result["response"], "baseline-result")
        self.assertEqual(request, request_before)
        delegate.assert_awaited_once()
        grounding = RequirementGroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertIsNotNone(grounding.last_error)

    async def test_ask_user_answer_uses_rule_updater_without_duplicate_slot(self):
        state = {
            "task_id": "task-rule",
            "current_phase": 1,
            "budget_remaining": 10.0,
            "initial_budget": 10.0,
            "tool_trajectory": [],
            "system_agent_llm_calls": [{"actions": []}],
            "_active_llm_call_index": 0,
        }
        query = _observation("Show customer names from 2024.")
        initial = process_observation(
            RequirementGroundingRuntime(), query, updater=RuleUpdater()
        ).runtime
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY] = initial.model_dump(
            mode="json"
        )
        context = SimpleNamespace(
            state=state,
            function_call_id="call-answer",
            invocation_id="inv-answer",
        )
        tool = SimpleNamespace(name="ask_user")
        args = {"question": "Which year?"}
        await grounding_callbacks.before_tool_callback(tool, args, context)
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            override = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                "Use 2023 instead.",
            )
        self.assertIn("Remaining budget", override)
        updated = RequirementGroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        values = updated.grounding_state.requirement_frame.value_slots
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].mention, "2023")
        self.assertEqual(updated.grounding_state.ambiguity_index, ())


if __name__ == "__main__":
    unittest.main()
