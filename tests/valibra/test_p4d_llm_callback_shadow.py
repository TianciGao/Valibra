import asyncio
import copy
import inspect
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent import server as valibra_server
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.service import (
    process_observation_with_llm,
)
from valibra_agent.requirement_grounding.updater import (
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    LLMUpdater,
    load_grounding_llm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROMPT_SHA256 = (
    "b5c55caa8dd06d8d8e9c54670e118bae1f2e5753ed48a9a1329f1d89ff812e00"
)
EXPECTED_FORM_SHA256 = (
    "441a59c410a99ef0db53b8e974aeeeaea1bcd1735aabdc3cb51f59e0b6e069a2"
)


def _llm_config(*, timeout="0.05", max_calls="4"):
    return load_grounding_llm_config(
        PROJECT_ROOT,
        {
            "GROUNDING_UPDATER_MODE": "llm",
            "GROUNDING_MODEL_PRESET": "glm52_high_32768",
            "GROUNDING_TIMEOUT_SECONDS": timeout,
            "GROUNDING_MAX_TOKENS": "32768",
            "GROUNDING_MAX_CALLS_PER_TASK": max_calls,
            "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
        },
    )


def _content(
    *,
    value_mention=None,
    schema_mention=None,
    operation_mention=None,
    operation_type="order",
):
    value_slots = []
    if value_mention is not None:
        value_slots.append(
            {
                "slot_role": "time_constraint",
                "mention": value_mention,
                "interpretation": f"calendar year {value_mention}",
                "value_type": "time",
            }
        )
    schema_slots = []
    if schema_mention is not None:
        schema_slots.append(
            {
                "slot_role": "schema_candidate",
                "mention": schema_mention,
                "interpretation": f"candidate concept {schema_mention}",
            }
        )
    operation_slots = []
    if operation_mention is not None:
        operation_slots.append(
            {
                "slot_role": "ordering",
                "mention": operation_mention,
                "interpretation": "descending result order",
                "operation_type": operation_type,
                "parameters": {"direction": "desc"},
            }
        )
    return json.dumps(
        {
            "value_slots": value_slots,
            "schema_slots": schema_slots,
            "operation_slots": operation_slots,
            "ambiguities": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


QUERY = "Show customer names from 2024 sorted by total descending."
QUERY_CONTENT = _content(
    value_mention="2024",
    schema_mention="customer names",
    operation_mention="sorted by total descending",
)
ANSWER = "Use 2023 instead."
ANSWER_CONTENT = _content(value_mention="2023")


class FakeClient:
    provider_may_continue_after_cancel = False
    provider_may_bill_after_cancel = False

    def __init__(self, *, content=QUERY_CONTENT, error=None, block=False):
        self.content = content
        self.error = error
        self.block = block
        self.calls = 0
        self.requests = []

    async def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        await asyncio.sleep(0)
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        content = self.content(request) if callable(self.content) else self.content
        return {
            "content": content,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 9,
                "reasoning_tokens": 4,
                "total_tokens": 21,
                "cost": None,
            },
            "model": "openai/glm-5.2",
            "provider": "offline-fake",
            "credential_source": "file",
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "raw_audit_ref": "private://grounding-llm/offline-fake.json",
        }


def _state(task_id="task-p4d", *, phase=1, budget=20.0):
    return {
        "task_id": task_id,
        "current_phase": phase,
        "budget_remaining": budget,
        "initial_budget": budget,
        "tool_trajectory": [],
        "system_agent_llm_calls": [{"actions": []}],
        "_active_llm_call_index": 0,
    }


def _context(state, call_id):
    return SimpleNamespace(
        state=state,
        function_call_id=call_id,
        invocation_id=f"inv-{call_id}",
    )


def _runtime(state):
    return RequirementGroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


async def _run_bound_query(state, message, request, *, model_calls=1):
    token = grounding_callbacks._bind_turn_message(
        state["task_id"],
        "a-interact",
        message,
    )
    try:
        result = None
        for _ in range(model_calls):
            result = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=state),
                request,
            )
        return result
    finally:
        grounding_callbacks._reset_turn_message(token)


class LLMCallbackShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_mode_preserves_rule_shadow(self):
        state = _state("task-rule-default")
        delegate = AsyncMock(return_value="baseline")
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                side_effect=AssertionError("LLM path must stay disabled"),
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            result = await _run_bound_query(state, QUERY, {"prompt": "same"})

        self.assertEqual(result, "baseline")
        delegate.assert_awaited_once()
        runtime = _runtime(state)
        self.assertEqual(runtime.metrics.root.get("llm_updater_calls", 0), 0)
        self.assertEqual(
            runtime.grounding_state.requirement_frame.value_slots[0].mention,
            "2024",
        )

    async def test_user_query_calls_fake_once_across_multiple_before_model(self):
        state = _state("task-llm-query")
        request = {"contents": [{"text": "MAIN PROMPT MUST NOT CHANGE"}]}
        request_before = copy.deepcopy(request)
        client = FakeClient()
        updater = LLMUpdater(client, _llm_config())
        delegate_result = object()
        delegate = AsyncMock(return_value=delegate_result)
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            result = await _run_bound_query(
                state,
                QUERY,
                request,
                model_calls=2,
            )

        self.assertIs(result, delegate_result)
        self.assertEqual(client.calls, 1)
        self.assertEqual(delegate.await_count, 2)
        self.assertEqual(request, request_before)
        runtime = _runtime(state)
        self.assertGreater(runtime.grounding_revision, 0)
        self.assertEqual(runtime.metrics.root["llm_updater_calls"], 1)
        self.assertEqual(runtime.metrics.root["llm_updater_total_tokens"], 21)
        slots = runtime.grounding_state.requirement_frame
        self.assertTrue(
            all(
                item.grounding_status == "hypothesized"
                for item in (*slots.value_slots, *slots.operation_slots)
            )
        )
        self.assertEqual(slots.schema_slots[0].binding_type, "unknown")
        self.assertIsNone(slots.schema_slots[0].bound_identifier)
        self.assertEqual(slots.operation_slots[0].operation_type, "order")
        self.assertEqual(runtime.grounding_state.ambiguity_index, ())
        self.assertNotIn("system_agent_token_usage", state)
        serialized = json.dumps(state, sort_keys=True)
        self.assertNotIn(QUERY_CONTENT, serialized)
        self.assertNotIn("FakeClient", serialized)
        self.assertNotIn("api_key", serialized.lower())
        self.assertNotIn("prompt_view", serialized.lower())

    async def test_ask_user_uses_raw_answer_and_returns_baseline_override(self):
        state = _state("task-user-answer")
        context = _context(state, "call-answer")
        tool = SimpleNamespace(name="ask_user")
        args = {"question": "Which year?"}
        client = FakeClient(content=ANSWER_CONTENT)
        updater = LLMUpdater(client, _llm_config())
        override = {"baseline": "override-must-be-returned"}
        after_delegate = AsyncMock(return_value=override)
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                after_delegate,
            ),
        ):
            rejection = await grounding_callbacks.before_tool_callback(
                tool,
                args,
                context,
            )
            result = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                ANSWER,
            )

        self.assertIsNone(rejection)
        self.assertIs(result, override)
        after_delegate.assert_awaited_once_with(tool, args, context, ANSWER)
        self.assertEqual(client.calls, 1)
        self.assertIn(ANSWER, client.requests[0].prompt)
        self.assertNotIn("override-must-be-returned", client.requests[0].prompt)
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertGreater(runtime.grounding_revision, 0)
        self.assertEqual(
            runtime.grounding_state.requirement_frame.value_slots[0].mention,
            "2023",
        )

    async def test_non_user_tool_observations_are_noop_without_llm_budget(self):
        state = _state("task-non-user", budget=100.0)
        tools = [
            ("execute_sql", {"sql": "SELECT 1"}),
            ("get_schema", {}),
            ("get_all_column_meanings", {}),
            ("get_column_meaning", {"table_name": "t", "column_name": "c"}),
            ("get_all_external_knowledge_names", {}),
            ("get_knowledge_definition", {"knowledge_name": "k"}),
            ("get_all_knowledge_definitions", {}),
            ("submit_sql", {"sql": "SELECT 1"}),
        ]
        before_delegate = AsyncMock(return_value=None)
        after_delegate = AsyncMock(return_value=None)
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                side_effect=AssertionError("non-user Observation called LLM"),
            ),
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                before_delegate,
            ),
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                after_delegate,
            ),
        ):
            for index, (name, args) in enumerate(tools, start=1):
                tool = SimpleNamespace(name=name)
                context = _context(state, f"call-{index}")
                await grounding_callbacks.before_tool_callback(tool, args, context)
                await grounding_callbacks.after_tool_callback(
                    tool,
                    args,
                    context,
                    {"result": name},
                )

        self.assertEqual(before_delegate.await_count, len(tools))
        self.assertEqual(after_delegate.await_count, len(tools))
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertEqual(len(runtime.processed_observation_ids), len(tools))
        self.assertEqual(runtime.metrics.root.get("llm_updater_calls", 0), 0)
        frame = runtime.grounding_state.requirement_frame
        self.assertEqual(frame.value_slots, ())
        self.assertEqual(frame.schema_slots, ())
        self.assertEqual(frame.operation_slots, ())

    async def test_llm_failure_timeout_and_invalid_form_all_fail_open(self):
        scenarios = (
            ("provider-error", FakeClient(error=RuntimeError("offline")), "0.05"),
            ("timeout", FakeClient(block=True), "0.001"),
            ("invalid-json", FakeClient(content="not-json"), "0.05"),
        )
        for name, client, timeout in scenarios:
            with self.subTest(name=name):
                state = _state(f"task-{name}")
                context = _context(state, f"call-{name}")
                tool = SimpleNamespace(name="ask_user")
                args = {"question": "Which year?"}
                updater = LLMUpdater(client, _llm_config(timeout=timeout))
                override = object()
                with (
                    patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                    patch.object(
                        grounding_callbacks,
                        "_build_llm_updater",
                        return_value=updater,
                    ),
                    patch.object(
                        baseline_callbacks,
                        "after_tool_callback",
                        AsyncMock(return_value=override),
                    ),
                ):
                    await grounding_callbacks.before_tool_callback(
                        tool,
                        args,
                        context,
                    )
                    result = await grounding_callbacks.after_tool_callback(
                        tool,
                        args,
                        context,
                        ANSWER,
                    )

                self.assertIs(result, override)
                self.assertEqual(client.calls, 1)
                runtime = _runtime(state)
                self.assertEqual(runtime.pending_tool_calls, {})
                self.assertEqual(runtime.grounding_revision, 0)
                self.assertIsNotNone(runtime.last_error)
                self.assertEqual(
                    runtime.grounding_state.requirement_frame.value_slots,
                    (),
                )
                if name == "timeout":
                    self.assertEqual(
                        runtime.metrics.root["llm_updater_timeouts"],
                        1,
                    )

    async def test_call_limit_fails_without_rule_fallback_or_second_call(self):
        state = _state("task-call-limit")
        client = FakeClient()
        updater = LLMUpdater(client, _llm_config(max_calls="1"))
        delegate = AsyncMock(return_value=None)
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            await _run_bound_query(state, QUERY, {})
            second_token = grounding_callbacks._bind_turn_message(
                state["task_id"],
                "a-interact",
                "Show orders in 2025.",
            )
            try:
                await grounding_callbacks.before_model_callback(
                    SimpleNamespace(state=state),
                    {},
                )
            finally:
                grounding_callbacks._reset_turn_message(second_token)

        self.assertEqual(client.calls, 1)
        runtime = _runtime(state)
        self.assertEqual(runtime.grounding_revision, 1)
        self.assertEqual(runtime.metrics.root["llm_updater_calls"], 1)
        self.assertIsNotNone(runtime.last_error)
        mentions = {
            slot.mention
            for slot in runtime.grounding_state.requirement_frame.value_slots
        }
        self.assertEqual(mentions, {"2024"})

    async def test_invalid_mode_is_fail_open_and_never_falls_back_to_rule(self):
        state = _state("task-invalid-mode")
        request = {"prompt": "unchanged"}
        delegate = AsyncMock(return_value="baseline")
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "LLM"}),
            patch.object(
                grounding_callbacks._RULE_UPDATER,
                "propose",
                side_effect=AssertionError("invalid mode fell back to Rule"),
            ),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                side_effect=AssertionError("invalid mode built LLM"),
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            result = await _run_bound_query(state, QUERY, request)

        self.assertEqual(result, "baseline")
        self.assertEqual(request, {"prompt": "unchanged"})
        runtime = _runtime(state)
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertIsNotNone(runtime.last_error)
        self.assertEqual(
            runtime.grounding_state.requirement_frame.value_slots,
            (),
        )

    async def test_llm_configuration_error_is_fail_open_without_rule_fallback(self):
        state = _state("task-invalid-llm-config")
        delegate = AsyncMock(return_value="baseline")
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks._RULE_UPDATER,
                "propose",
                side_effect=AssertionError("config error fell back to Rule"),
            ),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                side_effect=ValueError("synthetic invalid provider config"),
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            result = await _run_bound_query(state, QUERY, {"prompt": "same"})

        self.assertEqual(result, "baseline")
        delegate.assert_awaited_once()
        runtime = _runtime(state)
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertEqual(runtime.metrics.root.get("llm_updater_calls", 0), 0)
        self.assertIsNotNone(runtime.last_error)
        self.assertEqual(
            runtime.grounding_state.requirement_frame.value_slots,
            (),
        )

    async def test_llm_reducer_rejection_is_fail_open_to_baseline(self):
        state = _state("task-llm-reducer-reject")
        client = FakeClient()
        updater = LLMUpdater(client, _llm_config())
        delegate = AsyncMock(return_value="baseline")

        def rejecting_reducer(runtime, proposed_patch):
            del runtime, proposed_patch
            raise RuntimeError("synthetic reducer rejection")

        async def process_with_rejection(runtime, observation, *, updater):
            return await process_observation_with_llm(
                runtime,
                observation,
                updater=updater,
                reducer=rejecting_reducer,
            )

        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
            patch.object(
                grounding_callbacks,
                "process_observation_with_llm",
                side_effect=process_with_rejection,
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            result = await _run_bound_query(state, QUERY, {"prompt": "same"})

        self.assertEqual(result, "baseline")
        self.assertEqual(client.calls, 1)
        runtime = _runtime(state)
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertIsNotNone(runtime.last_error)
        self.assertEqual(
            runtime.grounding_state.requirement_frame.operation_slots,
            (),
        )

    async def test_concurrent_tasks_keep_context_frame_and_metrics_isolated(self):
        def content_for_request(request):
            if "2022" in request.prompt:
                return _content(value_mention="2022")
            if "2023" in request.prompt:
                return _content(value_mention="2023")
            raise AssertionError("unexpected synthetic request")

        client = FakeClient(content=content_for_request)
        updater = LLMUpdater(client, _llm_config())
        states = {
            "task-a": _state("task-a"),
            "task-b": _state("task-b"),
        }
        delegate = AsyncMock(return_value=None)

        async def run(task_id, message):
            return await _run_bound_query(states[task_id], message, {})

        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                delegate,
            ),
        ):
            await asyncio.gather(
                run("task-a", "Show orders in 2022."),
                run("task-b", "Show orders in 2023."),
            )

        self.assertEqual(client.calls, 2)
        for task_id, mention in (("task-a", "2022"), ("task-b", "2023")):
            runtime = _runtime(states[task_id])
            self.assertEqual(runtime.metrics.root["llm_updater_calls"], 1)
            self.assertEqual(
                runtime.grounding_state.requirement_frame.value_slots[0].mention,
                mention,
            )
            self.assertNotIn(
                "2023" if mention == "2022" else "2022",
                runtime.model_dump_json(),
            )

    async def test_submit_phase_transition_stays_deterministic_without_llm(self):
        state = _state("task-phase")
        context = _context(state, "call-submit")
        tool = SimpleNamespace(name="submit_sql")
        args = {"sql": "SELECT 1"}
        before_delegate = AsyncMock(return_value=None)
        after_delegate = AsyncMock(return_value="baseline-submit")
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                side_effect=AssertionError("phase transition called LLM"),
            ),
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                before_delegate,
            ),
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                after_delegate,
            ),
        ):
            await grounding_callbacks.before_tool_callback(tool, args, context)
            state["current_phase"] = 2
            result = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                "submitted",
            )

        self.assertEqual(result, "baseline-submit")
        before_delegate.assert_awaited_once()
        after_delegate.assert_awaited_once()
        runtime = _runtime(state)
        self.assertEqual(runtime.phase, 2)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.metrics.root.get("llm_updater_calls", 0), 0)

    async def test_after_model_delegates_once_without_response_mutation(self):
        response = {"model": "response-must-not-change"}
        before = copy.deepcopy(response)
        sentinel = object()
        delegate = AsyncMock(return_value=sentinel)
        with patch.object(
            baseline_callbacks,
            "after_model_callback",
            delegate,
        ):
            result = await grounding_callbacks.after_model_callback(
                SimpleNamespace(state={}),
                response,
            )
        self.assertIs(result, sentinel)
        self.assertEqual(response, before)
        delegate.assert_awaited_once()


class ModeAndBoundaryTests(unittest.TestCase):
    def test_health_reports_rule_llm_and_invalid_without_prompt_injection(self):
        with patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}):
            rule = valibra_server._configuration_summary()
        self.assertEqual(rule["grounding_updater"], "rule")
        self.assertTrue(rule["grounding_configuration_valid"])

        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=object(),
            ),
        ):
            llm = valibra_server._configuration_summary()
        self.assertEqual(llm["grounding_updater"], "llm")
        self.assertTrue(llm["grounding_configuration_valid"])

        with patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "unknown"}):
            invalid = valibra_server._configuration_summary()
        self.assertEqual(invalid["grounding_updater"], "invalid")
        self.assertFalse(invalid["grounding_configuration_valid"])
        for summary in (rule, llm, invalid):
            self.assertFalse(summary["prompt_view_injected"])

    def test_llm_configuration_error_is_reported_as_invalid(self):
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                side_effect=ValueError("synthetic invalid config"),
            ),
        ):
            status = grounding_callbacks.grounding_updater_status()
        self.assertEqual(status["requested_mode"], "llm")
        self.assertEqual(status["effective_mode"], "invalid")
        self.assertFalse(status["configuration_valid"])
        self.assertEqual(status["error_type"], "ValueError")

    def test_frozen_contract_and_shadow_source_boundaries_are_unchanged(self):
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, EXPECTED_PROMPT_SHA256)
        self.assertEqual(LLM_FRAME_FORM_SCHEMA_SHA256, EXPECTED_FORM_SHA256)
        source = inspect.getsource(grounding_callbacks)
        self.assertNotIn("create_task", source)
        self.assertNotIn("render_prompt_view", source)
        self.assertNotIn("activate_model_preset", source)
        self.assertNotIn("SYSTEM_AGENT_API", source)
        self.assertNotIn("task_data", source)
        self.assertNotIn("follow_up", source)
        self.assertNotIn("test_cases", source)


if __name__ == "__main__":
    unittest.main()
