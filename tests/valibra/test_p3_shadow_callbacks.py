import asyncio
import copy
import importlib.metadata
import inspect
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.tool_context import ToolContext

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent.agent import build_agent as build_valibra_agent
from valibra_agent.requirement_grounding.models import RequirementGroundingRuntime
from valibra_agent.requirement_grounding.observations import stable_digest


_RULE_MODE_PATCHER = None


def setUpModule():
    """P3 回归固定验证原有 Rule Shadow，不依赖本机 .env。"""

    global _RULE_MODE_PATCHER
    _RULE_MODE_PATCHER = patch.dict(
        os.environ,
        {"GROUNDING_UPDATER_MODE": ""},
    )
    _RULE_MODE_PATCHER.start()


def tearDownModule():
    if _RULE_MODE_PATCHER is not None:
        _RULE_MODE_PATCHER.stop()


def _state(task_id="task-shadow", phase=1, budget=10.0):
    return {
        "task_id": task_id,
        "current_phase": phase,
        "budget_remaining": budget,
        "initial_budget": budget,
        "tool_trajectory": [],
        "system_agent_llm_calls": [{"actions": []}],
        "_active_llm_call_index": 0,
    }


def _context(state, call_id, invocation_id="invocation-shadow"):
    return SimpleNamespace(
        state=state,
        function_call_id=call_id,
        invocation_id=invocation_id,
    )


def _runtime(state):
    return RequirementGroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


class AdkApiContractTests(unittest.TestCase):
    def test_adk_250_exposes_exact_function_call_id_and_error_callback(self):
        self.assertEqual(importlib.metadata.version("google-adk"), "2.5.0")
        self.assertIs(ToolContext, Context)
        self.assertIn("function_call_id", inspect.signature(ToolContext).parameters)
        self.assertIsInstance(ToolContext.function_call_id, property)
        self.assertIn("on_tool_error_callback", LlmAgent.model_fields)

    def test_valibra_agent_registers_official_tool_error_callback_only(self):
        agent = build_valibra_agent("a-interact")
        self.assertIs(
            agent.on_tool_error_callback,
            grounding_callbacks.on_tool_error_callback,
        )
        self.assertIsNone(agent.on_model_error_callback)

    def test_shadow_source_has_no_hidden_data_or_prompt_view_access(self):
        source = inspect.getsource(grounding_callbacks)
        for forbidden in (
            "task_data",
            "_last_submit_raw",
            "sol_sql",
            "test_cases",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class ShadowModelCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_before_model_initializes_runtime_without_changing_request(self):
        context = SimpleNamespace(state={"task_id": "task-model"})
        request = {"contents": [{"role": "user", "text": "unchanged"}]}
        before = copy.deepcopy(request)
        sentinel = object()
        delegate = AsyncMock(return_value=sentinel)
        with patch.object(
            baseline_callbacks,
            "before_model_callback",
            delegate,
        ):
            result = await grounding_callbacks.before_model_callback(
                context,
                request,
            )
        self.assertIs(result, sentinel)
        delegate.assert_awaited_once_with(context, request)
        self.assertEqual(request, before)
        runtime = _runtime(context.state)
        self.assertEqual(runtime, RequirementGroundingRuntime())
        self.assertNotIn("prompt_view", json.dumps(context.state))

    async def test_corrupt_runtime_is_fail_open_and_request_stays_identical(self):
        context = SimpleNamespace(
            state={
                "task_id": "task-corrupt",
                grounding_callbacks.GROUNDING_RUNTIME_KEY: {"bad": "shape"},
            }
        )
        request = {"prompt": "must remain byte-for-byte visible"}
        delegate = AsyncMock(return_value=None)
        with patch.object(
            baseline_callbacks,
            "before_model_callback",
            delegate,
        ):
            await grounding_callbacks.before_model_callback(context, request)
        self.assertEqual(request, {"prompt": "must remain byte-for-byte visible"})
        runtime = _runtime(context.state)
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertIsNotNone(runtime.last_error)

    async def test_after_model_returns_baseline_object_without_mutation(self):
        context = SimpleNamespace(state={})
        response = SimpleNamespace(content="model-visible")
        sentinel = object()
        delegate = AsyncMock(return_value=sentinel)
        with patch.object(
            baseline_callbacks,
            "after_model_callback",
            delegate,
        ):
            result = await grounding_callbacks.after_model_callback(
                context,
                response,
            )
        self.assertIs(result, sentinel)
        self.assertEqual(response.content, "model-visible")
        delegate.assert_awaited_once_with(context, response)


class ShadowToolLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_rejection_creates_no_pending(self):
        state = _state(budget=1.0)
        context = _context(state, "call-rejected")
        tool = SimpleNamespace(name="ask_user")
        result = await grounding_callbacks.before_tool_callback(
            tool,
            {"question": "q"},
            context,
        )
        self.assertIn("MUST call submit_sql", result["error"])
        self.assertNotIn(grounding_callbacks.GROUNDING_RUNTIME_KEY, state)
        self.assertEqual(state["budget_remaining"], 1.0)

    async def test_original_response_not_baseline_override_builds_observation(self):
        state = _state()
        context = _context(state, "call-original")
        tool = SimpleNamespace(name="execute_sql")
        args = {"sql": "SELECT 1"}
        original = {"rows": [{"value": 1}]}

        await grounding_callbacks.before_tool_callback(tool, args, context)
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            override = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                original,
            )

        self.assertEqual(
            override,
            str(original) + "\n\n[SYSTEM NOTE: Remaining budget: 9.0/10.0]",
        )
        event = state["tool_trajectory"][0]
        self.assertEqual(event["result"], original)
        shadow = event[grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(shadow["raw_digest"], stable_digest(original))
        self.assertNotEqual(shadow["raw_digest"], stable_digest(override))
        self.assertEqual(shadow["raw_log_ref"], "session://tool_trajectory/0")
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.grounding_revision, 1)
        self.assertEqual(len(runtime.processed_observation_ids), 1)
        self.assertEqual(len(runtime.grounding_state.evidence), 1)

    async def test_db_error_text_remains_the_baseline_visible_response(self):
        state = _state()
        context = _context(state, "call-db-error")
        tool = SimpleNamespace(name="execute_sql")
        args = {"sql": "SELECT broken"}
        original = "Error calling DB environment: ConnectError: offline"

        await grounding_callbacks.before_tool_callback(tool, args, context)
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            override = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                original,
            )

        self.assertEqual(
            override,
            original + "\n\n[SYSTEM NOTE: Remaining budget: 9.0/10.0]",
        )
        self.assertNotIn("Valibra", override)
        shadow = state["tool_trajectory"][0][
            grounding_callbacks.SHADOW_AUDIT_KEY
        ]
        self.assertEqual(shadow["raw_digest"], stable_digest(original))
        runtime = _runtime(state)
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertEqual(runtime.grounding_state.evidence, ())
        self.assertEqual(len(runtime.processed_observation_ids), 1)

    async def test_two_same_name_calls_pair_and_clean_by_distinct_ids(self):
        state = _state()
        tool = SimpleNamespace(name="execute_sql")
        first = _context(state, "call-same-a")
        second = _context(state, "call-same-b")

        async def fake_after(tool, args, tool_context, tool_response):
            tool_context.state["tool_trajectory"].append(
                {"tool": tool.name, "args": args, "result": tool_response}
            )
            return {"override_for": tool_context.function_call_id}

        before_delegate = AsyncMock(return_value=None)
        after_delegate = AsyncMock(side_effect=fake_after)
        with (
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
            await grounding_callbacks.before_tool_callback(
                tool, {"sql": "SELECT 'a'"}, first
            )
            await grounding_callbacks.before_tool_callback(
                tool, {"sql": "SELECT 'b'"}, second
            )
            pending = _runtime(state).pending_tool_calls
            self.assertEqual(set(pending), {"call-same-a", "call-same-b"})

            second_result = await grounding_callbacks.after_tool_callback(
                tool, {"sql": "SELECT 'b'"}, second, "response-b"
            )
            self.assertEqual(
                set(_runtime(state).pending_tool_calls),
                {"call-same-a"},
            )
            first_result = await grounding_callbacks.after_tool_callback(
                tool, {"sql": "SELECT 'a'"}, first, "response-a"
            )

        self.assertEqual(second_result, {"override_for": "call-same-b"})
        self.assertEqual(first_result, {"override_for": "call-same-a"})
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(len(runtime.processed_observation_ids), 2)
        self.assertEqual(
            [
                event[grounding_callbacks.SHADOW_AUDIT_KEY]["function_call_id"]
                for event in state["tool_trajectory"]
            ],
            ["call-same-b", "call-same-a"],
        )
        self.assertEqual(before_delegate.await_count, 2)
        self.assertEqual(after_delegate.await_count, 2)

    async def test_two_tasks_run_concurrently_without_cross_contamination(self):
        tool = SimpleNamespace(name="get_schema")

        async def fake_before(tool, args, tool_context):
            await asyncio.sleep(0)
            return None

        async def fake_after(tool, args, tool_context, tool_response):
            await asyncio.sleep(0)
            tool_context.state["tool_trajectory"].append(
                {"tool": tool.name, "args": args, "result": tool_response}
            )
            return None

        async def run_task(task_id):
            state = _state(task_id=task_id)
            context = _context(state, f"call-{task_id}", f"inv-{task_id}")
            await grounding_callbacks.before_tool_callback(tool, {}, context)
            await grounding_callbacks.after_tool_callback(
                tool,
                {},
                context,
                {"schema": task_id},
            )
            return state

        with (
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                AsyncMock(side_effect=fake_before),
            ),
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                AsyncMock(side_effect=fake_after),
            ),
        ):
            first, second = await asyncio.gather(
                run_task("task-one"),
                run_task("task-two"),
            )

        first_runtime = _runtime(first)
        second_runtime = _runtime(second)
        self.assertEqual(len(first_runtime.processed_observation_ids), 1)
        self.assertEqual(len(second_runtime.processed_observation_ids), 1)
        self.assertNotEqual(
            first_runtime.processed_observation_ids,
            second_runtime.processed_observation_ids,
        )
        self.assertIn("task-one", first["tool_trajectory"][0]["result"]["schema"])
        self.assertIn("task-two", second["tool_trajectory"][0]["result"]["schema"])

    async def test_legal_submit_moves_only_internal_runtime_to_phase_two(self):
        tool = SimpleNamespace(name="submit_sql")
        state = _state()
        context = _context(state, "call-submit")
        await grounding_callbacks.before_tool_callback(
            tool,
            {"sql": "SELECT 1"},
            context,
        )
        state["current_phase"] = 2
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            await grounding_callbacks.after_tool_callback(
                tool,
                {"sql": "SELECT 1"},
                context,
                "phase one passed",
            )
        runtime = _runtime(state)
        self.assertEqual(runtime.phase, 2)
        self.assertEqual(runtime.grounding_revision, 2)
        self.assertEqual(len(runtime.processed_observation_ids), 2)
        self.assertEqual(len(runtime.grounding_state.evidence), 1)
        self.assertEqual(runtime.grounding_state.requirement_frame.value_slots, ())
        self.assertEqual(runtime.grounding_state.ambiguity_index, ())
        shadow = state["tool_trajectory"][0][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertIsNotNone(shadow["transition_observation_id"])

    async def test_phase_transition_requires_submit_phase_one_and_phase_after_two(self):
        cases = [
            ("submit_sql", 1),
            ("execute_sql", 2),
        ]
        for index, (tool_name, phase_after) in enumerate(cases, start=1):
            with self.subTest(tool=tool_name, phase_after=phase_after):
                state = _state()
                context = _context(state, f"call-no-transition-{index}")
                tool = SimpleNamespace(name=tool_name)
                args = {"sql": "SELECT 1"}
                await grounding_callbacks.before_tool_callback(tool, args, context)
                state["current_phase"] = phase_after
                with patch.object(
                    baseline_callbacks,
                    "utc_now",
                    return_value="fixed",
                ):
                    await grounding_callbacks.after_tool_callback(
                        tool,
                        args,
                        context,
                        "no legal transition",
                    )
                runtime = _runtime(state)
                self.assertEqual(runtime.phase, 1)
                self.assertEqual(runtime.grounding_revision, 1)
                self.assertEqual(len(runtime.processed_observation_ids), 1)
                self.assertEqual(len(runtime.grounding_state.evidence), 1)

    async def test_missing_function_call_id_never_creates_or_guesses_pending(self):
        state = _state()
        context = _context(state, None)
        tool = SimpleNamespace(name="execute_sql")
        args = {"sql": "SELECT 1"}
        await grounding_callbacks.before_tool_callback(tool, args, context)
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.processed_observation_ids, ())
        self.assertIsNotNone(runtime.last_error)
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            override = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                "raw",
            )
        self.assertIn("Remaining budget", override)
        self.assertEqual(_runtime(state).pending_tool_calls, {})
        self.assertEqual(_runtime(state).processed_observation_ids, ())

    async def test_tool_error_cleans_exact_pending_and_returns_none(self):
        state = _state()
        state["_infrastructure_error"] = "User Simulator outage"
        context = _context(state, "call-error")
        tool = SimpleNamespace(name="ask_user")
        await grounding_callbacks.before_tool_callback(
            tool,
            {"question": "q"},
            context,
        )
        self.assertIn("call-error", _runtime(state).pending_tool_calls)
        result = await grounding_callbacks.on_tool_error_callback(
            tool,
            {"question": "q"},
            context,
            RuntimeError("Bearer super-secret-token infrastructure failure"),
        )
        self.assertIsNone(result)
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertEqual(runtime.grounding_state.evidence, ())
        self.assertEqual(len(runtime.processed_observation_ids), 1)
        self.assertEqual(state["_infrastructure_error"], "User Simulator outage")
        self.assertNotIn("super-secret-token", runtime.model_dump_json())
        self.assertIn("<redacted>", runtime.last_error.message_preview)

    async def test_large_response_is_not_copied_into_runtime_or_shadow_metadata(self):
        state = _state()
        context = _context(state, "call-large")
        tool = SimpleNamespace(name="get_schema")
        tail = "PRIVATE-LARGE-TAIL"
        original = {"schema": "x" * 100_000 + tail}
        await grounding_callbacks.before_tool_callback(tool, {}, context)
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            await grounding_callbacks.after_tool_callback(
                tool,
                {},
                context,
                original,
            )
        runtime_json = _runtime(state).model_dump_json()
        shadow = state["tool_trajectory"][0][grounding_callbacks.SHADOW_AUDIT_KEY]
        shadow_json = json.dumps(shadow)
        self.assertNotIn(tail, runtime_json)
        self.assertNotIn(tail, shadow_json)
        self.assertLessEqual(len(shadow["summary"]), 512)
        self.assertLess(shadow["runtime_bytes"], 524_288)
        self.assertEqual(state["tool_trajectory"][0]["result"], original)


class ShadowFailOpenTests(unittest.IsolatedAsyncioTestCase):
    async def _run_with_failure(self, target, *, phase_transition=False):
        state = _state()
        context = _context(state, f"call-{target}")
        tool_name = "submit_sql" if phase_transition else "execute_sql"
        tool = SimpleNamespace(name=tool_name)
        args = {"sql": "SELECT 1"}
        await grounding_callbacks.before_tool_callback(tool, args, context)
        if phase_transition:
            state["current_phase"] = 2
        with (
            patch.object(baseline_callbacks, "utc_now", return_value="fixed"),
            patch(
                f"valibra_agent.grounding_callbacks.{target}",
                side_effect=RuntimeError(f"synthetic {target} failure"),
            ),
        ):
            override = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                "original tool response",
            )
        return state, override

    async def test_observation_normalization_failure_is_fail_open(self):
        state, override = await self._run_with_failure("build_observation")
        self.assertIn("Remaining budget", override)
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertIsNotNone(runtime.last_error)

    async def test_updater_service_failure_is_fail_open(self):
        state, override = await self._run_with_failure("process_observation")
        self.assertIn("Remaining budget", override)
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertIsNotNone(runtime.last_error)

    async def test_phase_reducer_failure_is_fail_open(self):
        state, override = await self._run_with_failure(
            "process_phase_transition",
            phase_transition=True,
        )
        self.assertIn("Remaining budget", override)
        runtime = _runtime(state)
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.phase, 1)
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertIsNotNone(runtime.last_error)

    async def test_reuse_retains_runtime_while_reset_state_starts_empty(self):
        delegate = AsyncMock(return_value=None)
        reused_state = _state(task_id="task-reuse")
        reset_state = _state(task_id="task-reuse")
        with patch.object(
            baseline_callbacks,
            "before_model_callback",
            delegate,
        ):
            await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=reused_state),
                {},
            )
            tool = SimpleNamespace(name="get_schema")
            context = _context(reused_state, "call-reuse")
            with (
                patch.object(
                    baseline_callbacks,
                    "before_tool_callback",
                    AsyncMock(return_value=None),
                ),
                patch.object(
                    baseline_callbacks,
                    "after_tool_callback",
                    AsyncMock(return_value=None),
                ),
            ):
                await grounding_callbacks.before_tool_callback(tool, {}, context)
                await grounding_callbacks.after_tool_callback(
                    tool, {}, context, "schema"
                )
            before_reuse = _runtime(reused_state)
            await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=reused_state),
                {},
            )
            await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=reset_state),
                {},
            )
        self.assertEqual(_runtime(reused_state), before_reuse)
        self.assertEqual(_runtime(reset_state), RequirementGroundingRuntime())


if __name__ == "__main__":
    unittest.main()
