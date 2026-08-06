import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from system_agent.tools import get_ainteract_tools
from valibra_agent import grounding_callbacks
from valibra_agent.requirement_grounding.models import RequirementGroundingRuntime


class CallbackDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_callback_delegates_exactly_once_and_returns_identity(self):
        cases = [
            ("before_model_callback", (object(), object())),
            ("after_model_callback", (object(), object())),
            ("before_tool_callback", (object(), {}, object())),
            ("after_tool_callback", (object(), {}, object(), object())),
        ]
        for name, args in cases:
            with self.subTest(callback=name):
                sentinel = object()
                delegate = AsyncMock(return_value=sentinel)
                with patch.object(baseline_callbacks, name, delegate):
                    result = await getattr(grounding_callbacks, name)(*args)
                self.assertIs(result, sentinel)
                delegate.assert_awaited_once_with(*args)


class CallbackBehaviorParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_rejection_matches_baseline(self):
        tool = SimpleNamespace(name="ask_user")
        baseline_context = SimpleNamespace(
            state={"budget_remaining": 1.0, "current_phase": 1}
        )
        valibra_context = SimpleNamespace(
            state={"budget_remaining": 1.0, "current_phase": 1},
            function_call_id="call-rejected",
        )
        expected = await baseline_callbacks.before_tool_callback(
            tool, {"question": "q"}, baseline_context
        )
        actual = await grounding_callbacks.before_tool_callback(
            tool, {"question": "q"}, valibra_context
        )
        self.assertEqual(actual, expected)
        self.assertEqual(valibra_context.state, baseline_context.state)
        self.assertIn("MUST call submit_sql", actual["error"])
        self.assertNotIn(
            grounding_callbacks.GROUNDING_RUNTIME_KEY,
            valibra_context.state,
        )

    async def test_tool_response_override_matches_baseline(self):
        initial_state = {
            "task_id": "task-parity",
            "budget_remaining": 5.0,
            "initial_budget": 5.0,
            "current_phase": 1,
            "tool_trajectory": [],
            "system_agent_llm_calls": [{"actions": []}],
            "_active_llm_call_index": 0,
        }
        baseline_context = SimpleNamespace(state=_copy_state(initial_state))
        valibra_context = SimpleNamespace(
            state=_copy_state(initial_state),
            function_call_id="call-parity",
            invocation_id="invocation-parity",
        )
        tool = SimpleNamespace(name="execute_sql")
        expected_rejection = await baseline_callbacks.before_tool_callback(
            tool, {"sql": "SELECT 1"}, baseline_context
        )
        actual_rejection = await grounding_callbacks.before_tool_callback(
            tool, {"sql": "SELECT 1"}, valibra_context
        )
        self.assertIsNone(expected_rejection)
        self.assertIsNone(actual_rejection)
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            expected = await baseline_callbacks.after_tool_callback(
                tool, {"sql": "SELECT 1"}, baseline_context, "ok"
            )
            actual = await grounding_callbacks.after_tool_callback(
                tool, {"sql": "SELECT 1"}, valibra_context, "ok"
            )
        self.assertEqual(actual, expected)
        self.assertEqual(actual, "ok\n\n[SYSTEM NOTE: Remaining budget: 4.0/5.0]")
        self.assertEqual(
            valibra_context.state["budget_remaining"],
            baseline_context.state["budget_remaining"],
        )
        valibra_event = dict(valibra_context.state["tool_trajectory"][0])
        shadow = valibra_event.pop(grounding_callbacks.SHADOW_AUDIT_KEY)
        self.assertEqual(valibra_event, baseline_context.state["tool_trajectory"][0])
        self.assertEqual(shadow["function_call_id"], "call-parity")
        runtime = RequirementGroundingRuntime.model_validate(
            valibra_context.state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(runtime.grounding_revision, 0)

    async def test_two_function_calls_in_one_turn_share_baseline_state_machine(self):
        tools = {tool.name: tool for tool in get_ainteract_tools()}
        context = SimpleNamespace(
            state={
                "task_id": "task-two-calls",
                "budget_remaining": 5.0,
                "initial_budget": 5.0,
                "current_phase": 1,
                "tool_trajectory": [],
                "system_agent_llm_calls": [{"actions": []}],
                "_active_llm_call_index": 0,
            },
            function_call_id="",
            invocation_id="invocation-two-calls",
        )
        calls = [
            (tools["execute_sql"], {"sql": "SELECT 1"}, "row"),
            (
                tools["get_column_meaning"],
                {"table_name": "t", "column_name": "c"},
                "meaning",
            ),
        ]
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            for index, (tool, args, response) in enumerate(calls, start=1):
                context.function_call_id = f"call-{index}"
                rejection = await grounding_callbacks.before_tool_callback(
                    tool, args, context
                )
                self.assertIsNone(rejection)
                override = await grounding_callbacks.after_tool_callback(
                    tool, args, context, response
                )
                self.assertIn("Remaining budget", override)
        self.assertEqual(context.state["budget_remaining"], 3.5)
        self.assertEqual(
            [event["tool"] for event in context.state["tool_trajectory"]],
            ["execute_sql", "get_column_meaning"],
        )
        runtime = RequirementGroundingRuntime.model_validate(
            context.state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.pending_tool_calls, {})
        self.assertEqual(len(runtime.processed_observation_ids), 2)
        self.assertEqual(
            [event["tool"] for event in context.state["system_agent_llm_calls"][0]["actions"]],
            ["execute_sql", "get_column_meaning"],
        )


class SubmitPhaseParityTests(unittest.TestCase):
    def test_submit_phase_transition_uses_shared_baseline_function(self):
        baseline_submit = get_ainteract_tools()[-1].func
        valibra_submit = get_ainteract_tools()[-1].func
        self.assertIs(valibra_submit, baseline_submit)

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "passed": True,
                    "reward": 1.0,
                    "phase_completed": 1,
                    "has_follow_up": True,
                    "follow_up_query": "follow up",
                    "message": "passed",
                }

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, json):
                return FakeResponse()

        context = SimpleNamespace(
            state={
                "task_id": "task",
                "budget_remaining": 3.0,
                "current_phase": 1,
            }
        )
        with patch("system_agent.tools.httpx.Client", FakeClient):
            result = valibra_submit("SELECT 1", context)
        self.assertIn("Reward: 1.0", result)
        self.assertTrue(context.state["phase1_completed"])
        self.assertEqual(context.state["current_phase"], 2)
        self.assertFalse(context.state.get("task_done", False))


def _copy_state(state):
    return copy.deepcopy(state)


if __name__ == "__main__":
    unittest.main()
