import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from system_agent.tools import get_ainteract_tools
from valibra_agent import grounding_callbacks
from valibra_agent.requirement_grounding.models import RequirementGroundingRuntime


_RULE_MODE_PATCHER = None


def setUpModule():
    """V0 parity 固定走 Rule 默认路径，不读取开发者本机模式。"""

    global _RULE_MODE_PATCHER
    _RULE_MODE_PATCHER = patch.dict(
        os.environ,
        {"GROUNDING_UPDATER_MODE": ""},
    )
    _RULE_MODE_PATCHER.start()


def tearDownModule():
    if _RULE_MODE_PATCHER is not None:
        _RULE_MODE_PATCHER.stop()


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
