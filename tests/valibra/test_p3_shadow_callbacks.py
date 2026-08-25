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

if __name__ == "__main__":
    unittest.main()
