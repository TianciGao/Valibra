import inspect
import unittest

from system_agent.agent import AINTERACT_INSTRUCTION, build_agent as build_baseline
from valibra_agent import grounding_callbacks
from valibra_agent.agent import build_agent as build_valibra


EXPECTED_TOOLS = [
    "execute_sql",
    "get_schema",
    "get_all_column_meanings",
    "get_column_meaning",
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
    "ask_user",
    "submit_sql",
]


class ValibraAgentParityTests(unittest.TestCase):
    def setUp(self):
        self.baseline = build_baseline("a-interact")
        self.valibra = build_valibra("a-interact")

    def test_prompt_and_identity_fields_match_baseline(self):
        self.assertEqual(self.valibra.instruction, AINTERACT_INSTRUCTION)
        self.assertEqual(self.valibra.instruction, self.baseline.instruction)
        self.assertEqual(self.valibra.name, self.baseline.name)
        self.assertEqual(self.valibra.description, self.baseline.description)
        self.assertEqual(
            self.valibra.generate_content_config,
            self.baseline.generate_content_config,
        )
        self.assertEqual(type(self.valibra.model), type(self.baseline.model))
        self.assertEqual(self.valibra.model.model, self.baseline.model.model)

    def test_nine_tool_names_order_signatures_and_functions_match(self):
        baseline_tools = self.baseline.tools
        valibra_tools = self.valibra.tools
        self.assertEqual([tool.name for tool in valibra_tools], EXPECTED_TOOLS)
        self.assertEqual(
            [tool.name for tool in valibra_tools],
            [tool.name for tool in baseline_tools],
        )
        self.assertEqual(len(valibra_tools), 9)
        for baseline_tool, valibra_tool in zip(baseline_tools, valibra_tools):
            self.assertIs(valibra_tool.func, baseline_tool.func)
            self.assertEqual(
                inspect.signature(valibra_tool.func),
                inspect.signature(baseline_tool.func),
            )
        self.assertFalse(
            any("ground" in tool.name.lower() for tool in valibra_tools)
        )

    def test_only_callbacks_are_replaced(self):
        self.assertIs(
            self.valibra.before_model_callback,
            grounding_callbacks.before_model_callback,
        )
        self.assertIs(
            self.valibra.after_model_callback,
            grounding_callbacks.after_model_callback,
        )
        self.assertIs(
            self.valibra.before_tool_callback,
            grounding_callbacks.before_tool_callback,
        )
        self.assertIs(
            self.valibra.after_tool_callback,
            grounding_callbacks.after_tool_callback,
        )

    def test_non_ainteract_mode_delegates_to_baseline(self):
        baseline = build_baseline("c-interact")
        valibra = build_valibra("c-interact")
        self.assertEqual(valibra.instruction, baseline.instruction)
        self.assertEqual(
            [tool.name for tool in valibra.tools],
            [tool.name for tool in baseline.tools],
        )


if __name__ == "__main__":
    unittest.main()
