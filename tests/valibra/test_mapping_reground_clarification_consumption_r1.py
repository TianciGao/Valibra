from __future__ import annotations

import unittest

from valibra_agent.sql_grounding import updater


class MappingRegroundClarificationConsumptionR1Tests(unittest.TestCase):
    def test_exception_is_isolated_to_mapping_reground_prompt(self) -> None:
        marker = (
            "用户有权定义的 predicate、classification、filter、entity scope 或组合条件"
        )
        self.assertNotIn(marker, updater.MAPPING_GROUNDING_PROMPT)
        self.assertIn(marker, updater.MAPPING_REGROUNDING_CONSUMPTION_RULES)
        self.assertIn(marker, updater.MAPPING_REGROUNDING_GROUNDING_PROMPT)

        ordinary = {"query": "q", "current_state": {}}
        mapping_restart = {
            **ordinary,
            "regrounding_context": {"decision": "REGROUND_MAPPING"},
        }
        structure_restart = {
            **ordinary,
            "regrounding_context": {"decision": "REGROUND_STRUCTURE"},
        }
        self.assertEqual(
            updater._prompt_for_input_payload("mapping", ordinary),
            updater.MAPPING_GROUNDING_PROMPT,
        )
        self.assertEqual(
            updater._prompt_for_input_payload("mapping", structure_restart),
            updater.MAPPING_REGROUNDING_GROUNDING_PROMPT,
        )
        self.assertEqual(
            updater._prompt_for_input_payload("mapping", mapping_restart),
            updater.MAPPING_REGROUNDING_GROUNDING_PROMPT,
        )

    def test_validation_correction_keeps_prompt_precedence_during_restart(self) -> None:
        payload = {
            "query": "q",
            "current_state": {},
            "regrounding_context": {"decision": "REGROUND_STRUCTURE"},
            "mapping_validation_correction": {
                "rejected_mapping": {},
                "validation_error": "bad target",
            },
        }
        self.assertEqual(
            updater._prompt_for_input_payload("mapping", payload),
            updater.MAPPING_VALIDATION_CORRECTION_PROMPT,
        )

    def test_non_mapping_stage_does_not_inherit_mapping_reground_prompt(self) -> None:
        payload = {
            "query": "q",
            "current_state": {},
            "regrounding_context": {"decision": "REGROUND_STRUCTURE"},
        }
        self.assertEqual(
            updater._prompt_for_input_payload("knowledge", payload),
            updater.KNOWLEDGE_GROUNDING_PROMPT,
        )

    def test_exception_preserves_exact_phrase_and_metadata_authority(self) -> None:
        prompt = updater.MAPPING_REGROUNDING_CONSUMPTION_RULES
        for fragment in (
            "unresolved_mappings 中一个 exact Query",
            "SQL identifier 仍只能来自 Official metadata",
            "不得从 clarification 创建新的 mapping.phrase",
            "原 Query phrase 映射到全部必要 canonical targets",
            "并清除它的 omission",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, prompt)

    def test_exception_does_not_promote_user_answer_to_official_formula(self) -> None:
        prompt = updater.MAPPING_REGROUNDING_CONSUMPTION_RULES
        for fragment in (
            "用户回答仍不能替代必须由 Official Knowledge 定义的 business formula / derived",
            "此类情况继续遵守下面的 Official formula consumption contract",
            "只有同一 Official evidence 明确证明该字段与该公式结果语义等价时",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, prompt)


if __name__ == "__main__":
    unittest.main()
