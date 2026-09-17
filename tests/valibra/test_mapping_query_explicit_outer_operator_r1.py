from __future__ import annotations

import hashlib
import unittest

from valibra_agent.sql_grounding import updater


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MappingQueryExplicitOuterOperatorR1Tests(unittest.TestCase):
    def test_mapping_prompt_uses_operator_provenance_contract(self) -> None:
        prompt = updater.MAPPING_GROUNDING_PROMPT

        self.assertIn(
            "先判断其计算或判定关系的来源",
            prompt,
        )
        self.assertIn(
            "本身已经明确给出外层 operator、operands 和方向",
            prompt,
        )
        self.assertIn(
            "不得仅因该\n  外层关系创建 derived_rule_required",
            prompt,
        )
        self.assertIn(
            "指标名称中出现\n  ratio / score / index 等词，本身不构成 operator provenance",
            prompt,
        )

    def test_mapping_prompt_preserves_no_inferred_computation_guards(self) -> None:
        prompt = updater.MAPPING_GROUNDING_PROMPT

        self.assertIn(
            "不授权推断 operand 内部组成、aggregation",
            prompt,
        )
        self.assertIn(
            "多个 targets 只表示“这些字段都需要”，不表示 SUM / AVG / 加法 / 比率 / 排序 / 优先级",
            prompt,
        )
        self.assertIn(
            "这一阶段不要猜业务公式、阈值或聚合方式",
            prompt,
        )

    def test_reground_mapping_inherits_same_base_contract(self) -> None:
        self.assertTrue(
            updater.MAPPING_REGROUNDING_GROUNDING_PROMPT.startswith(
                updater.MAPPING_GROUNDING_PROMPT
            )
        )
        self.assertIn(
            "本身已经明确给出外层 operator、operands 和方向",
            updater.MAPPING_REGROUNDING_GROUNDING_PROMPT,
        )

    def test_other_stage_prompts_remain_frozen(self) -> None:
        expected = {
            "structure": "04fe3f57e52e09c4ad41e059afd8a211f9808f54f7cfca15e67524594b0f61e1",
            "knowledge": "02a8adcb72c1dcb1ee817c3ca1dc0bb6be5d2dd6521b1ea67a0f8a6dd986cfa7",
            "check": "d70eedd5785774cec00fa681e8c76f862a41fb5755d3a2af10c5634f29892d44",
            "final_gate": "bbd960f1a14fb13746f7bd50a703dc9b874b411c7aa3cfc404fe6ba2a9960cda",
        }
        actual = {
            "structure": _sha(updater.STRUCTURE_GROUNDING_PROMPT),
            "knowledge": _sha(updater.KNOWLEDGE_GROUNDING_PROMPT),
            "check": _sha(updater.CHECK_GROUNDING_PROMPT),
            "final_gate": _sha(updater.FINAL_REGROUNDING_GATE_PROMPT),
        }

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
