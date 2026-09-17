"""Archived assertions for a superseded Check Prompt; see README.md."""

from __future__ import annotations

import unittest

from valibra_agent.sql_grounding.updater import CHECK_GROUNDING_PROMPT


class CheckSemanticClosureContractR1Tests(unittest.TestCase):
    def test_check_defers_restart_stage_choice_to_final_gate(self) -> None:
        required = (
            "Check 不决定从哪个",
            "Final Regrounding Gate 会结合本轮新获得的 actionable evidence",
            "REGROUND_MAPPING、REGROUND_STRUCTURE 或 TERMINAL",
            "没有新材料的盲目重跑不会产生有效修复",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)
        self.assertNotIn(
            "Runtime 会在同一 Official phase 启动新的\n"
            "    Structure → Mapping → Knowledge → Check cycle",
            CHECK_GROUNDING_PROMPT,
        )

    def test_concept_closure_has_three_authorized_modes(self) -> None:
        required = (
            "authorized closure mode",
            "current mapping 中存在由 Official metadata 直接、语义等价支持的 target",
            "exact Official rule",
            "SQL operands 已由合法 mapping Ground",
            "stored target 与该 exact rule result 语义等价",
            "A / B / C 不自动授权",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)

    def test_directness_preserves_full_semantic_span(self) -> None:
        required = (
            "用户实际要求的完整 concept",
            "modifier、condition",
            "entity / grain 与 semantic role",
            "不得删除这些信息后",
            "related / proxy、raw measure",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)

    def test_required_transformation_uses_authority_first_contract(self) -> None:
        required = (
            "Required transformation / grain authority",
            "确实是回答 Query 必需的",
            "不能因为表中可能有多行",
            "权威来源只能是 Query / follow_up、有效 clarification 或 Official",
            "stored metric 名称或常见 SQL 习惯不能授权 SUM / AVG /",
            "total / average / latest / highest",
            "未出现 SQL\n  关键字而制造缺口",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)

    def test_clarification_is_read_only_for_check_but_visible_to_regrounding(self) -> None:
        required = (
            "不得由本轮 Check 直接复制、改写",
            "materialize 到四维 State",
            "作为 Final Gate / 下一轮 re-Grounding 的只读输入",
            "结合合法 Official evidence 生成",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)
        self.assertNotIn(
            "user_clarifications / answered_clarifications 绝不能成为 State 修改依据",
            CHECK_GROUNDING_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
