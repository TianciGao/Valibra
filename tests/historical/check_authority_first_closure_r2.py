"""Archived assertions for a non-KEEP Prompt experiment; see README.md."""

from __future__ import annotations

import unittest

from valibra_agent.sql_grounding.updater import CHECK_GROUNDING_PROMPT


class CheckAuthorityFirstClosureR2Tests(unittest.TestCase):
    def test_authority_first_procedure_is_explicit_and_ordered(self) -> None:
        required = (
            "Authority-first closure procedure",
            "1. Query / follow_up、仍有效 user clarification 或 Official evidence 是否已经明确闭合",
            "2. 是否缺少一个可由新的合法 Official tool 获取的事实",
            "3. 本轮新获得的 actionable evidence 是否已明确使 current State 失效",
            "4. 是否仍缺一个必须由用户业务意图决定的 SQL-relevant choice",
            "5. 以上都不是：terminal incomplete",
        )
        positions = []
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)
                positions.append(CHECK_GROUNDING_PROMPT.index(fragment))
        self.assertEqual(positions, sorted(positions))

    def test_missing_authority_applies_only_to_required_business_choices(self) -> None:
        required = (
            "已经确认必需、会改变业务含义或 SQL 结果的 choice",
            "缺少权威来源\n只对这种已确认必需的业务语义 choice 构成 unresolved",
            "implementation choice 不是用户业务歧义",
            "不得把纯 SQL implementation choice 升级成",
            "semantic gap",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)

    def test_natural_language_is_valid_transformation_authority(self) -> None:
        for fragment in (
            "“total revenue”可授权 SUM",
            "“average repair time”可授权 AVG",
            "“latest snapshot”可授权 latest-row",
            "“highest loss”可授权 MAX / 相应排序",
            "不要求用户写 SQL 关键字",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)

    def test_ask_user_no_longer_requires_enumerating_two_interpretations(self) -> None:
        self.assertIn(
            "不要求 Check 枚举或证明存在两个合理解释",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertNotIn(
            "至少存在两个合理的用户业务解释",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "如果尚未确认某个业务\nchoice 是回答 Query 所必需",
            CHECK_GROUNDING_PROMPT,
        )

    def test_execution_contracts_are_preserved(self) -> None:
        required = (
            "Exact Official formula operand completeness",
            "explicit Official proof",
            "Check 永远不得创建原本不存在的新 phrase",
            "bounded correction 或删除该一个 phrase",
            "Evidence progress / handoff",
            "next_tool 必须是上述对象之一或 null",
            "previous_official_calls 中已经执行过的 exact tool + arguments 不得重复调用",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)


if __name__ == "__main__":
    unittest.main()
