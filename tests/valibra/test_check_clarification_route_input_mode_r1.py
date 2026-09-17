from __future__ import annotations

import hashlib
import unittest

from valibra_agent.sql_grounding import updater


class CheckClarificationRouteInputModeR1Tests(unittest.TestCase):
    def test_route_is_scoped_to_current_top_level_latest_answer(self) -> None:
        prompt = updater.CHECK_GROUNDING_PROMPT

        self.assertIn(
            "clarification_route 只描述当前这一次 Check request 对顶层 latest_user_answer 的即时路由",
            prompt,
        )
        self.assertIn(
            "只有当前输入 JSON 顶层明确存在 latest_user_answer 字段时",
            prompt,
        )
        self.assertIn(
            "历史 evidence，绝不等同于 latest_user_answer",
            prompt,
        )

    def test_historical_clarifications_cannot_carry_stay_check(self) -> None:
        prompt = updater.CHECK_GROUNDING_PROMPT

        self.assertIn(
            "当前输入顶层没有 latest_user_answer 时，clarification_route 必须为 none",
            prompt,
        )
        self.assertIn(
            "也不能返回 stay_check、restart_grounding 或 terminal",
            prompt,
        )
        self.assertIn(
            "仍可 status=complete，但此时 clarification_route 必须为 none",
            prompt,
        )

    def test_existing_latest_answer_routes_remain_available(self) -> None:
        prompt = updater.CHECK_GROUNDING_PROMPT

        for route in ("stay_check", "restart_grounding", "terminal"):
            self.assertIn(f"- {route}：", prompt)
        self.assertIn("继续本轮完整性检查，可 complete 或继续取证", prompt)

    def test_only_check_prompt_identity_changes(self) -> None:
        sha = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()

        self.assertEqual(
            sha(updater.MAPPING_GROUNDING_PROMPT),
            "a1bad3f28e8328fefea85c87b7092c0069649a0f4d73f2ad7a3135a0d37ac60b",
        )
        self.assertEqual(
            sha(updater.MAPPING_REGROUNDING_GROUNDING_PROMPT),
            "964f4bbf552b80fc0bb6fe5aa1fe7fbd3d410d7196435a454dda80cabedbe9e0",
        )
        self.assertEqual(
            sha(updater.KNOWLEDGE_GROUNDING_PROMPT),
            "02a8adcb72c1dcb1ee817c3ca1dc0bb6be5d2dd6521b1ea67a0f8a6dd986cfa7",
        )
        self.assertEqual(
            sha(updater.FINAL_REGROUNDING_GATE_PROMPT),
            "bbd960f1a14fb13746f7bd50a703dc9b874b411c7aa3cfc404fe6ba2a9960cda",
        )
        self.assertEqual(
            sha(updater.CHECK_GROUNDING_PROMPT),
            "d70eedd5785774cec00fa681e8c76f862a41fb5755d3a2af10c5634f29892d44",
        )


if __name__ == "__main__":
    unittest.main()
