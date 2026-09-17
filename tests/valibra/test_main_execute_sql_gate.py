from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    GroundingRuntime,
    SQLGroundingState,
    sql_grounding_state_sha256,
)


def ready_state(*, tables: tuple[str, ...] = ("orders",)) -> dict[str, object]:
    grounding_state = SQLGroundingState(
        tables=tables,
        join_keys=(),
        column_mapping=(),
        domain_knowledge=(),
    )
    runtime = GroundingRuntime(
        grounding_revision=1,
        stage="SQL_ATTEMPT",
        focus_dimension="none",
        grounding_state=grounding_state,
    )
    return {
        "task_id": "main-execute-gate-task",
        "current_phase": 1,
        "initial_budget": 10.0,
        "budget_remaining": 10.0,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PHASE_OUTCOMES_KEY: {
            "1": {
                "phase": 1,
                "status": "succeeded",
                "observation_id": "main-execute-gate-ready",
                "error_type": None,
                "provider_attempted": True,
                "grounding_revision": runtime.grounding_revision,
                "state_sha256": sql_grounding_state_sha256(grounding_state),
            }
        },
    }


def call_context(state: dict[str, object], call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        function_call_id=call_id,
        invocation_id="main-execute-gate-invocation",
    )


class MainExecuteSQLGateTests(unittest.IsolatedAsyncioTestCase):
    async def assert_rejected(
        self,
        sql: str,
        reason: str,
        *,
        tables: tuple[str, ...] = ("orders",),
        call_id: str = "main-execute-rejected",
    ) -> tuple[dict[str, object], dict[str, str]]:
        state = ready_state(tables=tables)
        before_budget = state["budget_remaining"]
        runtime_before = state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        baseline_before = AsyncMock(return_value=None)
        baseline_after = AsyncMock(return_value="official-override")
        tool = SimpleNamespace(name="execute_sql")
        context = call_context(state, call_id)
        args = {"sql": sql}
        with (
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                baseline_before,
            ),
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                baseline_after,
            ),
        ):
            response = await grounding_callbacks.before_tool_callback(
                tool,
                args,
                context,
            )
            self.assertIsInstance(response, dict)
            self.assertEqual(response["reason"], reason)
            returned = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context,
                response,
            )
        self.assertEqual(returned, response)
        self.assertEqual(baseline_before.await_count, 0)
        self.assertEqual(baseline_after.await_count, 0)
        self.assertEqual(state["budget_remaining"], before_budget)
        self.assertEqual(state["tool_trajectory"], [])
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY],
            runtime_before,
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_FAILED_CLOSED_CALLS_KEY],
            {},
        )
        return state, response

    async def test_information_schema_and_pg_catalog_are_rejected_before_cost(self) -> None:
        cases = (
            "SELECT column_name FROM information_schema.columns",
            "SELECT relname FROM pg_catalog.pg_class",
        )
        for index, sql in enumerate(cases, start=1):
            with self.subTest(sql=sql):
                await self.assert_rejected(
                    sql,
                    "system_catalog_forbidden",
                    call_id=f"main-system-catalog-{index}",
                )

    async def test_table_outside_frozen_state_is_rejected(self) -> None:
        await self.assert_rejected(
            "SELECT id FROM customers",
            "table_outside_frozen_state",
        )

    async def test_unparseable_or_multiple_statements_fail_closed(self) -> None:
        for index, sql in enumerate(("SELECT FROM", "SELECT 1; SELECT 2"), start=1):
            with self.subTest(sql=sql):
                await self.assert_rejected(
                    sql,
                    "sql_validation_failed",
                    call_id=f"main-invalid-sql-{index}",
                )

    async def test_frozen_schema_qualified_table_alias_and_cte_are_allowed(self) -> None:
        state = ready_state(tables=("public.orders",))
        baseline_before = AsyncMock(return_value=None)
        sql = (
            "WITH ranked AS (SELECT o.id FROM public.orders AS o) "
            "SELECT ranked.id FROM ranked"
        )
        with patch.object(
            baseline_callbacks,
            "before_tool_callback",
            baseline_before,
        ):
            response = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="execute_sql"),
                {"sql": sql},
                call_context(state, "main-allowed-cte"),
            )
        self.assertIsNone(response)
        self.assertEqual(baseline_before.await_count, 1)
        self.assertIn(
            "main-allowed-cte",
            state[grounding_callbacks.GROUNDING_PENDING_KEY],
        )

    async def test_select_star_is_rejected_but_count_star_is_not(self) -> None:
        await self.assert_rejected(
            "SELECT o.* FROM orders AS o",
            "star_projection_forbidden",
        )

        state = ready_state()
        baseline_before = AsyncMock(return_value=None)
        with patch.object(
            baseline_callbacks,
            "before_tool_callback",
            baseline_before,
        ):
            response = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="execute_sql"),
                {"sql": "SELECT COUNT(*) FROM orders"},
                call_context(state, "main-count-star-allowed"),
            )
        self.assertIsNone(response)
        self.assertEqual(baseline_before.await_count, 1)

    async def test_second_canonical_sql_is_rejected_without_second_cost(self) -> None:
        state = ready_state()
        executions = 0

        async def baseline_before(_tool, _args, context):
            nonlocal executions
            executions += 1
            context.state["budget_remaining"] -= 1.0
            return None

        tool = SimpleNamespace(name="execute_sql")
        first_sql = "SELECT id FROM orders WHERE id = 1"
        with patch.object(
            baseline_callbacks,
            "before_tool_callback",
            side_effect=baseline_before,
        ):
            first = await grounding_callbacks.before_tool_callback(
                tool,
                {"sql": first_sql},
                call_context(state, "main-duplicate-first"),
            )
        self.assertIsNone(first)
        state[grounding_callbacks.GROUNDING_PENDING_KEY] = {}
        state["tool_trajectory"].append(
            {
                "type": "tool",
                "tool": "execute_sql",
                "phase": 1,
                "args": {"sql": first_sql},
                "result": "id\n--\n1",
            }
        )
        budget_after_first = state["budget_remaining"]

        baseline_second = AsyncMock(return_value=None)
        baseline_after = AsyncMock(return_value=None)
        second_context = call_context(state, "main-duplicate-second")
        second_args = {"sql": "select ID from ORDERS where ID=1"}
        with (
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                baseline_second,
            ),
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                baseline_after,
            ),
        ):
            denial = await grounding_callbacks.before_tool_callback(
                tool,
                second_args,
                second_context,
            )
            returned = await grounding_callbacks.after_tool_callback(
                tool,
                second_args,
                second_context,
                denial,
            )
        self.assertEqual(denial["reason"], "duplicate_sql")
        self.assertEqual(returned, denial)
        self.assertEqual(executions, 1)
        self.assertEqual(baseline_second.await_count, 0)
        self.assertEqual(baseline_after.await_count, 0)
        self.assertEqual(state["budget_remaining"], budget_after_first)
        self.assertEqual(len(state["tool_trajectory"]), 1)

    async def test_exact_check_execute_sql_dispatch_bypasses_main_gate(self) -> None:
        state = ready_state()
        pending = grounding_callbacks._PendingCheckTool(
            phase=1,
            function_call_id="check-execute-exact",
            missing_information="A bounded runtime fact is required.",
            tool_name="execute_sql",
            arguments={"sql": "SELECT value FROM check_probe"},
            request_digest="a" * 64,
        )
        grounding_callbacks._store_pending_check_tool(state, pending)
        baseline_before = AsyncMock(return_value=None)
        with patch.object(
            baseline_callbacks,
            "before_tool_callback",
            baseline_before,
        ):
            response = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="execute_sql"),
                pending.arguments,
                call_context(state, pending.function_call_id),
            )
        self.assertIsNone(response)
        self.assertEqual(baseline_before.await_count, 1)
        self.assertEqual(grounding_callbacks._pending_check_tool(state), pending)

    async def test_submit_sql_is_not_affected_by_main_execute_gate(self) -> None:
        state = ready_state()
        baseline_before = AsyncMock(return_value=None)
        with patch.object(
            baseline_callbacks,
            "before_tool_callback",
            baseline_before,
        ):
            response = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="submit_sql"),
                {"sql": "SELECT id FROM orders"},
                call_context(state, "main-submit-unaffected"),
            )
        self.assertIsNone(response)
        self.assertEqual(baseline_before.await_count, 1)


if __name__ == "__main__":
    unittest.main()
