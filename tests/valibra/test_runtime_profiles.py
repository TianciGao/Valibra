from __future__ import annotations

import hashlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent import server
from valibra_agent.runtime_profile import valibra_execution_profile
from valibra_agent.sql_grounding.models import (
    GroundingRuntime,
    SQLGroundingState,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_STAGE_PROMPT_SHA256,
)


def _ready_state(task_id: str = "profile-task") -> dict[str, object]:
    grounding_state = SQLGroundingState(
        tables=("orders",),
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
        "task_id": task_id,
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
                "observation_id": "profile-ready",
                "error_type": None,
                "provider_attempted": True,
                "grounding_revision": runtime.grounding_revision,
                "state_sha256": sql_grounding_state_sha256(grounding_state),
            }
        },
    }


def _initial_state(task_id: str = "profile-initial") -> dict[str, object]:
    runtime = GroundingRuntime()
    return {
        "task_id": task_id,
        "current_phase": 1,
        "initial_budget": 10.0,
        "budget_remaining": 10.0,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
    }


def _failed_state(task_id: str = "profile-failed") -> dict[str, object]:
    state = _initial_state(task_id)
    runtime = GroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )
    state[grounding_callbacks.GROUNDING_PHASE_OUTCOMES_KEY] = {
        "1": {
            "phase": 1,
            "status": "failed",
            "observation_id": "profile-failed-observation",
            "error_type": "GroundingFailed",
            "provider_attempted": True,
            "grounding_revision": runtime.grounding_revision,
            "state_sha256": sql_grounding_state_sha256(runtime.grounding_state),
        }
    }
    return state


def _context(state: dict[str, object], call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        function_call_id=call_id,
        invocation_id="profile-invocation",
    )


class RuntimeProfileContractTests(unittest.TestCase):
    def test_profile_is_strict_and_defaults_to_research(self) -> None:
        self.assertEqual(valibra_execution_profile({}), "research")
        self.assertEqual(
            valibra_execution_profile({"VALIBRA_EXECUTION_PROFILE": "research"}),
            "research",
        )
        self.assertEqual(
            valibra_execution_profile(
                {"VALIBRA_EXECUTION_PROFILE": "leaderboard"}
            ),
            "leaderboard",
        )
        with self.assertRaisesRegex(ValueError, "research.*leaderboard"):
            valibra_execution_profile({"VALIBRA_EXECUTION_PROFILE": "shadow"})

    def test_grounding_check_writer_contract_hashes_are_unchanged(self) -> None:
        self.assertEqual(
            SQL_GROUNDING_PROMPT_SHA256,
            "34bfc4a5682510e4f9963cbc3f5fa55505f3715fffe50b6dd3a98890f25be425",
        )
        self.assertEqual(
            SQL_GROUNDING_FORM_SCHEMA_SHA256,
            "728fc43c6ed85e72e60c9ebf85b00487059a2871020a28764b9641c69b84ed81",
        )
        self.assertEqual(
            SQL_GROUNDING_CONFIGURATION_SHA256,
            "b1881e01b13314bf244591b406b86558ad3d30d07ed1a9197630ba40ccdf264a",
        )
        self.assertEqual(
            SQL_GROUNDING_STAGE_PROMPT_SHA256,
            {
                "structure": "dacb466200beafb6dfc6ba6d1f8cf3da40cfd0ea791d7f77e8d940f84f6528fd",
                "mapping": "2dced1fcc8aaeb5b22dc5f861209d22f8a21b64613d31d3b4472f1ddd4cde3c3",
                "knowledge": "4f805cabaa53200dfac4b5226d0bc90306e9c35164c8aaf1ae7ad47f56e42e8f",
                "check": "4eb5c8b3a8a9dd4bbbc6bcc1aa4208a6b72791bf1bfedcbca2dbd90c4d471801",
            },
        )
        self.assertEqual(
            SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
            {
                "structure": "d040bb89edcd2331b8ab51e5169dedcc6b87fefadc39abdabcbfd11a2fdcc0b5",
                "mapping": "7a1e8cb588d1c0b89546bfd02b8be0d254ca015cdee753e5e2d23e0da5ea3b44",
                "knowledge": "c80f6dc31b8bc4aad425fb73fe62e361dd06362551827473ee8e9302052ed246",
                "check": "ed4bdaf32e5415be3b4f30bd041c0c4dba4225705736c55bd0b516f5079327e6",
            },
        )
        self.assertEqual(
            hashlib.sha256(
                grounding_callbacks._SQL_WRITER_PROMPT.encode("utf-8")
            ).hexdigest(),
            "61deab4ea63bdef3a8c511a0a625abdbe87969f9df36130f76344d7bcc8ea711",
        )

    def test_health_summary_identifies_effective_profile(self) -> None:
        with patch.dict(
            os.environ,
            {"VALIBRA_EXECUTION_PROFILE": "research"},
        ):
            research = server._configuration_summary()
        self.assertEqual(research["execution_profile"], "research")
        self.assertTrue(research["attempt_gate_enabled"])
        self.assertEqual(
            server._variant(research),
            "SQL-Grounding-V1-SG6b-Active-Gate",
        )

        with patch.dict(
            os.environ,
            {"VALIBRA_EXECUTION_PROFILE": "leaderboard"},
        ):
            leaderboard = server._configuration_summary()
        self.assertEqual(leaderboard["execution_profile"], "leaderboard")
        self.assertFalse(leaderboard["attempt_gate_enabled"])
        self.assertEqual(
            leaderboard["attempt_gate_mode"],
            "official_passthrough",
        )
        self.assertEqual(
            server._variant(leaderboard),
            "SQL-Grounding-V1-Leaderboard-PassThrough",
        )

    def test_bootstrap_model_result_is_profiled_without_payload_rewrite(self) -> None:
        official = "official schema\n\n[SYSTEM NOTE: Remaining budget: 9.0/10.0]"
        with patch.dict(
            os.environ,
            {"VALIBRA_EXECUTION_PROFILE": "research"},
        ):
            research = grounding_callbacks._profiled_bootstrap_model_visible_result(
                "get_schema",
                official,
                succeeded=True,
            )
        self.assertIn("VALIBRA_BOOTSTRAP_EVIDENCE_STORED", research)
        self.assertNotEqual(research, official)

        with patch.dict(
            os.environ,
            {"VALIBRA_EXECUTION_PROFILE": "leaderboard"},
        ):
            leaderboard = (
                grounding_callbacks._profiled_bootstrap_model_visible_result(
                    "get_schema",
                    official,
                    succeeded=True,
                )
            )
        self.assertIs(leaderboard, official)


class CallbackProfileRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_research_main_gate_keeps_zero_cost_synthetic_rejection(self) -> None:
        state = _ready_state("research-gate")
        baseline_before = AsyncMock(return_value=None)
        with (
            patch.dict(
                os.environ,
                {"VALIBRA_EXECUTION_PROFILE": "research"},
            ),
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                baseline_before,
            ),
        ):
            result = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="execute_sql"),
                {"sql": "SELECT column_name FROM information_schema.columns"},
                _context(state, "research-gate-call"),
            )
        self.assertEqual(result["reason"], "system_catalog_forbidden")
        baseline_before.assert_not_awaited()
        self.assertEqual(state["budget_remaining"], 10.0)
        self.assertEqual(state["tool_trajectory"], [])

    async def test_research_keeps_submit_duplicate_and_failed_phase_guards(
        self,
    ) -> None:
        submit_state = _initial_state("research-submit")
        duplicate_state = _initial_state("research-duplicate")
        duplicate_state["tool_trajectory"] = [
            {
                "type": "tool",
                "tool": "get_schema",
                "phase": 1,
                "args": {},
                "result": "official schema",
                "cost": 1.0,
            }
        ]
        failed_state = _failed_state("research-failed")
        cases = (
            (
                "submit_sql",
                submit_state,
                {"sql": "SELECT 1"},
                "research-submit-call",
                "VALIBRA_FIRST_SUBMIT_BLOCKED",
            ),
            (
                "get_schema",
                duplicate_state,
                {},
                "research-duplicate-call",
                "already_stored",
            ),
            (
                "execute_sql",
                failed_state,
                {"sql": "SELECT 1"},
                "research-failed-call",
                "VALIBRA_SQL_GROUNDING_FAILED_CLOSED",
            ),
        )
        for tool_name, state, args, call_id, expected_status in cases:
            with self.subTest(tool=tool_name):
                baseline_before = AsyncMock(return_value=None)
                with (
                    patch.dict(
                        os.environ,
                        {"VALIBRA_EXECUTION_PROFILE": "research"},
                    ),
                    patch.object(
                        baseline_callbacks,
                        "before_tool_callback",
                        baseline_before,
                    ),
                ):
                    result = await grounding_callbacks.before_tool_callback(
                        SimpleNamespace(name=tool_name),
                        args,
                        _context(state, call_id),
                    )
                self.assertEqual(result["status"], expected_status)
                baseline_before.assert_not_awaited()

    async def test_leaderboard_emitted_execute_has_one_official_chain(self) -> None:
        state = _ready_state("leaderboard-chain")
        args = {"sql": "SELECT column_name FROM information_schema.columns"}
        tool = SimpleNamespace(name="execute_sql")
        context = _context(state, "leaderboard-chain-call")
        tool_calls: list[dict[str, str]] = []
        official_before = baseline_callbacks.before_tool_callback
        official_after = baseline_callbacks.after_tool_callback
        tracked_before = AsyncMock(wraps=official_before)
        tracked_after = AsyncMock(wraps=official_after)
        turn_token = grounding_callbacks._bind_turn_message(
            "leaderboard-chain",
            "a-interact",
            "User Query:\nList columns.",
        )
        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "VALIBRA_EXECUTION_PROFILE": "leaderboard",
                        "GROUNDING_UPDATER_MODE": "",
                    },
                ),
                patch.object(
                    baseline_callbacks,
                    "before_tool_callback",
                    tracked_before,
                ),
                patch.object(
                    baseline_callbacks,
                    "after_tool_callback",
                    tracked_after,
                ),
            ):
                before = await grounding_callbacks.before_tool_callback(
                    tool,
                    args,
                    context,
                )
                self.assertIsNone(before)
                tool_calls.append(dict(args))
                raw_result = "column_name\n-----------\nid"
                returned = await grounding_callbacks.after_tool_callback(
                    tool,
                    args,
                    context,
                    raw_result,
                )
        finally:
            grounding_callbacks._reset_turn_message(turn_token)

        tracked_before.assert_awaited_once_with(tool, args, context)
        tracked_after.assert_awaited_once_with(tool, args, context, raw_result)
        self.assertEqual(tool_calls, [args])
        self.assertEqual(state["budget_remaining"], 9.0)
        self.assertEqual(
            returned,
            raw_result + "\n\n[SYSTEM NOTE: Remaining budget: 9.0/10.0]",
        )
        self.assertEqual(len(state["tool_trajectory"]), 1)
        event = state["tool_trajectory"][0]
        self.assertEqual(event["tool"], "execute_sql")
        self.assertEqual(event["args"], args)
        self.assertEqual(event["result"], raw_result)
        self.assertEqual(event["cost"], 1.0)
        self.assertNotIn(grounding_callbacks.SHADOW_AUDIT_KEY, event)
        self.assertNotIn(grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY, event)

    async def test_leaderboard_submit_duplicate_bootstrap_and_failure_pass_through(
        self,
    ) -> None:
        cases: list[tuple[str, dict[str, object], dict[str, str], str]] = []

        submit_state = _initial_state("leaderboard-submit")
        cases.append(
            ("submit_sql", submit_state, {"sql": "SELECT 1"}, "submit-call")
        )

        duplicate_state = _initial_state("leaderboard-duplicate-bootstrap")
        duplicate_state["tool_trajectory"] = [
            {
                "type": "tool",
                "tool": "get_schema",
                "phase": 1,
                "args": {},
                "result": "official schema",
                "cost": 1.0,
            }
        ]
        cases.append(("get_schema", duplicate_state, {}, "duplicate-bootstrap-call"))

        failed_state = _failed_state("leaderboard-failed-phase")
        cases.append(
            (
                "execute_sql",
                failed_state,
                {"sql": "SELECT 1"},
                "failed-phase-call",
            )
        )

        for tool_name, state, args, call_id in cases:
            with self.subTest(tool=tool_name):
                baseline_before = AsyncMock(return_value=None)
                with (
                    patch.dict(
                        os.environ,
                        {"VALIBRA_EXECUTION_PROFILE": "leaderboard"},
                    ),
                    patch.object(
                        baseline_callbacks,
                        "before_tool_callback",
                        baseline_before,
                    ),
                ):
                    result = await grounding_callbacks.before_tool_callback(
                        SimpleNamespace(name=tool_name),
                        args,
                        _context(state, call_id),
                    )
                self.assertIsNone(result)
                baseline_before.assert_awaited_once()
                self.assertIn(call_id, state[grounding_callbacks.GROUNDING_PENDING_KEY])
                self.assertFalse(
                    grounding_callbacks._failed_closed_call_present(state, call_id)
                )
                self.assertFalse(
                    grounding_callbacks._suppressed_bootstrap_present(state, call_id)
                )
                self.assertFalse(
                    grounding_callbacks._blocked_submit_present(state, call_id)
                )


if __name__ == "__main__":
    unittest.main()
