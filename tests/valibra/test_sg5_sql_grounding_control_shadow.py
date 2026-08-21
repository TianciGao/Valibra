from __future__ import annotations

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks, server
from valibra_agent.sql_grounding.control import (
    evaluate_first_submit_gate as pure_evaluate_first_submit_gate,
)
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
)


PROMPT_SHA = "812a189320a2f77efed13c99f5f4ba56538570542e341e46167d36f3b2a6f9d6"
FORM_SHA = "1f7e3c1f1ae86876f63de951bcade30fc1ba338e046416fe033331d447775d15"
CONFIG_SHA = "a507a6f3513e53d4c8c784589d15679569070b14251500e74ae77d4597dcf143"
QUERY = "Show the maintenance cost."
SCHEMA = """CREATE TABLE operational_metrics (
  maintcost NUMERIC
);"""
ALL_COLUMN_MEANINGS = "{}"
ALL_KNOWLEDGE_DEFINITIONS = "[]"
_ORIGINAL_MODE = os.environ.get("GROUNDING_UPDATER_MODE")


def setUpModule() -> None:
    os.environ.pop("GROUNDING_UPDATER_MODE", None)


def tearDownModule() -> None:
    if _ORIGINAL_MODE is None:
        os.environ.pop("GROUNDING_UPDATER_MODE", None)
    else:
        os.environ["GROUNDING_UPDATER_MODE"] = _ORIGINAL_MODE


def empty_complete_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=(),
        join_keys=(),
        column_mapping=(),
        domain_knowledge=(),
    )


def grounded_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("operational_metrics",),
        join_keys=(),
        column_mapping=(
            ColumnMapping(
                phrase="maintenance cost",
                targets=("operational_metrics.maintcost",),
            ),
        ),
        domain_knowledge=(),
    )


def runtime_payload(
    *,
    stage: str = "SQL_ATTEMPT",
    focus: str = "none",
    state: SQLGroundingState | None = None,
    revision: int = 0,
) -> dict:
    return GroundingRuntime(
        grounding_revision=revision,
        stage=stage,
        focus_dimension=focus,
        grounding_state=state or empty_complete_state(),
    ).model_dump(mode="json")


def task_state(task_id: str, *, budget: float = 12.0) -> dict:
    return {
        "task_id": task_id,
        "current_phase": 1,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": budget,
        "initial_budget": budget,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
    }


def add_bootstrap_trajectory(current: dict) -> None:
    for tool, result in (
        ("get_schema", SCHEMA),
        ("get_all_column_meanings", ALL_COLUMN_MEANINGS),
        ("get_all_knowledge_definitions", ALL_KNOWLEDGE_DEFINITIONS),
    ):
        current["tool_trajectory"].append(
            {
                "type": "tool",
                "tool": tool,
                "phase": 1,
                "args": {},
                "result": result,
            }
        )


def tool_context(current: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=current,
        function_call_id=call_id,
        invocation_id=f"inv-{call_id}",
    )


def load_runtime(current: dict) -> GroundingRuntime:
    return GroundingRuntime.model_validate(
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


class QueueUpdater:
    def __init__(self, *responses: GroundingLLMResponse) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.observation_types: list[str] = []

    async def propose(self, runtime, observation, **kwargs):
        self.observation_types.append(observation.observation_type)
        grounding_input = kwargs.get("grounding_input", {})
        fields = set(grounding_input)
        target = self.responses[0]
        if "schema" in fields:
            response = GroundingLLMResponse(
                sql_grounding_state=SQLGroundingState(
                    tables=target.sql_grounding_state.tables,
                    join_keys=target.sql_grounding_state.join_keys,
                    column_mapping=runtime.grounding_state.column_mapping,
                    domain_knowledge=runtime.grounding_state.domain_knowledge,
                ),
                user_clarification_requests=(),
                next_focus_dimension="column_mapping",
            )
        elif "column_meanings" in fields:
            response = GroundingLLMResponse(
                sql_grounding_state=SQLGroundingState(
                    tables=target.sql_grounding_state.tables,
                    join_keys=target.sql_grounding_state.join_keys,
                    column_mapping=target.sql_grounding_state.column_mapping,
                    domain_knowledge=runtime.grounding_state.domain_knowledge,
                ),
                user_clarification_requests=(),
                next_focus_dimension="domain_knowledge",
            )
        else:
            response = self.responses.pop(0)
        self.calls += 1
        return GroundingUpdaterResult(
            response=response,
            telemetry=GroundingLLMTelemetry(
                attempted=False,
                status="succeeded",
                request_sha256="",
                response_sha256="",
                prompt_sha256=PROMPT_SHA,
                form_schema_sha256=FORM_SHA,
                configuration_sha256=CONFIG_SHA,
            ),
            transport_normalization="none",
        )


class SG5ControlShadowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.turn_token = None

    def tearDown(self) -> None:
        if self.turn_token is not None:
            grounding_callbacks._reset_turn_message(self.turn_token)

    def bind(self, current: dict) -> None:
        self.turn_token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", QUERY
        )

    async def run_tool(
        self,
        current: dict,
        tool_name: str,
        call_id: str,
        response: object,
        *,
        args: dict | None = None,
    ) -> object:
        tool = SimpleNamespace(name=tool_name)
        call_context = tool_context(current, call_id)
        call_args = args or {}
        await grounding_callbacks.before_tool_callback(tool, call_args, call_context)
        return await grounding_callbacks.after_tool_callback(
            tool,
            call_args,
            call_context,
            response,
        )

    async def test_intermediate_schema_is_evidence_only(self):
        current = task_state("sg5-initial")
        self.bind(current)
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=grounded_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            )
        )
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await self.run_tool(current, "get_schema", "schema-1", SCHEMA)

        final = load_runtime(current)
        self.assertEqual(final.stage, "INITIAL_GROUNDING")
        self.assertEqual(final.grounding_revision, 0)
        control = current["tool_trajectory"][0][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertEqual(control["stage_before"], "INITIAL_GROUNDING")
        self.assertEqual(control["stage_after"], "INITIAL_GROUNDING")
        self.assertEqual(updater.calls, 0)
        self.assertFalse(control["control_hint_injected"])

    async def test_submit_failure_does_not_reopen_focus(self):
        current = task_state("sg5-focus")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload(
            stage="SQL_ATTEMPT",
            focus="none",
        )
        add_bootstrap_trajectory(current)
        self.bind(current)
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=empty_complete_state(),
                user_clarification_requests=(),
                next_focus_dimension="domain_knowledge",
            )
        )
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await self.run_tool(
                current,
                "submit_sql",
                "focus-only-submit",
                "incorrect",
                args={"sql": "SELECT 1"},
            )

        focused = load_runtime(current)
        self.assertEqual(focused.stage, "SQL_ATTEMPT")
        self.assertEqual(focused.focus_dimension, "none")
        self.assertEqual(focused.grounding_revision, 0)
        requests = [
            {"contents": [{"role": "user", "text": "same"}]},
            {"contents": [{"role": "user", "text": "same"}]},
        ]
        originals = copy.deepcopy(requests)

        async def baseline(callback_context, request):
            del request
            calls = callback_context.state.setdefault("system_agent_llm_calls", [])
            calls.append({"actions": []})
            callback_context.state["_active_llm_call_index"] = len(calls) - 1
            return None

        with patch.object(baseline_callbacks, "before_model_callback", baseline):
            for request in requests:
                await grounding_callbacks.before_model_callback(
                    SimpleNamespace(state=current), request
                )

        self.assertEqual(requests, originals)
        for call in current["system_agent_llm_calls"]:
            control = call[grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY]
            self.assertEqual(control["focus_dimension"], "none")
            self.assertEqual(control["tool_directions"], [])
            self.assertFalse(control["control_hint_injected"])
        self.assertEqual(load_runtime(current).focus_dimension, "none")
        self.assertEqual(updater.calls, 0)

    async def test_closed_first_submit_is_blocked_before_baseline_cost(self):
        valibra = task_state("sg5-closed-v")
        self.bind(valibra)
        order: list[tuple[str, float]] = []

        def gate(runtime, *, first_submit, pending_clarifications=0):
            order.append(("gate", valibra["budget_remaining"]))
            return pure_evaluate_first_submit_gate(
                runtime,
                first_submit=first_submit,
                pending_clarifications=pending_clarifications,
            )

        tool = SimpleNamespace(name="submit_sql")
        baseline_mock = AsyncMock(return_value=None)
        with (
            patch.object(grounding_callbacks, "evaluate_first_submit_gate", gate),
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                baseline_mock,
            ),
        ):
            valibra_before = await grounding_callbacks.before_tool_callback(
                tool, {"sql": "SELECT 1"}, tool_context(valibra, "closed-v")
            )
        self.assertEqual(valibra_before["status"], "VALIBRA_FIRST_SUBMIT_BLOCKED")
        baseline_mock.assert_not_awaited()
        valibra_after = await grounding_callbacks.after_tool_callback(
            tool,
            {"sql": "SELECT 1"},
            tool_context(valibra, "closed-v"),
            valibra_before,
        )

        self.assertEqual(order, [("gate", 12.0)])
        self.assertEqual(valibra_after, valibra_before)
        self.assertEqual(valibra["budget_remaining"], 12.0)
        self.assertEqual(load_runtime(valibra).stage, "INITIAL_GROUNDING")
        self.assertEqual(valibra["tool_trajectory"], [])
        control = valibra[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY][0]
        self.assertEqual(control["effective_gate_action"], "blocked")
        self.assertTrue(control["after_tool_seen"])

    async def test_open_first_submit_preserves_baseline_without_repair(self):
        current = task_state("sg5-open")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload()
        add_bootstrap_trajectory(current)
        self.bind(current)
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=empty_complete_state(),
                user_clarification_requests=(),
                next_focus_dimension="column_mapping",
            )
        )
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await self.run_tool(
                current,
                "submit_sql",
                "open-1",
                "incorrect",
                args={"sql": "SELECT 1"},
            )
        final = load_runtime(current)
        self.assertEqual(final.stage, "SQL_ATTEMPT")
        self.assertEqual(final.focus_dimension, "none")
        self.assertEqual(final.grounding_revision, 0)
        control = current["tool_trajectory"][-1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertTrue(control["attempt_gate"]["open"])
        self.assertFalse(control["attempt_gate"]["would_block"])
        self.assertEqual(control["event"], "official_submit_failed")
        self.assertEqual(updater.observation_types, [])

    async def test_p1_failures_and_tools_never_reopen_grounding(self):
        current = task_state("sg5-p1-loop")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload()
        add_bootstrap_trajectory(current)
        self.bind(current)
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=grounded_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            ),
        )
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await self.run_tool(
                current, "submit_sql", "p1-fail-1", "incorrect", args={"sql": "S1"}
            )
            await self.run_tool(current, "get_schema", "p1-repair", SCHEMA)
            repaired = load_runtime(current)
            self.assertEqual(repaired.stage, "SQL_ATTEMPT")
            self.assertEqual(repaired.focus_dimension, "none")
            self.assertEqual(repaired.grounding_revision, 0)
            await self.run_tool(
                current, "submit_sql", "p1-fail-2", "incorrect", args={"sql": "S2"}
            )

        final = load_runtime(current)
        self.assertEqual(final.stage, "SQL_ATTEMPT")
        self.assertEqual(final.focus_dimension, "none")
        self.assertEqual(updater.calls, 0)
        second_gate = current["tool_trajectory"][-1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]["attempt_gate"]
        self.assertFalse(second_gate["applicable"])
        self.assertEqual(second_gate["reason"], "subsequent_submit_not_gated")

    async def test_p1_success_follow_up_grounds_once_without_rebootstrap(self):
        current = task_state("sg5-p2-follow-up")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload()
        add_bootstrap_trajectory(current)
        self.bind(current)
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=grounded_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            )
        )
        tool = SimpleNamespace(name="submit_sql")
        call = tool_context(current, "p1-pass")
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await grounding_callbacks.before_tool_callback(tool, {"sql": "S"}, call)
            current["phase1_completed"] = True
            current["current_phase"] = 2
            raw = "passed\nFollow-up question: Add active users.\nBudget remaining: 9"
            await grounding_callbacks.after_tool_callback(tool, {"sql": "S"}, call, raw)

        final = load_runtime(current)
        self.assertEqual(final.stage, "P2_INCREMENTAL")
        self.assertEqual(updater.calls, 3)
        self.assertEqual(updater.observation_types, ["p2_follow_up"] * 3)
        audit = current["tool_trajectory"][-1][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(
            [
                item["service_status"]
                for item in audit["p2_follow_up"]["staged_grounding"]
            ],
            ["accepted", "accepted", "accepted"],
        )
        control = current["tool_trajectory"][-1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertEqual(control["official_outcome"], "p1_follow_up")
        self.assertEqual(control["event"], "official_p2_follow_up")

    async def test_p1_failure_then_success_enters_p2_incremental(self):
        current = task_state("sg5-p1-repair-pass")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload()
        add_bootstrap_trajectory(current)
        self.bind(current)
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=grounded_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            ),
        )
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await self.run_tool(
                current,
                "submit_sql",
                "repair-pass-fail",
                "incorrect",
                args={"sql": "S1"},
            )
            await self.run_tool(current, "get_schema", "repair-pass-schema", SCHEMA)
            self.assertEqual(load_runtime(current).stage, "SQL_ATTEMPT")

            tool = SimpleNamespace(name="submit_sql")
            call = tool_context(current, "repair-pass-success")
            await grounding_callbacks.before_tool_callback(tool, {"sql": "S2"}, call)
            current["phase1_completed"] = True
            current["current_phase"] = 2
            await grounding_callbacks.after_tool_callback(
                tool,
                {"sql": "S2"},
                call,
                "passed\nFollow-up question: Add active users.\nBudget remaining: 3",
            )

        final = load_runtime(current)
        self.assertEqual(final.stage, "P2_INCREMENTAL")
        self.assertEqual(final.grounding_revision, 2)
        self.assertEqual(updater.calls, 3)
        control = current["tool_trajectory"][-1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertEqual(control["official_outcome"], "p1_follow_up")
        self.assertEqual(control["event"], "official_p2_follow_up")

    async def test_p2_failure_after_prior_submit_never_calls_repair(self):
        current = task_state("sg5-p2-no-repair")
        current["current_phase"] = 2
        current["phase1_completed"] = True
        current["tool_trajectory"].append({"type": "tool", "tool": "submit_sql"})
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload(
            stage="P2_INCREMENTAL"
        )
        self.bind(current)
        updater = QueueUpdater()
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await self.run_tool(
                current, "submit_sql", "p2-fail", "incorrect", args={"sql": "P2"}
            )
            self.assertEqual(load_runtime(current).stage, "P2_INCREMENTAL")
            await self.run_tool(current, "get_schema", "p2-schema", SCHEMA)

            tool = SimpleNamespace(name="submit_sql")
            call = tool_context(current, "p2-retry-pass")
            await grounding_callbacks.before_tool_callback(tool, {"sql": "P2B"}, call)
            current["phase2_completed"] = True
            current["task_done"] = True
            await grounding_callbacks.after_tool_callback(
                tool,
                {"sql": "P2B"},
                call,
                "passed",
            )

        final = load_runtime(current)
        self.assertEqual(final.stage, "DONE")
        self.assertEqual(final.focus_dimension, "none")
        self.assertEqual(final.grounding_revision, 0)
        self.assertEqual(updater.calls, 0)
        retry_gate = current["tool_trajectory"][-1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]["attempt_gate"]
        self.assertFalse(retry_gate["applicable"])
        self.assertEqual(retry_gate["reason"], "subsequent_submit_not_gated")

    async def test_p1_without_follow_up_and_p2_pass_reach_done(self):
        cases = (
            ("p1", 1, True, False),
            ("p2", 2, True, True),
        )
        for label, phase, phase1_done, phase2_done in cases:
            with self.subTest(label=label):
                current = task_state(f"sg5-done-{label}")
                current["current_phase"] = phase
                current["phase1_completed"] = phase == 2
                current["phase2_completed"] = False
                if phase == 2:
                    current["tool_trajectory"].append(
                        {"type": "tool", "tool": "submit_sql"}
                    )
                current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload(
                    stage="SQL_ATTEMPT" if phase == 1 else "P2_INCREMENTAL"
                )
                self.bind(current)
                tool = SimpleNamespace(name="submit_sql")
                call = tool_context(current, f"done-{label}")
                await grounding_callbacks.before_tool_callback(
                    tool, {"sql": "S"}, call
                )
                current["current_phase"] = 2
                current["phase1_completed"] = phase1_done
                current["phase2_completed"] = phase2_done
                current["task_done"] = True
                await grounding_callbacks.after_tool_callback(
                    tool, {"sql": "S"}, call, "passed"
                )
                grounding_callbacks._reset_turn_message(self.turn_token)
                self.turn_token = None
                self.assertEqual(load_runtime(current).stage, "DONE")

    async def test_active_closed_gate_never_forces_illegal_terminal_transition(self):
        current = task_state("sg5-closed-terminal")
        self.bind(current)
        tool = SimpleNamespace(name="submit_sql")
        call = tool_context(current, "closed-terminal")
        await grounding_callbacks.before_tool_callback(tool, {"sql": "S"}, call)
        current["phase1_completed"] = True
        current["current_phase"] = 2
        current["task_done"] = True
        await grounding_callbacks.after_tool_callback(tool, {"sql": "S"}, call, "passed")

        self.assertEqual(load_runtime(current).stage, "INITIAL_GROUNDING")
        self.assertEqual(current["tool_trajectory"], [])
        control = current[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY][0]
        self.assertEqual(control["effective_gate_action"], "blocked")
        self.assertTrue(control["after_tool_seen"])

    async def test_control_transition_error_fails_open_with_last_valid_runtime(self):
        current = task_state("sg5-control-fail-open")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload()
        add_bootstrap_trajectory(current)
        self.bind(current)
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=grounded_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            )
        )
        with (
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
            patch.object(
                grounding_callbacks,
                "transition_grounding_stage",
                side_effect=RuntimeError("private control detail"),
            ),
        ):
            returned = await self.run_tool(
                current,
                "submit_sql",
                "control-fail",
                "incorrect",
                args={"sql": "SELECT 1"},
            )

        self.assertIsNotNone(returned)
        final = load_runtime(current)
        self.assertEqual(final.stage, "SQL_ATTEMPT")
        self.assertEqual(final.grounding_revision, 0)
        control = current["tool_trajectory"][-1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertEqual(control["control_status"], "failed_open")
        self.assertEqual(control["error_type"], "RuntimeError")
        self.assertNotIn("private control detail", str(current))

    async def test_official_failure_does_not_enter_service_or_repair(self):
        current = task_state("sg5-service-fail-open")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload()
        add_bootstrap_trajectory(current)
        self.bind(current)
        with patch.object(
            grounding_callbacks,
            "_build_validation_context",
            side_effect=RuntimeError("private context detail"),
        ):
            await self.run_tool(
                current,
                "submit_sql",
                "service-fail",
                "incorrect",
                args={"sql": "S"},
            )

        final = load_runtime(current)
        self.assertEqual(final.stage, "SQL_ATTEMPT")
        self.assertEqual(final.grounding_revision, 0)
        control = current["tool_trajectory"][-1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertEqual(control["control_status"], "succeeded")
        self.assertNotIn("error_type", control)
        self.assertEqual(control["events"][0]["event"], "official_submit_failed")
        self.assertNotIn("private context detail", str(current))

    async def test_official_p2_transition_survives_follow_up_parse_failure(self):
        current = task_state("sg5-follow-up-fail-open")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime_payload()
        self.bind(current)
        tool = SimpleNamespace(name="submit_sql")
        call = tool_context(current, "follow-up-fail")
        await grounding_callbacks.before_tool_callback(tool, {"sql": "S"}, call)
        current["phase1_completed"] = True
        current["current_phase"] = 2
        await grounding_callbacks.after_tool_callback(
            tool,
            {"sql": "S"},
            call,
            "passed but malformed follow-up transport",
        )

        self.assertEqual(load_runtime(current).stage, "P2_INCREMENTAL")
        control = current["tool_trajectory"][0][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertEqual(control["control_status"], "failed_open")
        self.assertEqual(control["event"], "official_p2_follow_up")
        self.assertEqual(control["error_type"], "ValueError")


class SG5HealthAndFreezeTests(unittest.TestCase):
    def test_health_reports_active_visibility_and_active_gate_truthfully(self):
        summary = server._configuration_summary()
        self.assertEqual(summary["grounding_prompt_view_effective_mode"], "active")
        self.assertTrue(summary["prompt_view_injection_enabled"])
        self.assertEqual(summary["control_mode"], "active_hint")
        self.assertTrue(summary["control_hint_injection_enabled"])
        self.assertEqual(summary["attempt_gate_mode"], "active_first_submit")
        self.assertTrue(summary["attempt_gate_blocking_enabled"])
        self.assertTrue(summary["attempt_gate_budget_liveness_bypass"])
        self.assertTrue(summary["control_enabled"])
        self.assertTrue(summary["attempt_gate_enabled"])
        self.assertEqual(server._variant(summary), "SQL-Grounding-V1-SG6b-Active-Gate")

    def test_sg4_contract_hashes_are_unchanged(self):
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)


if __name__ == "__main__":
    unittest.main()
