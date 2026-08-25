from __future__ import annotations

import asyncio
import copy
import math
import os
import unittest
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks, server
from valibra_agent.sql_grounding.control import (
    evaluate_first_submit_gate,
    tool_directions_for_focus,
    transition_grounding_stage,
)
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    SQLGroundingValidationError,
    canonical_json,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
)


PROMPT_SHA = "db5a44e94a0e7fba92e78e6ca99e6b324affdee92aaed4dcd45010ff03876454"
FORM_SHA = "728fc43c6ed85e72e60c9ebf85b00487059a2871020a28764b9641c69b84ed81"
CONFIG_SHA = "31115d248a048c76a9ffd947aade7c4fa725915f82243809f06b8e14cbd98f0c"
QUERY = "Show the maintenance cost."
SCHEMA = """CREATE TABLE operational_metrics (
  maintcost NUMERIC
);"""


def task_state(task_id: str, *, budget: Any = 10.0) -> dict[str, Any]:
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


def incomplete_runtime(focus: str = "tables") -> GroundingRuntime:
    return GroundingRuntime(focus_dimension=focus)


def complete_state() -> SQLGroundingState:
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


def complete_runtime() -> GroundingRuntime:
    return GroundingRuntime(
        grounding_revision=1,
        stage="SQL_ATTEMPT",
        focus_dimension="none",
        grounding_state=complete_state(),
    )


def store_runtime(state: dict[str, Any], runtime: GroundingRuntime) -> None:
    state[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime.model_dump(mode="json")


def load_runtime(state: dict[str, Any]) -> GroundingRuntime:
    return GroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


def tool_context(state: dict[str, Any], call_id: str | None) -> SimpleNamespace:
    payload = {"state": state, "invocation_id": f"inv-{call_id or 'missing'}"}
    if call_id is not None:
        payload["function_call_id"] = call_id
    return SimpleNamespace(**payload)


class QueueUpdater:
    def __init__(self, *responses: GroundingLLMResponse) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def propose(self, runtime, observation, **kwargs):
        del runtime, observation, kwargs
        response = self.responses[self.calls]
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


class ScriptedLlm(BaseLlm):
    script: list[tuple[str, dict[str, Any]]]
    calls: int = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        self.calls += 1
        if self.calls <= len(self.script):
            name, args = self.script[self.calls - 1]
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id=f"sg6b-call-{self.calls}",
                                name=name,
                                args=args,
                            )
                        )
                    ],
                )
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="SG6B_LOCAL_DONE")],
            )
        )


async def run_actual_adk(
    *,
    task_id: str,
    budget: float,
    script: list[tuple[str, dict[str, Any]]],
    updater: Any | None = None,
) -> tuple[dict[str, Any], dict[str, int], ScriptedLlm, list[Any]]:
    counts = {"get_schema": 0, "submit_sql": 0}

    def get_schema() -> str:
        counts["get_schema"] += 1
        return SCHEMA

    def submit_sql(sql: str) -> dict[str, str]:
        del sql
        counts["submit_sql"] += 1
        return {"status": "incorrect"}

    model = ScriptedLlm(model="local-sg6b", script=script)
    agent = LlmAgent(
        name="sg6b_local_agent",
        model=model,
        instruction="Offline SG6b ADK lifecycle fixture.",
        tools=[get_schema, submit_sql],
        before_model_callback=grounding_callbacks.before_model_callback,
        after_model_callback=grounding_callbacks.after_model_callback,
        before_tool_callback=grounding_callbacks.before_tool_callback,
        after_tool_callback=grounding_callbacks.after_tool_callback,
        on_tool_error_callback=grounding_callbacks.on_tool_error_callback,
    )
    runner = InMemoryRunner(agent=agent, app_name=f"sg6b_{task_id}")
    session = await runner.session_service.create_session(
        app_name=f"sg6b_{task_id}",
        user_id=f"user-{task_id}",
        state=task_state(task_id, budget=budget),
    )
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=QUERY)],
    )
    events: list[Any] = []
    turn_token = grounding_callbacks._bind_turn_message(task_id, "a-interact", QUERY)
    replacement = updater or grounding_callbacks._SQL_GROUNDING_UPDATER
    try:
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}, clear=False),
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", replacement),
        ):
            async for event in runner.run_async(
                user_id=f"user-{task_id}",
                session_id=session.id,
                new_message=message,
            ):
                events.append(event)
    finally:
        grounding_callbacks._reset_turn_message(turn_token)
    final = await runner.session_service.get_session(
        app_name=f"sg6b_{task_id}",
        user_id=f"user-{task_id}",
        session_id=session.id,
    )
    return dict(final.state), counts, model, events


class SG6bExecutionPolicyTests(unittest.IsolatedAsyncioTestCase):
    def policy(self, *, focus: str, budget: float):
        runtime = incomplete_runtime(focus)
        gate = evaluate_first_submit_gate(runtime, first_submit=True)
        return grounding_callbacks._evaluate_gate_execution_policy(
            runtime,
            gate=gate,
            state={"budget_remaining": budget},
            tool_costs=baseline_callbacks.TOOL_COSTS,
        )

    def test_affordability_matrix_uses_only_baseline_costs(self):
        cases = (
            ("tables", 0.25, "budget_liveness_bypass", ()),
            ("tables", 1.0, "blocked", ("get_schema",)),
            ("column_mapping", 0.5, "blocked", ("get_column_meaning",)),
            ("column_mapping", 0.49, "budget_liveness_bypass", ()),
            (
                "domain_knowledge",
                0.5,
                "blocked",
                (
                    "get_all_external_knowledge_names",
                    "get_knowledge_definition",
                ),
            ),
            ("join_keys", math.nextafter(1.0, 0.0), "budget_liveness_bypass", ()),
        )
        for focus, budget, action, affordable in cases:
            with self.subTest(focus=focus, budget=budget):
                result = self.policy(focus=focus, budget=budget)
                self.assertEqual(result.action, action)
                self.assertEqual(result.affordable_directions, affordable)
                self.assertEqual(
                    result.focus_directions,
                    tool_directions_for_focus(focus),
                )

    def test_open_and_subsequent_submit_skip_liveness_policy(self):
        opened = complete_runtime()
        for first_submit in (True, False):
            with self.subTest(first_submit=first_submit):
                gate = evaluate_first_submit_gate(
                    opened,
                    first_submit=first_submit,
                )
                policy = grounding_callbacks._evaluate_gate_execution_policy(
                    opened,
                    gate=gate,
                    state={"budget_remaining": "not-read"},
                    tool_costs={},
                )
                self.assertEqual(policy.action, "open")

    async def test_hard_block_is_exact_bounded_and_not_official(self):
        state = task_state("sg6b-direct-block")
        original_runtime = incomplete_runtime()
        store_runtime(state, original_runtime)
        tool = SimpleNamespace(name="submit_sql")
        context = tool_context(state, "blocked-1")
        sql = "SELECT secret_payload FROM hidden_table"
        baseline_before = AsyncMock(wraps=baseline_callbacks.before_tool_callback)
        baseline_after = AsyncMock(wraps=baseline_callbacks.after_tool_callback)
        with (
            patch.object(baseline_callbacks, "before_tool_callback", baseline_before),
            patch.object(baseline_callbacks, "after_tool_callback", baseline_after),
        ):
            denial = await grounding_callbacks.before_tool_callback(
                tool,
                {"sql": sql},
                context,
            )
            returned = await grounding_callbacks.after_tool_callback(
                tool,
                {"sql": sql},
                context,
                denial,
            )

        self.assertEqual(returned, denial)
        self.assertEqual(denial["status"], "VALIBRA_FIRST_SUBMIT_BLOCKED")
        baseline_before.assert_not_awaited()
        baseline_after.assert_not_awaited()
        self.assertEqual(state["budget_remaining"], 10.0)
        self.assertEqual(state["tool_trajectory"], [])
        self.assertEqual(load_runtime(state), original_runtime)
        self.assertEqual(state[grounding_callbacks.GROUNDING_BLOCKED_SUBMITS_KEY], {})
        self.assertNotIn(sql, canonical_json(state))
        audits = state[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY]
        self.assertEqual(len(audits), 1)
        self.assertTrue(audits[0]["after_tool_seen"])
        self.assertFalse(audits[0]["actual_tool_executed"])

    async def test_repeated_blocked_submit_remains_first_and_cleans_exact_ids(self):
        state = task_state("sg6b-repeat")
        store_runtime(state, incomplete_runtime())
        tool = SimpleNamespace(name="submit_sql")
        for call_id in ("blocked-a", "blocked-b"):
            context = tool_context(state, call_id)
            denial = await grounding_callbacks.before_tool_callback(
                tool,
                {"sql": f"SELECT '{call_id}'"},
                context,
            )
            await grounding_callbacks.after_tool_callback(
                tool,
                {"sql": f"SELECT '{call_id}'"},
                context,
                denial,
            )
        self.assertEqual(state["tool_trajectory"], [])
        self.assertEqual(state[grounding_callbacks.GROUNDING_BLOCKED_SUBMITS_KEY], {})
        self.assertEqual(len(state[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY]), 2)
        self.assertTrue(grounding_callbacks._is_first_official_submit(state))

    async def test_missing_id_runtime_and_policy_errors_fail_open_once(self):
        cases: list[tuple[str, dict[str, Any], Any]] = []
        missing_id = task_state("sg6b-missing-id")
        store_runtime(missing_id, incomplete_runtime())
        cases.append(("missing_id", missing_id, None))
        invalid_runtime = task_state("sg6b-invalid-runtime")
        invalid_runtime[grounding_callbacks.GROUNDING_RUNTIME_KEY] = {"bad": "shape"}
        cases.append(("runtime", invalid_runtime, "runtime"))
        invalid_budget = task_state("sg6b-invalid-budget", budget="invalid")
        store_runtime(invalid_budget, incomplete_runtime())
        cases.append(("budget", invalid_budget, "budget"))

        for label, state, marker in cases:
            with self.subTest(label=label):
                baseline = AsyncMock(return_value={"baseline": marker})
                context = tool_context(state, None if label == "missing_id" else label)
                with patch.object(
                    baseline_callbacks,
                    "before_tool_callback",
                    baseline,
                ):
                    result = await grounding_callbacks.before_tool_callback(
                        SimpleNamespace(name="submit_sql"),
                        {"sql": "SELECT 1"},
                        context,
                    )
                self.assertEqual(result, {"baseline": marker})
                baseline.assert_awaited_once()
                self.assertFalse(
                    grounding_callbacks._blocked_submit_present(state, label)
                )

        evaluation_state = task_state("sg6b-evaluation-error")
        store_runtime(evaluation_state, incomplete_runtime())
        baseline = AsyncMock(return_value={"baseline": "evaluation"})
        with (
            patch.object(
                grounding_callbacks,
                "evaluate_first_submit_gate",
                side_effect=RuntimeError("private"),
            ),
            patch.object(baseline_callbacks, "before_tool_callback", baseline),
        ):
            result = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="submit_sql"),
                {"sql": "SELECT 1"},
                tool_context(evaluation_state, "evaluation"),
            )
        self.assertEqual(result, {"baseline": "evaluation"})
        baseline.assert_awaited_once()

    async def test_cost_mapping_anomaly_fails_open_without_guessing(self):
        state = task_state("sg6b-cost-anomaly")
        store_runtime(state, incomplete_runtime())
        baseline = AsyncMock(return_value={"baseline": "cost"})
        with (
            patch.object(
                baseline_callbacks,
                "TOOL_COSTS",
                {"submit_sql": 3.0},
            ),
            patch.object(baseline_callbacks, "before_tool_callback", baseline),
        ):
            returned = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="submit_sql"),
                {"sql": "SELECT 1"},
                tool_context(state, "cost-anomaly"),
            )
        self.assertEqual(returned, {"baseline": "cost"})
        baseline.assert_awaited_once()

    async def test_task_scoped_block_and_bypass_do_not_cross(self):
        blocked = task_state("sg6b-concurrent-a", budget=10.0)
        bypassed = task_state("sg6b-concurrent-b", budget=0.25)
        store_runtime(blocked, incomplete_runtime())
        store_runtime(bypassed, incomplete_runtime())
        tool = SimpleNamespace(name="submit_sql")

        async def execute(state, call_id):
            context = tool_context(state, call_id)
            before = await grounding_callbacks.before_tool_callback(
                tool,
                {"sql": f"SELECT '{call_id}'"},
                context,
            )
            response = before if before is not None else {"status": "incorrect"}
            after = await grounding_callbacks.after_tool_callback(
                tool,
                {"sql": f"SELECT '{call_id}'"},
                context,
                response,
            )
            return before, after

        (block_before, _), (bypass_before, _) = await asyncio.gather(
            execute(blocked, "task-a"),
            execute(bypassed, "task-b"),
        )
        self.assertEqual(block_before["status"], "VALIBRA_FIRST_SUBMIT_BLOCKED")
        self.assertIsNone(bypass_before)
        self.assertEqual(blocked["budget_remaining"], 10.0)
        self.assertEqual(blocked["tool_trajectory"], [])
        self.assertEqual(bypassed["budget_remaining"], -1)
        self.assertEqual(len(bypassed["tool_trajectory"]), 1)
        self.assertNotIn("task-b", canonical_json(blocked))
        self.assertNotIn("task-a", canonical_json(bypassed))

    def test_initial_forced_exit_transition_is_explicit_and_revision_neutral(self):
        runtime = incomplete_runtime()
        with self.assertRaises(SQLGroundingValidationError):
            transition_grounding_stage(runtime, "official_submit_failed")
        after_failure = transition_grounding_stage(
            runtime,
            "official_submit_failed",
            allow_initial_forced_exit=True,
        )
        p2 = transition_grounding_stage(
            runtime,
            "official_p2_follow_up",
            allow_initial_forced_exit=True,
        )
        self.assertEqual(after_failure.stage, "INITIAL_GROUNDING")
        self.assertEqual(p2.stage, "P2_INCREMENTAL")
        self.assertEqual(after_failure.grounding_revision, runtime.grounding_revision)
        self.assertEqual(p2.grounding_revision, runtime.grounding_revision)


class SG6bActualAdkLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_adk_closed_override_skips_tool_and_official_trajectory(self):
        before_calls: list[str] = []
        after_calls: list[str] = []
        original_before = baseline_callbacks.before_tool_callback
        original_after = baseline_callbacks.after_tool_callback

        async def tracked_before(tool, args, context):
            before_calls.append(tool.name)
            return await original_before(tool, args, context)

        async def tracked_after(tool, args, context, response):
            after_calls.append(tool.name)
            return await original_after(tool, args, context, response)

        with (
            patch.object(baseline_callbacks, "before_tool_callback", tracked_before),
            patch.object(baseline_callbacks, "after_tool_callback", tracked_after),
        ):
            state, counts, model, events = await run_actual_adk(
                task_id="sg6b-adk-block",
                budget=10.0,
                script=[("submit_sql", {"sql": "SELECT 1"})],
            )
        self.assertEqual(counts["submit_sql"], 0)
        self.assertEqual(before_calls, [])
        self.assertEqual(after_calls, [])
        self.assertEqual(state["budget_remaining"], 10.0)
        self.assertEqual(state["tool_trajectory"], [])
        self.assertEqual(load_runtime(state).stage, "INITIAL_GROUNDING")
        self.assertEqual(model.calls, 2)
        serialized = canonical_json(
            [event.model_dump(mode="json") for event in events]
        )
        self.assertIn("VALIBRA_FIRST_SUBMIT_BLOCKED", serialized)
        audit = state[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY][0]
        self.assertTrue(audit["after_tool_seen"])
        self.assertFalse(audit["baseline_before_tool_called"])
        self.assertFalse(audit["baseline_after_tool_called"])

    async def test_budget_liveness_restores_actual_official_termination(self):
        before_calls: list[str] = []
        after_calls: list[str] = []
        original_before = baseline_callbacks.before_tool_callback
        original_after = baseline_callbacks.after_tool_callback

        async def tracked_before(tool, args, context):
            before_calls.append(tool.name)
            return await original_before(tool, args, context)

        async def tracked_after(tool, args, context, response):
            after_calls.append(tool.name)
            return await original_after(tool, args, context, response)

        with (
            patch.object(baseline_callbacks, "before_tool_callback", tracked_before),
            patch.object(baseline_callbacks, "after_tool_callback", tracked_after),
        ):
            state, counts, model, _ = await run_actual_adk(
                task_id="sg6b-adk-bypass",
                budget=0.25,
                script=[
                    ("get_schema", {}),
                    ("submit_sql", {"sql": "SELECT 1"}),
                ],
            )
        self.assertEqual(counts, {"get_schema": 0, "submit_sql": 1})
        self.assertEqual(before_calls, ["get_schema", "submit_sql"])
        self.assertEqual(after_calls, ["get_schema", "submit_sql"])
        self.assertEqual(model.calls, 2)
        self.assertEqual(state["budget_remaining"], -1)
        self.assertEqual(
            [event["tool"] for event in state["tool_trajectory"]],
            ["get_schema", "submit_sql"],
        )
        submit_audit = state["tool_trajectory"][1][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        gate = submit_audit["attempt_gate"]
        self.assertTrue(gate["would_block"])
        self.assertFalse(gate["blocked"])
        self.assertTrue(gate["liveness_bypass"])
        self.assertEqual(gate["budget_remaining"], 0.25)
        self.assertEqual(gate["focus_direction_count"], 1)
        self.assertEqual(gate["affordable_direction_count"], 0)
        self.assertEqual(gate["effective_gate_action"], "budget_liveness_bypass")
        self.assertEqual(submit_audit["control_status"], "succeeded")
        self.assertNotIn("error_type", submit_audit)
        self.assertEqual(load_runtime(state).stage, "INITIAL_GROUNDING")

    async def test_intermediate_schema_no_longer_opens_blocked_submit(self):
        updater = QueueUpdater(
            GroundingLLMResponse(
                sql_grounding_state=complete_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            ),
            GroundingLLMResponse(
                sql_grounding_state=complete_state(),
                user_clarification_requests=(),
                next_focus_dimension="column_mapping",
            ),
        )
        state, counts, model, events = await run_actual_adk(
            task_id="sg6b-adk-progress",
            budget=10.0,
            script=[
                ("submit_sql", {"sql": "SELECT 0"}),
                ("get_schema", {}),
                ("submit_sql", {"sql": "SELECT maintcost FROM operational_metrics"}),
            ],
            updater=updater,
        )
        self.assertEqual(counts, {"get_schema": 1, "submit_sql": 0})
        self.assertEqual(model.calls, 4)
        self.assertEqual(state["budget_remaining"], 9.0)
        self.assertEqual(
            [event["tool"] for event in state["tool_trajectory"]],
            ["get_schema"],
        )
        self.assertEqual(len(state[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY]), 2)
        self.assertTrue(
            all(
                audit["effective_gate_action"] == "blocked"
                for audit in state[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY]
            )
        )
        serialized = canonical_json(
            [event.model_dump(mode="json") for event in events]
        )
        self.assertIn("VALIBRA_FIRST_SUBMIT_BLOCKED", serialized)


class SG6bHealthAndFreezeTests(unittest.TestCase):
    def test_health_truthfully_reports_active_gate_and_liveness_bypass(self):
        summary = server._configuration_summary()
        self.assertEqual(summary["attempt_gate_mode"], "active_first_submit")
        self.assertTrue(summary["attempt_gate_blocking_enabled"])
        self.assertTrue(summary["attempt_gate_enabled"])
        self.assertTrue(summary["attempt_gate_budget_liveness_bypass"])
        self.assertEqual(
            server._variant(summary),
            "SQL-Grounding-V1-SG6b-Active-Gate",
        )

    def test_sg6a_visibility_and_frozen_contracts_are_unchanged(self):
        summary = server._configuration_summary()
        self.assertEqual(summary["grounding_prompt_view_effective_mode"], "active")
        self.assertTrue(summary["prompt_view_injection_enabled"])
        self.assertEqual(summary["control_mode"], "active_hint")
        self.assertTrue(summary["control_hint_injection_enabled"])
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)


if __name__ == "__main__":
    unittest.main()
