from __future__ import annotations

import copy
import json
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingClientResponse,
    GroundingUpdaterResult,
    SQLGroundingUpdater,
)


QUERY = "Show the maintenance cost."
RULE = "Use maintcost as reported."
PROMPT_SHA = "34bfc4a5682510e4f9963cbc3f5fa55505f3715fffe50b6dd3a98890f25be425"
FORM_SHA = "728fc43c6ed85e72e60c9ebf85b00487059a2871020a28764b9641c69b84ed81"
CONFIG_SHA = "b1881e01b13314bf244591b406b86558ad3d30d07ed1a9197630ba40ccdf264a"
SCHEMA = """CREATE TABLE operational_metrics (
  maintcost NUMERIC,
  payload JSONB
);"""
COLUMN_MEANINGS = json.dumps(
    {
        "stage3|operational_metrics|maintcost": "Maintenance cost.",
        "stage3|operational_metrics|payload": {
            "column_meaning": "Structured maintenance data.",
            "fields_meaning": {"cost": "Reported maintenance cost."},
        },
    },
    sort_keys=True,
)
KNOWLEDGE_DEFINITIONS = json.dumps(
    [{"id": 1, "definition": RULE}],
    sort_keys=True,
)


def primary_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("operational_metrics",),
        join_keys=(),
        column_mapping=(
            ColumnMapping(
                phrase="maintenance cost",
                targets=("operational_metrics.maintcost",),
            ),
        ),
        domain_knowledge=(
            DomainKnowledge(kind="business_rule", content=RULE),
        ),
    )


def repaired_state() -> SQLGroundingState:
    return primary_state().model_copy(
        update={
            "column_mapping": (
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("operational_metrics.payload ->> 'cost'",),
                ),
            )
        }
    )


def official_event(
    tool: str,
    result: Any,
    *,
    args: dict[str, Any] | None = None,
    phase: int = 1,
    cost: float = 1.0,
) -> dict[str, Any]:
    return {
        "type": "tool",
        "tool": tool,
        "phase": phase,
        "args": copy.deepcopy(args or {}),
        "result": copy.deepcopy(result),
        "cost": cost,
        "budget_before": None,
        "budget_after": None,
        "action_input_tokens": 0,
        "action_output_tokens": 0,
        "timestamp": "2026-08-20T00:00:00Z",
    }


def bootstrap_trajectory() -> list[dict[str, Any]]:
    return [
        official_event("get_schema", SCHEMA),
        official_event("get_all_column_meanings", COLUMN_MEANINGS),
        official_event(
            "get_all_knowledge_definitions",
            KNOWLEDGE_DEFINITIONS,
        ),
    ]


def task_state(task_id: str) -> dict[str, Any]:
    runtime = GroundingRuntime(
        grounding_revision=1,
        stage="SQL_ATTEMPT",
        focus_dimension="none",
        grounding_state=primary_state(),
    )
    return {
        "task_id": task_id,
        "current_phase": 1,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 7.0,
        "initial_budget": 10.0,
        "tool_trajectory": bootstrap_trajectory(),
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 1,
    }


def load_runtime(state: dict[str, Any]) -> GroundingRuntime:
    return GroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


class RepairUpdater:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[dict[str, Any]] = []

    async def propose(self, runtime: GroundingRuntime, *args: Any, **kwargs: Any):
        del runtime, args
        self.calls += 1
        self.inputs.append(copy.deepcopy(kwargs["grounding_input"]))
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=repaired_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            ),
            telemetry=GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="a" * 64,
                response_sha256="b" * 64,
                prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
                form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


class UnexpectedUpdater:
    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, *args: Any, **kwargs: Any):
        del args, kwargs
        self.calls += 1
        raise AssertionError("Grounding must not be called")


class CapturingClient:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def complete(self, request: Any) -> GroundingClientResponse:
        self.requests.append(request)
        raise AssertionError("oversized Repair must not reach the client")


class Stage3SubmitDrivenRepairTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.turn_token = None

    def tearDown(self) -> None:
        if self.turn_token is not None:
            grounding_callbacks._reset_turn_message(self.turn_token)

    def bind(self, state: dict[str, Any]) -> None:
        self.turn_token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )

    async def run_tool(
        self,
        state: dict[str, Any],
        tool_name: str,
        call_id: str,
        result: Any,
        *,
        args: dict[str, Any] | None = None,
        official_updates: dict[str, Any] | None = None,
    ) -> Any:
        tool = SimpleNamespace(name=tool_name)
        context = SimpleNamespace(
            state=state,
            function_call_id=call_id,
            invocation_id=f"inv-{call_id}",
        )
        call_args = args or {}
        before = await grounding_callbacks.before_tool_callback(
            tool, call_args, context
        )
        self.assertIsNone(before)
        if official_updates:
            state.update(official_updates)
        return await grounding_callbacks.after_tool_callback(
            tool,
            call_args,
            context,
            result,
        )

    @contextmanager
    def provider_context(self, updater: Any):
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
            patch.object(
                grounding_callbacks,
                "load_sql_grounding_llm_config",
                return_value=SimpleNamespace(max_calls_per_task=8),
            ),
        ):
            yield

    async def test_every_execute_result_is_evidence_only(self):
        state = task_state("stage3-execute-only")
        self.bind(state)
        updater = UnexpectedUpdater()
        before_runtime = load_runtime(state)
        with self.provider_context(updater):
            await self.run_tool(
                state,
                "execute_sql",
                "execute-success",
                [{"maintcost": 7}],
                args={"sql": "SELECT maintcost FROM operational_metrics"},
            )
            await self.run_tool(
                state,
                "execute_sql",
                "execute-error",
                "SQL Error: synthetic failure",
                args={"sql": "SELECT missing FROM operational_metrics"},
            )
            await self.run_tool(
                state,
                "execute_sql",
                "execute-empty",
                [],
                args={"sql": "SELECT maintcost FROM operational_metrics WHERE FALSE"},
            )

        self.assertEqual(updater.calls, 0)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 1
        )
        self.assertEqual(load_runtime(state), before_runtime)
        self.assertEqual(
            [event["tool"] for event in state["tool_trajectory"]],
            [
                "get_schema",
                "get_all_column_meanings",
                "get_all_knowledge_definitions",
                "execute_sql",
                "execute_sql",
                "execute_sql",
            ],
        )
        statuses = [
            state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY][call_id][
                grounding_callbacks.SHADOW_AUDIT_KEY
            ]["service_status"]
            for call_id in ("execute-success", "execute-error", "execute-empty")
        ]
        self.assertEqual(
            statuses,
            [
                "stored_official_evidence_only",
                "skipped_tool_error_audit_only",
                "stored_official_evidence_only",
            ],
        )

    async def test_first_submit_pass_never_calls_grounding_again(self):
        state = task_state("stage3-submit-pass")
        self.bind(state)
        updater = UnexpectedUpdater()
        with self.provider_context(updater):
            await self.run_tool(
                state,
                "submit_sql",
                "submit-pass",
                "passed",
                args={"sql": "SELECT maintcost FROM operational_metrics"},
                official_updates={
                    "phase1_completed": True,
                    "task_done": True,
                    "current_phase": 2,
                },
            )
        self.assertEqual(updater.calls, 0)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 1
        )
        self.assertEqual(load_runtime(state).stage, "DONE")
        self.assertEqual(
            [event["tool"] for event in state["tool_trajectory"]].count(
                "get_schema"
            ),
            1,
        )

    async def test_submit_failures_never_call_grounding_or_change_state(self):
        state = task_state("stage3-submit-fail")
        self.bind(state)
        updater = UnexpectedUpdater()
        frozen = load_runtime(state)
        with self.provider_context(updater):
            await self.run_tool(
                state,
                "execute_sql",
                "execute-before-repair",
                [{"maintcost": 7}],
                args={"sql": "SELECT maintcost FROM operational_metrics"},
            )
            await self.run_tool(
                state,
                "execute_sql",
                "execute-error-before-repair",
                "SQL Error: wrong JSON path",
                args={"sql": "SELECT payload ->> 'bad' FROM operational_metrics"},
            )
            await self.run_tool(
                state,
                "submit_sql",
                "submit-fail-1",
                "incorrect",
                args={"sql": "SELECT maintcost FROM operational_metrics"},
            )

            self.assertEqual(updater.calls, 0)
            self.assertEqual(
                state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 1
            )
            self.assertEqual(load_runtime(state), frozen)

            await self.run_tool(
                state,
                "submit_sql",
                "submit-fail-2",
                "incorrect again",
                args={"sql": "SELECT payload ->> 'cost' FROM operational_metrics"},
            )

        self.assertEqual(updater.calls, 0)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 1
        )
        self.assertEqual(load_runtime(state), frozen)
        self.assertEqual(
            [event["tool"] for event in state["tool_trajectory"]].count(
                "get_all_knowledge_definitions"
            ),
            1,
        )
        second_audit = state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY][
            "submit-fail-2"
        ][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(
            second_audit["service_status"],
            "skipped_submit_failure_no_repair",
        )

    async def test_large_execute_history_still_does_not_trigger_grounding(self):
        state = task_state("stage3-size-bound")
        state["tool_trajectory"].extend(
            official_event(
                "execute_sql",
                "x" * 60_000,
                args={"sql": f"SELECT {index}"},
            )
            for index in range(5)
        )
        self.bind(state)
        updater = UnexpectedUpdater()
        frozen = load_runtime(state)
        with self.provider_context(updater):
            await self.run_tool(
                state,
                "submit_sql",
                "submit-size-bound",
                "incorrect",
                args={"sql": "SELECT maintcost FROM operational_metrics"},
            )
        self.assertEqual(updater.calls, 0)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 1
        )
        self.assertEqual(load_runtime(state), frozen)
        audit = state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY][
            "submit-size-bound"
        ][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(audit["service_status"], "skipped_submit_failure_no_repair")

    def test_no_repair_prompt_and_form_contract_are_frozen(self):
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)
        self.assertNotIn("execute_sql_evidence", SQL_GROUNDING_PROMPT)
        self.assertNotIn("submit_failure", SQL_GROUNDING_PROMPT)
        self.assertNotIn("REPAIR", SQL_GROUNDING_PROMPT)


if __name__ == "__main__":
    unittest.main()
