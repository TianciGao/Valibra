from __future__ import annotations

import copy
import os
import unittest
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    UserClarificationRecord,
    UserClarificationRequest,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    DEFAULT_GROUNDING_MAX_CALLS_PER_TASK,
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
)
from tests.valibra.test_stage3_submit_driven_repair import (
    QUERY,
    bootstrap_trajectory,
    load_runtime,
    primary_state,
    repaired_state,
    task_state,
)


FOLLOW_UP = "Now include the current power reading."
PROMPT_SHA = "8d4e53fd2f53ea2635d4ddfb8bcca5c32da271546e0c181614e5cb5690b1635c"
FORM_SHA = "9d3cef810801de43bf9d6537a9811641252652cb910f4beb0248d6b129b52642"
CONFIG_SHA = "24b00b82a2ee3a8219fadba8728a31fe7d29c9b97e51d8892e0c87a3093f7557"
P2_SUBMIT_RESPONSE = (
    f"passed\nFollow-up question: {FOLLOW_UP}\nBudget remaining: 4"
)


class QueueUpdater:
    def __init__(
        self,
        *states: SQLGroundingState,
        clarifications: tuple[UserClarificationRequest, ...] = (),
    ) -> None:
        self.states = deque(states)
        self.clarifications = clarifications
        self.calls = 0
        self.inputs: list[dict[str, Any]] = []

    async def propose(
        self,
        runtime: GroundingRuntime,
        *args: Any,
        **kwargs: Any,
    ) -> GroundingUpdaterResult:
        del args
        self.calls += 1
        grounding_input = copy.deepcopy(kwargs["grounding_input"])
        self.inputs.append(grounding_input)
        fields = set(grounding_input)
        if "schema" in fields:
            candidate = runtime.grounding_state
            focus = "column_mapping"
            clarifications = ()
        elif "column_meanings" in fields:
            candidate = runtime.grounding_state
            focus = "domain_knowledge"
            clarifications = ()
        elif "knowledge_definitions" in fields:
            if not self.states:
                raise AssertionError("Grounding received an unexpected Knowledge call")
            candidate = self.states.popleft()
            focus = "none"
            clarifications = self.clarifications
        else:
            candidate = runtime.grounding_state
            focus = "none"
            clarifications = ()
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=candidate,
                user_clarification_requests=clarifications,
                next_focus_dimension=focus,
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


class Stage4AP2GroundingLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.turn_token = None

    def tearDown(self) -> None:
        if self.turn_token is not None:
            grounding_callbacks._reset_turn_message(self.turn_token)

    def bind(self, state: dict[str, Any]) -> None:
        self.turn_token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )

    @contextmanager
    def provider_context(self, updater: QueueUpdater):
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
            tool, call_args, context, result
        )

    async def enter_p2(
        self,
        state: dict[str, Any],
        updater: QueueUpdater,
    ) -> None:
        with self.provider_context(updater):
            await self.run_tool(
                state,
                "submit_sql",
                "p1-submit-pass",
                P2_SUBMIT_RESPONSE,
                args={"sql": "SELECT maintcost FROM operational_metrics"},
                official_updates={
                    "phase1_completed": True,
                    "phase2_completed": False,
                    "task_done": False,
                    "current_phase": 2,
                },
            )

    def assert_bootstrap_once(self, state: dict[str, Any]) -> None:
        tools = [event["tool"] for event in state["tool_trajectory"]]
        for tool_name in (
            "get_schema",
            "get_all_column_meanings",
            "get_all_knowledge_definitions",
        ):
            self.assertEqual(tools.count(tool_name), 1)

    async def test_p2_follow_up_triggers_three_staged_calls_and_execute_never_does(self):
        state = task_state("stage4a-p2-follow-up")
        self.bind(state)
        updater = QueueUpdater(primary_state())

        await self.enter_p2(state, updater)

        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            [set(item) for item in updater.inputs],
            [
                {
                    "query", "follow_up", "schema", "current_state",
                    "user_clarifications",
                },
                {
                    "query", "follow_up", "column_meanings", "current_state",
                    "user_clarifications",
                },
                {
                    "query", "follow_up", "knowledge_definitions",
                    "relevant_column_meanings", "current_state",
                    "user_clarifications",
                },
            ],
        )
        self.assertEqual(updater.inputs[0]["query"], QUERY)
        self.assertEqual(updater.inputs[0]["follow_up"], FOLLOW_UP)
        self.assertEqual(updater.inputs[0]["user_clarifications"], [])
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 4
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 1, "2": 3},
        )
        self.assertEqual(load_runtime(state).stage, "P2_INCREMENTAL")
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PHASE_OUTCOMES_KEY]["2"][
                "status"
            ],
            "succeeded",
        )
        self.assert_bootstrap_once(state)
        frozen_runtime = load_runtime(state)
        frozen_sha = sql_grounding_state_sha256(
            frozen_runtime.grounding_state
        )

        with self.provider_context(updater):
            await self.run_tool(
                state,
                "execute_sql",
                "p2-execute-success",
                [{"maintcost": 7}],
                args={"sql": "SELECT maintcost FROM operational_metrics"},
            )
            await self.run_tool(
                state,
                "execute_sql",
                "p2-execute-error",
                "SQL Error: synthetic failure",
                args={"sql": "SELECT missing FROM operational_metrics"},
            )

        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 4
        )
        self.assertEqual(load_runtime(state), frozen_runtime)
        self.assertEqual(
            sql_grounding_state_sha256(load_runtime(state).grounding_state),
            frozen_sha,
        )
        with self.provider_context(updater):
            await self.run_tool(
                state,
                "submit_sql",
                "p2-submit-pass",
                "passed",
                args={"sql": "SELECT maintcost FROM operational_metrics"},
                official_updates={
                    "phase2_completed": True,
                    "task_done": True,
                    "current_phase": 2,
                },
            )
        self.assertEqual(updater.calls, 3)
        self.assertEqual(load_runtime(state).stage, "DONE")

    async def test_p2_request_contains_answered_phase_one_clarification(self):
        state = task_state("stage4a-p2-p1-clarification")
        self.bind(state)
        record = UserClarificationRecord(
            phase=1,
            phrase="maintenance cost",
            kind="user_intent",
            question="Which maintenance cost definition should be used?",
            answer="Use the reported maintenance cost.",
        )
        grounding_callbacks._store_clarification_records(state, (record,))
        updater = QueueUpdater(primary_state())

        await self.enter_p2(state, updater)

        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            [item["user_clarifications"] for item in updater.inputs],
            [[record.model_dump(mode="json")]] * 3,
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 1, "2": 3},
        )
        self.assert_bootstrap_once(state)

    async def test_p2_submit_failures_never_trigger_grounding(self):
        state = task_state("stage4a-p2-no-repair")
        self.bind(state)
        updater = QueueUpdater(primary_state())
        await self.enter_p2(state, updater)
        frozen = load_runtime(state)

        with self.provider_context(updater):
            await self.run_tool(
                state,
                "execute_sql",
                "p2-execute-before-repair",
                "SQL Error: wrong JSON path",
                args={"sql": "SELECT payload ->> 'bad' FROM operational_metrics"},
            )
            await self.run_tool(
                state,
                "submit_sql",
                "p2-submit-fail-1",
                "incorrect",
                args={"sql": "SELECT maintcost FROM operational_metrics"},
            )

        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 1, "2": 3},
        )
        self.assertEqual(load_runtime(state), frozen)

        with self.provider_context(updater):
            await self.run_tool(
                state,
                "submit_sql",
                "p2-submit-fail-2",
                "incorrect again",
                args={"sql": "SELECT payload ->> 'cost' FROM operational_metrics"},
            )

        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 4
        )
        audit = state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY][
            "p2-submit-fail-2"
        ][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(
            audit["service_status"], "skipped_submit_failure_no_repair"
        )

    @unittest.skip("retired 1.2 Clarification Patch; v1.3 resumes unified Check")
    async def test_p2_clarification_answer_triggers_one_targeted_patch(self):
        state = task_state("stage4a-p2-clarification")
        self.bind(state)
        question = "Should current power mean the latest recorded reading?"
        updater = QueueUpdater(
            primary_state(),
            clarifications=(
                UserClarificationRequest(
                    phrase="current power",
                    kind="user_intent",
                    question=question,
                ),
            ),
        )
        await self.enter_p2(state, updater)
        frozen = load_runtime(state)
        pending = grounding_callbacks._next_pending_clarification(state)
        self.assertIsNotNone(pending)
        call_id = grounding_callbacks._clarification_function_call_id(
            state,
            pending,
        )
        with self.provider_context(updater):
            await self.run_tool(
                state,
                "ask_user",
                call_id,
                "Use the latest reading.",
                args={"question": question},
            )
        self.assertEqual(updater.calls, 4)
        self.assertEqual(load_runtime(state).grounding_state, frozen.grounding_state)
        self.assertEqual(
            load_runtime(state).grounding_revision,
            frozen.grounding_revision,
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 1, "2": 4},
        )
        self.assertEqual(
            set(updater.inputs[-1]),
            {
                "query",
                "follow_up",
                "user_clarifications",
                "current_state",
                "clarification_qa",
                "relevant_column_meanings",
                "relevant_knowledge_definitions",
            },
        )
        records = grounding_callbacks._clarification_records(state)
        self.assertEqual(records[-1].phase, 2)
        self.assertEqual(records[-1].answer, "Use the latest reading.")

    async def test_p1_and_p2_allow_three_or_four_staged_calls_only(self):
        state = task_state("stage4a-two-call-cap")
        self.bind(state)
        updater = QueueUpdater(primary_state())

        with self.provider_context(updater):
            await self.run_tool(
                state,
                "submit_sql",
                "p1-submit-fail",
                "incorrect",
                args={"sql": "SELECT maintcost FROM operational_metrics"},
            )
            await self.run_tool(
                state,
                "submit_sql",
                "p1-submit-pass-after-repair",
                P2_SUBMIT_RESPONSE,
                args={"sql": "SELECT payload ->> 'cost' FROM operational_metrics"},
                official_updates={
                    "phase1_completed": True,
                    "phase2_completed": False,
                    "task_done": False,
                    "current_phase": 2,
                },
            )
            await self.run_tool(
                state,
                "submit_sql",
                "p2-submit-fail",
                "incorrect phase 2",
                args={"sql": "SELECT payload ->> 'cost' FROM operational_metrics"},
            )

        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY], 4
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 1, "2": 3},
        )
        self.assertEqual(load_runtime(state).stage, "P2_INCREMENTAL")
        self.assert_bootstrap_once(state)

    async def test_duplicate_bootstrap_tools_are_suppressed_before_charge(self):
        state = task_state("stage4a-no-rebootstrap")
        original_budget = state["budget_remaining"]
        original_trajectory = copy.deepcopy(state["tool_trajectory"])

        from system_agent import callbacks as baseline_callbacks

        with (
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                new=AsyncMock(side_effect=AssertionError("Baseline must not charge")),
            ) as baseline_before,
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                new=AsyncMock(side_effect=AssertionError("Official tool did not run")),
            ) as baseline_after,
        ):
            for index, tool_name in enumerate(
                (
                    "get_schema",
                    "get_all_column_meanings",
                    "get_all_knowledge_definitions",
                ),
                start=1,
            ):
                tool = SimpleNamespace(name=tool_name)
                context = SimpleNamespace(
                    state=state,
                    function_call_id=f"duplicate-bootstrap-{index}",
                    invocation_id=f"inv-duplicate-bootstrap-{index}",
                )
                response = await grounding_callbacks.before_tool_callback(
                    tool, {}, context
                )
                self.assertEqual(response["status"], "already_stored")
                returned = await grounding_callbacks.after_tool_callback(
                    tool, {}, context, response
                )
                self.assertEqual(returned, response)

            failed_state = task_state("stage4a-no-rebootstrap-after-error")
            failed_state["tool_trajectory"] = [
                {
                    "type": "tool",
                    "tool": "get_schema",
                    "phase": 1,
                    "args": {},
                    "result": "Error: synthetic schema outage",
                }
            ]
            failed_tool = SimpleNamespace(name="get_schema")
            failed_context = SimpleNamespace(
                state=failed_state,
                function_call_id="duplicate-bootstrap-after-error",
                invocation_id="inv-duplicate-bootstrap-after-error",
            )
            failed_response = await grounding_callbacks.before_tool_callback(
                failed_tool, {}, failed_context
            )
            self.assertEqual(failed_response["status"], "already_stored")
            self.assertEqual(
                await grounding_callbacks.after_tool_callback(
                    failed_tool, {}, failed_context, failed_response
                ),
                failed_response,
            )
            self.assertEqual(len(failed_state["tool_trajectory"]), 1)
            self.assertEqual(
                failed_state.get(
                    grounding_callbacks.GROUNDING_SUPPRESSED_BOOTSTRAP_KEY
                ),
                {},
            )

            degraded_context = SimpleNamespace(
                state=state,
                function_call_id="",
                invocation_id="inv-duplicate-bootstrap-degraded",
            )
            degraded_response = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="get_schema"), {}, degraded_context
            )
            self.assertEqual(degraded_response["status"], "already_stored")
            self.assertEqual(
                await grounding_callbacks.after_tool_callback(
                    SimpleNamespace(name="get_schema"),
                    {},
                    degraded_context,
                    degraded_response,
                ),
                degraded_response,
            )

        baseline_before.assert_not_awaited()
        baseline_after.assert_not_awaited()
        self.assertEqual(state["budget_remaining"], original_budget)
        self.assertEqual(state["tool_trajectory"], original_trajectory)
        self.assertEqual(
            state.get(grounding_callbacks.GROUNDING_SUPPRESSED_BOOTSTRAP_KEY),
            {},
        )
        self.assert_bootstrap_once(state)

    def test_stage4a_contract_and_hashes_are_frozen(self):
        self.assertEqual(DEFAULT_GROUNDING_MAX_CALLS_PER_TASK, 32)
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)

    def test_stage4a_does_not_add_a_fifth_dimension_or_evidence_store(self):
        state = primary_state().model_dump(mode="json")
        self.assertEqual(
            set(state),
            {"tables", "join_keys", "column_mapping", "domain_knowledge"},
        )
        runtime = GroundingRuntime().model_dump(mode="json")
        self.assertEqual(
            set(runtime),
            {"grounding_revision", "stage", "focus_dimension", "grounding_state"},
        )
        self.assertNotIn("evidence", runtime)
        self.assertNotIn("evidence_store", runtime)
        self.assertEqual(len(bootstrap_trajectory()), 3)


if __name__ == "__main__":
    unittest.main()
