from __future__ import annotations

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding import updater as grounding_updater
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingCheckResponse,
    GroundingCheckToolRequest,
    GroundingRuntime,
    SQLGroundingState,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_STAGE_PROMPT_SHA256,
    GroundingUpdaterResult,
    classify_grounding_input,
)


QUERY = "How many bot accounts are dormant?"
FOLLOW_UP = "What percentage of all accounts are dormant bots?"
ACCT_ARGS = {"table_name": "accounts", "column_name": "acct_form"}
STATE_ARGS = {"table_name": "accounts", "column_name": "StateFlag"}
ACCT_RESULT = (
    "TEXT. Account form or type. Possible values: Bot, Business, Hybrid, Personal."
)
STATE_RESULT = (
    "TEXT. Current state of the account. Possible values: Active, Deleted, "
    "Dormant, Suspended."
)
SCHEMA = """CREATE TABLE accounts (
  account_id INTEGER PRIMARY KEY,
  acct_form TEXT,
  StateFlag TEXT
);
First 3 rows:
account_id | acct_form | StateFlag
1 | Bot | Dormant
...
"""


def _digest(arguments: dict[str, str]) -> str:
    return grounding_callbacks._sha256_text(
        "get_column_meaning:"
        + grounding_callbacks.canonical_json(arguments)
    )


def _runtime() -> GroundingRuntime:
    return GroundingRuntime(
        grounding_revision=4,
        stage="P2_INCREMENTAL",
        focus_dimension="none",
        grounding_state=SQLGroundingState(
            tables=("accounts",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="bot accounts",
                    targets=("accounts.acct_form",),
                ),
                ColumnMapping(
                    phrase="dormant",
                    targets=("accounts.StateFlag",),
                ),
            ),
            domain_knowledge=(),
        ),
    )


def _p1_check_audit(arguments: dict[str, str]) -> dict[str, object]:
    return {
        "phase": 1,
        "status": "incomplete",
        "missing_information": "Need exact category evidence.",
        "tool_name": "get_column_meaning",
        "request_digest": _digest(arguments),
        "budget_before": 11.0,
        "tool_cost": 0.5,
        "budget_after": 10.5,
        "blocked_reason": None,
    }


def _state() -> dict[str, object]:
    runtime = _runtime()
    return {
        "task_id": "fake_account_23",
        "selected_database": "fake_account",
        "current_phase": 2,
        "initial_budget": 14.0,
        "budget_remaining": 5.0,
        grounding_callbacks.GROUNDING_SEQUENCE_KEY: 9,
        "tool_trajectory": [
            {
                "type": "tool",
                "tool": "get_schema",
                "phase": 1,
                "args": {},
                "result": SCHEMA,
            },
            {
                "type": "tool",
                "tool": "get_column_meaning",
                "phase": 1,
                "args": dict(ACCT_ARGS),
                "result": ACCT_RESULT,
            },
            {
                "type": "tool",
                "tool": "get_column_meaning",
                "phase": 1,
                "args": dict(STATE_ARGS),
                "result": STATE_RESULT,
            },
            {
                "type": "tool",
                "tool": "submit_sql",
                "phase": 1,
                "args": {"sql": "SELECT COUNT(*) FROM accounts"},
                "result": (
                    f"Follow-up question: {FOLLOW_UP}\n"
                    "Budget remaining: 5 bird-coins"
                ),
            },
        ],
        grounding_callbacks.GROUNDING_CHECK_AUDITS_KEY: [
            _p1_check_audit(ACCT_ARGS),
            _p1_check_audit(STATE_ARGS),
        ],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 3,
        grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
            "1": 0,
            "2": 3,
        },
        "system_agent_llm_calls": [],
    }


def _incomplete(arguments: dict[str, str]) -> GroundingCheckResponse:
    return GroundingCheckResponse(
        status="incomplete",
        clarification_route="none",
        missing_information=f"Need {arguments['column_name']} category evidence.",
        next_tool=GroundingCheckToolRequest(
            tool_name="get_column_meaning",
            arguments=arguments,
            user_clarification_request=None,
        ),
        column_mapping=_runtime().grounding_state.column_mapping or (),
        domain_knowledge=_runtime().grounding_state.domain_knowledge or (),
    )


def _complete() -> GroundingCheckResponse:
    return GroundingCheckResponse(
        status="complete",
        clarification_route="none",
        missing_information=None,
        next_tool=None,
        column_mapping=_runtime().grounding_state.column_mapping or (),
        domain_knowledge=_runtime().grounding_state.domain_knowledge or (),
    )


def _telemetry() -> GroundingLLMTelemetry:
    return GroundingLLMTelemetry(
        attempted=True,
        status="succeeded",
        request_sha256="",
        response_sha256="",
        prompt_sha256=SQL_GROUNDING_STAGE_PROMPT_SHA256["check"],
        form_schema_sha256=SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256["check"],
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


class _SequenceUpdater:
    def __init__(self, *responses: GroundingCheckResponse) -> None:
        self.responses = list(responses)
        self.inputs: list[dict[str, object]] = []

    async def propose(self, *args: object, **kwargs: object) -> GroundingUpdaterResult:
        del args
        self.inputs.append(copy.deepcopy(dict(kwargs["grounding_input"])))
        return GroundingUpdaterResult(
            response=self.responses.pop(0),
            call_kind="check",
            telemetry=_telemetry(),
            transport_normalization="none",
        )


class P2CheckCumulativeOfficialEvidenceR1Tests(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {
                "VALIBRA_P2_EXACT_OFFICIAL_EVIDENCE_REUSE": "1",
                "VALIBRA_P2_CHECK_CUMULATIVE_OFFICIAL_EVIDENCE": "1",
            },
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def _start(self, state: dict[str, object]) -> None:
        grounding_callbacks._start_p2_check_cumulative_evidence(
            state,
            runtime=_runtime(),
            query=QUERY,
            follow_up=FOLLOW_UP,
        )

    def test_feature_disabled_preserves_the_existing_check_contract(self) -> None:
        state = _state()
        with patch.dict(
            os.environ,
            {"VALIBRA_P2_CHECK_CUMULATIVE_OFFICIAL_EVIDENCE": "0"},
        ):
            self._start(state)
            self.assertNotIn(
                grounding_callbacks.P2_CHECK_CUMULATIVE_EVIDENCE_KEY,
                state,
            )
            request = grounding_callbacks._build_check_grounding_request(
                state,
                query=QUERY,
                follow_up=FOLLOW_UP,
                runtime=_runtime(),
                phase=2,
                initial=True,
            )
        self.assertNotIn("check_evidence_context", request)

    def test_cumulative_context_keeps_prior_result_without_latest_duplication(
        self,
    ) -> None:
        state = _state()
        self._start(state)
        balance = state["budget_remaining"]
        self.assertTrue(
            grounding_callbacks._append_p2_check_cumulative_evidence(
                state,
                tool_name="get_column_meaning",
                arguments=ACCT_ARGS,
                result=ACCT_RESULT,
                source="p2_exact_p1_replay",
            )
        )
        first = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            follow_up=FOLLOW_UP,
            runtime=_runtime(),
            phase=2,
            latest_tool_name="get_column_meaning",
            latest_tool_arguments=ACCT_ARGS,
            latest_tool_result=ACCT_RESULT,
        )
        self.assertEqual(first["check_evidence_context"], [])

        self.assertTrue(
            grounding_callbacks._append_p2_check_cumulative_evidence(
                state,
                tool_name="get_column_meaning",
                arguments=STATE_ARGS,
                result=STATE_RESULT,
                source="p2_exact_p1_replay",
            )
        )
        second = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            follow_up=FOLLOW_UP,
            runtime=_runtime(),
            phase=2,
            latest_tool_name="get_column_meaning",
            latest_tool_arguments=STATE_ARGS,
            latest_tool_result=STATE_RESULT,
        )
        self.assertEqual(len(second["check_evidence_context"]), 1)
        self.assertEqual(
            second["check_evidence_context"][0]["arguments"], ACCT_ARGS
        )
        self.assertEqual(
            second["check_evidence_context"][0]["result"], ACCT_RESULT
        )
        self.assertEqual(second["latest_tool"]["result"], STATE_RESULT)
        self.assertEqual(classify_grounding_input(second, phase=2), "check")
        observation = build_sql_grounding_observation(
            task_id="fake_account_23",
            phase=2,
            sequence=10,
            observation_type="metadata",
            content=STATE_RESULT,
            summary="P2 column evidence",
            tool_name="get_column_meaning",
            function_call_id="valibra-check-2-test",
        )
        validated = grounding_updater._validated_bundled_grounding_input(
            second,
            runtime=_runtime(),
            original_query=QUERY,
            follow_up_query=FOLLOW_UP,
            observation=observation,
        )
        self.assertEqual(
            validated["check_evidence_context"],
            second["check_evidence_context"],
        )
        self.assertEqual(state["budget_remaining"], balance)

        invalid = copy.deepcopy(second)
        invalid["check_evidence_context"][0]["result_sha256"] = "0" * 64
        with self.assertRaises(grounding_updater.GroundingUpdaterError):
            grounding_updater._validated_bundled_grounding_input(
                invalid,
                runtime=_runtime(),
                original_query=QUERY,
                follow_up_query=FOLLOW_UP,
                observation=observation,
            )

    def test_same_digest_is_deduplicated_and_conflict_fails_closed(self) -> None:
        state = _state()
        self._start(state)
        self.assertTrue(
            grounding_callbacks._append_p2_check_cumulative_evidence(
                state,
                tool_name="get_column_meaning",
                arguments=ACCT_ARGS,
                result=ACCT_RESULT,
                source="official_call",
            )
        )
        self.assertFalse(
            grounding_callbacks._append_p2_check_cumulative_evidence(
                state,
                tool_name="get_column_meaning",
                arguments=ACCT_ARGS,
                result=ACCT_RESULT,
                source="p2_exact_p1_replay",
            )
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            grounding_callbacks._append_p2_check_cumulative_evidence(
                state,
                tool_name="get_column_meaning",
                arguments=ACCT_ARGS,
                result="A different result.",
                source="official_call",
            )

    def test_new_cycle_replaces_old_context_and_task_mismatch_fails_closed(
        self,
    ) -> None:
        state = _state()
        self._start(state)
        grounding_callbacks._append_p2_check_cumulative_evidence(
            state,
            tool_name="get_column_meaning",
            arguments=ACCT_ARGS,
            result=ACCT_RESULT,
            source="official_call",
        )
        state[grounding_callbacks.GROUNDING_SEQUENCE_KEY] = 10
        self._start(state)
        payload = grounding_callbacks._project_p2_check_cumulative_evidence(
            state,
            query=QUERY,
            follow_up=FOLLOW_UP,
        )
        self.assertEqual(payload, [])
        state["task_id"] = "another_task"
        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            grounding_callbacks._project_p2_check_cumulative_evidence(
                state,
                query=QUERY,
                follow_up=FOLLOW_UP,
            )

    def test_p1_and_atomic_requests_never_receive_the_carrier(self) -> None:
        state = _state()
        self._start(state)
        grounding_callbacks._append_p2_check_cumulative_evidence(
            state,
            tool_name="get_column_meaning",
            arguments=ACCT_ARGS,
            result=ACCT_RESULT,
            source="official_call",
        )
        p1 = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            runtime=_runtime(),
            phase=1,
            initial=True,
        )
        atomic = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            follow_up=FOLLOW_UP,
            runtime=_runtime(),
            phase=2,
            initial=True,
            include_p2_cumulative_evidence=False,
        )
        self.assertNotIn("check_evidence_context", p1)
        self.assertNotIn("check_evidence_context", atomic)

    def test_entry_and_total_caps_are_fail_open_for_the_latest_result(self) -> None:
        state = _state()
        self._start(state)
        for index in range(5):
            added = grounding_callbacks._append_p2_check_cumulative_evidence(
                state,
                tool_name="get_column_meaning",
                arguments={
                    "table_name": "accounts",
                    "column_name": f"field_{index}",
                },
                result=f"Meaning {index}",
                source="official_call",
            )
            self.assertEqual(added, index < 4)
        record = grounding_callbacks._p2_check_cumulative_evidence(state)
        assert record is not None
        self.assertEqual(len(record.entries), 4)
        self.assertFalse(
            grounding_callbacks._append_p2_check_cumulative_evidence(
                state,
                tool_name="get_column_meaning",
                arguments=ACCT_ARGS,
                result="x" * 2_049,
                source="official_call",
            )
        )

    def test_phase_outcome_and_terminal_failure_destroy_the_carrier(self) -> None:
        state = _state()
        self._start(state)
        grounding_callbacks._record_phase_grounding_outcome(
            state,
            observation=build_sql_grounding_observation(
                task_id="fake_account_23",
                phase=2,
                sequence=10,
                observation_type="p2_follow_up",
                content=FOLLOW_UP,
                summary="complete P2 Check",
            ),
            runtime=_runtime(),
            status="succeeded",
            provider_attempted=False,
        )
        self.assertIsNone(
            state[grounding_callbacks.P2_CHECK_CUMULATIVE_EVIDENCE_KEY]
        )

        terminal_state = _state()
        self._start(terminal_state)
        grounding_callbacks._failed_phase_grounding_result(
            terminal_state,
            runtime=_runtime(),
            observation=build_sql_grounding_observation(
                task_id="fake_account_23",
                phase=2,
                sequence=11,
                observation_type="p2_follow_up",
                content=FOLLOW_UP,
                summary="terminal P2 Check",
            ),
            service_status="terminal_incomplete",
            error_type="CheckTerminalIncomplete",
            provider_attempted=False,
        )
        self.assertIsNone(
            terminal_state[grounding_callbacks.P2_CHECK_CUMULATIVE_EVIDENCE_KEY]
        )

    async def test_two_exact_replays_accumulate_for_the_third_check(self) -> None:
        state = _state()
        runtime = _runtime()
        self._start(state)
        updater = _SequenceUpdater(
            _incomplete(ACCT_ARGS),
            _incomplete(STATE_ARGS),
            _complete(),
        )
        observation = build_sql_grounding_observation(
            task_id="fake_account_23",
            phase=2,
            sequence=10,
            observation_type="p2_follow_up",
            content=FOLLOW_UP,
            summary="initial Phase-2 unified Grounding Check",
        )
        check_input = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            follow_up=FOLLOW_UP,
            runtime=runtime,
            phase=2,
            initial=True,
        )
        token = grounding_callbacks._bind_turn_message(
            "fake_account_23", "a-interact", QUERY
        )
        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "GROUNDING_UPDATER_MODE": "llm",
                        "VALIBRA_FINAL_REGROUNDING_GATE": "0",
                    },
                ),
                patch.object(
                    grounding_callbacks,
                    "_SQL_GROUNDING_UPDATER",
                    updater,
                ),
                patch.object(
                    grounding_callbacks,
                    "load_sql_grounding_llm_config",
                    return_value=SimpleNamespace(max_calls_per_task=32),
                ),
            ):
                result = await grounding_callbacks._handle_observation(
                    state,
                    observation,
                    runtime,
                    grounding_input=check_input,
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(len(updater.inputs), 3)
        final_input = updater.inputs[2]
        self.assertEqual(final_input["latest_tool"]["result"], STATE_RESULT)
        self.assertEqual(len(final_input["check_evidence_context"]), 1)
        self.assertEqual(
            final_input["check_evidence_context"][0]["result"], ACCT_RESULT
        )
        self.assertEqual(state["budget_remaining"], 5.0)
        self.assertEqual(result.service_status, "noop")
        self.assertIsNone(
            state[grounding_callbacks.P2_CHECK_CUMULATIVE_EVIDENCE_KEY]
        )


if __name__ == "__main__":
    unittest.main()
