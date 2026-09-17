from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from valibra_agent import grounding_callbacks
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
)


QUERY = "How many bot accounts are dormant?"
FOLLOW_UP = "What percentage of all accounts are dormant bots?"
ARGS = {"table_name": "accounts", "column_name": "acct_form"}
RESULT = "TEXT. Account form or type. Possible values: Bot, Business, Hybrid, Personal."
SCHEMA = """CREATE TABLE accounts (
  account_id INTEGER PRIMARY KEY,
  acct_form TEXT
);
First 3 rows:
account_id | acct_form
1 | Bot
...
"""


def _request_digest(arguments: dict[str, str] = ARGS) -> str:
    arguments_json = grounding_callbacks.canonical_json(arguments)
    return grounding_callbacks._sha256_text(
        f"get_column_meaning:{arguments_json}"
    )


def _state(
    *,
    task_id: str = "fake_account_23",
    budget: float = 3.0,
    result: str = RESULT,
) -> dict[str, object]:
    runtime = GroundingRuntime(
        grounding_revision=5,
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
            ),
            domain_knowledge=(),
        ),
    )
    return {
        "task_id": task_id,
        "selected_database": "fake_account",
        "current_phase": 2,
        "initial_budget": 14.0,
        "budget_remaining": budget,
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
                "args": dict(ARGS),
                "result": result,
            },
            {
                "type": "tool",
                "tool": "submit_sql",
                "phase": 1,
                "args": {"sql": "SELECT COUNT(*) FROM accounts"},
                "result": (
                    f"Follow-up question: {FOLLOW_UP}\n"
                    "Budget remaining: 3 bird-coins"
                ),
            },
        ],
        grounding_callbacks.GROUNDING_CHECK_AUDITS_KEY: [
            {
                "phase": 1,
                "status": "incomplete",
                "missing_information": "Need the bot account category.",
                "tool_name": "get_column_meaning",
                "request_digest": _request_digest(),
                "budget_before": 11.0,
                "tool_cost": 0.5,
                "budget_after": 10.5,
                "blocked_reason": None,
            }
        ],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 3,
        grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
            "1": 0,
            "2": 3,
        },
        "system_agent_llm_calls": [],
    }


def _incomplete(
    *,
    arguments: dict[str, str] = ARGS,
    tool_name: str = "get_column_meaning",
) -> GroundingCheckResponse:
    return GroundingCheckResponse(
        status="incomplete",
        clarification_route="none",
        missing_information="Need exact Official category evidence.",
        next_tool=GroundingCheckToolRequest(
            tool_name=tool_name,
            arguments=arguments,
            user_clarification_request=None,
        ),
        column_mapping=(
            ColumnMapping(
                phrase="bot accounts",
                targets=("accounts.acct_form",),
            ),
        ),
        domain_knowledge=(),
    )


def _complete() -> GroundingCheckResponse:
    return GroundingCheckResponse(
        status="complete",
        clarification_route="none",
        missing_information=None,
        next_tool=None,
        column_mapping=(
            ColumnMapping(
                phrase="bot accounts",
                targets=("accounts.acct_form",),
            ),
        ),
        domain_knowledge=(),
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
        self.inputs.append(dict(kwargs["grounding_input"]))
        return GroundingUpdaterResult(
            response=self.responses.pop(0),
            call_kind="check",
            telemetry=_telemetry(),
            transport_normalization="none",
        )


class P2ExactOfficialEvidenceReuseR1Tests(unittest.IsolatedAsyncioTestCase):
    def _resolve(
        self,
        state: dict[str, object],
        response: GroundingCheckResponse | None = None,
        *,
        phase: int = 2,
        allow_repeated_gap: bool = False,
    ) -> object:
        with patch.dict(
            os.environ,
            {"VALIBRA_P2_EXACT_OFFICIAL_EVIDENCE_REUSE": "1"},
        ):
            return grounding_callbacks._resolve_p2_exact_official_evidence_replay(
                state,
                phase=phase,
                response=response or _incomplete(),
                allow_repeated_gap=allow_repeated_gap,
            )

    def test_exact_same_task_p1_result_is_eligible(self) -> None:
        state = _state()
        before = state["budget_remaining"]
        replay = self._resolve(state)

        self.assertIsNotNone(replay)
        assert replay is not None
        self.assertEqual(replay.task_id, "fake_account_23")
        self.assertEqual(replay.source_phase, 1)
        self.assertEqual(replay.target_phase, 2)
        self.assertEqual(replay.arguments, ARGS)
        self.assertEqual(replay.result, RESULT)
        self.assertEqual(state["budget_remaining"], before)

    def test_different_arguments_are_a_cache_miss(self) -> None:
        response = _incomplete(
            arguments={"table_name": "accounts", "column_name": "StateFlag"}
        )
        self.assertIsNone(self._resolve(_state(), response))

    def test_cache_is_state_local_and_cannot_cross_tasks(self) -> None:
        self.assertIsNotNone(self._resolve(_state(task_id="task-a")))
        other = _state(task_id="task-b")
        other["tool_trajectory"] = []
        self.assertIsNone(self._resolve(other))

    def test_mutable_or_non_column_tool_is_not_eligible(self) -> None:
        response = _incomplete(
            arguments={"sql": "SELECT 1"},
            tool_name="execute_sql",
        )
        self.assertIsNone(self._resolve(_state(), response))

    def test_atomic_or_p1_check_is_not_eligible(self) -> None:
        self.assertIsNone(self._resolve(_state(), allow_repeated_gap=True))
        self.assertIsNone(self._resolve(_state(), phase=1))

    def test_conflicting_historical_results_fail_closed(self) -> None:
        state = _state()
        state["tool_trajectory"] = list(state["tool_trajectory"]) + [
            {
                "type": "tool",
                "tool": "get_column_meaning",
                "phase": 1,
                "args": dict(ARGS),
                "result": "A conflicting result.",
            }
        ]
        self.assertIsNone(self._resolve(state))

    def test_cache_miss_preserves_the_six_coin_floor(self) -> None:
        state = _state(budget=6.4)
        response = _incomplete(
            arguments={"table_name": "accounts", "column_name": "StateFlag"}
        )
        self.assertIsNone(self._resolve(state, response))
        with self.assertRaisesRegex(ValueError, "budget_exhausted"):
            grounding_callbacks._schedule_check_tool(
                state,
                phase=2,
                response=response,
            )

    async def test_replay_continues_p2_check_without_tool_or_coin(self) -> None:
        state = _state()
        runtime_before = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        trajectory_before = list(state["tool_trajectory"])
        updater = _SequenceUpdater(_incomplete(), _complete())
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
            runtime=runtime_before,
            phase=2,
            initial=True,
        )
        token = grounding_callbacks._bind_turn_message(
            "fake_account_23",
            "a-interact",
            QUERY,
        )
        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "GROUNDING_UPDATER_MODE": "llm",
                        "VALIBRA_FINAL_REGROUNDING_GATE": "0",
                        "VALIBRA_P2_EXACT_OFFICIAL_EVIDENCE_REUSE": "1",
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
                    runtime_before,
                    grounding_input=check_input,
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(len(updater.inputs), 2)
        self.assertEqual(state["budget_remaining"], 3.0)
        self.assertEqual(state["tool_trajectory"], trajectory_before)
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))
        latest_input = updater.inputs[1]
        self.assertEqual(latest_input["latest_tool"]["result"], RESULT)
        self.assertEqual(
            latest_input["previous_official_calls"],
            [
                {
                    "tool_name": "get_column_meaning",
                    "arguments": ARGS,
                    "request_digest": _request_digest(),
                }
            ],
        )
        replay_audits = [
            item
            for item in grounding_callbacks._check_audits(state)
            if item.get("evidence_replay")
            == "p2_exact_p1_get_column_meaning"
        ]
        self.assertEqual(len(replay_audits), 1)
        self.assertEqual(replay_audits[0]["tool_cost"], 0.0)
        self.assertEqual(replay_audits[0]["budget_before"], 3.0)
        self.assertEqual(replay_audits[0]["budget_after"], 3.0)
        self.assertEqual(result.service_status, "noop")
        self.assertTrue(grounding_callbacks._phase_grounding_succeeded(state, 2))


if __name__ == "__main__":
    unittest.main()
