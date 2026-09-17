from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    GroundingRuntime,
    SQLGroundingState,
    canonical_json,
)


QUERY = "Show the average delay for qualifying shipments."
OLD_SQL = "SELECT AVG(delay_hours) FROM shipments WHERE delay_hours > 24"
NEW_SQL = "SELECT AVG(actual_hours - planned_hours) FROM shipments WHERE actual_hours - planned_hours > 24"
GENERIC_FAIL = (
    "SQL failed Phase 1. Your SQL is not correct.\n"
    "Budget remaining: 4 bird-coins"
)


def ready_state() -> dict:
    grounding_state = SQLGroundingState(
        tables=("shipments",),
        join_keys=(),
        column_mapping=(),
        domain_knowledge=(),
    )
    runtime = GroundingRuntime(
        grounding_revision=3,
        stage="SQL_ATTEMPT",
        focus_dimension="none",
        grounding_state=grounding_state,
    )
    return {
        "task_id": "main-fresh-recompile-r1",
        "current_phase": 1,
        "budget_remaining": 4.0,
        "tool_trajectory": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY: [],
    }


def pending(sql: str, *, call_id: str = "submit-old"):
    args_json = canonical_json({"sql": sql})
    return grounding_callbacks._PendingToolCall(
        function_call_id=call_id,
        tool_name="submit_sql",
        phase_before=1,
        args_digest=grounding_callbacks._sha256_text(args_json),
        args_summary=args_json,
        sequence=1,
    )


def request_with_history(*, post_failure_sql: str | None = None) -> LlmRequest:
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=QUERY)]),
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="execute-old", name="execute_sql", args={"sql": OLD_SQL}
                    )
                )
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="execute-old",
                        name="execute_sql",
                        response={"result": "avg\n---\n31"},
                    )
                )
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="submit-old", name="submit_sql", args={"sql": OLD_SQL}
                    )
                )
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="submit-old",
                        name="submit_sql",
                        response={"result": GENERIC_FAIL},
                    )
                )
            ],
        ),
    ]
    if post_failure_sql is not None:
        contents.extend(
            [
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="execute-fresh",
                                name="execute_sql",
                                args={"sql": post_failure_sql},
                            )
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="execute-fresh",
                                name="execute_sql",
                                response={"result": "SQL Error: bad cast"},
                            )
                        )
                    ],
                ),
            ]
        )
    request = LlmRequest(
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name="execute_sql"),
                        types.FunctionDeclaration(name="submit_sql"),
                    ]
                )
            ]
        ),
    )
    request.tools_dict = {
        "execute_sql": SimpleNamespace(),
        "submit_sql": SimpleNamespace(),
    }
    return request


class MainHealthyHandoffFreshRecompileR1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {"VALIBRA_MAIN_FRESH_RECOMPILE_R1": "1"},
            clear=True,
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def arm(self, state: dict) -> None:
        grounding_callbacks._update_main_fresh_recompile_after_submit(
            state,
            phase=1,
            event="official_submit_failed",
            tool_response=GENERIC_FAIL,
            pending=pending(OLD_SQL),
        )

    def test_exact_generic_failure_arms_once_and_nonexact_does_not(self) -> None:
        state = ready_state()
        grounding_callbacks._update_main_fresh_recompile_after_submit(
            state,
            phase=1,
            event="official_submit_failed",
            tool_response="SQL Error: syntax error",
            pending=pending(OLD_SQL),
        )
        self.assertIsNone(
            grounding_callbacks._current_main_fresh_recompile_episode(state)
        )

        self.arm(state)
        episode = grounding_callbacks._current_main_fresh_recompile_episode(state)
        self.assertEqual(episode["status"], "ARMED")

        response = grounding_callbacks._observe_main_fresh_recompile_candidate(
            state,
            tool_name="submit_sql",
            args={"sql": NEW_SQL},
        )
        self.assertIsNone(response)
        grounding_callbacks._update_main_fresh_recompile_after_submit(
            state,
            phase=1,
            event="official_submit_failed",
            tool_response=GENERIC_FAIL,
            pending=pending(NEW_SQL, call_id="submit-fresh"),
        )
        self.assertIsNone(
            grounding_callbacks._current_main_fresh_recompile_episode(state)
        )
        self.assertEqual(
            state[grounding_callbacks.MAIN_FRESH_RECOMPILE_EPISODES_KEY]["1"][
                "status"
            ],
            "COMPLETED",
        )

    def test_fresh_context_hides_old_sql_and_retains_new_db_error(self) -> None:
        state = ready_state()
        self.arm(state)
        request = request_with_history(post_failure_sql=NEW_SQL)
        grounding_callbacks._inject_sql_writer_context(
            request,
            phase=1,
            original_query=QUERY,
            follow_up=None,
            view_text="[VALIBRA DATABASE GROUNDING]",
            budget_remaining=4,
            fresh_recompile=True,
        )
        serialized = json.dumps(
            [item.model_dump(mode="json") for item in request.contents],
            sort_keys=True,
        )
        self.assertIn(QUERY, serialized)
        self.assertNotIn(OLD_SQL, serialized)
        self.assertIn(NEW_SQL, serialized)
        self.assertIn("SQL Error: bad cast", serialized)
        self.assertIn(
            "FRESH RECOMPILE MODE",
            request.config.system_instruction,
        )

    def test_history_marker_accepts_only_the_deterministic_adk_budget_note(self) -> None:
        request = request_with_history()
        marker_part = request.contents[-1].parts[0]
        marker_part.function_response.response["result"] = (
            f"{GENERIC_FAIL}\n\n[SYSTEM NOTE: Remaining budget: 4.0/18.0]"
        )
        grounding_callbacks._filter_fresh_recompile_contents(request, phase=1)
        serialized = json.dumps(
            [item.model_dump(mode="json") for item in request.contents],
            sort_keys=True,
        )
        self.assertIn(QUERY, serialized)
        self.assertNotIn(OLD_SQL, serialized)

        request = request_with_history()
        request.contents[-1].parts[0].function_response.response["result"] = (
            f"{GENERIC_FAIL}\n\n[UNTRUSTED NOTE]"
        )
        with self.assertRaisesRegex(ValueError, "exact generic submit"):
            grounding_callbacks._filter_fresh_recompile_contents(request, phase=1)

    def test_exact_old_sql_duplicate_is_rejected_without_ending_normal_path(self) -> None:
        state = ready_state()
        self.arm(state)
        response = grounding_callbacks._observe_main_fresh_recompile_candidate(
            state,
            tool_name="execute_sql",
            args={"sql": OLD_SQL},
        )
        self.assertEqual(
            response["status"],
            grounding_callbacks._MAIN_FRESH_RECOMPILE_DUPLICATE_STATUS,
        )
        self.assertIsNone(
            grounding_callbacks._current_main_fresh_recompile_episode(state)
        )

    def test_feature_flag_is_exact_and_default_off(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                grounding_callbacks._main_fresh_recompile_r1_enabled()
            )
        with patch.dict(
            os.environ,
            {"VALIBRA_MAIN_FRESH_RECOMPILE_R1": "true"},
            clear=True,
        ):
            self.assertFalse(
                grounding_callbacks._main_fresh_recompile_r1_enabled()
            )


if __name__ == "__main__":
    unittest.main()
