from __future__ import annotations

import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingRuntime,
    SQLGroundingState,
    UserClarificationRecord,
)
from valibra_agent.sql_grounding.prompt_view import render_grounding_view


QUERY = "What is the average CO2 impact for shipments that are very late?"
FOLLOW_UP = "Order the same result by average CO2 impact descending."


def frozen_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("reviewsandimprovements", "shipments"),
        join_keys=(
            "shipments.reckey = reviewsandimprovements.reckeyrev",
        ),
        column_mapping=(
            ColumnMapping(
                phrase="CO2 impact",
                targets=("reviewsandimprovements.carbonkg",),
            ),
            ColumnMapping(
                phrase="very late",
                targets=(
                    "shipments.shipment_overview -> 'timing_performance' ->> 'actual_duration_hrs'",
                    "shipments.shipment_overview -> 'timing_performance' ->> 'planned_eta_hrs'",
                ),
            ),
        ),
        domain_knowledge=(),
    )


def clarification(answer: str = "The difference between actual duration and planned time exceeds 24 hours.") -> UserClarificationRecord:
    return UserClarificationRecord(
        phase=1,
        phrase="very late",
        kind="missing_knowledge",
        question="What rule defines very late?",
        answer=answer,
    )


def session_state(*, revision: int = 3, phase: int = 1) -> dict:
    runtime = GroundingRuntime(
        grounding_revision=revision,
        stage=("SQL_ATTEMPT" if phase == 1 else "P2_INCREMENTAL"),
        focus_dimension="none",
        grounding_state=frozen_state(),
    )
    return {
        "task_id": "answer-contract-shadow-test",
        "current_phase": phase,
        "phase1_completed": phase == 2,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 9.0,
        "initial_budget": 18.0,
        "tool_trajectory": [
            {
                "type": "tool",
                "tool": "get_all_knowledge_definitions",
                "phase": 1,
                "args": {},
                "result": json.dumps([]),
            }
        ],
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY: [
            clarification().model_dump(mode="json")
        ],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
    }


def grounding_input(*, phase: int = 1) -> dict:
    result = {
        "query": QUERY,
        "current_state": frozen_state().model_dump(mode="json"),
    }
    if phase == 2:
        result["follow_up"] = FOLLOW_UP
    return result


def writer_request() -> LlmRequest:
    names = ["execute_sql", "get_schema", "ask_user", "submit_sql"]
    request = LlmRequest(
        config=types.GenerateContentConfig(
            system_instruction="BASE",
            temperature=0.0,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name=name) for name in names
                    ]
                )
            ],
        )
    )
    request.tools_dict = {name: SimpleNamespace() for name in names}
    return request


class AnswerContractLiteShadowTests(unittest.TestCase):
    def test_feature_flag_is_exact_and_default_off(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(grounding_callbacks._answer_contract_shadow_enabled())
        for disabled in ("", "0", "true", "TRUE", "yes"):
            with patch.dict(
                os.environ,
                {"VALIBRA_ANSWER_CONTRACT_SHADOW": disabled},
                clear=True,
            ):
                self.assertFalse(grounding_callbacks._answer_contract_shadow_enabled())
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
            clear=True,
        ):
            self.assertTrue(grounding_callbacks._answer_contract_shadow_enabled())

    def test_complete_snapshot_emits_verified_predicate_and_aggregation(self) -> None:
        state = session_state()
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
            clear=True,
        ):
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=runtime,
                phase=1,
                grounding_input=grounding_input(),
            )
        audits = state[grounding_callbacks.ANSWER_CONTRACT_SHADOW_AUDITS_KEY]
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["status"], "EMITTED")
        self.assertEqual(audits[0]["lifecycle_status"], "ACTIVE")
        self.assertEqual(audits[0]["verifier"], "VERIFIED")
        self.assertEqual(audits[0]["invariant_count"], 2)
        self.assertEqual(audits[0]["invariant_kinds"], ["aggregation", "predicate"])

    def test_runtime_adapter_preserves_fail_closed_omission(self) -> None:
        state = session_state()
        state[grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY] = [
            UserClarificationRecord(
                phase=1,
                phrase="mostly",
                kind="user_intent",
                question="What does mostly mean?",
                answer="More than 50 percent.",
            ).model_dump(mode="json")
        ]
        omitted_state = SQLGroundingState(
            tables=("cases",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="Australian cases",
                    targets=("cases.visacls",),
                ),
            ),
            domain_knowledge=(),
        )
        runtime = GroundingRuntime(
            grounding_revision=3,
            stage="SQL_ATTEMPT",
            focus_dimension="none",
            grounding_state=omitted_state,
        )
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
            clear=True,
        ):
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=runtime,
                phase=1,
                grounding_input={
                    "query": "How many attorneys have mostly Australian cases?",
                    "current_state": omitted_state.model_dump(mode="json"),
                },
            )
        audit = state[grounding_callbacks.ANSWER_CONTRACT_SHADOW_AUDITS_KEY][0]
        self.assertEqual(audit["status"], "OMITTED")
        self.assertEqual(audit["invariant_count"], 0)

    def test_clarification_and_revision_changes_stale_old_contract(self) -> None:
        state = session_state()
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
            clear=True,
        ):
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=runtime,
                phase=1,
                grounding_input=grounding_input(),
            )
            grounding_callbacks._store_clarification_records(
                state,
                (clarification("The difference exceeds 48 hours."),),
            )
            changed_runtime = runtime.model_copy(
                update={"grounding_revision": runtime.grounding_revision + 1}
            )
            grounding_callbacks._refresh_answer_contract_shadow_lifecycle(
                state,
                changed_runtime,
            )
        audit = state[grounding_callbacks.ANSWER_CONTRACT_SHADOW_AUDITS_KEY][0]
        self.assertEqual(audit["lifecycle_status"], "STALE")
        self.assertIn("GROUNDING_REVISION_CHANGED", audit["stale_reasons"])
        self.assertIn("CLARIFICATION_DIGEST_CHANGED", audit["stale_reasons"])

    def test_restart_rebuilds_from_new_revision_only(self) -> None:
        state = session_state()
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
            clear=True,
        ):
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=runtime,
                phase=1,
                grounding_input=grounding_input(),
            )
            grounding_callbacks._store_clarification_records(
                state,
                (clarification("The difference exceeds 48 hours."),),
            )
            restarted = runtime.model_copy(
                update={"grounding_revision": 4}
            )
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=restarted,
                phase=1,
                grounding_input=grounding_input(),
            )
        audits = state[grounding_callbacks.ANSWER_CONTRACT_SHADOW_AUDITS_KEY]
        self.assertEqual(len(audits), 2)
        self.assertEqual(audits[0]["lifecycle_status"], "STALE")
        self.assertEqual(audits[1]["lifecycle_status"], "ACTIVE")
        self.assertEqual(audits[1]["grounding_revision"], 4)
        self.assertNotEqual(audits[0]["contract_sha256"], audits[1]["contract_sha256"])

    def test_p2_stales_p1_and_builds_again_without_provider(self) -> None:
        state = session_state()
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
            clear=True,
        ):
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=runtime,
                phase=1,
                grounding_input=grounding_input(),
            )
            state["current_phase"] = 2
            p2_runtime = runtime.model_copy(
                update={"grounding_revision": 4, "stage": "P2_INCREMENTAL"}
            )
            grounding_callbacks._refresh_answer_contract_shadow_lifecycle(
                state,
                p2_runtime,
            )
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=p2_runtime,
                phase=2,
                grounding_input=grounding_input(phase=2),
            )
        audits = state[grounding_callbacks.ANSWER_CONTRACT_SHADOW_AUDITS_KEY]
        self.assertEqual(len(audits), 2)
        self.assertEqual(audits[0]["lifecycle_status"], "STALE")
        self.assertIn("PHASE_CHANGED", audits[0]["stale_reasons"])
        self.assertEqual(audits[1]["phase"], 2)
        self.assertEqual(audits[1]["grounding_revision"], 4)
        self.assertEqual(audits[1]["lifecycle_status"], "ACTIVE")

    def test_builder_exception_is_logged_and_never_raised(self) -> None:
        state = session_state()
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        with (
            patch.dict(
                os.environ,
                {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
                clear=True,
            ),
            patch.object(
                grounding_callbacks.AnswerContractLiteBuilder,
                "build",
                side_effect=RuntimeError("synthetic shadow failure"),
            ),
        ):
            grounding_callbacks._emit_answer_contract_shadow(
                state,
                runtime=runtime,
                phase=1,
                grounding_input=grounding_input(),
            )
        audit = state[grounding_callbacks.ANSWER_CONTRACT_SHADOW_AUDITS_KEY][0]
        self.assertEqual(audit["status"], "SHADOW_ERROR")
        self.assertEqual(audit["error_type"], "RuntimeError")
        self.assertEqual(runtime.stage, "SQL_ATTEMPT")

    def test_shadow_on_off_main_request_digest_is_identical(self) -> None:
        off_state = session_state()
        on_state = copy.deepcopy(off_state)
        runtime = GroundingRuntime.model_validate(
            off_state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_SHADOW": "1"},
            clear=True,
        ):
            grounding_callbacks._emit_answer_contract_shadow(
                on_state,
                runtime=runtime,
                phase=1,
                grounding_input=grounding_input(),
            )
        off_request = writer_request()
        on_request = writer_request()
        off_view = render_grounding_view(
            runtime.grounding_state,
            clarifications=grounding_callbacks._clarification_records(off_state),
        )
        on_view = render_grounding_view(
            runtime.grounding_state,
            clarifications=grounding_callbacks._clarification_records(on_state),
        )
        self.assertEqual(off_view.text, on_view.text)
        for request, view in ((off_request, off_view), (on_request, on_view)):
            grounding_callbacks._inject_sql_writer_context(
                request,
                phase=1,
                original_query=QUERY,
                follow_up=None,
                view_text=view.text,
                budget_remaining=9,
            )
        self.assertEqual(
            grounding_callbacks._request_sha256(off_request),
            grounding_callbacks._request_sha256(on_request),
        )


if __name__ == "__main__":
    unittest.main()
