from __future__ import annotations

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    GroundingCheckClarificationProposal,
    GroundingCheckResponse,
    GroundingCheckToolRequest,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    MappingGroundingResponse,
    StructureGroundingResponse,
    UserClarificationRecord,
    UserClarificationRequest,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
    classify_grounding_input,
)
from tests.valibra.test_stage3_submit_driven_repair import (
    QUERY,
    bootstrap_trajectory,
    primary_state,
)


class _FourStageUpdater:
    def __init__(self, *, answer_route: str = "restart_grounding") -> None:
        self.inputs: list[dict] = []
        self.answer_route = answer_route

    async def propose(self, runtime, observation, *args, **kwargs):
        del args
        grounding_input = copy.deepcopy(kwargs["grounding_input"])
        self.inputs.append(grounding_input)
        kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
        state = runtime.grounding_state
        if kind == "structure":
            response = StructureGroundingResponse(
                tables=state.tables or (),
                join_keys=state.join_keys or (),
            )
        elif kind == "mapping":
            response = MappingGroundingResponse(
                tables=state.tables or (),
                join_keys=state.join_keys or (),
                column_mapping=state.column_mapping or (),
                unresolved_mappings=(),
            )
        elif kind == "knowledge":
            response = KnowledgeGroundingResponse(
                column_mapping=state.column_mapping or (),
                selected_knowledge_ids=(1,),
            )
        else:
            route = (
                self.answer_route
                if "latest_user_answer" in grounding_input
                else "none"
            )
            response = GroundingCheckResponse(
                status=("incomplete" if route in {"restart_grounding", "terminal"} else "complete"),
                clarification_route=route,
                missing_information=(
                    "The clarification changes the requested business concept."
                    if route == "restart_grounding"
                    else "The clarification did not resolve the business question."
                    if route == "terminal"
                    else None
                ),
                next_tool=None,
                column_mapping=state.column_mapping or (),
                domain_knowledge=state.domain_knowledge or (),
            )
        return GroundingUpdaterResult(
            response=response,
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


def _state() -> dict:
    runtime = GroundingRuntime(
        grounding_revision=3,
        stage="INITIAL_GROUNDING",
        focus_dimension="none",
        grounding_state=primary_state(),
    )
    return {
        "task_id": "v3-clarification-regrounding",
        "current_phase": 1,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 7.0,
        "initial_budget": 10.0,
        "tool_trajectory": bootstrap_trajectory(),
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 4,
        grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
            "1": 4,
            "2": 0,
        },
    }


class ClarificationRegroundingV3Tests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_callback_dispatches_one_complete_regrounding_cycle(
        self,
    ) -> None:
        state = _state()
        state["budget_remaining"] = 100.0
        state["initial_budget"] = 100.0
        question = "Which business meaning of maintenance cost do you intend?"
        pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=GroundingCheckResponse(
                status="incomplete",
                clarification_route="none",
                missing_information="The intended maintenance-cost meaning is ambiguous.",
                next_tool=GroundingCheckToolRequest(
                    tool_name="ask_user",
                    arguments={"question": question},
                    user_clarification_request=(
                        GroundingCheckClarificationProposal(
                            phrase="maintenance cost",
                            kind="user_intent",
                        )
                    ),
                ),
                column_mapping=primary_state().column_mapping or (),
                domain_knowledge=primary_state().domain_knowledge or (),
            ),
        )
        tool = SimpleNamespace(name="ask_user")
        context = SimpleNamespace(
            state=state,
            function_call_id=pending.function_call_id,
            invocation_id="v3-regrounding-callback",
        )
        self.assertIsNone(
            await grounding_callbacks.before_tool_callback(
                tool,
                {"question": question},
                context,
            )
        )
        updater = _FourStageUpdater()
        token = grounding_callbacks._bind_turn_message(
            state["task_id"],
            "a-interact",
            QUERY,
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
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
                await grounding_callbacks.after_tool_callback(
                    tool,
                    {"question": question},
                    context,
                    "Use reported maintenance cost for each asset.",
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(len(updater.inputs), 5)
        self.assertIn("latest_user_answer", updater.inputs[0])
        self.assertEqual(
            [
                classify_grounding_input(item, phase=1)
                for item in updater.inputs[1:]
            ],
            ["structure", "mapping", "knowledge", "check"],
        )
        self.assertEqual(
            GroundingRuntime.model_validate(
                state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
            ).stage,
            "SQL_ATTEMPT",
        )
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))
        records = grounding_callbacks._clarification_records(state)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].answer,
            "Use reported maintenance cost for each asset.",
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 9, "2": 0},
        )

    async def test_narrow_answer_stays_in_check_without_regrounding(self) -> None:
        state = _state()
        state["budget_remaining"] = 100.0
        state["initial_budget"] = 100.0
        question = "Which business meaning of maintenance cost do you intend?"
        pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=GroundingCheckResponse(
                status="incomplete",
                clarification_route="none",
                missing_information="The maintenance-cost literal is missing.",
                next_tool=GroundingCheckToolRequest(
                    tool_name="ask_user",
                    arguments={"question": question},
                    user_clarification_request=GroundingCheckClarificationProposal(
                        phrase="maintenance cost",
                        kind="missing_knowledge",
                    ),
                ),
                column_mapping=primary_state().column_mapping or (),
                domain_knowledge=primary_state().domain_knowledge or (),
            ),
        )
        tool = SimpleNamespace(name="ask_user")
        context = SimpleNamespace(
            state=state,
            function_call_id=pending.function_call_id,
            invocation_id="v31-narrow-callback",
        )
        self.assertIsNone(
            await grounding_callbacks.before_tool_callback(
                tool,
                {"question": question},
                context,
            )
        )
        updater = _FourStageUpdater(answer_route="stay_check")
        token = grounding_callbacks._bind_turn_message(
            state["task_id"],
            "a-interact",
            QUERY,
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
                patch.object(
                    grounding_callbacks,
                    "load_sql_grounding_llm_config",
                    return_value=SimpleNamespace(max_calls_per_task=32),
                ),
            ):
                await grounding_callbacks.after_tool_callback(
                    tool,
                    {"question": question},
                    context,
                    "Use the literal Severe.",
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(len(updater.inputs), 1)
        self.assertIn("latest_user_answer", updater.inputs[0])
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.stage, "SQL_ATTEMPT")
        self.assertEqual(runtime.grounding_state, primary_state())
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 5, "2": 0},
        )

    async def test_answer_runs_four_stages_with_one_cumulative_overlay(self) -> None:
        state = _state()
        request = UserClarificationRequest(
            phrase="maintenance cost",
            kind="user_intent",
            question="Which business meaning of maintenance cost do you intend?",
        )
        grounding_callbacks._register_clarification_requests(
            state,
            phase=1,
            requests=(request,),
        )
        answer = "Use the reported maintenance cost for each asset."
        grounding_callbacks._record_clarification_answer(
            state,
            phase=1,
            question=request.question,
            answer=answer,
        )
        observation = build_sql_grounding_observation(
            task_id=state["task_id"],
            phase=1,
            sequence=grounding_callbacks._next_sequence(state),
            observation_type="user_answer",
            content=answer,
            summary="answered clarification",
            tool_name="ask_user",
            function_call_id="v3-clarification-call",
        )
        updater = _FourStageUpdater()
        token = grounding_callbacks._bind_turn_message(
            state["task_id"],
            "a-interact",
            QUERY,
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
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
                result, audits = (
                    await grounding_callbacks._run_clarification_regrounding(
                        state,
                        answer_observation=observation,
                        query=QUERY,
                        runtime=GroundingRuntime.model_validate(
                            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
                        ),
                        phase=1,
                        follow_up=None,
                    )
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(
            [item["staged_grounding_kind"] for item in audits],
            ["structure", "mapping", "knowledge", "check"],
            audits,
        )
        self.assertEqual(len(updater.inputs), 4)
        for payload in updater.inputs:
            self.assertEqual(
                payload["user_clarifications"],
                [
                    {
                        "phase": 1,
                        "phrase": request.phrase,
                        "kind": request.kind,
                        "question": request.question,
                        "answer": answer,
                    }
                ],
            )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 8, "2": 0},
        )
        self.assertEqual(result.runtime.stage, "SQL_ATTEMPT")
        self.assertEqual(result.runtime.grounding_state, primary_state())

    def test_answered_phrase_cannot_be_asked_again(self) -> None:
        state = _state()
        first = UserClarificationRequest(
            phrase="maintenance cost",
            kind="user_intent",
            question="Which maintenance-cost meaning do you intend?",
        )
        grounding_callbacks._register_clarification_requests(
            state,
            phase=1,
            requests=(first,),
        )
        grounding_callbacks._record_clarification_answer(
            state,
            phase=1,
            question=first.question,
            answer="Use reported maintenance cost.",
        )
        with self.assertRaisesRegex(ValueError, "phrase cannot be requested again"):
            grounding_callbacks._register_clarification_requests(
                state,
                phase=1,
                requests=(
                    UserClarificationRequest(
                        phrase="maintenance cost",
                        kind="user_intent",
                        question="Can you clarify maintenance cost once more?",
                    ),
                ),
            )

    def test_phase_two_regrounding_keeps_official_phase_and_cumulative_overlay(
        self,
    ) -> None:
        state = _state()
        state["current_phase"] = 2
        grounding_callbacks._store_clarification_records(
            state,
            (
                UserClarificationRecord(
                    phase=1,
                    phrase="maintenance cost",
                    kind="user_intent",
                    question="Which maintenance-cost meaning do you intend?",
                    answer="Use reported maintenance cost.",
                ),
                UserClarificationRecord(
                    phase=2,
                    phrase="top assets",
                    kind="user_intent",
                    question="Which assets should be included?",
                    answer="Only currently active assets.",
                ),
            ),
        )
        runtime = GroundingRuntime(
            grounding_revision=7,
            stage="P2_INCREMENTAL",
            focus_dimension="none",
            grounding_state=primary_state(),
        )
        payload = grounding_callbacks._phase_request_common(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up="Now show the top assets.",
        )
        self.assertEqual(state["current_phase"], 2)
        self.assertEqual(payload["follow_up"], "Now show the top assets.")
        self.assertEqual(
            [item["phase"] for item in payload["user_clarifications"]],
            [1, 2],
        )
        self.assertEqual(runtime.stage, "P2_INCREMENTAL")
        self.assertEqual(runtime.focus_dimension, "none")


if __name__ == "__main__":
    unittest.main()
