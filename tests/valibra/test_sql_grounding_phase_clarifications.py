from __future__ import annotations

import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import ValidationError

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    UserClarificationRecord,
    UserClarificationRequest,
    ValidationContext,
    canonical_json,
    sql_grounding_state_sha256,
    validate_grounding_llm_response,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.prompt_view import render_grounding_view
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    SQL_GROUNDING_STAGE_FORM_SCHEMAS,
    GroundingUpdaterResult,
)
from tests.valibra.test_stage3_submit_driven_repair import bootstrap_trajectory


QUERY = "Show active artists and their revenue."
PROMPT_SHA = "c18b3e366e8bf14de1f5cf1fa5144977e352cb96fabae9de6da579a46d03f6ed"
FORM_SHA = "58f44fbc8ea1ed1d38603b06fe13ec4a8a60d6d3512597de7ee6554b7e73df48"
CONFIG_SHA = "20503be3af181db8cc3df1d9680d5e62e5fc50e4436f22e658303023546e24cb"


def complete_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=(),
        join_keys=(),
        column_mapping=(),
        domain_knowledge=(),
    )


def runtime() -> GroundingRuntime:
    return GroundingRuntime(
        grounding_revision=1,
        stage="SQL_ATTEMPT",
        focus_dimension="none",
        grounding_state=complete_state(),
    )


def task_state(task_id: str) -> dict:
    current = runtime()
    return {
        "task_id": task_id,
        "current_phase": 1,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 10.0,
        "initial_budget": 10.0,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: current.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 1,
        grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
            "1": 1,
            "2": 0,
        },
    }


def request() -> LlmRequest:
    return LlmRequest(
        model="local-clarification-test",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=QUERY)],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction="BASE SYSTEM INSTRUCTION",
            temperature=0.0,
        ),
    )


class ClarificationFormContractTests(unittest.TestCase):
    def test_outer_form_and_frozen_hashes_are_exact(self) -> None:
        self.assertEqual(
            set(SQL_GROUNDING_FORM_SCHEMA),
            {"structure", "mapping", "knowledge", "check"},
        )
        self.assertEqual(
            SQL_GROUNDING_FORM_SCHEMA,
            SQL_GROUNDING_STAGE_FORM_SCHEMAS,
        )
        self.assertTrue(
            all(
                schema.get("additionalProperties") is False
                for schema in SQL_GROUNDING_FORM_SCHEMA.values()
            )
        )
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)
        with self.assertRaises(ValidationError):
            GroundingLLMResponse.model_validate(
                {
                    "sql_grounding_state": complete_state().model_dump(mode="json"),
                    "next_focus_dimension": "none",
                }
            )

    def test_only_user_owned_clarifications_are_accepted(self) -> None:
        clarification = UserClarificationRequest(
            phrase="active",
            kind="user_intent",
            question="What should active mean for this request?",
        )
        response = GroundingLLMResponse(
            sql_grounding_state=complete_state(),
            user_clarification_requests=(clarification,),
            next_focus_dimension="none",
        )
        context = ValidationContext(
            current_query=QUERY,
            latest_observation_id="observation-1",
        )
        self.assertIs(
            validate_grounding_llm_response(
                response,
                stage="INITIAL_GROUNDING",
                context=context,
            ),
            response,
        )
        for question in (
            "Which table should be queried?",
            "Which SQL should be written?",
            "What caused this runtime error?",
        ):
            with self.subTest(question=question), self.assertRaises(ValidationError):
                UserClarificationRequest(
                    phrase="active",
                    kind="user_intent",
                    question=question,
                )
        with self.assertRaises(ValidationError):
            UserClarificationRequest(
                phrase="active",
                kind="schema",  # type: ignore[arg-type]
                question="What should active mean?",
            )
        invalid_phrase = GroundingLLMResponse(
            sql_grounding_state=complete_state(),
            user_clarification_requests=(
                UserClarificationRequest(
                    phrase="currently enabled",
                    kind="user_intent",
                    question="What should currently enabled mean?",
                ),
            ),
            next_focus_dimension="none",
        )
        with self.assertRaisesRegex(ValueError, "verbatim"):
            validate_grounding_llm_response(
                invalid_phrase,
                stage="INITIAL_GROUNDING",
                context=context,
            )

    def test_answered_overlay_is_bounded_and_not_a_fifth_dimension(self) -> None:
        record = UserClarificationRecord(
            phase=1,
            phrase="active",
            kind="missing_knowledge",
            question="What business rule defines active?",
            answer="Use the artist's current contract status.",
        )
        rendered = render_grounding_view(
            complete_state(),
            clarifications=(record,),
        )
        self.assertIn("[USER CLARIFICATIONS]", rendered.text)
        self.assertIn("current contract status", rendered.text)
        self.assertNotIn("user_clarification", canonical_json(complete_state()))
        self.assertEqual(
            set(GroundingRuntime.model_fields),
            {"grounding_revision", "stage", "focus_dimension", "grounding_state"},
        )

    def test_same_phase_question_must_be_unique_in_response_contract(self) -> None:
        with self.assertRaisesRegex(ValidationError, "question must be unique"):
            GroundingLLMResponse(
                sql_grounding_state=complete_state(),
                user_clarification_requests=(
                    UserClarificationRequest(
                        phrase="active",
                        kind="user_intent",
                        question="What does active mean?",
                    ),
                    UserClarificationRequest(
                        phrase="artists",
                        kind="missing_knowledge",
                        question="What does active mean?",
                    ),
                ),
                next_focus_dimension="none",
            )

    def test_answered_clarification_survives_item_budget_before_state(self) -> None:
        record = UserClarificationRecord(
            phase=1,
            phrase="active",
            kind="user_intent",
            question="What does active mean?",
            answer="Use signed contracts only.",
        )
        rendered = render_grounding_view(
            complete_state(),
            clarifications=(record,),
            max_items=1,
        )
        self.assertIn("[USER CLARIFICATIONS]", rendered.text)
        self.assertIn("Use signed contracts only.", rendered.text)
        self.assertEqual(rendered.included_items, 1)
        self.assertEqual(rendered.omitted_items, 4)

        constrained = render_grounding_view(
            complete_state(),
            clarifications=(record,),
            max_chars=rendered.char_count,
            max_tokens=rendered.token_count,
        )
        self.assertIn("[USER CLARIFICATIONS]", constrained.text)
        self.assertIn("Use signed contracts only.", constrained.text)
        self.assertEqual(constrained.included_items, 1)
        self.assertEqual(constrained.omitted_items, 4)


class IncompleteGroundingUpdater:
    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=SQLGroundingState(),
                user_clarification_requests=(),
                next_focus_dimension="tables",
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


class ClarificationPatchUpdater:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs = []

    async def propose(self, runtime, *args, **kwargs):
        del args
        self.calls += 1
        self.inputs.append(copy.deepcopy(kwargs["grounding_input"]))
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=runtime.grounding_state,
                user_clarification_requests=(),
                next_focus_dimension="none",
            ),
            telemetry=GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="c" * 64,
                response_sha256="d" * 64,
                prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
                form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


class ReplacingClarificationPatchUpdater(ClarificationPatchUpdater):
    def __init__(self, corrected_state: SQLGroundingState) -> None:
        super().__init__()
        self.corrected_state = corrected_state

    async def propose(self, runtime, *args, **kwargs):
        del runtime, args
        self.calls += 1
        self.inputs.append(copy.deepcopy(kwargs["grounding_input"]))
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=self.corrected_state,
                user_clarification_requests=(),
                next_focus_dimension="none",
            ),
            telemetry=GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="e" * 64,
                response_sha256="f" * 64,
                prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
                form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


class ClarificationCallbackLifecycleTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skip("retired 1.2 Clarification Patch; v1.3 Check A→B is covered separately")
    async def test_clarification_replaces_a_with_candidate_table_field_b_once(
        self,
    ) -> None:
        query = "Show active artists and their revenue."
        schema = """CREATE TABLE artists (
  active_flag BOOLEAN,
  revenue NUMERIC,
  status_label TEXT
);
CREATE TABLE unrelated (
  decoy TEXT
);"""
        meanings = json.dumps(
            {
                "clarification|artists|active_flag": "True for active artists.",
                "clarification|artists|revenue": "Booked artist revenue.",
                "clarification|artists|status_label": "Legacy status text.",
                "clarification|unrelated|decoy": "Must remain excluded.",
            },
            sort_keys=True,
        )
        mapping_a = (
            ColumnMapping(phrase="active", targets=("artists.status_label",)),
            ColumnMapping(phrase="revenue", targets=("artists.revenue",)),
        )
        mapping_b = (
            ColumnMapping(phrase="active", targets=("artists.active_flag",)),
            mapping_a[1],
        )
        initial_state = SQLGroundingState(
            tables=("artists",),
            join_keys=(),
            column_mapping=mapping_a,
            domain_knowledge=(),
        )
        current = task_state("clarification-a-to-b")
        pending_runtime = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=initial_state,
        )
        current.update(
            {
                grounding_callbacks.GROUNDING_RUNTIME_KEY: pending_runtime.model_dump(
                    mode="json"
                ),
                grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 3,
                grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
                    "1": 3,
                    "2": 0,
                },
                "tool_trajectory": [
                    {
                        "type": "tool",
                        "tool": "get_schema",
                        "phase": 1,
                        "args": {},
                        "result": schema,
                    },
                    {
                        "type": "tool",
                        "tool": "get_all_column_meanings",
                        "phase": 1,
                        "args": {},
                        "result": meanings,
                    },
                    {
                        "type": "tool",
                        "tool": "get_all_knowledge_definitions",
                        "phase": 1,
                        "args": {},
                        "result": "[]",
                    },
                ],
            }
        )
        clarification = UserClarificationRequest(
            phrase="active",
            kind="user_intent",
            question="Should active mean the boolean active flag?",
        )
        grounding_callbacks._register_clarification_requests(
            current,
            phase=1,
            requests=(clarification,),
        )
        updater = ReplacingClarificationPatchUpdater(
            initial_state.model_copy(update={"column_mapping": mapping_b})
        )
        token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", query
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
                patch.object(
                    grounding_callbacks,
                    "load_sql_grounding_llm_config",
                    return_value=SimpleNamespace(max_calls_per_task=8),
                ),
            ):
                forced = await grounding_callbacks.before_model_callback(
                    SimpleNamespace(state=current),
                    request(),
                )
                function_call = forced.content.parts[0].function_call
                context = SimpleNamespace(
                    state=current,
                    function_call_id=function_call.id,
                    invocation_id="inv-clarification-a-to-b",
                )
                await grounding_callbacks.before_tool_callback(
                    SimpleNamespace(name="ask_user"),
                    {"question": function_call.args["question"]},
                    context,
                )
                await grounding_callbacks.after_tool_callback(
                    SimpleNamespace(name="ask_user"),
                    {"question": function_call.args["question"]},
                    context,
                    "Yes, use the boolean active flag.",
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(updater.calls, 1)
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            4,
        )
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 4, "2": 0},
        )
        relevant = updater.inputs[0]["relevant_column_meanings"]
        self.assertIn("clarification|artists|status_label", relevant)
        self.assertIn("clarification|artists|active_flag", relevant)
        self.assertIn("clarification|artists|revenue", relevant)
        self.assertNotIn("clarification|unrelated|decoy", relevant)
        final_runtime = GroundingRuntime.model_validate(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(final_runtime.grounding_revision, 4)
        self.assertEqual(final_runtime.stage, "SQL_ATTEMPT")
        self.assertEqual(
            final_runtime.grounding_state.tables,
            initial_state.tables,
        )
        self.assertEqual(
            final_runtime.grounding_state.join_keys,
            initial_state.join_keys,
        )
        self.assertEqual(final_runtime.grounding_state.column_mapping, mapping_b)
        self.assertEqual(
            final_runtime.grounding_state.column_mapping[1],
            mapping_a[1],
        )

    async def test_failed_phase_grounding_blocks_model_and_official_tools(self) -> None:
        initial = GroundingRuntime()
        current = {
            "task_id": "grounding-failed-closed",
            "current_phase": 1,
            "phase1_completed": False,
            "phase2_completed": False,
            "task_done": False,
            "budget_remaining": 10.0,
            "initial_budget": 10.0,
            "tool_trajectory": [],
            "system_agent_llm_calls": [],
            grounding_callbacks.GROUNDING_RUNTIME_KEY: initial.model_dump(mode="json"),
        }
        observation = build_sql_grounding_observation(
            task_id=current["task_id"],
            phase=1,
            sequence=1,
            observation_type="schema",
            content="CREATE TABLE t (c INTEGER)",
            summary="Official schema evidence observed",
            tool_name="get_schema",
            function_call_id="failed-closed-bootstrap-1",
            private_raw_ref="session://tool_trajectory/0",
        )
        grounding_input = {
            "query": QUERY,
            "schema": "CREATE TABLE t (c INTEGER)",
            "current_state": initial.grounding_state.model_dump(mode="json"),
        }
        updater = IncompleteGroundingUpdater()
        context = ValidationContext(
            current_query=QUERY,
            latest_observation_id=observation.observation_id,
            official_trajectory_observation_ids=(observation.observation_id,),
        )
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
            patch.object(
                grounding_callbacks,
                "load_sql_grounding_llm_config",
                return_value=SimpleNamespace(max_calls_per_task=8),
            ),
            patch.object(
                grounding_callbacks,
                "_build_validation_context",
                return_value=context,
            ),
        ):
            first = await grounding_callbacks._handle_observation(
                current,
                observation,
                grounding_input=grounding_input,
            )
            second = await grounding_callbacks._handle_observation(
                current,
                observation,
                grounding_input=grounding_input,
            )

            self.assertEqual(first.service_status, "rejected")
            self.assertEqual(second.service_status, "skipped_phase_grounding_terminal")
            self.assertEqual(updater.calls, 1)
            self.assertEqual(
                current[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
                1,
            )
            outcome = current[grounding_callbacks.GROUNDING_PHASE_OUTCOMES_KEY]["1"]
            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(outcome["error_type"], "state_validation_failed")
            self.assertEqual(outcome["grounding_revision"], 0)

            model_result = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=current),
                request(),
            )
            self.assertIn(
                "VALIBRA_SQL_GROUNDING_FAILED_CLOSED",
                model_result.content.parts[0].text,
            )
            budget_before = current["budget_remaining"]
            tool_context = SimpleNamespace(
                state=current,
                function_call_id="failed-closed-execute",
                invocation_id="inv-failed-closed-execute",
            )
            denial = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="execute_sql"),
                {"sql": "SELECT 1"},
                tool_context,
            )
            self.assertEqual(
                denial["status"],
                "VALIBRA_SQL_GROUNDING_FAILED_CLOSED",
            )
            self.assertEqual(current["budget_remaining"], budget_before)
            self.assertEqual(
                await grounding_callbacks.after_tool_callback(
                    SimpleNamespace(name="execute_sql"),
                    {"sql": "SELECT 1"},
                    tool_context,
                    denial,
                ),
                denial,
            )
            self.assertEqual(current.get("tool_trajectory"), [])

    async def test_empty_clarifications_continue_directly_to_main(self) -> None:
        current = task_state("clarification-empty")
        original = copy.deepcopy(current[grounding_callbacks.GROUNDING_RUNTIME_KEY])
        result = await grounding_callbacks.before_model_callback(
            SimpleNamespace(state=current),
            request(),
        )
        self.assertIsNone(result)
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY],
            original,
        )

    @unittest.skip("retired 1.2 Clarification Patch; v1.3 resumes unified Check")
    async def test_forced_ask_answer_overlay_does_not_reground_or_change_state(self) -> None:
        current = task_state("clarification-answer")
        pending_runtime = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=complete_state(),
        )
        current.update(
            {
                grounding_callbacks.GROUNDING_RUNTIME_KEY: pending_runtime.model_dump(
                    mode="json"
                ),
                grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 3,
                grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
                    "1": 3,
                    "2": 0,
                },
                "tool_trajectory": bootstrap_trajectory(),
            }
        )
        clarification = UserClarificationRequest(
            phrase="active",
            kind="user_intent",
            question="What should active mean for this request?",
        )
        grounding_callbacks._register_clarification_requests(
            current,
            phase=1,
            requests=(clarification,),
        )
        frozen_runtime = copy.deepcopy(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        frozen_sha = sql_grounding_state_sha256(runtime().grounding_state)

        first_request = request()
        forced = await grounding_callbacks.before_model_callback(
            SimpleNamespace(state=current),
            first_request,
        )
        self.assertIsNotNone(forced)
        function_call = forced.content.parts[0].function_call
        self.assertEqual(function_call.name, "ask_user")
        self.assertEqual(function_call.args, {"question": clarification.question})

        tool = SimpleNamespace(name="ask_user")
        context = SimpleNamespace(
            state=current,
            function_call_id=function_call.id,
            invocation_id="inv-clarification-answer",
        )
        before = await grounding_callbacks.before_tool_callback(
            tool,
            {"question": clarification.question},
            context,
        )
        self.assertIsNone(before)
        updater = ClarificationPatchUpdater()
        token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", QUERY
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
                    return_value=SimpleNamespace(max_calls_per_task=8),
                ),
            ):
                await grounding_callbacks.after_tool_callback(
                    tool,
                    {"question": clarification.question},
                    context,
                    "Use signed contracts only.",
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            4,
        )
        self.assertEqual(updater.calls, 1)
        self.assertEqual(
            set(updater.inputs[0]),
            {
                "query",
                "current_state",
                "clarification_qa",
                "relevant_column_meanings",
                "relevant_knowledge_definitions",
            },
        )
        updated_runtime = GroundingRuntime.model_validate(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(updated_runtime.grounding_revision, 3)
        self.assertEqual(updated_runtime.stage, "SQL_ATTEMPT")
        self.assertEqual(
            sql_grounding_state_sha256(
                updated_runtime.grounding_state
            ),
            frozen_sha,
        )
        records = grounding_callbacks._clarification_records(current)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].answer, "Use signed contracts only.")

        second_request = request()
        result = await grounding_callbacks.before_model_callback(
            SimpleNamespace(state=current),
            second_request,
        )
        self.assertIsNone(result)
        instruction = second_request.config.system_instruction
        self.assertIn("[USER CLARIFICATIONS]", instruction)
        self.assertIn("Use signed contracts only.", instruction)

    async def test_pending_clarification_blocks_every_submit_before_cost(self) -> None:
        current = task_state("clarification-submit-gate")
        clarification = UserClarificationRequest(
            phrase="active",
            kind="user_intent",
            question="What should active mean for this request?",
        )
        grounding_callbacks._register_clarification_requests(
            current,
            phase=1,
            requests=(clarification,),
        )
        budget = current["budget_remaining"]
        for index in (1, 2):
            context = SimpleNamespace(
                state=current,
                function_call_id=f"pending-submit-{index}",
                invocation_id=f"inv-pending-submit-{index}",
            )
            denial = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="submit_sql"),
                {"sql": "SELECT 1"},
                context,
            )
            self.assertEqual(denial["status"], "VALIBRA_FIRST_SUBMIT_BLOCKED")
            self.assertEqual(current["budget_remaining"], budget)
            self.assertFalse(
                current.get(grounding_callbacks.GROUNDING_PENDING_KEY)
            )

    @unittest.skip("retired batched clarification flow; v1.3 Check asks one question at a time")
    async def test_multiple_answers_trigger_one_patch_only_after_all_are_complete(self) -> None:
        current = task_state("clarification-multiple")
        pending_runtime = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=complete_state(),
        )
        current.update(
            {
                grounding_callbacks.GROUNDING_RUNTIME_KEY: pending_runtime.model_dump(
                    mode="json"
                ),
                grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 3,
                grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
                    "1": 3,
                    "2": 0,
                },
                "tool_trajectory": bootstrap_trajectory(),
            }
        )
        requests = (
            UserClarificationRequest(
                phrase="active",
                kind="user_intent",
                question="What should active mean for this request?",
            ),
            UserClarificationRequest(
                phrase="revenue",
                kind="missing_knowledge",
                question="Which revenue rule should be used?",
            ),
        )
        grounding_callbacks._register_clarification_requests(
            current,
            phase=1,
            requests=requests,
        )
        updater = ClarificationPatchUpdater()
        token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", QUERY
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
                    return_value=SimpleNamespace(max_calls_per_task=8),
                ),
            ):
                for index, answer in enumerate(
                    ("Use signed contracts.", "Use booked revenue."),
                    start=1,
                ):
                    forced = await grounding_callbacks.before_model_callback(
                        SimpleNamespace(state=current),
                        request(),
                    )
                    function_call = forced.content.parts[0].function_call
                    context = SimpleNamespace(
                        state=current,
                        function_call_id=function_call.id,
                        invocation_id=f"inv-clarification-{index}",
                    )
                    await grounding_callbacks.before_tool_callback(
                        SimpleNamespace(name="ask_user"),
                        {"question": function_call.args["question"]},
                        context,
                    )
                    await grounding_callbacks.after_tool_callback(
                        SimpleNamespace(name="ask_user"),
                        {"question": function_call.args["question"]},
                        context,
                        answer,
                    )
                    self.assertEqual(updater.calls, int(index == 2))
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 4, "2": 0},
        )
        self.assertEqual(
            len(updater.inputs[0]["clarification_qa"]),
            2,
        )
        runtime_after = GroundingRuntime.model_validate(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime_after.stage, "SQL_ATTEMPT")
        self.assertEqual(runtime_after.grounding_revision, 3)

    def test_provider_ledger_uses_the_v13_task_safety_bound(self) -> None:
        current = {"task_id": "clarification-ledger"}
        for _ in range(16):
            grounding_callbacks._record_provider_call(current, 1)
        for _ in range(16):
            grounding_callbacks._record_provider_call(current, 2)
        with self.assertRaises(ValueError):
            grounding_callbacks._record_provider_call(current, 2)
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            32,
        )
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 16, "2": 16},
        )


if __name__ == "__main__":
    unittest.main()
