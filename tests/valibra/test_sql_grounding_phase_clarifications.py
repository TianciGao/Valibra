from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import ValidationError

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
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
from valibra_agent.sql_grounding.prompt_view import render_grounding_view
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
)


QUERY = "Show active artists and their revenue."
PROMPT_SHA = "8f13e7ecc0551b2d940e22546889f6d19a908a380be43c1d50b4f1128bec2fb7"
FORM_SHA = "1f7e3c1f1ae86876f63de951bcade30fc1ba338e046416fe033331d447775d15"
CONFIG_SHA = "ee00b4d7190f6dd2041b0a0ddae6c2059fca5024a6c068b4269b85fc070e61d6"


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
            set(SQL_GROUNDING_FORM_SCHEMA["properties"]),
            {
                "sql_grounding_state",
                "user_clarification_requests",
                "next_focus_dimension",
            },
        )
        self.assertEqual(
            set(SQL_GROUNDING_FORM_SCHEMA["required"]),
            {
                "sql_grounding_state",
                "user_clarification_requests",
                "next_focus_dimension",
            },
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


class ClarificationCallbackLifecycleTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_forced_ask_answer_overlay_does_not_reground_or_change_state(self) -> None:
        current = task_state("clarification-answer")
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
        await grounding_callbacks.after_tool_callback(
            tool,
            {"question": clarification.question},
            context,
            "Use signed contracts only.",
        )

        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            1,
        )
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY],
            frozen_runtime,
        )
        self.assertEqual(
            sql_grounding_state_sha256(
                GroundingRuntime.model_validate(frozen_runtime).grounding_state
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

    def test_provider_ledger_is_one_per_phase_two_per_task(self) -> None:
        current = {"task_id": "clarification-ledger"}
        grounding_callbacks._record_provider_call(current, 1)
        with self.assertRaises(ValueError):
            grounding_callbacks._record_provider_call(current, 1)
        grounding_callbacks._record_provider_call(current, 2)
        with self.assertRaises(ValueError):
            grounding_callbacks._record_provider_call(current, 2)
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            2,
        )
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY],
            {"1": 1, "2": 1},
        )


if __name__ == "__main__":
    unittest.main()
