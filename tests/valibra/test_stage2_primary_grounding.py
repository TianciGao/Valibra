from __future__ import annotations

import copy
import json
import os
import unittest
from collections.abc import AsyncGenerator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from shared.audit import to_jsonable
from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    SQLGroundingValidationError,
    ValidationContext,
    UserClarificationRecord,
    canonical_json,
    validate_sql_grounding_state,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.service import process_sql_grounding_observation
from valibra_agent.sql_grounding.telemetry import (
    GroundingLLMTelemetry,
    GroundingTokenUsage,
)
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingClientResponse,
    GroundingUpdaterResult,
    SQLGroundingUpdater,
)


QUERY = "Show the maintenance cost."
RULE = "Use maintcost as reported."
SCHEMA_SENTINEL = "stage2-private-schema-sentinel"
MEANINGS_SENTINEL = "stage2-private-meanings-sentinel"
SCHEMA = f"""CREATE TABLE operational_metrics (
  maintcost NUMERIC,
  payload JSONB
);
-- {SCHEMA_SENTINEL}
"""
COLUMN_MEANINGS = json.dumps(
    {
        "stage2|operational_metrics|maintcost": (
            f"Maintenance cost; {MEANINGS_SENTINEL}"
        ),
        "stage2|operational_metrics|payload": {
            "column_meaning": "Structured maintenance data.",
            "fields_meaning": {
                "cost": {
                    "reported": "Reported maintenance cost."
                }
            },
        },
    },
    sort_keys=True,
)
KNOWLEDGE_DEFINITIONS = json.dumps(
    [{"id": "rule-1", "definition": RULE}],
    sort_keys=True,
)


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
        domain_knowledge=(
            DomainKnowledge(kind="business_rule", content=RULE),
        ),
    )


def structure_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("operational_metrics",),
        join_keys=(),
    )


def mapping_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("operational_metrics",),
        join_keys=(),
        column_mapping=complete_state().column_mapping,
    )


def state(task_id: str) -> dict[str, Any]:
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
    }


def primary_observation() -> Any:
    return build_sql_grounding_observation(
        task_id="stage2-service-task",
        phase=1,
        sequence=1,
        observation_type="knowledge",
        content=KNOWLEDGE_DEFINITIONS,
        summary="complete Official bootstrap evidence observed",
        tool_name="get_all_knowledge_definitions",
        function_call_id="stage2-bootstrap-call-3",
        private_raw_ref="session://tool_trajectory/2",
    )


def primary_input(runtime: GroundingRuntime) -> dict[str, Any]:
    return {
        "query": QUERY,
        "schema": SCHEMA,
        "column_meanings": COLUMN_MEANINGS,
        "knowledge_definitions": KNOWLEDGE_DEFINITIONS,
        "current_state": runtime.grounding_state.model_dump(mode="json"),
    }


def validation_context(observation: Any) -> ValidationContext:
    return ValidationContext(
        current_query=QUERY,
        latest_observation_id=observation.observation_id,
        official_trajectory_observation_ids=(observation.observation_id,),
        known_tables=frozenset({"operational_metrics"}),
        known_columns=frozenset(
            {
                "operational_metrics.maintcost",
                "operational_metrics.payload",
            }
        ),
        supported_json_paths=frozenset(
            {
                ("operational_metrics.payload", ("cost",)),
                ("operational_metrics.payload", ("cost", "reported")),
            }
        ),
        supported_domain_knowledge=frozenset({("business_rule", RULE)}),
    )


class PrimaryFakeUpdater:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[dict[str, Any]] = []

    async def propose(self, runtime: GroundingRuntime, *args: Any, **kwargs: Any):
        del runtime, args
        self.calls += 1
        self.inputs.append(copy.deepcopy(kwargs["grounding_input"]))
        fields = set(self.inputs[-1])
        if "schema" in fields:
            candidate = structure_state()
            focus = "column_mapping"
        elif "column_meanings" in fields:
            candidate = mapping_state()
            focus = "domain_knowledge"
        else:
            candidate = complete_state()
            focus = "none"
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=candidate,
                user_clarification_requests=(),
                next_focus_dimension=focus,
            ),
            telemetry=GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                usage=GroundingTokenUsage(
                    input_tokens=31,
                    output_tokens=11,
                    reasoning_tokens=7,
                    total_tokens=42,
                ),
                request_sha256="a" * 64,
                response_sha256="b" * 64,
                prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
                form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


class FinalLocalModel(BaseLlm):
    calls: int = 0
    requests: list[dict[str, Any]] = []

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.calls += 1
        self.requests.append(to_jsonable(copy.deepcopy(llm_request)))
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="STAGE2_LOCAL_DONE")],
            )
        )


class CapturingClient:
    def __init__(self, response: GroundingLLMResponse) -> None:
        self.response = response
        self.requests: list[Any] = []

    async def complete(self, request: Any) -> GroundingClientResponse:
        self.requests.append(request)
        return GroundingClientResponse(content=canonical_json(self.response))


class Stage2PrimaryGroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_adk_three_tools_trigger_three_staged_updates(self):
        executions: list[str] = []

        def get_schema() -> str:
            executions.append("get_schema")
            return SCHEMA

        def get_all_column_meanings() -> str:
            executions.append("get_all_column_meanings")
            return COLUMN_MEANINGS

        def get_all_knowledge_definitions() -> str:
            executions.append("get_all_knowledge_definitions")
            return KNOWLEDGE_DEFINITIONS

        model = FinalLocalModel(model="stage2-local")
        updater = PrimaryFakeUpdater()
        agent = LlmAgent(
            name="stage2_primary_agent",
            model=model,
            instruction="Offline Stage 2 lifecycle test.",
            tools=[
                get_schema,
                get_all_column_meanings,
                get_all_knowledge_definitions,
            ],
            before_model_callback=grounding_callbacks.before_model_callback,
            after_model_callback=grounding_callbacks.after_model_callback,
            before_tool_callback=grounding_callbacks.before_tool_callback,
            after_tool_callback=grounding_callbacks.after_tool_callback,
            on_tool_error_callback=grounding_callbacks.on_tool_error_callback,
        )
        runner = InMemoryRunner(agent=agent, app_name="stage2_primary")
        session = await runner.session_service.create_session(
            app_name="stage2_primary",
            user_id="stage2-user",
            state=state("stage2-primary-task"),
        )
        token = grounding_callbacks._bind_turn_message(
            "stage2-primary-task", "a-interact", QUERY
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
                events = [
                    event
                    async for event in runner.run_async(
                        user_id="stage2-user",
                        session_id=session.id,
                        new_message=types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=QUERY)],
                        ),
                    )
                ]
        finally:
            grounding_callbacks._reset_turn_message(token)
        self.assertTrue(events)
        final = await runner.session_service.get_session(
            app_name="stage2_primary",
            user_id="stage2-user",
            session_id=session.id,
        )
        self.assertIsNotNone(final)
        session_state = final.state

        self.assertEqual(executions, list(grounding_callbacks._BOOTSTRAP_TOOL_SEQUENCE))
        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            session_state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            3,
        )
        self.assertEqual(set(updater.inputs[0]), {
            "query",
            "schema",
            "current_state",
        })
        self.assertEqual(updater.inputs[0]["schema"], SCHEMA)
        self.assertEqual(
            [set(item) for item in updater.inputs],
            [
                {"query", "schema", "current_state"},
                {"query", "column_meanings", "current_state"},
                {
                    "query",
                    "knowledge_definitions",
                    "relevant_column_meanings",
                    "current_state",
                },
            ],
        )
        runtime = GroundingRuntime.model_validate(
            session_state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.grounding_revision, 3)
        self.assertEqual(runtime.grounding_state, complete_state())
        phase_outcome = session_state[
            grounding_callbacks.GROUNDING_PHASE_OUTCOMES_KEY
        ]["1"]
        self.assertEqual(phase_outcome["status"], "succeeded")
        self.assertEqual(phase_outcome["grounding_revision"], 3)
        self.assertEqual(
            phase_outcome["state_sha256"],
            grounding_callbacks.sql_grounding_state_sha256(
                runtime.grounding_state
            ),
        )
        self.assertEqual(runtime.stage, "SQL_ATTEMPT")
        self.assertEqual(runtime.focus_dimension, "none")
        self.assertEqual(session_state["budget_remaining"], 7.0)
        self.assertEqual(model.calls, 1)

        audits = session_state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY]
        ordered = [audits[key][grounding_callbacks.SHADOW_AUDIT_KEY] for key in sorted(audits)]
        self.assertEqual(
            [item["service_status"] for item in ordered],
            ["accepted", "accepted", "accepted"],
        )
        self.assertEqual(
            [item["changed_dimensions"] for item in ordered],
            [
                ["tables", "join_keys"],
                ["column_mapping"],
                ["domain_knowledge"],
            ],
        )
        self.assertTrue(ordered[-1]["provider_attempted"])
        self.assertTrue(ordered[-1]["staged_grounding_triggered"])

        model_visible = canonical_json(model.requests)
        self.assertNotIn(SCHEMA_SENTINEL, model_visible)
        self.assertNotIn(MEANINGS_SENTINEL, model_visible)
        self.assertNotIn(COLUMN_MEANINGS, model_visible)

    async def test_updater_receives_exact_stage_fields_and_service_is_atomic(self):
        runtime = GroundingRuntime()
        stages = (
            (
                "schema",
                "get_schema",
                SCHEMA,
                {"schema": SCHEMA},
                structure_state(),
                "column_mapping",
            ),
            (
                "metadata",
                "get_all_column_meanings",
                COLUMN_MEANINGS,
                {"column_meanings": json.loads(COLUMN_MEANINGS)},
                mapping_state(),
                "domain_knowledge",
            ),
            (
                "knowledge",
                "get_all_knowledge_definitions",
                KNOWLEDGE_DEFINITIONS,
                {
                    "knowledge_definitions": json.loads(KNOWLEDGE_DEFINITIONS),
                    "relevant_column_meanings": json.loads(COLUMN_MEANINGS),
                },
                complete_state(),
                "none",
            ),
        )
        observations = []
        for sequence, (
            observation_type,
            tool_name,
            content,
            evidence,
            candidate,
            focus,
        ) in enumerate(stages, start=1):
            observation = build_sql_grounding_observation(
                task_id="stage2-service-task",
                phase=1,
                sequence=sequence,
                observation_type=observation_type,
                content=content,
                summary=f"{tool_name} evidence",
                tool_name=tool_name,
                function_call_id=f"stage2-call-{sequence}",
                private_raw_ref=f"session://tool_trajectory/{sequence - 1}",
            )
            observations.append(observation)
            grounding_input = {
                "query": QUERY,
                "current_state": runtime.grounding_state.model_dump(mode="json"),
                **evidence,
            }
            client = CapturingClient(
                GroundingLLMResponse(
                    sql_grounding_state=candidate,
                    user_clarification_requests=(),
                    next_focus_dimension=focus,
                )
            )
            result = await process_sql_grounding_observation(
                runtime,
                observation,
                validation_context(observation),
                SQLGroundingUpdater(client),
                grounding_input=grounding_input,
            )
            self.assertEqual(result.state_update.status, "accepted")
            request_payload = json.loads(client.requests[0].input_json)
            self.assertEqual(request_payload, grounding_input)
            self.assertNotIn("latest_observation", request_payload)
            runtime = result.runtime

        self.assertEqual(runtime.grounding_revision, 3)
        self.assertEqual(runtime.grounding_state, complete_state())

        invalid_state = complete_state().model_copy(
            update={
                "domain_knowledge": (
                    DomainKnowledge(
                        kind="business_rule",
                        content="Invented rule is forbidden.",
                    ),
                )
            }
        )
        invalid_client = CapturingClient(
            GroundingLLMResponse(
                sql_grounding_state=invalid_state,
                user_clarification_requests=(),
                next_focus_dimension="none",
            )
        )
        rejected = await process_sql_grounding_observation(
            GroundingRuntime(
                grounding_revision=2,
                grounding_state=mapping_state(),
                focus_dimension="domain_knowledge",
            ),
            observations[-1],
            validation_context(observations[-1]),
            SQLGroundingUpdater(invalid_client),
            grounding_input={
                "query": QUERY,
                "current_state": mapping_state().model_dump(mode="json"),
                "knowledge_definitions": json.loads(KNOWLEDGE_DEFINITIONS),
                "relevant_column_meanings": json.loads(COLUMN_MEANINGS),
            },
        )
        self.assertEqual(rejected.state_update.status, "rejected")
        self.assertEqual(rejected.runtime.grounding_revision, 2)

        partial_client = CapturingClient(
            GroundingLLMResponse(
                sql_grounding_state=SQLGroundingState(tables=()),
                user_clarification_requests=(),
                next_focus_dimension="join_keys",
            )
        )
        partial = await process_sql_grounding_observation(
            GroundingRuntime(),
            observations[0],
            validation_context(observations[0]),
            SQLGroundingUpdater(partial_client),
            grounding_input={
                "query": QUERY,
                "current_state": GroundingRuntime().grounding_state.model_dump(mode="json"),
                "schema": SCHEMA,
            },
        )
        self.assertEqual(partial.state_update.status, "rejected")
        self.assertEqual(partial.runtime, GroundingRuntime())

    async def test_incomplete_bundle_is_rejected_before_client_call(self):
        runtime = GroundingRuntime()
        observation = primary_observation()
        client = CapturingClient(
            GroundingLLMResponse(
                sql_grounding_state=complete_state(),
                user_clarification_requests=(),
                next_focus_dimension="none",
            )
        )
        incomplete = {
            "query": QUERY,
            "current_state": runtime.grounding_state.model_dump(mode="json"),
            "schema": SCHEMA,
            "column_meanings": json.loads(COLUMN_MEANINGS),
        }
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            validation_context(observation),
            SQLGroundingUpdater(client),
            grounding_input=incomplete,
        )
        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(client.requests, [])

    async def test_stage_permissions_and_clarification_scope_fail_closed(self):
        alternate_rule = "Use active-contract maintenance cost only."
        base_context = ValidationContext(
            current_query="Show active maintenance cost.",
            latest_observation_id="placeholder",
            known_tables=frozenset({"operational_metrics", "other_metrics"}),
            known_columns=frozenset(
                {
                    "operational_metrics.maintcost",
                    "operational_metrics.payload",
                }
            ),
            supported_json_paths=frozenset(
                {("operational_metrics.payload", ("cost",))}
            ),
            supported_domain_knowledge=frozenset(
                {
                    ("business_rule", RULE),
                    ("business_rule", alternate_rule),
                }
            ),
        )

        structure_observation = build_sql_grounding_observation(
            task_id="stage2-permissions",
            phase=1,
            sequence=1,
            observation_type="schema",
            content=SCHEMA,
            summary="schema evidence",
            tool_name="get_schema",
            function_call_id="stage2-permission-structure",
            private_raw_ref="session://tool_trajectory/0",
        )
        invalid_structure = structure_state().model_copy(
            update={"column_mapping": complete_state().column_mapping}
        )
        result = await process_sql_grounding_observation(
            GroundingRuntime(),
            structure_observation,
            replace(
                base_context,
                latest_observation_id=structure_observation.observation_id,
            ),
            SQLGroundingUpdater(
                CapturingClient(
                    GroundingLLMResponse(
                        sql_grounding_state=invalid_structure,
                        user_clarification_requests=(),
                        next_focus_dimension="column_mapping",
                    )
                )
            ),
            grounding_input={
                "query": "Show active maintenance cost.",
                "current_state": SQLGroundingState().model_dump(mode="json"),
                "schema": SCHEMA,
            },
        )
        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.runtime, GroundingRuntime())

        current = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=complete_state(),
        )
        answer = UserClarificationRecord(
            phase=1,
            phrase="active",
            kind="user_intent",
            question="What should active mean?",
            answer="Use active contracts.",
        )
        patch_observation = build_sql_grounding_observation(
            task_id="stage2-permissions",
            phase=1,
            sequence=2,
            observation_type="user_answer",
            content=answer.answer,
            summary="clarification answer",
            tool_name="ask_user",
            function_call_id="stage2-permission-answer",
            private_raw_ref="session://tool_trajectory/1",
        )
        unrelated_mapping = complete_state().model_copy(
            update={
                "column_mapping": (
                    ColumnMapping(
                        phrase="maintenance cost",
                        targets=("operational_metrics.payload ->> 'cost'",),
                    ),
                )
            }
        )
        patch_context = replace(
            base_context,
            latest_observation_id=patch_observation.observation_id,
        )
        rejected = await process_sql_grounding_observation(
            current,
            patch_observation,
            patch_context,
            SQLGroundingUpdater(
                CapturingClient(
                    GroundingLLMResponse(
                        sql_grounding_state=unrelated_mapping,
                        user_clarification_requests=(),
                        next_focus_dimension="none",
                    )
                )
            ),
            grounding_input={
                "query": "Show active maintenance cost.",
                "current_state": complete_state().model_dump(mode="json"),
                "clarification_qa": [answer.model_dump(mode="json")],
                "relevant_column_meanings": {},
                "relevant_knowledge_definitions": [],
            },
        )
        self.assertEqual(rejected.state_update.status, "rejected")
        self.assertEqual(rejected.runtime, current)

        unrelated_knowledge = complete_state().model_copy(
            update={
                "domain_knowledge": (
                    DomainKnowledge(
                        kind="business_rule",
                        content=alternate_rule,
                    ),
                )
            }
        )
        rejected_knowledge = await process_sql_grounding_observation(
            current,
            patch_observation,
            patch_context,
            SQLGroundingUpdater(
                CapturingClient(
                    GroundingLLMResponse(
                        sql_grounding_state=unrelated_knowledge,
                        user_clarification_requests=(),
                        next_focus_dimension="none",
                    )
                )
            ),
            grounding_input={
                "query": "Show active maintenance cost.",
                "current_state": complete_state().model_dump(mode="json"),
                "clarification_qa": [answer.model_dump(mode="json")],
                "relevant_column_meanings": {},
                "relevant_knowledge_definitions": [],
            },
        )
        self.assertEqual(rejected_knowledge.state_update.status, "rejected")
        self.assertEqual(rejected_knowledge.runtime, current)

    def test_json_paths_require_exact_fields_meaning_projection(self):
        known_columns = {
            "operational_metrics.maintcost",
            "operational_metrics.payload",
        }
        paths = grounding_callbacks._column_meaning_json_paths(
            COLUMN_MEANINGS,
            known_columns=known_columns,
        )
        self.assertIn(
            ("operational_metrics.payload", ("cost", "reported")),
            paths,
        )
        context = ValidationContext(
            current_query="Show reported maintenance cost.",
            latest_observation_id="stage2-json-path",
            known_tables=frozenset({"operational_metrics"}),
            known_columns=frozenset(known_columns),
            supported_json_paths=paths,
        )
        valid = SQLGroundingState(
            tables=("operational_metrics",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="reported maintenance cost",
                    targets=(
                        "operational_metrics.payload -> 'cost' ->> 'reported'",
                    ),
                ),
            ),
            domain_knowledge=(),
        )
        self.assertIs(validate_sql_grounding_state(valid, context), valid)
        invalid = valid.model_copy(
            update={
                "column_mapping": (
                    ColumnMapping(
                        phrase="reported maintenance cost",
                        targets=(
                            "operational_metrics.payload -> 'cost' ->> 'invented'",
                        ),
                    ),
                )
            }
        )
        with self.assertRaises(SQLGroundingValidationError):
            validate_sql_grounding_state(invalid, context)


if __name__ == "__main__":
    unittest.main()
