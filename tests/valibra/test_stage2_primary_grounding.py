from __future__ import annotations

import copy
import json
import os
import unittest
from collections.abc import AsyncGenerator
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
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=complete_state(),
                next_focus_dimension="none",
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
    async def test_real_adk_three_tools_trigger_exactly_one_primary_update(self):
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
                    return_value=SimpleNamespace(max_calls_per_task=4),
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
        self.assertEqual(updater.calls, 1)
        self.assertEqual(
            session_state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            1,
        )
        self.assertEqual(set(updater.inputs[0]), {
            "query",
            "schema",
            "column_meanings",
            "knowledge_definitions",
            "current_state",
        })
        self.assertEqual(updater.inputs[0]["schema"], SCHEMA)
        self.assertEqual(updater.inputs[0]["column_meanings"], COLUMN_MEANINGS)
        self.assertEqual(
            updater.inputs[0]["knowledge_definitions"], KNOWLEDGE_DEFINITIONS
        )
        runtime = GroundingRuntime.model_validate(
            session_state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.grounding_revision, 1)
        self.assertEqual(runtime.grounding_state, complete_state())
        self.assertEqual(runtime.stage, "SQL_ATTEMPT")
        self.assertEqual(runtime.focus_dimension, "none")
        self.assertEqual(session_state["budget_remaining"], 7.0)
        self.assertEqual(model.calls, 1)

        audits = session_state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY]
        ordered = [audits[key][grounding_callbacks.SHADOW_AUDIT_KEY] for key in sorted(audits)]
        self.assertEqual(
            [item["service_status"] for item in ordered],
            ["stored_bootstrap_evidence", "stored_bootstrap_evidence", "accepted"],
        )
        self.assertEqual(
            ordered[-1]["changed_dimensions"],
            ["tables", "join_keys", "column_mapping", "domain_knowledge"],
        )
        self.assertTrue(ordered[-1]["provider_attempted"])
        self.assertTrue(ordered[-1]["primary_grounding_triggered"])

        model_visible = canonical_json(model.requests)
        self.assertNotIn(SCHEMA_SENTINEL, model_visible)
        self.assertNotIn(MEANINGS_SENTINEL, model_visible)
        self.assertNotIn(COLUMN_MEANINGS, model_visible)

    async def test_updater_receives_exact_five_fields_and_service_is_atomic(self):
        runtime = GroundingRuntime()
        observation = primary_observation()
        context = validation_context(observation)
        valid_response = GroundingLLMResponse(
            sql_grounding_state=complete_state(),
            next_focus_dimension="none",
        )
        client = CapturingClient(valid_response)
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context,
            SQLGroundingUpdater(client),
            grounding_input=primary_input(runtime),
        )
        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(result.runtime.grounding_revision, 1)
        request_payload = json.loads(client.requests[0].input_json)
        self.assertEqual(request_payload, primary_input(runtime))
        self.assertNotIn("latest_observation", request_payload)
        self.assertEqual(client.requests[0].observation_id, observation.observation_id)

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
                next_focus_dimension="none",
            )
        )
        rejected = await process_sql_grounding_observation(
            runtime,
            observation,
            context,
            SQLGroundingUpdater(invalid_client),
            grounding_input=primary_input(runtime),
        )
        self.assertEqual(rejected.state_update.status, "rejected")
        self.assertEqual(rejected.runtime, runtime)
        self.assertEqual(rejected.runtime.grounding_revision, 0)

    async def test_incomplete_bundle_is_rejected_before_client_call(self):
        runtime = GroundingRuntime()
        observation = primary_observation()
        client = CapturingClient(
            GroundingLLMResponse(
                sql_grounding_state=complete_state(),
                next_focus_dimension="none",
            )
        )
        incomplete = primary_input(runtime)
        incomplete.pop("knowledge_definitions")
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
