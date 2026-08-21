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
    ValidationContext,
    validate_sql_grounding_state,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
)


QUERY = "Show the maintenance cost."
SCHEMA_SENTINEL = "stage1_private_schema_sentinel"
MEANINGS_SENTINEL = "stage1_private_meanings_sentinel"
KNOWLEDGE_SENTINEL = "stage1_private_knowledge_sentinel"
SCHEMA = f"""CREATE TABLE operational_metrics (
  maintcost NUMERIC,
  source_label TEXT
);
-- {SCHEMA_SENTINEL}
"""
COLUMN_MEANINGS = json.dumps(
    {
        "operational_metrics": {
            "maintcost": f"Maintenance cost; {MEANINGS_SENTINEL}",
            "source_label": "Source label",
        }
    },
    sort_keys=True,
)
KNOWLEDGE_ITEMS = [
    {
        "id": "k1",
        "knowledge": "ratio_rule",
        "description": "Ratio definition",
        "definition": f"Use maintcost as reported; {KNOWLEDGE_SENTINEL}",
    },
    {
        "id": "k2",
        "knowledge": "currency_rule",
        "description": "Currency definition",
        "definition": "All maintenance costs are denominated in USD.",
    },
]
KNOWLEDGE_DEFINITIONS = json.dumps(KNOWLEDGE_ITEMS, sort_keys=True)


def _state(task_id: str) -> dict[str, Any]:
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


class _StagedOfflineUpdater:
    calls = 0

    def __init__(self) -> None:
        self.inputs: list[dict[str, Any]] = []

    async def propose(self, runtime: GroundingRuntime, *args: Any, **kwargs: Any) -> Any:
        del args
        self.calls += 1
        grounding_input = copy.deepcopy(kwargs["grounding_input"])
        self.inputs.append(grounding_input)
        fields = set(grounding_input)
        if "schema" in fields:
            candidate = SQLGroundingState(
                tables=("operational_metrics",),
                join_keys=(),
            )
            focus = "column_mapping"
        elif "column_meanings" in fields:
            candidate = SQLGroundingState(
                tables=("operational_metrics",),
                join_keys=(),
                column_mapping=(
                    ColumnMapping(
                        phrase="maintenance cost",
                        targets=("operational_metrics.maintcost",),
                    ),
                ),
            )
            focus = "domain_knowledge"
        else:
            candidate = SQLGroundingState(
                tables=("operational_metrics",),
                join_keys=(),
                column_mapping=(
                    ColumnMapping(
                        phrase="maintenance cost",
                        targets=("operational_metrics.maintcost",),
                    ),
                ),
                domain_knowledge=(),
            )
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
                request_sha256="",
                response_sha256="",
                prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
                form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


class _FinalLocalModel(BaseLlm):
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
                parts=[types.Part.from_text(text="STAGE1_LOCAL_DONE")],
            )
        )


class Stage1BootstrapEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_adk_interleaves_three_tools_with_staged_grounding(self):
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

        model = _FinalLocalModel(model="stage1-local")
        updater = _StagedOfflineUpdater()
        agent = LlmAgent(
            name="stage1_bootstrap_agent",
            model=model,
            instruction="Offline Stage 1 lifecycle test.",
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
        app_name = "stage1_bootstrap"
        user_id = "stage1-user"
        runner = InMemoryRunner(agent=agent, app_name=app_name)
        session = await runner.session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            state=_state("stage1-bootstrap-task"),
        )
        turn_token = grounding_callbacks._bind_turn_message(
            "stage1-bootstrap-task", "a-interact", QUERY
        )
        try:
            with (
                patch.dict(
                    os.environ,
                    {"GROUNDING_UPDATER_MODE": "llm"},
                    clear=False,
                ),
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
                        user_id=user_id,
                        session_id=session.id,
                        new_message=types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=QUERY)],
                        ),
                    )
                ]
        finally:
            grounding_callbacks._reset_turn_message(turn_token)
        final = await runner.session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
        )
        self.assertIsNotNone(final)
        state = final.state

        self.assertEqual(
            executions,
            list(grounding_callbacks._BOOTSTRAP_TOOL_SEQUENCE),
        )
        self.assertEqual(updater.calls, 3)
        self.assertEqual(
            state.get(grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY, 0),
            3,
        )
        self.assertEqual(model.calls, 1)
        phase_outcome = state[
            grounding_callbacks.GROUNDING_PHASE_OUTCOMES_KEY
        ]["1"]
        self.assertEqual(phase_outcome["status"], "succeeded")
        self.assertEqual(state["budget_remaining"], 7.0)
        trajectory = state["tool_trajectory"]
        self.assertEqual([item["tool"] for item in trajectory], executions)
        self.assertEqual([item["cost"] for item in trajectory], [1.0, 1.0, 1.0])
        self.assertEqual(
            [item["result"] for item in trajectory],
            [SCHEMA, COLUMN_MEANINGS, KNOWLEDGE_DEFINITIONS],
        )
        exact_audits = state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY]
        self.assertEqual(len(exact_audits), 3)
        ordered_audits = [exact_audits[key] for key in sorted(exact_audits)]
        self.assertEqual(
            [
                record[grounding_callbacks.SHADOW_AUDIT_KEY]["service_status"]
                for record in ordered_audits
            ],
            ["accepted", "accepted", "accepted"],
        )
        self.assertTrue(
            all(
                record[grounding_callbacks.SHADOW_AUDIT_KEY][
                    "provider_attempted"
                ]
                for record in ordered_audits
            )
        )
        self.assertTrue(
            ordered_audits[-1][grounding_callbacks.SHADOW_AUDIT_KEY][
                "staged_grounding_triggered"
            ]
        )
        self.assertEqual(state[grounding_callbacks.GROUNDING_PENDING_KEY], {})

        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.grounding_revision, 3)
        self.assertEqual(runtime.stage, "SQL_ATTEMPT")
        self.assertTrue(runtime.grounding_state.all_dimensions_evaluated)
        request = grounding_callbacks._build_staged_grounding_request(
            state,
            call_kind="knowledge",
            query=QUERY,
            runtime=runtime,
            phase=1,
        )
        self.assertEqual(set(request), {
            "query",
            "current_state",
            "knowledge_definitions",
            "relevant_column_meanings",
        })
        self.assertEqual(
            [set(item) for item in updater.inputs],
            [
                {"query", "current_state", "schema"},
                {"query", "current_state", "column_meanings"},
                {
                    "query",
                    "current_state",
                    "knowledge_definitions",
                    "relevant_column_meanings",
                },
            ],
        )

        model_visible = json.dumps(model.requests, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(SCHEMA_SENTINEL, model_visible)
        self.assertNotIn(MEANINGS_SENTINEL, model_visible)
        self.assertNotIn(KNOWLEDGE_SENTINEL, model_visible)
        self.assertTrue(model.requests)

        trajectory_only = json.dumps(trajectory, ensure_ascii=False, sort_keys=True)
        self.assertIn(SCHEMA_SENTINEL, trajectory_only)
        self.assertIn(MEANINGS_SENTINEL, trajectory_only)
        self.assertIn(KNOWLEDGE_SENTINEL, trajectory_only)
        state_without_official_audits = dict(state)
        state_without_official_audits.pop("tool_trajectory", None)
        state_without_official_audits.pop("system_agent_llm_calls", None)
        private_elsewhere = json.dumps(
            to_jsonable(state_without_official_audits),
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertNotIn(SCHEMA_SENTINEL, private_elsewhere)
        self.assertNotIn(MEANINGS_SENTINEL, private_elsewhere)
        self.assertNotIn(KNOWLEDGE_SENTINEL, private_elsewhere)

    def test_bulk_definitions_project_individually_into_validation_context(self):
        state = _state("stage1-projection-task")
        state["tool_trajectory"] = [
            {"tool": "get_schema", "result": SCHEMA},
            {"tool": "get_all_column_meanings", "result": COLUMN_MEANINGS},
            {
                "tool": "get_all_knowledge_definitions",
                "result": KNOWLEDGE_DEFINITIONS,
            },
        ]
        definitions = grounding_callbacks._exact_knowledge_definitions(
            KNOWLEDGE_DEFINITIONS
        )
        self.assertEqual(
            definitions,
            tuple(item["definition"] for item in KNOWLEDGE_ITEMS),
        )
        known_tables: set[str] = set()
        known_columns: set[str] = set()
        supported: set[tuple[str, str]] = set()
        grounding_callbacks._project_official_evidence(
            tool_name="get_all_knowledge_definitions",
            observation_type="knowledge",
            content=KNOWLEDGE_DEFINITIONS,
            known_tables=known_tables,
            known_columns=known_columns,
            supported_knowledge=supported,
        )
        self.assertEqual(
            supported,
            {("business_rule", item["definition"]) for item in KNOWLEDGE_ITEMS},
        )
        context = ValidationContext(
            current_query=QUERY,
            latest_observation_id="stage1-bulk-knowledge",
            known_tables=frozenset({"operational_metrics"}),
            known_columns=frozenset(
                {
                    "operational_metrics.maintcost",
                    "operational_metrics.source_label",
                }
            ),
            supported_domain_knowledge=frozenset(supported),
        )
        for item in KNOWLEDGE_ITEMS:
            self.assertIn(
                ("business_rule", item["definition"]),
                context.supported_domain_knowledge,
            )
            validate_sql_grounding_state(
                SQLGroundingState(
                    tables=(),
                    join_keys=(),
                    column_mapping=(),
                    domain_knowledge=(
                        DomainKnowledge(
                            kind="business_rule",
                            content=item["definition"],
                        ),
                    ),
                ),
                context,
            )

    def test_bulk_parsers_and_bundle_fail_closed(self):
        invalid_bulk_values = (
            "not-json",
            "{}",
            '[{"definition":"ok","extra":"forbidden"}]',
            '[{"id":"missing-definition"}]',
            '["not-an-object"]',
            '[{"definition":"first","definition":"second"}]',
            '[{"definition":"ok","description":NaN}]',
        )
        for value in invalid_bulk_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    grounding_callbacks._exact_knowledge_definitions(value)

        state = _state("stage1-incomplete-task")
        state["tool_trajectory"] = [
            {"tool": "get_schema", "result": SCHEMA},
            {"tool": "get_all_column_meanings", "result": COLUMN_MEANINGS},
        ]
        with self.assertRaisesRegex(ValueError, "requires bootstrap knowledge"):
            grounding_callbacks._build_staged_grounding_request(
                state,
                call_kind="knowledge",
                query=QUERY,
                runtime=GroundingRuntime(),
                phase=1,
            )

        state["tool_trajectory"].append(
            {
                "tool": "get_all_knowledge_definitions",
                "result": "Error: unavailable",
            }
        )
        with self.assertRaisesRegex(ValueError, "Official tool error"):
            grounding_callbacks._build_staged_grounding_request(
                state,
                call_kind="knowledge",
                query=QUERY,
                runtime=GroundingRuntime(),
                phase=1,
            )

    def test_no_runtime_or_evidence_store_field_was_added(self):
        runtime_payload = GroundingRuntime().model_dump(mode="json")
        self.assertEqual(
            set(runtime_payload),
            {"grounding_revision", "stage", "focus_dimension", "grounding_state"},
        )
        self.assertFalse(
            any("evidence" in key.lower() for key in runtime_payload)
        )


if __name__ == "__main__":
    unittest.main()
