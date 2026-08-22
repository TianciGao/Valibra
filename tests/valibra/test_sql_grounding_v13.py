from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.state import State
from google.genai import types
from pydantic import ValidationError

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    GroundingCheckClarificationProposal,
    GroundingCheckResponse,
    GroundingCheckToolRequest,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    MappingGroundingResponse,
    SQLGroundingState,
    StructureGroundingResponse,
    UserClarificationRequest,
    ValidationContext,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.service import process_sql_grounding_observation
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    SQL_GROUNDING_STAGE_FORM_SCHEMAS,
    SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_STAGE_PROMPTS,
    SQL_GROUNDING_STAGE_PROMPT_SHA256,
    GroundingUpdaterResult,
    classify_grounding_input,
)


QUERY = "Show the maintenance cost for active assets."
PROMPT_SHA = "9d6ebec81d63a7741b6aa031c481d4382d781192c055e18c80fdb4e56ab6f7ed"
FORM_SHA = "728fc43c6ed85e72e60c9ebf85b00487059a2871020a28764b9641c69b84ed81"
CONFIG_SHA = "d2475dac834970e16f232d694a567831991a8e4f116618122b279962006c469c"
STAGE_PROMPT_SHA = {
    "structure": "dacb466200beafb6dfc6ba6d1f8cf3da40cfd0ea791d7f77e8d940f84f6528fd",
    "mapping": "2dced1fcc8aaeb5b22dc5f861209d22f8a21b64613d31d3b4472f1ddd4cde3c3",
    "knowledge": "4f805cabaa53200dfac4b5226d0bc90306e9c35164c8aaf1ae7ad47f56e42e8f",
    "check": "4487a54437fd229ad8f3bf1ebb77f5b48b4db9cd5a1b9505b8e5f3407880c796",
}
STAGE_FORM_SHA = {
    "structure": "d040bb89edcd2331b8ab51e5169dedcc6b87fefadc39abdabcbfd11a2fdcc0b5",
    "mapping": "7a1e8cb588d1c0b89546bfd02b8be0d254ca015cdee753e5e2d23e0da5ea3b44",
    "knowledge": "c80f6dc31b8bc4aad425fb73fe62e361dd06362551827473ee8e9302052ed246",
    "check": "ed4bdaf32e5415be3b4f30bd041c0c4dba4225705736c55bd0b516f5079327e6",
}
SCHEMA_WITH_ROWS = """CREATE TABLE operational_metrics (
  asset_id INTEGER PRIMARY KEY,
  maintcost NUMERIC,
  reported_cost NUMERIC,
  status TEXT,
  payload JSONB
);
First 3 rows:
asset_id | maintcost | reported_cost | status | payload
1 | 4 | 5 | active | {"quality": {"score": 0.98}}
...
"""


def telemetry(kind: str) -> GroundingLLMTelemetry:
    return GroundingLLMTelemetry(
        attempted=True,
        status="succeeded",
        request_sha256="",
        response_sha256="",
        prompt_sha256=SQL_GROUNDING_STAGE_PROMPT_SHA256[kind],
        form_schema_sha256=SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256[kind],
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


class FakeUpdater:
    def __init__(self, response: object, kind: str) -> None:
        self.response = response
        self.kind = kind
        self.calls = 0

    async def propose(self, *args: object, **kwargs: object) -> GroundingUpdaterResult:
        del args, kwargs
        self.calls += 1
        return GroundingUpdaterResult(
            response=self.response,
            call_kind=self.kind,
            telemetry=telemetry(self.kind),
            transport_normalization="none",
        )


def context(observation_id: str, *, rule: str | None = None) -> ValidationContext:
    supported = frozenset({("business_rule", rule)}) if rule else frozenset()
    return ValidationContext(
        current_query=QUERY,
        latest_observation_id=observation_id,
        known_tables=frozenset({"operational_metrics"}),
        known_columns=frozenset(
            {
                "operational_metrics.asset_id",
                "operational_metrics.maintcost",
                "operational_metrics.reported_cost",
                "operational_metrics.status",
            }
        ),
        supported_domain_knowledge=supported,
    )


def complete_state(target: str = "operational_metrics.maintcost") -> SQLGroundingState:
    return SQLGroundingState(
        tables=("operational_metrics",),
        join_keys=(),
        column_mapping=(
            ColumnMapping(phrase="maintenance cost", targets=(target,)),
        ),
        domain_knowledge=(),
    )


class SQLGroundingV13FormTests(unittest.TestCase):
    def test_four_prompts_and_forms_are_distinct_and_hashed(self) -> None:
        self.assertEqual(
            set(SQL_GROUNDING_STAGE_PROMPTS),
            {"structure", "mapping", "knowledge", "check"},
        )
        self.assertEqual(len(set(SQL_GROUNDING_STAGE_PROMPT_SHA256.values())), 4)
        self.assertEqual(len(set(SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256.values())), 4)
        self.assertEqual(SQL_GROUNDING_STAGE_PROMPT_SHA256, STAGE_PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256, STAGE_FORM_SHA)
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)
        for schema in SQL_GROUNDING_STAGE_FORM_SCHEMAS.values():
            self.assertFalse(schema.get("additionalProperties", True))
        knowledge_prompt = SQL_GROUNDING_STAGE_PROMPTS["knowledge"]
        self.assertLess(
            knowledge_prompt.index("选择当前 Query 真正需要的精确 knowledge"),
            knowledge_prompt.index("再用选中的 knowledge 检查并修正"),
        )
        self.assertIn(
            "current column_mapping 只是上一轮的暂定结果",
            knowledge_prompt,
        )
        self.assertIn("不是选择 knowledge 的依据", knowledge_prompt)
        self.assertIn("不生成 SQL", knowledge_prompt)
        mapping_prompt = SQL_GROUNDING_STAGE_PROMPTS["mapping"]
        self.assertIn('"targets": ["..."]', mapping_prompt)
        self.assertIn("字段名必须是 targets，不能是 target", mapping_prompt)
        self.assertIn('必须写成 ["table.column"]', mapping_prompt)
        check_prompt = SQL_GROUNDING_STAGE_PROMPTS["check"]
        self.assertIn(
            "next_tool 必须是上述对象之一或 null，不能是字符串",
            check_prompt,
        )
        self.assertIn(
            "user_clarification_request 只能位于 next_tool 内部",
            check_prompt,
        )
        self.assertIn('"tool_name": "get_column_meaning"', check_prompt)
        self.assertIn('"tool_name": "ask_user"', check_prompt)
        self.assertIn("question 只写一次", check_prompt)
        self.assertIn("只填 phrase 和 kind", check_prompt)
        self.assertIn("必须原样保留，不能随意清空", check_prompt)

    def test_stage_forms_have_only_authorized_fields(self) -> None:
        self.assertEqual(
            set(SQL_GROUNDING_STAGE_FORM_SCHEMAS["structure"]["properties"]),
            {"tables", "join_keys"},
        )
        self.assertEqual(
            set(SQL_GROUNDING_STAGE_FORM_SCHEMAS["mapping"]["properties"]),
            {"tables", "join_keys", "column_mapping"},
        )
        self.assertEqual(
            set(SQL_GROUNDING_STAGE_FORM_SCHEMAS["knowledge"]["properties"]),
            {"column_mapping", "selected_knowledge_ids"},
        )
        self.assertEqual(
            set(SQL_GROUNDING_STAGE_FORM_SCHEMAS["check"]["properties"]),
            {
                "status",
                "missing_information",
                "next_tool",
                "column_mapping",
                "domain_knowledge",
            },
        )
        clarification_schema = SQL_GROUNDING_STAGE_FORM_SCHEMAS["check"]["$defs"][
            "GroundingCheckClarificationProposal"
        ]
        self.assertEqual(
            set(clarification_schema["properties"]),
            {"phrase", "kind"},
        )
        self.assertFalse(clarification_schema.get("additionalProperties", True))

    def test_mapping_rejects_singular_target_field(self) -> None:
        with self.assertRaises(ValidationError):
            MappingGroundingResponse.model_validate(
                {
                    "tables": ["operational_metrics"],
                    "join_keys": [],
                    "column_mapping": [
                        {
                            "phrase": "maintenance cost",
                            "target": "operational_metrics.maintcost",
                        }
                    ],
                }
            )

    def test_knowledge_selection_form_is_strict_and_ids_are_unique(self) -> None:
        empty = KnowledgeGroundingResponse(
            column_mapping=(),
            selected_knowledge_ids=(),
        )
        self.assertEqual(empty.selected_knowledge_ids, ())
        with self.assertRaises(ValidationError):
            KnowledgeGroundingResponse(
                column_mapping=(),
                selected_knowledge_ids=(32, 32),
            )
        with self.assertRaises(ValidationError):
            KnowledgeGroundingResponse.model_validate(
                {
                    "column_mapping": [],
                    "selected_knowledge_ids": [],
                    "domain_knowledge": [],
                }
            )
        with self.assertRaises(ValidationError):
            KnowledgeGroundingResponse.model_validate(
                {
                    "column_mapping": [],
                    "selected_knowledge_ids": [],
                    "tables": [],
                    "join_keys": [],
                }
            )

    def test_check_requires_complete_or_one_concrete_gap_and_tool(self) -> None:
        complete = GroundingCheckResponse(
            status="complete",
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )
        self.assertIsNone(complete.next_tool)
        with self.assertRaises(ValidationError):
            GroundingCheckResponse(
                status="incomplete",
                column_mapping=(),
                domain_knowledge=(),
            )
        with self.assertRaises(ValidationError):
            GroundingCheckToolRequest(
                tool_name="execute_sql",
                arguments={"sql": "DELETE FROM operational_metrics"},
            )
        with self.assertRaises(ValidationError):
            GroundingCheckResponse.model_validate(
                {
                    "status": "incomplete",
                    "missing_information": "Need the exact maintenance field.",
                    "next_tool": "get_column_meaning",
                    "column_mapping": [],
                    "domain_knowledge": [],
                }
            )

    def test_check_ask_user_question_is_materialized_from_arguments_once(self) -> None:
        question = "Which maintenance cost meaning do you intend?"
        response = GroundingCheckResponse.model_validate(
            {
                "status": "incomplete",
                "missing_information": "The intended maintenance metric is unknown.",
                "next_tool": {
                    "tool_name": "ask_user",
                    "arguments": {"question": question},
                    "user_clarification_request": {
                        "phrase": "maintenance cost",
                        "kind": "user_intent",
                    },
                },
                "column_mapping": [],
                "domain_knowledge": [],
            }
        )
        materialized = (
            response.next_tool.materialize_user_clarification_request()
            if response.next_tool is not None
            else None
        )
        self.assertIsNotNone(materialized)
        self.assertEqual(materialized.question, question)
        self.assertEqual(materialized.phrase, "maintenance cost")
        self.assertEqual(materialized.kind, "user_intent")

        with self.assertRaises(ValidationError):
            GroundingCheckToolRequest(
                tool_name="get_column_meaning",
                arguments={
                    "table_name": "operational_metrics",
                    "column_name": "maintcost",
                },
                user_clarification_request=GroundingCheckClarificationProposal(
                    phrase="maintenance cost",
                    kind="user_intent",
                ),
            )
        ordinary = GroundingCheckToolRequest(
            tool_name="get_column_meaning",
            arguments={
                "table_name": "operational_metrics",
                "column_name": "maintcost",
            },
        )
        self.assertIsNone(ordinary.user_clarification_request)
        with self.assertRaises(ValidationError):
            GroundingCheckResponse.model_validate(
                {
                    "status": "incomplete",
                    "missing_information": "Need the user's intended measure.",
                    "next_tool": {
                        "tool_name": "ask_user",
                        "arguments": {"question": "Which measure do you mean?"},
                        "user_clarification_request": {
                            "phrase": "maintenance cost",
                            "kind": "user_intent",
                            "question": "Which measure do you mean?",
                        },
                    },
                    "user_clarification_request": {
                        "phrase": "maintenance cost",
                        "kind": "user_intent",
                        "question": "Which measure do you mean?",
                    },
                    "column_mapping": [],
                    "domain_knowledge": [],
                }
            )

    def test_column_targets_are_joint_requirements_not_candidates(self) -> None:
        text = "\n".join(SQL_GROUNDING_STAGE_PROMPTS.values())
        self.assertIn("targets 不是候选字段集合", text)
        self.assertIn("同一计算或判断确实同时需要多个字段", text)

    def test_classification_accepts_initial_tool_and_answer_check_only(self) -> None:
        base = {"query": QUERY, "current_state": complete_state().model_dump(mode="json")}
        self.assertEqual(
            classify_grounding_input({**base, "check_context": {"kind": "initial"}}, phase=1),
            "check",
        )
        self.assertEqual(
            classify_grounding_input({**base, "latest_tool": {}}, phase=1),
            "check",
        )
        self.assertEqual(
            classify_grounding_input({**base, "latest_user_answer": {}}, phase=1),
            "check",
        )

    def test_structure_request_preserves_raw_schema_rows_values_and_jsonb(self) -> None:
        state = {
            "task_id": "v13-raw-schema",
            "tool_trajectory": [
                {
                    "tool": "get_schema",
                    "args": {},
                    "result": SCHEMA_WITH_ROWS,
                    "phase": 1,
                }
            ],
        }
        payload = grounding_callbacks._build_staged_grounding_request(
            state,
            call_kind="structure",
            query=QUERY,
            runtime=GroundingRuntime(),
            phase=1,
        )
        self.assertEqual(payload["schema"], SCHEMA_WITH_ROWS)
        self.assertIn("First 3 rows", payload["schema"])
        self.assertIn("active", payload["schema"])
        self.assertIn('{"quality": {"score": 0.98}}', payload["schema"])


class SQLGroundingV13ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_can_replace_a_with_b_without_structure_change(self) -> None:
        rule = "Use reported_cost when the user asks for maintenance cost."
        old = complete_state("operational_metrics.maintcost").model_copy(
            update={"domain_knowledge": None}
        )
        runtime = GroundingRuntime(
            grounding_revision=2,
            stage="INITIAL_GROUNDING",
            focus_dimension="domain_knowledge",
            grounding_state=old,
        )
        response = KnowledgeGroundingResponse(
            column_mapping=(
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("operational_metrics.reported_cost",),
                ),
            ),
            selected_knowledge_ids=(32,),
        )
        observation = build_sql_grounding_observation(
            task_id="v13-a-b",
            phase=1,
            sequence=3,
            observation_type="knowledge",
            content=[{"id": 32, "definition": rule}],
            summary="knowledge",
            tool_name="get_all_knowledge_definitions",
            function_call_id="v13-a-b-call",
        )
        bundle = {
            "query": QUERY,
            "current_state": old.model_dump(mode="json"),
            "knowledge_definitions": [{"id": 32, "definition": rule}],
            "relevant_column_meanings": {
                "operational_metrics": {
                    "maintcost": "legacy value",
                    "reported_cost": "reported maintenance cost",
                }
            },
        }
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context(observation.observation_id, rule=rule),
            FakeUpdater(response, "knowledge"),
            grounding_input=bundle,
        )
        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(result.state_update.changed_dimensions, ("column_mapping", "domain_knowledge"))
        self.assertEqual(result.runtime.grounding_state.tables, old.tables)
        self.assertEqual(result.runtime.grounding_state.join_keys, old.join_keys)
        self.assertEqual(
            result.runtime.grounding_state.domain_knowledge,
            (DomainKnowledge(kind="business_rule", content=rule),),
        )
        self.assertEqual(
            result.runtime.grounding_state.domain_knowledge[0].content,
            rule,
        )

    async def test_knowledge_selection_rejects_unknown_id_atomically(self) -> None:
        rule = "Use reported_cost when the user asks for maintenance cost."
        old = complete_state().model_copy(update={"domain_knowledge": None})
        runtime = GroundingRuntime(
            grounding_revision=2,
            stage="INITIAL_GROUNDING",
            focus_dimension="domain_knowledge",
            grounding_state=old,
        )
        observation = build_sql_grounding_observation(
            task_id="v13-missing-knowledge-id",
            phase=1,
            sequence=3,
            observation_type="knowledge",
            content=[{"id": 32, "definition": rule}],
            summary="knowledge",
            tool_name="get_all_knowledge_definitions",
            function_call_id="v13-missing-knowledge-id-call",
        )
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context(observation.observation_id, rule=rule),
            FakeUpdater(
                KnowledgeGroundingResponse(
                    column_mapping=old.column_mapping or (),
                    selected_knowledge_ids=(33,),
                ),
                "knowledge",
            ),
            grounding_input={
                "query": QUERY,
                "current_state": old.model_dump(mode="json"),
                "knowledge_definitions": [{"id": 32, "definition": rule}],
                "relevant_column_meanings": {},
            },
        )
        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.runtime, runtime)

    async def test_knowledge_selection_rejects_duplicate_official_ids(self) -> None:
        rule = "Use reported_cost when the user asks for maintenance cost."
        old = complete_state().model_copy(update={"domain_knowledge": None})
        runtime = GroundingRuntime(
            grounding_revision=2,
            stage="INITIAL_GROUNDING",
            focus_dimension="domain_knowledge",
            grounding_state=old,
        )
        observation = build_sql_grounding_observation(
            task_id="v13-duplicate-official-id",
            phase=1,
            sequence=3,
            observation_type="knowledge",
            content=[
                {"id": 32, "definition": rule},
                {"id": 32, "definition": "Another exact definition."},
            ],
            summary="knowledge",
            tool_name="get_all_knowledge_definitions",
            function_call_id="v13-duplicate-official-id-call",
        )
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context(observation.observation_id, rule=rule),
            FakeUpdater(
                KnowledgeGroundingResponse(
                    column_mapping=old.column_mapping or (),
                    selected_knowledge_ids=(32,),
                ),
                "knowledge",
            ),
            grounding_input={
                "query": QUERY,
                "current_state": old.model_dump(mode="json"),
                "knowledge_definitions": [
                    {"id": 32, "definition": rule},
                    {"id": 32, "definition": "Another exact definition."},
                ],
                "relevant_column_meanings": {},
            },
        )
        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.runtime, runtime)

    async def test_check_updates_only_mapping_and_knowledge(self) -> None:
        old = complete_state()
        runtime = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=old,
        )
        response = GroundingCheckResponse(
            status="complete",
            column_mapping=old.column_mapping or (),
            domain_knowledge=(),
        )
        observation = build_sql_grounding_observation(
            task_id="v13-check",
            phase=1,
            sequence=4,
            observation_type="knowledge",
            content=[],
            summary="check",
            tool_name="get_all_knowledge_definitions",
            function_call_id="v13-check-call",
        )
        bundle = {
            "query": QUERY,
            "current_state": old.model_dump(mode="json"),
            "check_context": {"kind": "initial"},
        }
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context(observation.observation_id),
            FakeUpdater(response, "check"),
            grounding_input=bundle,
        )
        self.assertEqual(result.state_update.status, "noop")
        self.assertEqual(result.runtime.grounding_revision, 3)

    async def test_check_can_replace_a_with_official_candidate_table_field_b(self) -> None:
        old = complete_state("operational_metrics.maintcost")
        runtime = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=old,
        )
        response = GroundingCheckResponse(
            status="complete",
            column_mapping=(
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("operational_metrics.reported_cost",),
                ),
            ),
            domain_knowledge=(),
        )
        observation = build_sql_grounding_observation(
            task_id="v13-check-a-b",
            phase=1,
            sequence=5,
            observation_type="user_answer",
            content="Use the reported maintenance cost.",
            summary="answered bounded Check clarification",
            tool_name="ask_user",
            function_call_id="v13-check-a-b-call",
        )
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context(observation.observation_id),
            FakeUpdater(response, "check"),
            grounding_input={
                "query": QUERY,
                "current_state": old.model_dump(mode="json"),
                "latest_user_answer": {
                    "question": "Which maintenance cost meaning do you intend?",
                    "answer": "Use the reported maintenance cost.",
                },
            },
        )
        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(result.state_update.changed_dimensions, ("column_mapping",))
        self.assertEqual(result.runtime.grounding_state.tables, old.tables)
        self.assertEqual(result.runtime.grounding_state.join_keys, old.join_keys)
        self.assertEqual(
            result.runtime.grounding_state.column_mapping[0].targets,
            ("operational_metrics.reported_cost",),
        )


class SQLGroundingV13CallbackContractTests(unittest.IsolatedAsyncioTestCase):
    def state(self, *, budget: float = 10.0) -> dict[str, object]:
        return {
            "task_id": "v13-task",
            "current_phase": 1,
            "budget_remaining": budget,
            "tool_trajectory": [],
            "system_agent_llm_calls": [],
        }

    def incomplete_check(self, *, tool: str = "get_column_meaning") -> GroundingCheckResponse:
        args = (
            {"table_name": "operational_metrics", "column_name": "reported_cost"}
            if tool == "get_column_meaning"
            else {"question": "Which maintenance cost meaning do you intend?"}
        )
        clarification = (
            GroundingCheckClarificationProposal(
                phrase="maintenance cost",
                kind="user_intent",
            )
            if tool == "ask_user"
            else None
        )
        return GroundingCheckResponse(
            status="incomplete",
            missing_information="The exact maintenance cost field is unresolved.",
            next_tool=GroundingCheckToolRequest(
                tool_name=tool,
                arguments=args,
                user_clarification_request=clarification,
            ),
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )

    def adk_check_state(self) -> State:
        value = self.state()
        value.update(
            {
                "initial_budget": 10.0,
                grounding_callbacks.GROUNDING_RUNTIME_KEY: GroundingRuntime(
                    grounding_revision=3,
                    stage="INITIAL_GROUNDING",
                    focus_dimension="none",
                    grounding_state=complete_state(),
                ).model_dump(mode="json"),
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
                        "result": SCHEMA_WITH_ROWS,
                    }
                ],
            }
        )
        return State(value, {})

    async def run_check_tool(
        self,
        state: State,
        *,
        updater: FakeUpdater,
        build_error: BaseException | None = None,
    ) -> tuple[object, object, AsyncMock]:
        pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=self.incomplete_check(),
        )
        tool = SimpleNamespace(name=pending.tool_name)
        context = SimpleNamespace(
            state=state,
            function_call_id=pending.function_call_id,
            invocation_id="v13-check-exactly-once",
        )
        baseline_before = AsyncMock(return_value=None)

        async def baseline_after(
            _tool: object,
            args: dict,
            tool_context: object,
            tool_response: object,
        ) -> object:
            self.assertIsNone(
                grounding_callbacks._pending_check_tool(tool_context.state)
            )
            trajectory = list(tool_context.state.get("tool_trajectory", []))
            trajectory.append(
                {
                    "type": "tool",
                    "tool": pending.tool_name,
                    "phase": 1,
                    "args": args,
                    "result": tool_response,
                }
            )
            tool_context.state["tool_trajectory"] = trajectory
            return "baseline-after"

        build_patch = (
            patch.object(
                grounding_callbacks,
                "_build_check_grounding_request",
                side_effect=build_error,
            )
            if build_error is not None
            else patch.object(
                grounding_callbacks,
                "_build_check_grounding_request",
                wraps=grounding_callbacks._build_check_grounding_request,
            )
        )
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
                    return_value=SimpleNamespace(max_calls_per_task=8),
                ),
                patch.object(
                    baseline_callbacks,
                    "before_tool_callback",
                    baseline_before,
                ),
                patch.object(
                    baseline_callbacks,
                    "after_tool_callback",
                    side_effect=baseline_after,
                ),
                build_patch,
            ):
                await grounding_callbacks.before_tool_callback(
                    tool,
                    pending.arguments,
                    context,
                )
                state["budget_remaining"] = 9.5
                returned = await grounding_callbacks.after_tool_callback(
                    tool,
                    pending.arguments,
                    context,
                    "reported maintenance cost",
                )
        finally:
            grounding_callbacks._reset_turn_message(token)
        return pending, returned, baseline_before

    def test_check_requires_budget_after_call_at_least_six(self) -> None:
        state = self.state(budget=6.4)
        with self.assertRaisesRegex(ValueError, "budget_exhausted"):
            grounding_callbacks._schedule_check_tool(
                state,
                phase=1,
                response=self.incomplete_check(),
            )
        audit = grounding_callbacks._check_audits(state)[0]
        self.assertEqual(audit["blocked_reason"], "budget_exhausted")

    async def test_check_official_result_is_consumed_once_and_enters_next_check(
        self,
    ) -> None:
        state = self.adk_check_state()
        complete = GroundingCheckResponse(
            status="complete",
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )
        updater = FakeUpdater(complete, "check")

        pending, returned, baseline_before = await self.run_check_tool(
            state,
            updater=updater,
        )

        self.assertEqual(returned, "baseline-after")
        self.assertEqual(baseline_before.await_count, 1)
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))
        self.assertEqual(updater.calls, 1)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY],
            4,
        )
        self.assertEqual(
            [
                item["tool"]
                for item in state["tool_trajectory"]
                if item["tool"] == pending.tool_name
            ],
            [pending.tool_name],
        )
        self.assertTrue(grounding_callbacks._phase_grounding_succeeded(state, 1))

    async def test_check_after_tool_error_consumes_once_and_fails_closed(
        self,
    ) -> None:
        state = self.adk_check_state()
        updater = FakeUpdater(
            GroundingCheckResponse(
                status="complete",
                column_mapping=complete_state().column_mapping or (),
                domain_knowledge=(),
            ),
            "check",
        )
        pending, returned, baseline_before = await self.run_check_tool(
            state,
            updater=updater,
            build_error=AttributeError("synthetic Check result failure"),
        )

        self.assertEqual(returned, "baseline-after")
        self.assertEqual(baseline_before.await_count, 1)
        self.assertEqual(state["budget_remaining"], 9.5)
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))
        self.assertEqual(updater.calls, 0)
        failure = grounding_callbacks._phase_grounding_failure(state, 1)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.error_type, "AttributeError")
        error = state[grounding_callbacks.GROUNDING_ERROR_AUDIT_KEY][-1]
        self.assertEqual(error["service_status"], "failed_closed")
        self.assertEqual(error["error_type"], "AttributeError")
        self.assertEqual(error["function_call_id"], pending.function_call_id)

        before_model = AsyncMock(return_value=None)
        token = grounding_callbacks._bind_turn_message(
            state["task_id"],
            "a-interact",
            QUERY,
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                patch.object(
                    baseline_callbacks,
                    "before_model_callback",
                    before_model,
                ),
            ):
                response = await grounding_callbacks.before_model_callback(
                    SimpleNamespace(state=state),
                    LlmRequest(
                        contents=[
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=QUERY)],
                            )
                        ],
                        config=types.GenerateContentConfig(),
                    ),
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(state["budget_remaining"], 9.5)
        self.assertIsNotNone(response)
        self.assertIn(
            "VALIBRA_SQL_GROUNDING_FAILED_CLOSED",
            response.content.parts[0].text,
        )
        self.assertIsNone(response.content.parts[0].function_call)
        self.assertEqual(baseline_before.await_count, 1)

        with self.assertRaisesRegex(ValueError, "new concrete gap"):
            grounding_callbacks._schedule_check_tool(
                state,
                phase=1,
                response=self.incomplete_check(),
            )
        same_tool_new_gap = self.incomplete_check().model_copy(
            update={"missing_information": "A different concrete gap."}
        )
        with self.assertRaisesRegex(ValueError, "identical arguments"):
            grounding_callbacks._schedule_check_tool(
                state,
                phase=1,
                response=same_tool_new_gap,
            )

        duplicate = grounding_callbacks._PendingToolCall(
            function_call_id=pending.function_call_id,
            tool_name=pending.tool_name,
            phase_before=1,
            args_digest="0" * 64,
            args_summary="{}",
            sequence=99,
        )
        grounding_callbacks._add_pending(state, duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate pending function_call_id"):
            grounding_callbacks._add_pending(state, duplicate)
        grounding_callbacks._pop_pending(state, pending.function_call_id)

    def test_check_rejects_repeated_gap_and_duplicate_call(self) -> None:
        state = self.state()
        response = self.incomplete_check()
        grounding_callbacks._schedule_check_tool(state, phase=1, response=response)
        grounding_callbacks._store_pending_check_tool(state, None)
        with self.assertRaisesRegex(ValueError, "new concrete gap"):
            grounding_callbacks._schedule_check_tool(state, phase=1, response=response)

    def test_ask_user_is_registered_outside_state_and_one_tool_is_pending(self) -> None:
        state = self.state()
        response = self.incomplete_check(tool="ask_user")
        pending = grounding_callbacks._schedule_check_tool(state, phase=1, response=response)
        self.assertEqual(pending.tool_name, "ask_user")
        self.assertNotIn("clarification", complete_state().model_dump())
        records = grounding_callbacks._clarification_records(state)
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].answer)

    def test_answer_resumes_check_with_only_latest_qa_and_allows_new_question(self) -> None:
        state = self.state()
        first = self.incomplete_check(tool="ask_user")
        pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=first,
        )
        answer = "Use the reported maintenance cost."
        grounding_callbacks._record_clarification_answer(
            state,
            phase=1,
            question=pending.arguments["question"],
            answer=answer,
        )
        payload = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            runtime=GroundingRuntime(
                grounding_revision=3,
                focus_dimension="none",
                grounding_state=complete_state(),
            ),
            phase=1,
            latest_user_answer=answer,
        )
        self.assertEqual(
            set(payload),
            {"query", "current_state", "latest_user_answer"},
        )
        self.assertEqual(payload["latest_user_answer"]["answer"], answer)
        grounding_callbacks._store_pending_check_tool(state, None)

        second_question = "Which maintenance rule should be applied?"
        second = GroundingCheckResponse(
            status="incomplete",
            missing_information="The maintenance rule is still unresolved.",
            next_tool=GroundingCheckToolRequest(
                tool_name="ask_user",
                arguments={"question": second_question},
                user_clarification_request=GroundingCheckClarificationProposal(
                    phrase="maintenance cost",
                    kind="missing_knowledge",
                ),
            ),
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )
        grounding_callbacks._schedule_check_tool(state, phase=1, response=second)
        self.assertEqual(
            [item.question for item in grounding_callbacks._clarification_records(state)],
            [first.next_tool.arguments["question"], second_question],
        )

    def test_latest_check_request_does_not_accumulate_raw_history(self) -> None:
        state = self.state()
        state[grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY] = []
        payload = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            runtime=GroundingRuntime(
                grounding_revision=3,
                focus_dimension="none",
                grounding_state=complete_state(),
            ),
            phase=1,
            latest_tool_name="get_column_meaning",
            latest_tool_arguments={"table_name": "operational_metrics", "column_name": "reported_cost"},
            latest_tool_result="reported maintenance cost",
        )
        self.assertEqual(set(payload), {"query", "current_state", "latest_tool"})
        self.assertNotIn("trajectory", json.dumps(payload))

    def test_sql_writer_replaces_prompt_and_exposes_exactly_two_tools(self) -> None:
        names = ["execute_sql", "get_schema", "ask_user", "submit_sql"]
        request = LlmRequest(
            config=types.GenerateContentConfig(
                system_instruction="OLD EXPLORATION CONTROL HINT",
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
        grounding_callbacks._inject_sql_writer_context(
            request,
            phase=1,
            original_query=QUERY,
            follow_up=None,
            view_text="[VALIBRA DATABASE GROUNDING]",
            budget_remaining=7,
        )
        self.assertEqual(tuple(request.tools_dict), ("execute_sql", "submit_sql"))
        self.assertEqual(
            [item.name for item in request.config.tools[0].function_declarations],
            ["execute_sql", "submit_sql"],
        )
        self.assertNotIn("OLD EXPLORATION", request.config.system_instruction)
        self.assertNotIn("VALIBRA CONTROL", request.config.system_instruction)

    def test_sql_writer_hides_raw_bootstrap_but_keeps_query_and_execution_history(self) -> None:
        request = LlmRequest(
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=QUERY)]),
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="bootstrap-schema",
                                name="get_schema",
                                args={},
                            )
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="bootstrap-schema",
                                name="get_schema",
                                response={"result": SCHEMA_WITH_ROWS},
                            )
                        )
                    ],
                ),
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="execute-one",
                                name="execute_sql",
                                args={"sql": "SELECT 1"},
                            )
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="execute-one",
                                name="execute_sql",
                                response={"result": [[1]]},
                            )
                        )
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(name="get_schema"),
                            types.FunctionDeclaration(name="execute_sql"),
                            types.FunctionDeclaration(name="submit_sql"),
                        ]
                    )
                ]
            ),
        )
        request.tools_dict = {
            "get_schema": SimpleNamespace(),
            "execute_sql": SimpleNamespace(),
            "submit_sql": SimpleNamespace(),
        }
        grounding_callbacks._inject_sql_writer_context(
            request,
            phase=1,
            original_query=QUERY,
            follow_up=None,
            view_text="[VALIBRA DATABASE GROUNDING]",
            budget_remaining=7,
        )
        serialized = json.dumps(
            [item.model_dump(mode="json") for item in request.contents],
            sort_keys=True,
        )
        self.assertIn(QUERY, serialized)
        self.assertIn("execute_sql", serialized)
        self.assertNotIn("get_schema", serialized)
        self.assertNotIn("First 3 rows", serialized)
        self.assertNotIn('{"quality": {"score": 0.98}}', serialized)

    def test_main_execution_observation_does_not_change_frozen_state(self) -> None:
        state = complete_state()
        before = sql_grounding_state_sha256(state)
        runtime = GroundingRuntime(
            grounding_revision=4,
            stage="SQL_ATTEMPT",
            focus_dimension="none",
            grounding_state=state,
        )
        self.assertEqual(sql_grounding_state_sha256(runtime.grounding_state), before)
        self.assertNotIn("REPAIR", json.dumps(GroundingRuntime.model_json_schema()))


if __name__ == "__main__":
    unittest.main()
