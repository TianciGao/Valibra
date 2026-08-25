from __future__ import annotations

import copy
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
    UserClarificationRecord,
    UserClarificationRequest,
    ValidationContext,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.prompt_view import render_grounding_view
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
PROMPT_SHA = "102aa853ad0baef50a3d4f7841846348db1aa78ab1a19080e8d491a4cbf39bac"
FORM_SHA = "728fc43c6ed85e72e60c9ebf85b00487059a2871020a28764b9641c69b84ed81"
CONFIG_SHA = "fa3f4fd5f61546c0bc9fee9e2bf586001f4f82bba1d59807aaa783e3d1c02501"
WRITER_PROMPT_SHA = "61deab4ea63bdef3a8c511a0a625abdbe87969f9df36130f76344d7bcc8ea711"
STAGE_PROMPT_SHA = {
    "structure": "dacb466200beafb6dfc6ba6d1f8cf3da40cfd0ea791d7f77e8d940f84f6528fd",
    "mapping": "2dced1fcc8aaeb5b22dc5f861209d22f8a21b64613d31d3b4472f1ddd4cde3c3",
    "knowledge": "72ad5fd63b7a0e7a9107231326d2cd2468609ae1a275ab31d6c9c81f3b0c899c",
    "check": "446f45c12a7d954b438d8a913a8171c0b99e2d6f0aed59f627e407171769f5b2",
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
EMPTY_CHECK_PHASE_CONTEXT = {
    "previous_official_calls": [],
    "answered_clarifications": [],
}


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
        self.assertIn("previous_official_calls", check_prompt)
        self.assertIn("answered_clarifications", check_prompt)
        self.assertIn("不得通过无意义改写参数", check_prompt)
        self.assertIn("不包含 raw tool result", check_prompt)
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

    def test_check_prompt_requires_semantic_completeness_before_complete(self) -> None:
        check_prompt = SQL_GROUNDING_STAGE_PROMPTS["check"]
        requirements = {
            "formula": (
                "派生指标、计算公式、阈值或业务判断规则",
                "只有相关字段不够",
                "应如何组合计算",
                "缺少精确规则时必须 incomplete",
            ),
            "semantic_conflict": (
                "Query 与 State 的语义一致性",
                "明显不同、相反",
                "相邻/派生概念",
                "不得为了保留 current mapping 而放行错误语义",
            ),
            "literal": (
                "阈值、类别值或条件时",
                "必须在 current_state 或本轮合法 Official",
                "非示例性的依据",
                "不得采用示例值",
            ),
            "complete": (
                "字段、关系、精确业务规则/公式和关键 literal 都已齐全",
                "彼此语义一致时",
                "才能 complete",
            ),
        }
        for case, fragments in requirements.items():
            with self.subTest(case=case):
                for fragment in fragments:
                    self.assertIn(fragment, check_prompt)
        self.assertIn("只判断并补齐一个最具体的缺口", check_prompt)
        self.assertIn("不使用 execute_sql 探索", check_prompt)

    def test_check_prompt_distinguishes_entity_grain_and_literal_authority(self) -> None:
        check_prompt = SQL_GROUNDING_STAGE_PROMPTS["check"]
        cases = {
            "finer_grain_without_identity_is_incomplete": (
                "measure 来自更细粒度的 event、snapshot",
                "identity / output target",
                "grouping target / 关系",
                "仅有细粒度 measure 和一条可达 join path 不够",
                "缺少 entity identity 或 grouping grain 时必须 incomplete",
            ),
            "explicit_entity_grain_does_not_require_guessed_aggregate": (
                "不得自动猜 SUM / AVG / MAX",
                "不得自动补 mapping",
                "entity identity / grouping",
                "已明确，且 Query",
                "其他 completeness 条件满足时可以 complete",
            ),
            "illustrative_literal_is_not_authoritative": (
                "for example / e.g. / such as / 例如",
                "只是示例",
                "不能升级为 frozen mandatory",
                "不得采用示例值",
            ),
            "direct_predicate_can_be_authoritative": (
                "没有示例限定词的明确固定 predicate",
                "authoritative rule",
                "确实需要该 predicate",
            ),
            "irrelevant_example_does_not_create_threshold_gap": (
                "只要求排序、最值或返回观测值",
                "不得因为它不具权威性而制造 missing threshold",
                "不得强制加入对应谓词",
                "MAX / MIN / ORDER BY 等操作不要求在 State 中重复",
            ),
        }
        for case, fragments in cases.items():
            with self.subTest(case=case):
                for fragment in fragments:
                    self.assertIn(fragment, check_prompt)

    def test_check_prompt_requires_clarification_to_resolve_the_exact_gap(self) -> None:
        check_prompt = SQL_GROUNDING_STAGE_PROMPTS["check"]
        required_fragments = (
            "Clarification Scope Gate（最高优先级）",
            "A. 窄澄清",
            "B. 语义扩展",
            "C. 未解决",
            "只有 Gate 0 判定为 A（窄澄清）后",
            "明确、直接提供上一轮缺少的具体",
            "latest_user_answer 仍不等于缺口自动解决",
            "State 外的 phase-local clarification evidence",
            "不是 Official schema、metadata",
            "不得复制、改写或概括进",
            "Clarification overlay 只是为已有 Grounding 补充参数的通道",
            "不是替代新 Grounding 语义的通道",
            "out of scope",
            "不知道",
            "不确定",
            "回答模糊、拒绝",
            "与缺口无关",
            "不得生成或猜测任何缺失语义",
            "回答没有解决缺口时必须 incomplete",
            "next_tool = null 并 terminal",
            "不得为了满足 Form 重复",
            "不得伪造新的 gap 或 tool",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, check_prompt)
        self.assertIn("“1000 hours”可以解决该缺口", check_prompt)
        self.assertIn(
            "column_mapping 和 domain_knowledge 必须与 current_state",
            check_prompt,
        )
        self.assertLess(
            check_prompt.index("0. Clarification Scope Gate"),
            check_prompt.index("1. 业务规则完整性"),
        )
        self.assertLess(
            check_prompt.index("B. 语义扩展"),
            check_prompt.index("5. 窄澄清是否直接解决上一轮缺口"),
        )

    def test_check_completeness_examples_keep_the_strict_existing_form(self) -> None:
        incomplete_cases = (
            (
                "The exact formula for combining cost and impact is missing.",
                GroundingCheckToolRequest(
                    tool_name="get_all_external_knowledge_names",
                    arguments={},
                ),
            ),
            (
                "The requested availability concept conflicts with the mapped downtime field.",
                GroundingCheckToolRequest(
                    tool_name="get_column_meaning",
                    arguments={
                        "table_name": "operational_metrics",
                        "column_name": "status",
                    },
                ),
            ),
            (
                "The exact approved threshold literal is missing.",
                GroundingCheckToolRequest(
                    tool_name="get_knowledge_definition",
                    arguments={"knowledge_name": "Approved threshold rule"},
                ),
            ),
        )
        for gap, tool in incomplete_cases:
            with self.subTest(gap=gap):
                response = GroundingCheckResponse(
                    status="incomplete",
                    missing_information=gap,
                    next_tool=tool,
                    column_mapping=complete_state().column_mapping or (),
                    domain_knowledge=(),
                )
                self.assertEqual(response.status, "incomplete")
                self.assertIsNotNone(response.next_tool)

        rule = "Use the approved threshold 0.75 in the documented ratio formula."
        complete = GroundingCheckResponse(
            status="complete",
            missing_information=None,
            next_tool=None,
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(
                DomainKnowledge(kind="business_rule", content=rule),
            ),
        )
        self.assertEqual(complete.status, "complete")
        self.assertIsNone(complete.missing_information)
        self.assertIsNone(complete.next_tool)

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

    def test_check_requires_complete_or_one_concrete_gap(self) -> None:
        complete = GroundingCheckResponse(
            status="complete",
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )
        self.assertIsNone(complete.next_tool)
        terminal = GroundingCheckResponse(
            status="incomplete",
            missing_information="The exact maintenance threshold is unresolved.",
            next_tool=None,
            column_mapping=(),
            domain_knowledge=(),
        )
        self.assertIsNone(terminal.next_tool)
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
            classify_grounding_input(
                {
                    **base,
                    **EMPTY_CHECK_PHASE_CONTEXT,
                    "check_context": {"kind": "initial"},
                },
                phase=1,
            ),
            "check",
        )
        self.assertEqual(
            classify_grounding_input(
                {**base, **EMPTY_CHECK_PHASE_CONTEXT, "latest_tool": {}},
                phase=1,
            ),
            "check",
        )
        self.assertEqual(
            classify_grounding_input(
                {
                    **base,
                    **EMPTY_CHECK_PHASE_CONTEXT,
                    "latest_user_answer": {},
                },
                phase=1,
            ),
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
            **EMPTY_CHECK_PHASE_CONTEXT,
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
            observation_type="metadata",
            content="Official meaning: reported maintenance cost.",
            summary="Official column meaning observed",
            tool_name="get_column_meaning",
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
                **EMPTY_CHECK_PHASE_CONTEXT,
                "latest_tool": {
                    "name": "get_column_meaning",
                    "arguments": {
                        "table_name": "operational_metrics",
                        "column_name": "reported_cost",
                    },
                    "result": "Official meaning: reported maintenance cost.",
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

    async def test_latest_user_answer_cannot_change_mapping_or_knowledge(self) -> None:
        exact_rule = "The Official maintenance threshold is 1000 hours."
        old = complete_state()
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
            domain_knowledge=(
                DomainKnowledge(kind="business_rule", content=exact_rule),
            ),
        )
        observation = build_sql_grounding_observation(
            task_id="v13-answer-cannot-change-state",
            phase=1,
            sequence=6,
            observation_type="user_answer",
            content="Use 1000 hours and reported cost.",
            summary="answered bounded Check clarification",
            tool_name="ask_user",
            function_call_id="v13-answer-cannot-change-state-call",
        )
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context(observation.observation_id, rule=exact_rule),
            FakeUpdater(response, "check"),
            grounding_input={
                "query": QUERY,
                "current_state": old.model_dump(mode="json"),
                "previous_official_calls": [],
                "answered_clarifications": [
                    {
                        "question": "Which threshold and cost meaning apply?",
                        "answer": "Use 1000 hours and reported cost.",
                    }
                ],
                "latest_user_answer": {
                    "question": "Which threshold and cost meaning apply?",
                    "answer": "Use 1000 hours and reported cost.",
                },
            },
        )
        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.state_update.error_type, "authorization_rejected")
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(result.runtime.grounding_revision, 3)
        self.assertEqual(
            sql_grounding_state_sha256(result.runtime.grounding_state),
            sql_grounding_state_sha256(old),
        )

    async def test_unresolved_clarification_answers_keep_gap_and_state(self) -> None:
        gap = "The exact operating-hours threshold is missing."
        old = complete_state()
        unchanged_response = GroundingCheckResponse(
            status="incomplete",
            missing_information=gap,
            next_tool=GroundingCheckToolRequest(
                tool_name="get_all_external_knowledge_names",
                arguments={},
            ),
            column_mapping=old.column_mapping or (),
            domain_knowledge=old.domain_knowledge or (),
        )
        invalid_answers = (
            "out of scope",
            "I don't know",
            "Maybe it is a large value.",
            "Use the reported maintenance cost instead.",
        )
        for sequence, answer in enumerate(invalid_answers, start=10):
            with self.subTest(answer=answer):
                runtime = GroundingRuntime(
                    grounding_revision=3,
                    stage="INITIAL_GROUNDING",
                    focus_dimension="none",
                    grounding_state=old,
                )
                observation = build_sql_grounding_observation(
                    task_id=f"v13-unresolved-answer-{sequence}",
                    phase=1,
                    sequence=sequence,
                    observation_type="user_answer",
                    content=answer,
                    summary="answered bounded Check clarification",
                    tool_name="ask_user",
                    function_call_id=f"v13-unresolved-answer-call-{sequence}",
                )
                result = await process_sql_grounding_observation(
                    runtime,
                    observation,
                    context(observation.observation_id),
                    FakeUpdater(unchanged_response, "check"),
                    grounding_input={
                        "query": QUERY,
                        "current_state": old.model_dump(mode="json"),
                        "previous_official_calls": [],
                        "answered_clarifications": [
                            {
                                "question": (
                                    "What exact operating-hours threshold applies?"
                                ),
                                "answer": answer,
                            }
                        ],
                        "latest_user_answer": {
                            "question": "What exact operating-hours threshold applies?",
                            "answer": answer,
                        },
                    },
                )
                self.assertEqual(result.response.status, "incomplete")
                self.assertEqual(result.response.missing_information, gap)
                self.assertEqual(result.state_update.status, "noop")
                self.assertEqual(result.runtime, runtime)
                self.assertNotIn(
                    "threshold",
                    json.dumps(
                        result.runtime.grounding_state.model_dump(mode="json")
                    ).lower(),
                )

    async def test_explicit_clarification_can_supply_the_missing_threshold(self) -> None:
        answer = "1000 hours"
        old = complete_state()
        runtime = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=old,
        )
        response = GroundingCheckResponse(
            status="complete",
            missing_information=None,
            next_tool=None,
            column_mapping=old.column_mapping or (),
            domain_knowledge=old.domain_knowledge or (),
        )
        observation = build_sql_grounding_observation(
            task_id="v13-explicit-threshold-answer",
            phase=1,
            sequence=14,
            observation_type="user_answer",
            content=answer,
            summary="answered bounded Check clarification",
            tool_name="ask_user",
            function_call_id="v13-explicit-threshold-answer-call",
        )
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            context(observation.observation_id),
            FakeUpdater(response, "check"),
            grounding_input={
                "query": QUERY,
                "current_state": old.model_dump(mode="json"),
                "previous_official_calls": [],
                "answered_clarifications": [
                    {
                        "question": "What exact operating-hours threshold applies?",
                        "answer": answer,
                    }
                ],
                "latest_user_answer": {
                    "question": "What exact operating-hours threshold applies?",
                    "answer": answer,
                },
            },
        )
        self.assertEqual(result.response.status, "complete")
        self.assertIsNone(result.response.next_tool)
        self.assertEqual(result.state_update.status, "noop")
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(result.runtime.grounding_revision, 3)
        self.assertEqual(result.runtime.grounding_state.domain_knowledge, ())


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

    async def test_terminal_incomplete_fails_closed_without_tool_or_state_change(
        self,
    ) -> None:
        state = self.adk_check_state()
        before_runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        before_sha = sql_grounding_state_sha256(
            before_runtime.grounding_state
        )
        before_budget = state["budget_remaining"]
        before_trajectory = list(state["tool_trajectory"])
        terminal = GroundingCheckResponse(
            status="incomplete",
            missing_information=(
                "The exact operating-hours threshold is still missing."
            ),
            next_tool=None,
            column_mapping=(
                before_runtime.grounding_state.column_mapping or ()
            ),
            domain_knowledge=(
                before_runtime.grounding_state.domain_knowledge or ()
            ),
        )
        updater = FakeUpdater(terminal, "check")
        answer = "out of scope"
        observation = build_sql_grounding_observation(
            task_id=state["task_id"],
            phase=1,
            sequence=5,
            observation_type="user_answer",
            content=answer,
            summary="answered bounded Check clarification",
            tool_name="ask_user",
            function_call_id="v13-terminal-incomplete-answer",
        )
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
                    return_value=SimpleNamespace(max_calls_per_task=8),
                ),
            ):
                result = await grounding_callbacks._handle_observation(
                    state,
                    observation,
                    grounding_input={
                        "query": QUERY,
                        "current_state": before_runtime.grounding_state.model_dump(
                            mode="json"
                        ),
                        "previous_official_calls": [],
                        "answered_clarifications": [
                            {
                                "question": (
                                    "What exact operating-hours threshold applies?"
                                ),
                                "answer": answer,
                            }
                        ],
                        "latest_user_answer": {
                            "question": (
                                "What exact operating-hours threshold applies?"
                            ),
                            "answer": answer,
                        },
                    },
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        stored_runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(result.service_status, "terminal_incomplete")
        self.assertEqual(result.control_status, "failed_closed")
        self.assertEqual(
            result.control_error_type,
            "CheckTerminalIncomplete",
        )
        self.assertEqual(updater.calls, 1)
        self.assertEqual(result.service_result.state_update.status, "noop")
        self.assertEqual(stored_runtime, before_runtime)
        self.assertEqual(stored_runtime.grounding_revision, 3)
        self.assertEqual(
            sql_grounding_state_sha256(stored_runtime.grounding_state),
            before_sha,
        )
        self.assertEqual(state["budget_remaining"], before_budget)
        self.assertEqual(state["tool_trajectory"], before_trajectory)
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))
        audit = grounding_callbacks._check_audits(state)[-1]
        self.assertEqual(audit["blocked_reason"], "terminal_incomplete")
        self.assertIsNone(audit["tool_name"])
        self.assertEqual(audit["tool_cost"], 0.0)
        self.assertEqual(audit["budget_before"], audit["budget_after"])
        failure = grounding_callbacks._phase_grounding_failure(state, 1)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.error_type, "CheckTerminalIncomplete")

        baseline_before = AsyncMock(return_value=None)
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
                    baseline_before,
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

        self.assertIsNotNone(response)
        self.assertIn(
            "VALIBRA_SQL_GROUNDING_FAILED_CLOSED",
            response.content.parts[0].text,
        )
        self.assertIsNone(response.content.parts[0].function_call)
        self.assertEqual(baseline_before.await_count, 1)

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

    def test_answer_resumes_check_with_phase_context_and_allows_new_question(self) -> None:
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
            {
                "query",
                "current_state",
                "previous_official_calls",
                "answered_clarifications",
                "latest_user_answer",
            },
        )
        self.assertEqual(payload["latest_user_answer"]["answer"], answer)
        self.assertEqual(
            payload["answered_clarifications"],
            [
                {
                    "question": pending.arguments["question"],
                    "answer": answer,
                }
            ],
        )
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

    def test_latest_check_request_has_no_raw_history(self) -> None:
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
        self.assertEqual(
            set(payload),
            {
                "query",
                "current_state",
                "previous_official_calls",
                "answered_clarifications",
                "latest_tool",
            },
        )
        self.assertEqual(payload["previous_official_calls"], [])
        self.assertEqual(payload["answered_clarifications"], [])
        self.assertNotIn("trajectory", json.dumps(payload))

    def test_executed_check_call_is_projected_without_raw_result(self) -> None:
        state = self.state()
        response = GroundingCheckResponse(
            status="incomplete",
            missing_information="The applicable knowledge name is unresolved.",
            next_tool=GroundingCheckToolRequest(
                tool_name="get_all_external_knowledge_names",
                arguments={},
            ),
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )
        pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=response,
        )
        raw_result = "RAW-KNOWLEDGE-NAMES-MUST-NOT-BE-COPIED"
        state["tool_trajectory"] = [
            {
                "type": "tool",
                "tool": pending.tool_name,
                "phase": 1,
                "args": pending.arguments,
                "result": raw_result,
            }
        ]
        grounding_callbacks._store_pending_check_tool(state, None)

        payload = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            runtime=GroundingRuntime(
                grounding_revision=3,
                focus_dimension="none",
                grounding_state=complete_state(),
            ),
            phase=1,
            latest_tool_name=pending.tool_name,
            latest_tool_arguments=pending.arguments,
            latest_tool_result=raw_result,
        )

        self.assertEqual(
            payload["previous_official_calls"],
            [
                {
                    "tool_name": "get_all_external_knowledge_names",
                    "arguments": {},
                    "request_digest": pending.request_digest,
                }
            ],
        )
        self.assertNotIn(raw_result, json.dumps(payload["previous_official_calls"]))
        with self.assertRaisesRegex(ValueError, "identical arguments"):
            grounding_callbacks._schedule_check_tool(
                state,
                phase=1,
                response=response.model_copy(
                    update={
                        "missing_information": (
                            "A different gap still requests the same evidence."
                        )
                    }
                ),
            )

        next_response = GroundingCheckResponse(
            status="incomplete",
            missing_information="The exact business rule is unresolved.",
            next_tool=GroundingCheckToolRequest(
                tool_name="get_knowledge_definition",
                arguments={"knowledge_name": "Maintenance Cost"},
            ),
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )
        next_pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=next_response,
        )
        self.assertEqual(next_pending.tool_name, "get_knowledge_definition")

    def test_answered_clarification_survives_a_later_check_tool_turn(self) -> None:
        state = self.state()
        question = "Which maintenance cost meaning do you intend?"
        answer = "Use the reported maintenance cost."
        clarification_response = self.incomplete_check(tool="ask_user")
        clarification_pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=clarification_response,
        )
        grounding_callbacks._record_clarification_answer(
            state,
            phase=1,
            question=question,
            answer=answer,
        )
        state["tool_trajectory"] = [
            {
                "type": "tool",
                "tool": "ask_user",
                "phase": 1,
                "args": clarification_pending.arguments,
                "result": answer,
            }
        ]
        grounding_callbacks._store_pending_check_tool(state, None)

        evidence_response = GroundingCheckResponse(
            status="incomplete",
            missing_information="The exact cost column meaning is unresolved.",
            next_tool=GroundingCheckToolRequest(
                tool_name="get_column_meaning",
                arguments={
                    "table_name": "operational_metrics",
                    "column_name": "reported_cost",
                },
            ),
            column_mapping=complete_state().column_mapping or (),
            domain_knowledge=(),
        )
        evidence_pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=evidence_response,
        )
        state["tool_trajectory"].append(
            {
                "type": "tool",
                "tool": evidence_pending.tool_name,
                "phase": 1,
                "args": evidence_pending.arguments,
                "result": "Official reported-cost meaning",
            }
        )
        grounding_callbacks._store_pending_check_tool(state, None)
        before_state = complete_state()
        before_sha = sql_grounding_state_sha256(before_state)
        runtime = GroundingRuntime(
            grounding_revision=3,
            focus_dimension="none",
            grounding_state=before_state,
        )

        payload = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=1,
            latest_tool_name=evidence_pending.tool_name,
            latest_tool_arguments=evidence_pending.arguments,
            latest_tool_result="Official reported-cost meaning",
        )

        self.assertEqual(
            payload["answered_clarifications"],
            [{"question": question, "answer": answer}],
        )
        self.assertEqual(runtime.grounding_revision, 3)
        self.assertEqual(
            sql_grounding_state_sha256(runtime.grounding_state),
            before_sha,
        )
        self.assertEqual(
            set(runtime.grounding_state.model_dump(mode="json")),
            {"tables", "join_keys", "column_mapping", "domain_knowledge"},
        )

    def test_phase_local_context_does_not_cross_p1_and_p2(self) -> None:
        state = self.state()
        grounding_callbacks._store_clarification_records(
            state,
            (
                UserClarificationRecord(
                    phase=1,
                    phrase="maintenance cost",
                    kind="user_intent",
                    question="Which P1 cost meaning applies?",
                    answer="Use P1 reported cost.",
                ),
                UserClarificationRecord(
                    phase=2,
                    phrase="maintenance cost",
                    kind="user_intent",
                    question="Which P2 cost meaning applies?",
                    answer="Use P2 total cost.",
                ),
            ),
        )
        p1_response = self.incomplete_check()
        p1_pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=1,
            response=p1_response,
        )
        grounding_callbacks._store_pending_check_tool(state, None)
        p2_response = p1_response.model_copy(
            update={"missing_information": "The P2 cost field is unresolved."}
        )
        p2_pending = grounding_callbacks._schedule_check_tool(
            state,
            phase=2,
            response=p2_response,
        )
        grounding_callbacks._store_pending_check_tool(state, None)
        state["tool_trajectory"] = [
            {
                "type": "tool",
                "tool": p1_pending.tool_name,
                "phase": 1,
                "args": p1_pending.arguments,
                "result": "P1 meaning",
            },
            {
                "type": "tool",
                "tool": p2_pending.tool_name,
                "phase": 2,
                "args": p2_pending.arguments,
                "result": "P2 meaning",
            },
        ]
        p1_payload = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            runtime=GroundingRuntime(
                grounding_revision=3,
                focus_dimension="none",
                grounding_state=complete_state(),
            ),
            phase=1,
            initial=True,
        )
        p2_payload = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            follow_up="Compare that result with the previous period.",
            runtime=GroundingRuntime(
                grounding_revision=3,
                stage="P2_INCREMENTAL",
                focus_dimension="none",
                grounding_state=complete_state(),
            ),
            phase=2,
            initial=True,
        )

        self.assertEqual(
            [item["request_digest"] for item in p1_payload["previous_official_calls"]],
            [p1_pending.request_digest],
        )
        self.assertEqual(
            [item["request_digest"] for item in p2_payload["previous_official_calls"]],
            [p2_pending.request_digest],
        )
        self.assertEqual(
            p1_payload["answered_clarifications"],
            [
                {
                    "question": "Which P1 cost meaning applies?",
                    "answer": "Use P1 reported cost.",
                }
            ],
        )
        self.assertEqual(
            p2_payload["answered_clarifications"],
            [
                {
                    "question": "Which P2 cost meaning applies?",
                    "answer": "Use P2 total cost.",
                }
            ],
        )

    def test_sql_writer_replaces_prompt_and_exposes_exactly_two_tools(self) -> None:
        names = ["execute_sql", "get_schema", "ask_user", "submit_sql"]
        original_descriptions = {
            "execute_sql": "BASELINE execute description",
            "get_schema": "BASELINE schema description",
            "ask_user": "BASELINE ask description",
            "submit_sql": "BASELINE submit description",
        }
        original_group = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=name,
                    description=original_descriptions[name],
                )
                for name in names
            ]
        )
        request = LlmRequest(
            config=types.GenerateContentConfig(
                system_instruction="OLD EXPLORATION CONTROL HINT",
                tools=[copy.deepcopy(original_group)],
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
        writer_declarations = {
            item.name: item.description
            for item in request.config.tools[0].function_declarations
        }
        self.assertEqual(
            writer_declarations["execute_sql"],
            grounding_callbacks._SQL_WRITER_EXECUTE_TOOL_DESCRIPTION,
        )
        self.assertEqual(
            writer_declarations["submit_sql"],
            original_descriptions["submit_sql"],
        )
        self.assertIn("candidate PostgreSQL task-answer query", writer_declarations["execute_sql"])
        self.assertIn("not for schema discovery", writer_declarations["execute_sql"])
        self.assertIn("data sampling", writer_declarations["execute_sql"])
        self.assertEqual(
            {
                item.name: item.description
                for item in original_group.function_declarations
            },
            original_descriptions,
        )
        self.assertNotIn("OLD EXPLORATION", request.config.system_instruction)
        self.assertNotIn("VALIBRA CONTROL", request.config.system_instruction)

    def test_sql_writer_receives_answered_clarification_outside_state(self) -> None:
        state = complete_state()
        clarification = UserClarificationRecord(
            phase=1,
            phrase="maintenance cost",
            kind="missing_knowledge",
            question="What threshold applies?",
            answer="Use 1000 hours.",
        )
        view = render_grounding_view(state, clarifications=(clarification,))
        request = LlmRequest(
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(name="execute_sql"),
                            types.FunctionDeclaration(name="submit_sql"),
                        ]
                    )
                ]
            )
        )
        request.tools_dict = {
            "execute_sql": SimpleNamespace(),
            "submit_sql": SimpleNamespace(),
        }
        grounding_callbacks._inject_sql_writer_context(
            request,
            phase=1,
            original_query=QUERY,
            follow_up=None,
            view_text=view.text,
            budget_remaining=7,
        )
        instruction = request.config.system_instruction
        self.assertIn("[USER CLARIFICATIONS]", instruction)
        self.assertIn("Use 1000 hours.", instruction)
        self.assertNotIn("clarification", state.model_dump())

    def test_sql_writer_prompt_freezes_semantics_and_requires_convergence(self) -> None:
        prompt = grounding_callbacks._SQL_WRITER_PROMPT
        self.assertEqual(grounding_callbacks._sha256_text(prompt), WRITER_PROMPT_SHA)
        requirements = {
            "semantic_authority": (
                "Final Grounding State 和已回答澄清是语义权威",
                "不要自行替换、删除或重新解释其中的字段、阈值、公式、过滤条件或业务概念",
                "不要根据 execute_sql 结果发明新的语义条件",
            ),
            "aggregation_before_comparison": (
                "公式、阈值、过滤条件、多字段共同计算及聚合语义",
                "多个 targets 共同参与同一计算或判断时，必须共同使用",
                "先正确聚合，再排序或比较",
            ),
            "zero_rows_do_not_change_threshold": (
                "返回 0 rows 不能成为更换冻结阈值或业务规则的理由",
            ),
            "no_duplicate_execution": (
                "不要重复执行相同 SQL 或语义等价的无效查询",
            ),
            "successful_execution_converges_to_submit": (
                "成功 execute 且没有新的实现问题时，应立即 submit_sql",
                "剩余预算有限时，应提交当前最佳且与 State 一致的 SQL",
            ),
            "undiagnostic_submit_failure_does_not_thaw_semantics": (
                "该 FAIL 只表示本次提交未通过",
                "它不是 Frozen State、已回答澄清、冻结阈值、业务规则或字段语义错误的证据",
                "先前成功 execute 的结果仍是有效的 SQL 实现证据",
            ),
            "post_submit_review_uses_only_frozen_inputs_and_history": (
                "只能根据 Query / Follow-up、Frozen State、已回答澄清和已有 execute / submit history",
                "禁止重新查询 schema、DISTINCT values、枚举 JSON keys、row/data sampling、验证 frozen literals",
                "寻找新的业务规则和字段含义",
            ),
            "materially_different_candidate_requires_implementation_hypothesis": (
                "只有能够指出具体的 SQL implementation hypothesis 时",
                "materially different 且与 State 一致的新 candidate",
                "JOIN、aggregation grain、projection / result shape、CAST / NULL、GROUP BY / ORDER BY、latest-row handling",
            ),
            "post_submit_convergence_without_exploration": (
                "不得执行只是为了“看看数据”的 SQL",
                "不得重复执行相同或语义等价的 candidate",
                "只做必要的 implementation validation，然后尽快 submit_sql",
                "若找不到具体 implementation mismatch，不要返回数据库探索",
                "保持冻结语义，不发明新字段、条件、阈值、公式或业务规则",
            ),
        }
        for scenario, fragments in requirements.items():
            with self.subTest(scenario=scenario):
                for fragment in fragments:
                    self.assertIn(fragment, prompt)

        self.assertIn("execute_sql 只用于验证和修正", prompt)
        self.assertIn("不能用于新的语义探索", prompt)
        self.assertNotIn("solar_panel", prompt)

    def test_sql_writer_hides_raw_bootstrap_but_keeps_query_and_execution_history(self) -> None:
        sql = "SELECT 1"
        submit_result = "SQL failed Phase 1. Your SQL is not correct."
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
                                args={"sql": sql},
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
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="submit-one",
                                name="submit_sql",
                                args={"sql": sql},
                            )
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="submit-one",
                                name="submit_sql",
                                response={"result": submit_result},
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
        self.assertIn("submit_sql", serialized)
        self.assertEqual(serialized.count(sql), 2)
        self.assertIn(submit_result, serialized)
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
