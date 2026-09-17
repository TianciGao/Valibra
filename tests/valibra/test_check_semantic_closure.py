from __future__ import annotations

import unittest

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingCheckClarificationProposal,
    GroundingCheckResponse,
    GroundingCheckToolRequest,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    SQLGroundingState,
    ValidationContext,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.service import (
    process_sql_grounding_observation,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    CHECK_GROUNDING_PROMPT,
    KNOWLEDGE_GROUNDING_PROMPT,
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_STAGE_PROMPT_SHA256,
    GroundingUpdaterResult,
)


class _FakeUpdater:
    def __init__(self, response: object, call_kind: str) -> None:
        self.response = response
        self.call_kind = call_kind

    async def propose(self, *args: object, **kwargs: object) -> GroundingUpdaterResult:
        del args, kwargs
        return GroundingUpdaterResult(
            response=self.response,
            call_kind=self.call_kind,
            telemetry=GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="1" * 64,
                response_sha256="2" * 64,
                prompt_sha256=SQL_GROUNDING_STAGE_PROMPT_SHA256[self.call_kind],
                form_schema_sha256=SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256[
                    self.call_kind
                ],
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


def _observation(task: str, *, kind: str, tool: str):
    return build_sql_grounding_observation(
        task_id=task,
        phase=1,
        sequence=4,
        observation_type=kind,
        content="offline Official evidence",
        summary="offline contract fixture",
        tool_name=tool,
        function_call_id=f"{task}-call",
    )


def _context(
    query: str,
    observation_id: str,
    *,
    tables: tuple[str, ...],
    columns: tuple[str, ...],
    rules: tuple[str, ...] = (),
) -> ValidationContext:
    return ValidationContext(
        current_query=query,
        latest_observation_id=observation_id,
        known_tables=frozenset(tables),
        known_columns=frozenset(columns),
        supported_domain_knowledge=frozenset(
            ("business_rule", rule) for rule in rules
        ),
    )


class CheckClarificationContractTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _risky_state() -> SQLGroundingState:
        return SQLGroundingState(
            tables=("operational_metrics", "plants"),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="potential financial hit",
                    targets=("operational_metrics.revloss",),
                ),
                ColumnMapping(
                    phrase="risky plants",
                    targets=("plants.complyflag",),
                ),
            ),
            domain_knowledge=(),
        )

    async def _process_check(
        self,
        *,
        task: str,
        query: str,
        state: SQLGroundingState,
        response: GroundingCheckResponse,
        grounding_input: dict[str, object],
    ):
        runtime = GroundingRuntime(
            grounding_revision=3,
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=state,
        )
        if "latest_user_answer" in grounding_input:
            observation = _observation(task, kind="user_answer", tool="ask_user")
        elif isinstance(grounding_input.get("latest_tool"), dict):
            observation = _observation(
                task,
                kind="knowledge",
                tool=str(grounding_input["latest_tool"]["name"]),
            )
        else:
            observation = _observation(
                task,
                kind="knowledge",
                tool="get_all_knowledge_definitions",
            )
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            _context(
                query,
                observation.observation_id,
                tables=state.tables or (),
                columns=tuple(
                    target
                    for mapping in state.column_mapping or ()
                    for target in mapping.targets
                    if " -> " not in target
                ),
            ),
            _FakeUpdater(response, "check"),
            grounding_input=grounding_input,
        )
        return runtime, result

    async def test_verbatim_mapping_phrase_passes_and_schedules_ask_user(self) -> None:
        query = "What's the potential financial hit from our risky plants?"
        state = self._risky_state()
        response = GroundingCheckResponse(
            status="incomplete",
            clarification_route="none",
            missing_information="Which compliance values mean risky?",
            next_tool=GroundingCheckToolRequest(
                tool_name="ask_user",
                arguments={
                    "question": "Which compliance statuses should count as risky?"
                },
                user_clarification_request=GroundingCheckClarificationProposal(
                    phrase="risky plants",
                    kind="user_intent",
                ),
            ),
            column_mapping=state.column_mapping or (),
            domain_knowledge=(),
        )
        before_sha = sql_grounding_state_sha256(state)
        runtime, result = await self._process_check(
            task="verbatim-phrase",
            query=query,
            state=state,
            response=response,
            grounding_input={
                "query": query,
                "current_state": state.model_dump(mode="json"),
                "previous_official_calls": [],
                "unresolved_mappings": [],
                "answered_clarifications": [],
                "check_context": {"kind": "initial"},
            },
        )

        self.assertEqual(result.state_update.status, "noop")
        pending = grounding_callbacks._schedule_check_tool(
            {
                "task_id": "verbatim-phrase",
                "current_phase": 1,
                "budget_remaining": 10.0,
                "tool_trajectory": [],
            },
            phase=1,
            response=result.response,
        )
        self.assertEqual(pending.tool_name, "ask_user")
        self.assertEqual(
            response.next_tool.user_clarification_request.phrase,
            "risky plants",
        )
        self.assertEqual(runtime.grounding_revision, 3)
        self.assertEqual(result.runtime.grounding_revision, 3)
        self.assertEqual(
            sql_grounding_state_sha256(result.runtime.grounding_state), before_sha
        )

    async def test_extended_nonverbatim_phrase_fails_closed(self) -> None:
        query = "What's the potential financial hit from our risky plants?"
        state = self._risky_state()
        response = GroundingCheckResponse(
            status="incomplete",
            clarification_route="none",
            missing_information="Which compliance values mean risky?",
            next_tool=GroundingCheckToolRequest(
                tool_name="ask_user",
                arguments={
                    "question": "Which compliance statuses should count as risky?"
                },
                user_clarification_request=GroundingCheckClarificationProposal(
                    phrase="risky plants compliance status",
                    kind="user_intent",
                ),
            ),
            column_mapping=state.column_mapping or (),
            domain_knowledge=(),
        )
        before_sha = sql_grounding_state_sha256(state)
        runtime, result = await self._process_check(
            task="nonverbatim-phrase",
            query=query,
            state=state,
            response=response,
            grounding_input={
                "query": query,
                "current_state": state.model_dump(mode="json"),
                "previous_official_calls": [],
                "unresolved_mappings": [],
                "answered_clarifications": [],
                "check_context": {"kind": "initial"},
            },
        )

        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.state_update.error_type, "state_validation_failed")
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(
            sql_grounding_state_sha256(result.runtime.grounding_state), before_sha
        )

    async def test_existing_mapping_plus_literal_can_complete_without_state_change(
        self,
    ) -> None:
        query = "Show panels with bad busbar corrosion."
        state = SQLGroundingState(
            tables=("mechanical_condition",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="bad busbar corrosion",
                    targets=("mechanical_condition.busbar_corrosion",),
                ),
            ),
            domain_knowledge=(),
        )
        response = GroundingCheckResponse(
            status="complete",
            clarification_route="stay_check",
            column_mapping=state.column_mapping or (),
            domain_knowledge=(),
        )
        answer = "Use the Severe value."
        before_sha = sql_grounding_state_sha256(state)
        runtime, result = await self._process_check(
            task="literal-only-answer",
            query=query,
            state=state,
            response=response,
            grounding_input={
                "query": query,
                "current_state": state.model_dump(mode="json"),
                "previous_official_calls": [],
                "unresolved_mappings": [],
                "answered_clarifications": [
                    {"question": "Which level is bad?", "answer": answer}
                ],
                "latest_user_answer": {
                    "question": "Which level is bad?",
                    "answer": answer,
                },
            },
        )

        self.assertEqual(result.state_update.status, "noop")
        self.assertEqual(result.response.status, "complete")
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(
            sql_grounding_state_sha256(result.runtime.grounding_state), before_sha
        )

    async def test_answer_expansion_contract_requires_incomplete(self) -> None:
        self.assertIn("ask_user 是高成本 Grounding-cycle boundary", CHECK_GROUNDING_PROMPT)
        self.assertIn(
            "必须 status=incomplete、next_tool=null",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "runtime 会把它交给 Final Gate 并创建全新的 Draft",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "本轮 Check 不得一点点修旧 State",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "clarification 不是 schema、metadata 或 Official business",
            CHECK_GROUNDING_PROMPT,
        )

        state = SQLGroundingState(
            tables=("panel_models",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="aging terribly",
                    targets=("panel_models.degyrrate",),
                ),
            ),
            domain_knowledge=(),
        )
        answer = (
            "aging terribly means degyrrate > 0.5 AND severe busbar corrosion "
            "AND major delamination AND significant microcracking"
        )
        response = GroundingCheckResponse(
            status="incomplete",
            clarification_route="restart_grounding",
            missing_information=(
                "The added busbar-corrosion condition has no grounded field mapping."
            ),
            next_tool=None,
            column_mapping=state.column_mapping or (),
            domain_knowledge=(),
        )
        before_sha = sql_grounding_state_sha256(state)
        runtime, result = await self._process_check(
            task="expanded-answer",
            query="Show plants aging terribly.",
            state=state,
            response=response,
            grounding_input={
                "query": "Show plants aging terribly.",
                "current_state": state.model_dump(mode="json"),
                "previous_official_calls": [],
                "unresolved_mappings": [],
                "answered_clarifications": [
                    {"question": "What does aging terribly mean?", "answer": answer}
                ],
                "latest_user_answer": {
                    "question": "What does aging terribly mean?",
                    "answer": answer,
                },
            },
        )
        self.assertEqual(result.state_update.status, "noop")
        self.assertEqual(result.response.status, "incomplete")
        self.assertEqual(result.response.clarification_route, "restart_grounding")
        self.assertIsNone(result.response.next_tool)
        self.assertEqual(result.response.column_mapping, state.column_mapping)
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(
            sql_grounding_state_sha256(result.runtime.grounding_state), before_sha
        )

    async def test_grounded_historical_expansion_can_complete_after_tool_turn(
        self,
    ) -> None:
        self.assertIn(
            "user_clarifications 当前要求的每个 SQL-relevant concept",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "全部通过才 complete",
            CHECK_GROUNDING_PROMPT,
        )
        state = SQLGroundingState(
            tables=("mechanical_condition",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="aging terribly",
                    targets=(
                        "mechanical_condition.busbar_corrosion",
                        "mechanical_condition.degyrrate",
                        "mechanical_condition.delamination_status",
                        "mechanical_condition.microcrack_count",
                    ),
                ),
            ),
            domain_knowledge=(),
        )
        answer = (
            "aging terribly means degyrrate > 0.5 AND severe busbar corrosion "
            "AND major delamination AND significant microcracking"
        )
        response = GroundingCheckResponse(
            status="complete",
            clarification_route="none",
            column_mapping=state.column_mapping or (),
            domain_knowledge=(),
        )
        before_sha = sql_grounding_state_sha256(state)
        runtime, result = await self._process_check(
            task="grounded-expanded-answer",
            query="Show plants aging terribly.",
            state=state,
            response=response,
            grounding_input={
                "query": "Show plants aging terribly.",
                "current_state": state.model_dump(mode="json"),
                "previous_official_calls": [],
                "unresolved_mappings": [],
                "answered_clarifications": [
                    {"question": "What does aging terribly mean?", "answer": answer}
                ],
                "latest_tool": {
                    "name": "get_column_meaning",
                    "arguments": {
                        "table_name": "mechanical_condition",
                        "column_name": "microcrack_count",
                    },
                    "result": "Count of detected module microcracks.",
                },
            },
        )

        self.assertEqual(result.state_update.status, "noop")
        self.assertEqual(result.response.status, "complete")
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(
            sql_grounding_state_sha256(result.runtime.grounding_state), before_sha
        )

    def test_scope_gate_limits_user_literals_to_existing_mapped_concepts(self) -> None:
        self.assertIn(
            "user_clarifications 只定义用户真实意图",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "clarification 不是 schema、metadata 或 Official business",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "新 Grounding cycle 尚未覆盖它",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "不要在 Check 中一点点重做 Structure / Mapping / Knowledge",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn(
            "全部通过才 complete",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertNotIn(
            "回答已经直接解决缺口，且 Query + current_state + clarification 已足够时，必须返回 complete",
            CHECK_GROUNDING_PROMPT,
        )

    def test_check_prompt_requires_verbatim_phrase_and_mapping_reuse(self) -> None:
        self.assertIn(
            "user_clarification_request.phrase 必须是 query 或 follow_up 中逐字连续出现的片段",
            CHECK_GROUNDING_PROMPT,
        )
        self.assertIn("优先直接复用该 mapping.phrase", CHECK_GROUNDING_PROMPT)

    def test_ask_user_question_collects_only_foreseeable_user_owned_slots(
        self,
    ) -> None:
        required = (
            "已经能够明确预见、必须由用户决定且会实质改变 SQL 的未定项",
            "threshold、上下界、比较方向",
            "AND / OR / optional 关系",
            "一次仍然只询问一个 clarification phrase",
            "同一个问题中收齐这个 phrase 下已经能够预见的",
            "必要用户参数",
            "high、low、severe、major、multiple、significant、typical",
            "类别、cutoff、boundary 或业务判定标准",
            "不得为了“问完整”而猜测 Query",
            "不得要求用户提供可由 Official metadata / knowledge 确定的信息",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)


class KnowledgeMappingContractTests(unittest.IsolatedAsyncioTestCase):
    async def _process(
        self,
        *,
        old: SQLGroundingState,
        response: KnowledgeGroundingResponse,
        definitions: list[dict[str, object]],
        meanings: dict[str, dict[str, str]],
    ):
        runtime = GroundingRuntime(
            grounding_revision=2,
            stage="INITIAL_GROUNDING",
            focus_dimension="domain_knowledge",
            grounding_state=old,
        )
        observation = _observation(
            "knowledge-mapping-contract",
            kind="knowledge",
            tool="get_all_knowledge_definitions",
        )
        columns = tuple(
            f"{table}.{column}"
            for table, table_meanings in meanings.items()
            for column in table_meanings
        )
        rules = tuple(
            item["definition"]
            for item in definitions
            if isinstance(item.get("definition"), str)
        )
        query = "Show " + " and ".join(
            mapping.phrase for mapping in old.column_mapping or ()
        ) + "."
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            _context(
                query,
                observation.observation_id,
                tables=old.tables or (),
                columns=columns,
                rules=rules,
            ),
            _FakeUpdater(response, "knowledge"),
            grounding_input={
                "query": query,
                "current_state": old.model_dump(mode="json"),
                "knowledge_definitions": definitions,
                "unresolved_mappings": [],
                "relevant_column_meanings": meanings,
            },
        )
        return runtime, result

    @staticmethod
    def _base_state() -> SQLGroundingState:
        return SQLGroundingState(
            tables=("metrics",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("metrics.maintenance_cost",),
                ),
            ),
            domain_knowledge=None,
        )

    async def test_selected_empty_and_unchanged_mapping_passes(self) -> None:
        old = self._base_state()
        _, result = await self._process(
            old=old,
            response=KnowledgeGroundingResponse(
                column_mapping=old.column_mapping or (),
                selected_knowledge_ids=(),
            ),
            definitions=[],
            meanings={"metrics": {"maintenance_cost": "Maintenance cost."}},
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(
            result.runtime.grounding_state.column_mapping, old.column_mapping
        )
        self.assertEqual(result.runtime.grounding_state.domain_knowledge, ())

    async def test_selected_empty_and_changed_mapping_fails_closed(self) -> None:
        old = self._base_state()
        runtime, result = await self._process(
            old=old,
            response=KnowledgeGroundingResponse(
                column_mapping=(
                    ColumnMapping(
                        phrase="maintenance cost",
                        targets=("metrics.cleaning_cost", "metrics.replacement_cost"),
                    ),
                ),
                selected_knowledge_ids=(),
            ),
            definitions=[],
            meanings={
                "metrics": {
                    "maintenance_cost": "Maintenance cost.",
                    "cleaning_cost": "Cleaning cost.",
                    "replacement_cost": "Replacement cost.",
                }
            },
        )

        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.state_update.error_type, "state_validation_failed")
        self.assertEqual(result.runtime, runtime)

    async def test_selected_rule_changes_supported_phrase_only_fixture(self) -> None:
        rule = "Frequent breakdown is defined by the official MTBF measure."
        old = SQLGroundingState(
            tables=("metrics",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="breaking down all the time",
                    targets=("metrics.failure_count",),
                ),
                ColumnMapping(
                    phrase="running inefficiently",
                    targets=("metrics.efficiency_pct",),
                ),
            ),
            domain_knowledge=None,
        )
        response = KnowledgeGroundingResponse(
            column_mapping=(
                ColumnMapping(
                    phrase="breaking down all the time",
                    targets=("metrics.mtbf_hours",),
                ),
                ColumnMapping(
                    phrase="running inefficiently",
                    targets=("metrics.efficiency_pct",),
                ),
            ),
            selected_knowledge_ids=(47,),
        )
        _, result = await self._process(
            old=old,
            response=response,
            definitions=[
                {"id": 47, "knowledge": "MTBF", "definition": rule}
            ],
            meanings={
                "metrics": {
                    "failure_count": "Failure count.",
                    "mtbf_hours": "Mean time between failures.",
                    "efficiency_pct": "Efficiency percentage.",
                }
            },
        )

        self.assertEqual(result.state_update.status, "accepted")
        mappings = {
            item.phrase: item.targets
            for item in result.runtime.grounding_state.column_mapping or ()
        }
        self.assertEqual(
            mappings["breaking down all the time"], ("metrics.mtbf_hours",)
        )
        self.assertEqual(
            mappings["running inefficiently"], ("metrics.efficiency_pct",)
        )

    def test_knowledge_prompt_freezes_unsupported_mapping_changes(self) -> None:
        self.assertIn(
            "selected=[] 时，mapping 必须",
            KNOWLEDGE_GROUNDING_PROMPT,
        )
        self.assertIn("逐项、逐字、顺序完全不变", KNOWLEDGE_GROUNDING_PROMPT)
        self.assertIn(
            "只能修改该 knowledge 明确、直接证明错误的 phrase",
            KNOWLEDGE_GROUNDING_PROMPT,
        )
        self.assertIn("其他 phrase 必须原样保持", KNOWLEDGE_GROUNDING_PROMPT)


class FrozenContractTests(unittest.TestCase):
    def test_form_schema_hash_is_unchanged(self) -> None:
        self.assertEqual(
            SQL_GROUNDING_FORM_SCHEMA_SHA256,
            "3033213479034eb0b8879ae67145e9c34a1438b8f6391e956365bdec3c795afd",
        )

    def test_prompt_and_configuration_hashes_match_current_contract(self) -> None:
        self.assertEqual(
            SQL_GROUNDING_PROMPT_SHA256,
            "abcd64292037ba6fa5f6672c04383d47f9742da0ae63763afd66cc4ee8affccd",
        )
        self.assertEqual(
            SQL_GROUNDING_CONFIGURATION_SHA256,
            "f286cc3b0cf2361437d7503f6ec1eec24f2a2638e285bc23de59038eb5ee0110",
        )


if __name__ == "__main__":
    unittest.main()
