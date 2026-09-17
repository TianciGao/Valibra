from __future__ import annotations

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import patch

from google.adk.sessions.state import State as ADKState

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    FinalRegroundingGateResponse,
    GroundingCheckClarificationProposal,
    GroundingCheckResponse,
    GroundingCheckToolRequest,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    MappingGroundingResponse,
    SQLGroundingState,
    StructureGroundingResponse,
    UnresolvedMapping,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    SQL_GROUNDING_STAGE_FORM_SCHEMAS,
    SQL_GROUNDING_STAGE_PROMPTS,
    GroundingUpdaterResult,
    classify_grounding_input,
)


QUERY = "Show metric."
SCHEMA = """CREATE TABLE old_table (
metric NUMERIC
);

CREATE TABLE new_table (
metric NUMERIC
);"""
COLUMN_MEANINGS = json.dumps(
    {
        "atomic|old_table|metric": "Old metric.",
        "atomic|new_table|metric": "Canonical new metric.",
    },
    sort_keys=True,
)


def _telemetry() -> GroundingLLMTelemetry:
    return GroundingLLMTelemetry(
        attempted=True,
        status="succeeded",
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
        form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


def _tool_event(tool: str, result: str) -> dict:
    return {
        "type": "tool",
        "tool": tool,
        "phase": 1,
        "args": {},
        "result": result,
        "cost": 0.0,
        "budget_before": None,
        "budget_after": None,
        "action_input_tokens": 0,
        "action_output_tokens": 0,
        "timestamp": "2026-08-29T00:00:00Z",
    }


def _formal_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("old_table",),
        join_keys=(),
        column_mapping=(
            ColumnMapping(phrase="metric", targets=("old_table.metric",)),
        ),
        domain_knowledge=(),
    )


def _state() -> dict:
    runtime = GroundingRuntime(
        grounding_revision=7,
        stage="INITIAL_GROUNDING",
        focus_dimension="none",
        grounding_state=_formal_state(),
    )
    return {
        "task_id": "atomic-final-gate",
        "current_phase": 1,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 100.0,
        "initial_budget": 100.0,
        "tool_trajectory": [
            _tool_event("get_schema", SCHEMA),
            _tool_event("get_all_column_meanings", COLUMN_MEANINGS),
            _tool_event("get_all_knowledge_definitions", "[]"),
        ],
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 0,
        grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
            "1": 0,
            "2": 0,
        },
    }


class _Updater:
    def __init__(
        self,
        *,
        fail_kind: str | None = None,
        resolve_omissions: bool = False,
    ) -> None:
        self.fail_kind = fail_kind
        self.resolve_omissions = resolve_omissions
        self.kinds: list[str] = []
        self.inputs: list[dict] = []

    async def propose(self, runtime, observation, *args, **kwargs):
        del args
        self.inputs.append(copy.deepcopy(kwargs["grounding_input"]))
        kind = classify_grounding_input(
            kwargs["grounding_input"],
            phase=observation.phase,
        )
        self.kinds.append(kind)
        if kind == self.fail_kind:
            raise RuntimeError("injected Draft failure")
        if kind == "structure":
            response = StructureGroundingResponse(
                tables=("new_table",),
                join_keys=(),
            )
        elif kind == "mapping":
            unresolved = tuple(
                UnresolvedMapping.model_validate(item)
                for item in kwargs["grounding_input"]["unresolved_mappings"]
            )
            if self.resolve_omissions:
                unresolved = ()
            phrase = "Show" if self.resolve_omissions else "metric"
            response = MappingGroundingResponse(
                tables=runtime.grounding_state.tables or (),
                join_keys=runtime.grounding_state.join_keys or (),
                column_mapping=(
                    ColumnMapping(
                        phrase=phrase,
                        targets=("new_table.metric",),
                    ),
                ),
                unresolved_mappings=unresolved,
            )
        elif kind == "knowledge":
            response = KnowledgeGroundingResponse(
                column_mapping=runtime.grounding_state.column_mapping or (),
                selected_knowledge_ids=(),
            )
        elif kind == "check":
            response = GroundingCheckResponse(
                status="complete",
                clarification_route="none",
                missing_information=None,
                next_tool=None,
                column_mapping=runtime.grounding_state.column_mapping or (),
                domain_knowledge=runtime.grounding_state.domain_knowledge or (),
            )
        else:
            response = FinalRegroundingGateResponse(
                decision="NO_REGROUND",
                reason="The narrow literal does not invalidate State.",
            )
        return GroundingUpdaterResult(
            response=response,
            telemetry=_telemetry(),
            transport_normalization="none",
        )


class _OfficialLoopUpdater(_Updater):
    def __init__(
        self,
        *,
        terminal_after_tool: bool = False,
        resolve_omissions: bool = False,
    ) -> None:
        super().__init__(resolve_omissions=resolve_omissions)
        self.terminal_after_tool = terminal_after_tool
        self.inputs: list[dict] = []

    async def propose(self, runtime, observation, *args, **kwargs):
        grounding_input = kwargs["grounding_input"]
        kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
        self.kinds.append(kind)
        self.inputs.append(copy.deepcopy(grounding_input))
        if kind == "mapping":
            unresolved = tuple(
                UnresolvedMapping.model_validate(item)
                for item in grounding_input["unresolved_mappings"]
            )
            column_mapping = runtime.grounding_state.column_mapping or ()
            if self.resolve_omissions:
                unresolved = ()
                column_mapping = (
                    ColumnMapping(
                        phrase="Show",
                        targets=("old_table.metric",),
                    ),
                )
            response = MappingGroundingResponse(
                tables=runtime.grounding_state.tables or (),
                join_keys=runtime.grounding_state.join_keys or (),
                column_mapping=column_mapping,
                unresolved_mappings=unresolved,
            )
        elif kind == "knowledge":
            response = KnowledgeGroundingResponse(
                column_mapping=runtime.grounding_state.column_mapping or (),
                selected_knowledge_ids=(),
            )
        elif kind == "check" and "check_context" in grounding_input:
            response = GroundingCheckResponse(
                status="incomplete",
                clarification_route="none",
                missing_information="The canonical metric meaning is unresolved.",
                next_tool=GroundingCheckToolRequest(
                    tool_name="get_column_meaning",
                    arguments={"table_name": "old_table", "column_name": "metric"},
                ),
                column_mapping=runtime.grounding_state.column_mapping or (),
                domain_knowledge=runtime.grounding_state.domain_knowledge or (),
            )
        elif kind == "check" and "latest_tool" in grounding_input:
            if self.terminal_after_tool:
                response = GroundingCheckResponse(
                    status="incomplete",
                    clarification_route="terminal",
                    missing_information="Official evidence cannot resolve the gap.",
                    next_tool=None,
                    column_mapping=runtime.grounding_state.column_mapping or (),
                    domain_knowledge=runtime.grounding_state.domain_knowledge or (),
                )
            else:
                response = GroundingCheckResponse(
                    status="complete",
                    clarification_route="none",
                    missing_information=None,
                    next_tool=None,
                    column_mapping=runtime.grounding_state.column_mapping or (),
                    domain_knowledge=runtime.grounding_state.domain_knowledge or (),
                )
        else:
            raise AssertionError(kind)
        return GroundingUpdaterResult(
            response=response,
            telemetry=_telemetry(),
            transport_normalization="none",
        )


class _DraftClarificationUpdater(_Updater):
    def __init__(
        self,
        phrase: str,
        kind: str,
        *,
        requirement_type: str | None = None,
        related_mapping_phrases: tuple[str, ...] = (),
        resolve_omissions: bool = False,
    ) -> None:
        super().__init__()
        self.phrase = phrase
        self.kind = kind
        self.requirement_type = requirement_type
        self.related_mapping_phrases = related_mapping_phrases
        self.resolve_omissions = resolve_omissions

    async def propose(self, runtime, observation, *args, **kwargs):
        del args
        grounding_input = kwargs["grounding_input"]
        self.inputs.append(copy.deepcopy(grounding_input))
        kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
        self.kinds.append(kind)
        if kind == "mapping":
            unresolved = tuple(
                UnresolvedMapping.model_validate(item)
                for item in grounding_input["unresolved_mappings"]
            )
            column_mapping = runtime.grounding_state.column_mapping or ()
            if self.resolve_omissions:
                resolved = unresolved
                unresolved = ()
                target = column_mapping[0].targets
                column_mapping = column_mapping + tuple(
                    ColumnMapping(phrase=item.phrase, targets=target)
                    for item in resolved
                    if item.phrase
                    not in {mapping.phrase for mapping in column_mapping}
                )
            response = MappingGroundingResponse(
                tables=runtime.grounding_state.tables or (),
                join_keys=runtime.grounding_state.join_keys or (),
                column_mapping=column_mapping,
                unresolved_mappings=unresolved,
            )
        elif kind == "knowledge":
            response = KnowledgeGroundingResponse(
                column_mapping=runtime.grounding_state.column_mapping or (),
                selected_knowledge_ids=(),
            )
        elif kind == "check":
            response = GroundingCheckResponse(
                status="incomplete",
                clarification_route="none",
                missing_information=f"User intent for {self.phrase} is unresolved.",
                next_tool=GroundingCheckToolRequest(
                    tool_name="ask_user",
                    arguments={
                        "question": f"What should {self.phrase} mean?",
                    },
                    user_clarification_request=(
                        GroundingCheckClarificationProposal(
                            phrase=self.phrase,
                            kind=self.kind,
                            requirement_type=self.requirement_type,
                            related_mapping_phrases=(
                                self.related_mapping_phrases
                            ),
                        )
                    ),
                ),
                column_mapping=runtime.grounding_state.column_mapping or (),
                domain_knowledge=runtime.grounding_state.domain_knowledge or (),
            )
        else:
            raise AssertionError(kind)
        return GroundingUpdaterResult(
            response=response,
            telemetry=_telemetry(),
            transport_normalization="none",
        )


class _AtomicAnswerStayUpdater(_Updater):
    async def propose(self, runtime, observation, *args, **kwargs):
        del args
        grounding_input = kwargs["grounding_input"]
        self.inputs.append(copy.deepcopy(grounding_input))
        kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
        self.kinds.append(kind)
        if kind != "check":
            raise AssertionError(kind)
        response = GroundingCheckResponse(
            status="incomplete",
            clarification_route="stay_check",
            missing_information="More Official evidence might help.",
            next_tool=GroundingCheckToolRequest(
                tool_name="get_all_external_knowledge_names",
                arguments={},
            ),
            column_mapping=runtime.grounding_state.column_mapping or (),
            domain_knowledge=runtime.grounding_state.domain_knowledge or (),
        )
        return GroundingUpdaterResult(
            response=response,
            telemetry=_telemetry(),
            transport_normalization="none",
        )


class FinalRegroundingGateAtomicTests(unittest.IsolatedAsyncioTestCase):
    async def _run_draft_clarification_handoff(
        self,
        *,
        requested_phrase: str,
        omission_phrase: str = "Show",
        answered_phrase: str = "metric",
        requested_kind: str = "user_intent",
        requirement_type: str | None = None,
        related_mapping_phrases: tuple[str, ...] = (),
        resolve_omissions: bool = False,
    ):
        state = _state()
        grounding_callbacks._store_mapping_omission_carrier(
            state,
            grounding_callbacks._MappingOmissionCarrier(
                phase=1,
                grounding_revision=7,
                query_sha256=grounding_callbacks._mapping_omission_query_sha256(
                    QUERY, None
                ),
                mapping_evidence_sha256="a" * 64,
                unresolved_mappings=(
                    UnresolvedMapping(
                        phrase=omission_phrase,
                        reason="ambiguous_user_intent",
                    ),
                ),
            ),
        )
        first_question = f"What should {answered_phrase} mean?"
        grounding_callbacks._register_clarification_requests(
            state,
            phase=1,
            requests=(
                grounding_callbacks.UserClarificationRequest(
                    phrase=answered_phrase,
                    kind="user_intent",
                    question=first_question,
                ),
            ),
        )
        grounding_callbacks._record_clarification_answer(
            state,
            phase=1,
            question=first_question,
            answer="Use the clarified business meaning.",
        )
        formal_payload = copy.deepcopy(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        formal_carrier_payload = copy.deepcopy(
            state[grounding_callbacks.MAPPING_OMISSION_CARRIER_KEY]
        )
        formal = GroundingRuntime.model_validate(formal_payload)
        observation = build_sql_grounding_observation(
            task_id=state["task_id"],
            phase=1,
            sequence=grounding_callbacks._next_sequence(state),
            observation_type="user_answer",
            content="Use the clarified business meaning.",
            summary="answered first clarification",
            tool_name="ask_user",
            function_call_id="atomic-answer",
        )
        updater = _DraftClarificationUpdater(
            requested_phrase,
            requested_kind,
            requirement_type=requirement_type,
            related_mapping_phrases=related_mapping_phrases,
            resolve_omissions=resolve_omissions,
        )
        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}),
                patch.object(
                    grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater
                ),
            ):
                result = await grounding_callbacks._run_atomic_regrounding_draft(
                    state,
                    formal_result=grounding_callbacks._ObservationResult(
                        runtime=formal,
                        service_status="final_regrounding_gate_pending",
                        observation=observation,
                    ),
                    decision="REGROUND_MAPPING",
                    synchronization=(
                        grounding_callbacks._TaskGroundingSynchronization(
                            lock=asyncio.Lock()
                        )
                    ),
                    query=QUERY,
                    follow_up=None,
                )
        finally:
            grounding_callbacks._reset_turn_message(token)
        return (
            state,
            formal_payload,
            formal_carrier_payload,
            updater,
            result,
        )

    async def _run_mapping_official_loop(
        self,
        *,
        terminal_after_tool: bool,
        state=None,
        resolve_omissions: bool = False,
    ):
        state = _state() if state is None else state
        formal_payload = copy.deepcopy(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        formal = GroundingRuntime.model_validate(formal_payload)
        observation = build_sql_grounding_observation(
            task_id=state["task_id"],
            phase=1,
            sequence=grounding_callbacks._next_sequence(state),
            observation_type="user_answer",
            content="Use the clarified metric semantics.",
            summary="answered clarification",
            tool_name="ask_user",
            function_call_id="atomic-answer",
        )
        updater = _OfficialLoopUpdater(
            terminal_after_tool=terminal_after_tool,
            resolve_omissions=resolve_omissions,
        )
        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}),
                patch.object(
                    grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater
                ),
            ):
                waiting = await grounding_callbacks._run_atomic_regrounding_draft(
                    state,
                    formal_result=grounding_callbacks._ObservationResult(
                        runtime=formal,
                        service_status="final_regrounding_gate_pending",
                        observation=observation,
                        service_result=None,
                    ),
                    decision="REGROUND_MAPPING",
                    synchronization=(
                        grounding_callbacks._TaskGroundingSynchronization(
                            lock=asyncio.Lock()
                        )
                    ),
                    query=QUERY,
                    follow_up=None,
                )
                self.assertEqual(
                    waiting.service_status,
                    "atomic_regrounding_waiting_official",
                )
                self.assertEqual(
                    state[grounding_callbacks.GROUNDING_RUNTIME_KEY],
                    formal_payload,
                )
                pending = grounding_callbacks._pending_check_tool(state)
                self.assertIsNotNone(pending)
                assert pending is not None
                if resolve_omissions:
                    formal_carrier = grounding_callbacks._mapping_omission_carrier(
                        state
                    )
                    self.assertIsNotNone(formal_carrier)
                    assert formal_carrier is not None
                    self.assertEqual(
                        [item.phrase for item in formal_carrier.unresolved_mappings],
                        ["Show"],
                    )
                    draft_record = grounding_callbacks._atomic_regrounding_draft(
                        state
                    )
                    self.assertIsNotNone(draft_record)
                    assert draft_record is not None
                    self.assertEqual(draft_record.draft_unresolved_mappings, ())
                consumed = grounding_callbacks._consume_pending_check_tool(
                    state,
                    function_call_id=pending.function_call_id,
                    tool_name=pending.tool_name,
                )
                self.assertEqual(consumed, pending)
                official_observation = build_sql_grounding_observation(
                    task_id=state["task_id"],
                    phase=1,
                    sequence=grounding_callbacks._next_sequence(state),
                    observation_type="metadata",
                    content="Old metric.",
                    summary="Official metadata returned",
                    tool_name="get_column_meaning",
                    function_call_id=pending.function_call_id,
                )
                final = await (
                    grounding_callbacks._continue_atomic_regrounding_draft_after_official(
                        state,
                        pending_check=pending,
                        observation=official_observation,
                        latest_tool_arguments=pending.arguments,
                        latest_tool_result="Old metric.",
                    )
                )
        finally:
            grounding_callbacks._reset_turn_message(token)
        return state, formal_payload, updater, final

    def test_atomic_draft_snapshot_keeps_plain_dict_compatibility(self) -> None:
        state = _state()
        state["nested_probe"] = {"items": ["formal"]}

        snapshot = grounding_callbacks._snapshot_state_for_atomic_draft(state)
        snapshot["nested_probe"]["items"].append("draft")

        self.assertEqual(state["nested_probe"], {"items": ["formal"]})
        self.assertEqual(snapshot["nested_probe"]["items"], ["formal", "draft"])

    def test_atomic_draft_snapshot_supports_real_adk_state_and_delta(self) -> None:
        value = _state()
        value["nested_probe"] = {"items": ["formal"]}
        state = ADKState(
            value=value,
            delta={"delta_probe": {"items": ["pending"]}},
        )

        snapshot = grounding_callbacks._snapshot_state_for_atomic_draft(state)
        snapshot["nested_probe"]["items"].append("draft")
        snapshot["delta_probe"]["items"].append("draft")

        self.assertEqual(state["nested_probe"], {"items": ["formal"]})
        self.assertEqual(state["delta_probe"], {"items": ["pending"]})
        self.assertEqual(snapshot["delta_probe"]["items"], ["pending", "draft"])

    async def test_real_adk_state_runs_atomic_draft_to_commit(self) -> None:
        state = ADKState(value=_state(), delta={"adk_pending_delta": True})

        state, _formal_payload, updater, final = (
            await self._run_mapping_official_loop(
                terminal_after_tool=False,
                state=state,
            )
        )

        self.assertEqual(updater.kinds, ["mapping", "knowledge", "check", "check"])
        self.assertEqual(final.service_status, "atomic_regrounding_committed")
        self.assertTrue(state["adk_pending_delta"])

    async def test_official_continuation_uses_draft_carrier_after_mapping_clear(
        self,
    ) -> None:
        raw_state = _state()
        grounding_callbacks._store_mapping_omission_carrier(
            raw_state,
            grounding_callbacks._MappingOmissionCarrier(
                phase=1,
                grounding_revision=7,
                query_sha256=grounding_callbacks._mapping_omission_query_sha256(
                    QUERY, None
                ),
                mapping_evidence_sha256="a" * 64,
                unresolved_mappings=(
                    UnresolvedMapping(
                        phrase="Show",
                        reason="no_direct_metadata",
                    ),
                ),
            ),
        )
        state = ADKState(value=raw_state, delta={})

        state, _formal_payload, updater, final = (
            await self._run_mapping_official_loop(
                terminal_after_tool=False,
                state=state,
                resolve_omissions=True,
            )
        )

        self.assertEqual(updater.kinds, ["mapping", "knowledge", "check", "check"])
        self.assertEqual(updater.inputs[-1]["unresolved_mappings"], [])
        self.assertEqual(final.service_status, "atomic_regrounding_committed")
        committed_carrier = grounding_callbacks._mapping_omission_carrier(state)
        self.assertIsNotNone(committed_carrier)
        assert committed_carrier is not None
        self.assertEqual(committed_carrier.unresolved_mappings, ())

    async def test_draft_new_omission_clarification_rolls_back_to_outer_tool(
        self,
    ) -> None:
        state, formal_payload, carrier_payload, updater, result = (
            await self._run_draft_clarification_handoff(
                requested_phrase="Show",
                requested_kind="missing_knowledge",
            )
        )

        self.assertEqual(updater.kinds, ["mapping", "knowledge", "check"])
        self.assertEqual(
            result.service_status,
            "atomic_draft_needs_clarification",
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY], formal_payload
        )
        self.assertEqual(
            state[grounding_callbacks.MAPPING_OMISSION_CARRIER_KEY],
            carrier_payload,
        )
        self.assertIsNone(
            state.get(grounding_callbacks.ATOMIC_REGROUNDING_DRAFT_KEY)
        )
        pending = grounding_callbacks._pending_check_tool(state)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.tool_name, "ask_user")
        self.assertEqual(pending.arguments, {"question": "What should Show mean?"})
        self.assertEqual(pending.origin, "atomic_draft")
        self.assertIsNone(pending.requirement_type)
        self.assertEqual(pending.related_mapping_phrases, ())
        records = grounding_callbacks._clarification_records(state)
        self.assertEqual([item.phrase for item in records], ["metric", "Show"])
        self.assertEqual(records[1].kind, "missing_knowledge")
        self.assertIsNotNone(records[0].answer)
        self.assertIsNone(records[1].answer)
        self.assertIsNone(grounding_callbacks._phase_grounding_failure(state, 1))
        audit = state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1]
        self.assertEqual(audit["outcome"], "ATOMIC_DRAFT_NEEDS_CLARIFICATION")
        self.assertTrue(audit["rolled_back"])
        self.assertFalse(audit["committed"])
        self.assertTrue(audit["formal_state_unchanged"])

    async def test_draft_check_requirement_event_rolls_back_to_outer_tool(
        self,
    ) -> None:
        state, formal_payload, carrier_payload, _updater, result = (
            await self._run_draft_clarification_handoff(
                requested_phrase="Show",
                omission_phrase="Show metric",
                answered_phrase="Show metric",
                requirement_type="threshold",
                related_mapping_phrases=("metric",),
                resolve_omissions=True,
            )
        )

        self.assertEqual(result.service_status, "atomic_draft_needs_clarification")
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY], formal_payload
        )
        self.assertEqual(
            state[grounding_callbacks.MAPPING_OMISSION_CARRIER_KEY],
            carrier_payload,
        )
        self.assertIsNone(
            state.get(grounding_callbacks.ATOMIC_REGROUNDING_DRAFT_KEY)
        )
        pending = grounding_callbacks._pending_check_tool(state)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.origin, "atomic_draft")
        self.assertEqual(pending.requirement_type, "threshold")
        self.assertEqual(pending.related_mapping_phrases, ("metric",))
        audit = state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1]
        self.assertEqual(audit["outcome"], "ATOMIC_DRAFT_NEEDS_CLARIFICATION")
        self.assertEqual(audit["clarification_event_origin"], "atomic_draft")
        self.assertEqual(audit["clarification_requirement_type"], "threshold")
        self.assertEqual(
            audit["clarification_related_mapping_phrases"], ["metric"]
        )

    async def test_draft_check_requirement_relation_must_exist_in_mapping(
        self,
    ) -> None:
        state, _formal_payload, _carrier_payload, _updater, result = (
            await self._run_draft_clarification_handoff(
                requested_phrase="Show",
                omission_phrase="Show metric",
                answered_phrase="Show metric",
                requirement_type="threshold",
                related_mapping_phrases=("missing mapping phrase",),
                resolve_omissions=True,
            )
        )

        self.assertEqual(result.service_status, "atomic_regrounding_terminal")
        self.assertEqual(
            result.control_error_type,
            "AtomicDraftCheckRequirementRelationInvalid",
        )
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))

    async def test_draft_check_requirement_cannot_bypass_mapping_omission(
        self,
    ) -> None:
        state, _formal_payload, _carrier_payload, _updater, result = (
            await self._run_draft_clarification_handoff(
                requested_phrase="metric",
                omission_phrase="Show",
                answered_phrase="Show",
                requirement_type="category",
                related_mapping_phrases=("metric",),
            )
        )

        self.assertEqual(result.service_status, "atomic_regrounding_terminal")
        self.assertEqual(
            result.control_error_type,
            "AtomicDraftCheckRequirementBlockedByMappingOmission",
        )
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))

    async def test_draft_check_requirement_cannot_reask_answered_phrase(
        self,
    ) -> None:
        state, _formal_payload, _carrier_payload, _updater, result = (
            await self._run_draft_clarification_handoff(
                requested_phrase="Show",
                omission_phrase="Show",
                answered_phrase="Show",
                requirement_type="literal",
                related_mapping_phrases=("metric",),
                resolve_omissions=True,
            )
        )

        self.assertEqual(result.service_status, "atomic_regrounding_terminal")
        self.assertEqual(
            result.control_error_type,
            "AtomicDraftClarificationPhraseNotNew",
        )
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))

    async def test_draft_rolls_back_before_registering_outer_requirement(self) -> None:
        original = grounding_callbacks._schedule_check_tool
        draft_was_absent: list[bool] = []

        def checked_schedule(state, **kwargs):
            draft_was_absent.append(
                grounding_callbacks._atomic_regrounding_draft(state) is None
            )
            return original(state, **kwargs)

        with patch.object(
            grounding_callbacks,
            "_schedule_check_tool",
            side_effect=checked_schedule,
        ):
            _state_value, _formal, _carrier, _updater, result = (
                await self._run_draft_clarification_handoff(
                    requested_phrase="Show",
                    omission_phrase="Show metric",
                    answered_phrase="Show metric",
                    requirement_type="predicate",
                    related_mapping_phrases=("metric",),
                    resolve_omissions=True,
                )
            )

        self.assertEqual(result.service_status, "atomic_draft_needs_clarification")
        self.assertEqual(draft_was_absent, [True])

    async def test_atomic_requirement_answer_forces_fresh_gate_not_stay_check(
        self,
    ) -> None:
        state = _state()
        question = "What threshold should Show use?"
        grounding_callbacks._register_clarification_requests(
            state,
            phase=1,
            requests=(
                grounding_callbacks.UserClarificationRequest(
                    phrase="Show",
                    kind="user_intent",
                    question=question,
                ),
            ),
        )
        grounding_callbacks._record_clarification_answer(
            state,
            phase=1,
            question=question,
            answer="Use more than 10.",
        )
        pending = grounding_callbacks._PendingCheckTool(
            phase=1,
            function_call_id="atomic-answer",
            missing_information="The threshold for Show is unresolved.",
            tool_name="ask_user",
            arguments={"question": question},
            request_digest="a" * 64,
            origin="atomic_draft",
            requirement_type="threshold",
            related_mapping_phrases=("metric",),
        )
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        grounding_input = grounding_callbacks._build_check_grounding_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=1,
            latest_user_answer="Use more than 10.",
            paired_check_tool=pending,
        )
        self.assertEqual(
            grounding_input["latest_user_answer"]["clarification_event"],
            {
                "origin": "atomic_draft",
                "requirement_type": "threshold",
                "related_mapping_phrases": ["metric"],
            },
        )
        observation = build_sql_grounding_observation(
            task_id=state["task_id"],
            phase=1,
            sequence=grounding_callbacks._next_sequence(state),
            observation_type="user_answer",
            content="Use more than 10.",
            summary="answered atomic Draft requirement",
            tool_name="ask_user",
            function_call_id="atomic-answer",
        )
        updater = _AtomicAnswerStayUpdater()
        fresh_draft_calls = []

        async def run_fresh_draft(state_arg, *, formal_result, **kwargs):
            del kwargs
            self.assertIs(state_arg, state)
            self.assertIsNone(
                grounding_callbacks._atomic_regrounding_draft(state_arg)
            )
            self.assertIsNone(grounding_callbacks._pending_check_tool(state_arg))
            fresh_draft_calls.append(formal_result)
            return grounding_callbacks._ObservationResult(
                runtime=formal_result.runtime,
                service_status="fresh_atomic_draft_started",
                observation=formal_result.observation,
                service_result=formal_result.service_result,
                control_status="succeeded",
            )

        gate_result = GroundingUpdaterResult(
            response=FinalRegroundingGateResponse(
                decision="REGROUND_MAPPING",
                reason="The answered event must be applied to a fresh Draft.",
            ),
            telemetry=_telemetry(),
            transport_normalization="none",
        )
        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}),
                patch.object(
                    grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater
                ),
                patch.object(
                    grounding_callbacks,
                    "_propose_final_regrounding_gate",
                    return_value=gate_result,
                ),
                patch.object(
                    grounding_callbacks,
                    "_run_atomic_regrounding_draft",
                    side_effect=run_fresh_draft,
                ),
            ):
                result = await grounding_callbacks._handle_observation(
                    state,
                    observation,
                    runtime,
                    grounding_input=grounding_input,
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(updater.kinds, ["check"])
        self.assertEqual(result.service_status, "fresh_atomic_draft_started")
        self.assertEqual(len(fresh_draft_calls), 1)
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))
        self.assertIsNone(grounding_callbacks._atomic_regrounding_draft(state))
        forced_result = fresh_draft_calls[0]
        self.assertIsNotNone(forced_result.service_result)
        assert forced_result.service_result is not None
        response = forced_result.service_result.response
        self.assertIsInstance(response, GroundingCheckResponse)
        assert isinstance(response, GroundingCheckResponse)
        self.assertEqual(response.clarification_route, "restart_grounding")
        self.assertIsNone(response.next_tool)
        audit = state[grounding_callbacks.GROUNDING_CHECK_AUDITS_KEY][-1]
        self.assertEqual(
            audit["blocked_reason"],
            "atomic_draft_clarification_fresh_gate",
        )

    def test_check_requirement_type_rejects_business_formula(self) -> None:
        with self.assertRaises(ValueError):
            GroundingCheckClarificationProposal(
                phrase="Show",
                kind="user_intent",
                requirement_type="business_formula",
                related_mapping_phrases=("metric",),
            )

    def test_check_requirement_event_form_and_prompt_are_narrow(self) -> None:
        clarification_schema = SQL_GROUNDING_STAGE_FORM_SCHEMAS["check"][
            "$defs"
        ]["GroundingCheckClarificationProposal"]
        self.assertEqual(
            set(clarification_schema["properties"]),
            {
                "phrase",
                "kind",
                "requirement_type",
                "related_mapping_phrases",
            },
        )
        requirement_schema = clarification_schema["properties"][
            "requirement_type"
        ]
        self.assertEqual(
            set(requirement_schema["anyOf"][0]["enum"]),
            {"threshold", "category", "literal", "predicate"},
        )
        prompt = SQL_GROUNDING_STAGE_PROMPTS["check"]
        for fragment in (
            "related_mapping_phrases 必须非空",
            "逐项精确复用 current_state.column_mapping",
            "不得用于索取新的字段语义、formula、derived rule",
            "如果仍有 unresolved_mappings，不得使用这个 Check-level event",
            "产生问题的 private Draft 已经",
        ):
            self.assertIn(fragment, prompt)

    async def test_draft_cannot_reask_an_answered_omission_phrase(self) -> None:
        state, formal_payload, carrier_payload, _updater, result = (
            await self._run_draft_clarification_handoff(
                requested_phrase="Show",
                omission_phrase="Show",
                answered_phrase="Show",
            )
        )

        self.assertEqual(result.service_status, "atomic_regrounding_terminal")
        self.assertEqual(
            result.control_error_type,
            "AtomicDraftClarificationPhraseNotNew",
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY], formal_payload
        )
        self.assertEqual(
            state[grounding_callbacks.MAPPING_OMISSION_CARRIER_KEY],
            carrier_payload,
        )
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))

    async def test_draft_cannot_ask_phrase_outside_omission_carrier(self) -> None:
        state, formal_payload, carrier_payload, _updater, result = (
            await self._run_draft_clarification_handoff(
                requested_phrase="metric",
                omission_phrase="Show",
                answered_phrase="Show",
            )
        )

        self.assertEqual(result.service_status, "atomic_regrounding_terminal")
        self.assertEqual(
            result.control_error_type,
            "AtomicDraftClarificationNotUnresolved",
        )
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY], formal_payload
        )
        self.assertEqual(
            state[grounding_callbacks.MAPPING_OMISSION_CARRIER_KEY],
            carrier_payload,
        )
        self.assertIsNone(grounding_callbacks._pending_check_tool(state))

    async def test_incomplete_draft_runs_official_loop_then_commits_once(self) -> None:
        state, formal_payload, updater, final = await self._run_mapping_official_loop(
            terminal_after_tool=False
        )
        self.assertEqual(updater.kinds, ["mapping", "knowledge", "check", "check"])
        self.assertEqual(final.service_status, "atomic_regrounding_committed")
        self.assertEqual(final.runtime.grounding_revision, 7)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY],
            formal_payload,
        )
        self.assertIsNone(
            state.get(grounding_callbacks.ATOMIC_REGROUNDING_DRAFT_KEY)
        )
        self.assertTrue(
            all("regrounding_context" in item for item in updater.inputs)
        )
        self.assertIn("latest_tool", updater.inputs[-1])
        audit = state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1]
        self.assertTrue(audit["committed"])
        self.assertEqual(audit["official_call_count"], 1)

    async def test_terminal_after_official_discards_draft_and_keeps_s0(self) -> None:
        state, formal_payload, _updater, final = await self._run_mapping_official_loop(
            terminal_after_tool=True
        )
        self.assertEqual(final.service_status, "atomic_regrounding_terminal")
        self.assertEqual(final.runtime.grounding_revision, 7)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY],
            formal_payload,
        )
        self.assertIsNone(
            state.get(grounding_callbacks.ATOMIC_REGROUNDING_DRAFT_KEY)
        )
        audit = state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1]
        self.assertTrue(audit["formal_state_unchanged"])

    async def test_structure_draft_allows_transient_mixed_state_and_commits_once(
        self,
    ) -> None:
        state = _state()
        original_omission = grounding_callbacks._MappingOmissionCarrier(
            phase=1,
            grounding_revision=7,
            query_sha256=grounding_callbacks._mapping_omission_query_sha256(
                QUERY, None
            ),
            mapping_evidence_sha256="a" * 64,
            unresolved_mappings=(
                UnresolvedMapping(
                    phrase="Show",
                    reason="no_direct_metadata",
                ),
            ),
        )
        grounding_callbacks._store_mapping_omission_carrier(
            state, original_omission
        )
        formal = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        observation = build_sql_grounding_observation(
            task_id=state["task_id"],
            phase=1,
            sequence=grounding_callbacks._next_sequence(state),
            observation_type="user_answer",
            content="Use the new metric definition.",
            summary="answered clarification",
            tool_name="ask_user",
            function_call_id="atomic-answer",
        )
        result = grounding_callbacks._ObservationResult(
            runtime=formal,
            service_status="final_regrounding_gate_pending",
            observation=observation,
        )
        updater = _Updater(resolve_omissions=True)
        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
            ):
                committed = await grounding_callbacks._run_atomic_regrounding_draft(
                    state,
                    formal_result=result,
                    decision="REGROUND_STRUCTURE",
                    synchronization=grounding_callbacks._TaskGroundingSynchronization(
                        lock=asyncio.Lock()
                    ),
                    query=QUERY,
                    follow_up=None,
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(updater.kinds, ["structure", "mapping", "knowledge", "check"])
        knowledge_input = next(
            item
            for item in updater.inputs
            if classify_grounding_input(item, phase=1) == "knowledge"
        )
        self.assertEqual(knowledge_input["unresolved_mappings"], [])
        self.assertEqual(committed.service_status, "atomic_regrounding_committed")
        self.assertEqual(committed.runtime.grounding_revision, 8)
        self.assertEqual(committed.runtime.grounding_state.tables, ("new_table",))
        self.assertEqual(
            committed.runtime.grounding_state.column_mapping[0].targets,
            ("new_table.metric",),
        )
        self.assertEqual(
            committed.runtime.grounding_state.column_mapping[0].phrase,
            "Show",
        )
        # Direct Draft work never persisted an intermediate Runtime.
        self.assertEqual(
            GroundingRuntime.model_validate(
                state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
            ),
            formal,
        )
        audit = state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1]
        self.assertTrue(audit["committed"])
        self.assertEqual(audit["formal_revision_delta"], 1)
        self.assertEqual(audit["draft_stages"][0]["stage"], "structure")
        committed_omissions = grounding_callbacks._mapping_omission_carrier(state)
        self.assertIsNotNone(committed_omissions)
        self.assertEqual(committed_omissions.unresolved_mappings, ())
        self.assertEqual(committed_omissions.grounding_revision, 8)

    async def test_failed_draft_rolls_back_exact_formal_state(self) -> None:
        state = _state()
        original_omission = grounding_callbacks._MappingOmissionCarrier(
            phase=1,
            grounding_revision=7,
            query_sha256=grounding_callbacks._mapping_omission_query_sha256(
                QUERY, None
            ),
            mapping_evidence_sha256="a" * 64,
            unresolved_mappings=(
                UnresolvedMapping(
                    phrase="Show",
                    reason="no_direct_metadata",
                ),
            ),
        )
        grounding_callbacks._store_mapping_omission_carrier(
            state, original_omission
        )
        original_omission_payload = copy.deepcopy(
            state[grounding_callbacks.MAPPING_OMISSION_CARRIER_KEY]
        )
        formal_payload = copy.deepcopy(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        formal = GroundingRuntime.model_validate(formal_payload)
        observation = build_sql_grounding_observation(
            task_id=state["task_id"],
            phase=1,
            sequence=grounding_callbacks._next_sequence(state),
            observation_type="user_answer",
            content="Use the new metric definition.",
            summary="answered clarification",
            tool_name="ask_user",
            function_call_id="atomic-answer",
        )
        updater = _Updater(fail_kind="mapping")
        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}),
                patch.object(
                    grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater
                ),
            ):
                rolled_back = (
                    await grounding_callbacks._run_atomic_regrounding_draft(
                        state,
                        formal_result=grounding_callbacks._ObservationResult(
                            runtime=formal,
                            service_status="final_regrounding_gate_pending",
                            observation=observation,
                        ),
                        decision="REGROUND_STRUCTURE",
                        synchronization=(
                            grounding_callbacks._TaskGroundingSynchronization(
                                lock=asyncio.Lock()
                            )
                        ),
                        query=QUERY,
                        follow_up=None,
                    )
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(rolled_back.runtime, formal)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY],
            formal_payload,
        )
        audit = state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1]
        self.assertFalse(audit["committed"])
        self.assertTrue(audit["rolled_back"])
        self.assertEqual(audit["failed_stage"], "mapping")
        self.assertEqual(
            state[grounding_callbacks.MAPPING_OMISSION_CARRIER_KEY],
            original_omission_payload,
        )

    def test_feature_is_default_off_and_o3_compatibility_identity_is_frozen(
        self,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(grounding_callbacks._final_regrounding_gate_enabled())
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
