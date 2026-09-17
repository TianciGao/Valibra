from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    FinalRegroundingGateResponse,
    GroundingCheckResponse,
    GroundingRuntime,
    SQLGroundingState,
    UnresolvedMapping,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.service import SQLGroundingServiceResult
from valibra_agent.sql_grounding.telemetry import (
    GroundingLLMTelemetry,
    StateUpdateTelemetry,
)
from valibra_agent.sql_grounding.updater import (
    FINAL_REGROUNDING_GATE_FORM_SCHEMA_SHA256,
    FINAL_REGROUNDING_GATE_PROMPT,
    FINAL_REGROUNDING_GATE_PROMPT_SHA256,
    SQL_GROUNDING_CONFIGURATION_SHA256,
    GroundingUpdaterError,
    GroundingUpdaterResult,
    _build_request,
)


QUERY = "Show the average degradation."
FOLLOW_UP = "Return the Current Degradation Factor."
SCHEMA = """CREATE TABLE electrical_performance (
elec_perf_snapshot JSONB
);"""
COLUMN_MEANINGS = json.dumps(
    {
        "owner|electrical_performance|elec_perf_snapshot": {
            "column_meaning": "Electrical measurements.",
            "fields_meaning": {
                "imp_initial_a": "Initial maximum-power current.",
                "imp_now_a": "Current maximum-power current.",
            },
        }
    },
    sort_keys=True,
)
FORMULA = "I_deg = (I_initial - I_now) / I_initial"


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
        "timestamp": "2026-09-06T00:00:00Z",
    }


def _runtime(*, knowledge: tuple[DomainKnowledge, ...] = ()) -> GroundingRuntime:
    return GroundingRuntime(
        grounding_revision=7,
        stage="P2_INCREMENTAL",
        focus_dimension="none",
        grounding_state=SQLGroundingState(
            tables=("electrical_performance",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="initial and current Imp values",
                    targets=(
                        "electrical_performance.elec_perf_snapshot ->> "
                        "'imp_initial_a'",
                        "electrical_performance.elec_perf_snapshot ->> "
                        "'imp_now_a'",
                    ),
                ),
            ),
            domain_knowledge=knowledge,
        ),
    )


def _state(runtime: GroundingRuntime) -> dict:
    state = {
        "task_id": "owner-transition-r1",
        "current_phase": 2,
        "phase1_completed": True,
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
    grounding_callbacks._store_mapping_omission_carrier(
        state,
        grounding_callbacks._MappingOmissionCarrier(
            phase=2,
            grounding_revision=runtime.grounding_revision,
            query_sha256=grounding_callbacks._mapping_omission_query_sha256(
                QUERY,
                FOLLOW_UP,
            ),
            mapping_evidence_sha256="a" * 64,
            unresolved_mappings=(
                UnresolvedMapping(
                    phrase="Current Degradation Factor",
                    reason="derived_rule_required",
                ),
            ),
        ),
    )
    return state


def _check(runtime: GroundingRuntime, *, gap: str | None = None) -> GroundingCheckResponse:
    return GroundingCheckResponse(
        status="incomplete",
        clarification_route="none",
        missing_information=gap
        or (
            "The exact Official formula and both operands are present. "
            "Mapping should re-judge and clear Current Degradation Factor."
        ),
        next_tool=None,
        column_mapping=runtime.grounding_state.column_mapping or (),
        domain_knowledge=runtime.grounding_state.domain_knowledge or (),
    )


def _telemetry() -> GroundingLLMTelemetry:
    return GroundingLLMTelemetry(
        attempted=True,
        status="succeeded",
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        prompt_sha256=FINAL_REGROUNDING_GATE_PROMPT_SHA256,
        form_schema_sha256=FINAL_REGROUNDING_GATE_FORM_SCHEMA_SHA256,
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


def _observation_result(
    runtime: GroundingRuntime,
    check: GroundingCheckResponse,
) -> grounding_callbacks._ObservationResult:
    observation = build_sql_grounding_observation(
        task_id="owner-transition-r1",
        phase=2,
        sequence=8,
        observation_type="p2_follow_up",
        content=FOLLOW_UP,
        summary="P2 final Check",
    )
    state_sha = sql_grounding_state_sha256(runtime.grounding_state)
    service_result = SQLGroundingServiceResult(
        runtime=runtime,
        response=check,
        llm_telemetry=_telemetry(),
        state_update=StateUpdateTelemetry(
            observation_id=observation.observation_id,
            stage=runtime.stage,
            status="noop",
            old_state_sha256=state_sha,
            new_state_sha256=state_sha,
            changed_dimensions=(),
            revision_before=runtime.grounding_revision,
            revision_after=runtime.grounding_revision,
            focus_before=runtime.focus_dimension,
            focus_after=runtime.focus_dimension,
        ),
    )
    return grounding_callbacks._ObservationResult(
        runtime=runtime,
        service_status="final_regrounding_gate_pending",
        observation=observation,
        service_result=service_result,
        control_status="succeeded",
    )


class FinalGateMappingOwnerTransitionR1Tests(unittest.IsolatedAsyncioTestCase):
    def test_gate_receives_typed_mapping_owner_transition_carrier(self) -> None:
        runtime = _runtime(
            knowledge=(DomainKnowledge(kind="business_rule", content=FORMULA),)
        )
        state = _state(runtime)

        request = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=_check(runtime),
        )

        transition = request["final_gate_context"]["mapping_owner_transition"]
        self.assertEqual(
            transition["unresolved_mappings"],
            [
                {
                    "phrase": "Current Degradation Factor",
                    "reason": "derived_rule_required",
                }
            ],
        )
        self.assertEqual(
            transition["eligible_owner_transition_omissions"],
            transition["unresolved_mappings"],
        )
        self.assertTrue(transition["mapping_is_only_owner"])
        self.assertTrue(
            transition["actionable_evidence_changed_since_mapping"]
        )
        self.assertIsNotNone(
            grounding_callbacks._mapping_owner_transition_sha256(
                request,
                phase=2,
            )
        )
        observation = build_sql_grounding_observation(
            task_id="owner-transition-r1",
            phase=2,
            sequence=9,
            observation_type="p2_follow_up",
            content=FOLLOW_UP,
            summary="P2 Final Gate bundle validation",
        )
        validated = _build_request(
            runtime,
            observation,
            original_query=QUERY,
            follow_up_query=FOLLOW_UP,
            grounding_input=request,
        )
        self.assertEqual(validated.call_kind, "final_gate")

    def test_gate_bundle_rejects_tampered_owner_transition_carrier(self) -> None:
        runtime = _runtime(
            knowledge=(DomainKnowledge(kind="business_rule", content=FORMULA),)
        )
        state = _state(runtime)
        request = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=_check(runtime),
        )
        transition = request["final_gate_context"][
            "mapping_owner_transition"
        ]
        transition["eligible_owner_transition_omissions"] = [
            {
                "phrase": "Current Degradation Factor",
                "reason": "no_direct_metadata",
            }
        ]
        observation = build_sql_grounding_observation(
            task_id="owner-transition-r1",
            phase=2,
            sequence=9,
            observation_type="p2_follow_up",
            content=FOLLOW_UP,
            summary="P2 tampered Final Gate bundle",
        )

        with self.assertRaises(GroundingUpdaterError) as caught:
            _build_request(
                runtime,
                observation,
                original_query=QUERY,
                follow_up_query=FOLLOW_UP,
                grounding_input=request,
            )

        self.assertEqual(caught.exception.reason, "grounding_bundle_invalid")
        self.assertFalse(caught.exception.telemetry.attempted)

    def test_only_p2_derived_no_tool_check_is_owner_transition_eligible(self) -> None:
        runtime = _runtime(
            knowledge=(DomainKnowledge(kind="business_rule", content=FORMULA),)
        )
        state = _state(runtime)
        carrier = grounding_callbacks._mapping_omission_carrier(state)
        assert carrier is not None
        grounding_callbacks._store_mapping_omission_carrier(
            state,
            grounding_callbacks._MappingOmissionCarrier(
                phase=2,
                grounding_revision=runtime.grounding_revision,
                query_sha256=carrier.query_sha256,
                mapping_evidence_sha256=carrier.mapping_evidence_sha256,
                unresolved_mappings=(
                    UnresolvedMapping(
                        phrase="Current Degradation Factor",
                        reason="ambiguous_user_intent",
                    ),
                ),
            ),
        )

        request = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=_check(runtime),
        )

        transition = request["final_gate_context"]["mapping_owner_transition"]
        self.assertEqual(transition["eligible_owner_transition_omissions"], [])
        self.assertFalse(
            transition["actionable_evidence_changed_since_mapping"]
        )
        self.assertIsNone(
            grounding_callbacks._mapping_owner_transition_sha256(
                request,
                phase=2,
            )
        )

    def test_same_evidence_and_mapping_input_has_stable_transition_digest(self) -> None:
        runtime = _runtime(
            knowledge=(DomainKnowledge(kind="business_rule", content=FORMULA),)
        )
        state = _state(runtime)
        first = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=_check(runtime, gap="Mapping can clear the omission."),
        )
        second = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=_check(
                runtime,
                gap="The same actionable evidence permits Mapping reassessment.",
            ),
        )

        self.assertEqual(
            grounding_callbacks._mapping_owner_transition_sha256(first, phase=2),
            grounding_callbacks._mapping_owner_transition_sha256(second, phase=2),
        )

    def test_same_mapping_evidence_is_not_a_new_owner_transition(self) -> None:
        runtime = _runtime(
            knowledge=(DomainKnowledge(kind="business_rule", content=FORMULA),)
        )
        state = _state(runtime)
        first = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=_check(runtime),
        )
        context = first["final_gate_context"]
        same_evidence_sha = grounding_callbacks._mapping_actionable_evidence_sha256(
            {
                "column_meanings": context["column_meanings"],
                "user_clarifications": first.get("user_clarifications", []),
                "regrounding_context": {
                    "official_knowledge": first["current_state"][
                        "domain_knowledge"
                    ],
                    "answered_clarifications": context[
                        "answered_clarifications"
                    ],
                },
            }
        )
        carrier = grounding_callbacks._mapping_omission_carrier(state)
        assert carrier is not None
        grounding_callbacks._store_mapping_omission_carrier(
            state,
            grounding_callbacks._MappingOmissionCarrier(
                phase=carrier.phase,
                grounding_revision=carrier.grounding_revision,
                query_sha256=carrier.query_sha256,
                mapping_evidence_sha256=same_evidence_sha,
                unresolved_mappings=carrier.unresolved_mappings,
            ),
        )

        repeated = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=_check(runtime),
        )

        transition = repeated["final_gate_context"]["mapping_owner_transition"]
        self.assertFalse(
            transition["actionable_evidence_changed_since_mapping"]
        )
        self.assertIsNone(
            grounding_callbacks._mapping_owner_transition_sha256(
                repeated,
                phase=2,
            )
        )

    async def test_same_owner_transition_is_terminal_before_second_gate_call(
        self,
    ) -> None:
        runtime = _runtime(
            knowledge=(DomainKnowledge(kind="business_rule", content=FORMULA),)
        )
        state = _state(runtime)
        check = _check(runtime)
        gate_input = grounding_callbacks._build_final_regrounding_gate_request(
            state,
            query=QUERY,
            runtime=runtime,
            phase=2,
            follow_up=FOLLOW_UP,
            final_check=check,
        )
        transition_sha = grounding_callbacks._mapping_owner_transition_sha256(
            gate_input,
            phase=2,
        )
        assert transition_sha is not None
        grounding_callbacks._append_final_regrounding_gate_audit(
            state,
            {
                "phase": 2,
                "decision": "REGROUND_MAPPING",
                "reason": "First owner transition attempt.",
                "provider_status": "succeeded",
                "base_revision": runtime.grounding_revision,
                "committed": False,
                "rolled_back": True,
                "mapping_owner_transition_sha256": transition_sha,
            },
        )
        propose = AsyncMock()
        run_draft = AsyncMock()
        token = grounding_callbacks._bind_turn_message(
            state["task_id"],
            "a-interact",
            QUERY,
        )
        try:
            with (
                patch.object(
                    grounding_callbacks,
                    "_official_p2_follow_up",
                    return_value=FOLLOW_UP,
                ),
                patch.object(
                    grounding_callbacks,
                    "_propose_final_regrounding_gate",
                    propose,
                ),
                patch.object(
                    grounding_callbacks,
                    "_run_atomic_regrounding_draft",
                    run_draft,
                ),
            ):
                result = await grounding_callbacks._resolve_final_regrounding_gate(
                    state,
                    result=_observation_result(runtime, check),
                    synchronization=(
                        grounding_callbacks._TaskGroundingSynchronization(
                            lock=asyncio.Lock()
                        )
                    ),
                    grounding_input=None,
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        propose.assert_not_awaited()
        run_draft.assert_not_awaited()
        self.assertEqual(result.service_status, "final_regrounding_gate_terminal")
        self.assertEqual(
            result.control_error_type,
            "MappingOwnerTransitionExhausted",
        )
        self.assertEqual(
            state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1][
                "blocked_reason"
            ],
            "mapping_owner_transition_exhausted",
        )

    async def test_first_owner_transition_routes_to_one_atomic_mapping_draft(
        self,
    ) -> None:
        runtime = _runtime(
            knowledge=(DomainKnowledge(kind="business_rule", content=FORMULA),)
        )
        state = _state(runtime)
        check = _check(runtime)
        gate_result = GroundingUpdaterResult(
            response=FinalRegroundingGateResponse(
                decision="REGROUND_MAPPING",
                reason=(
                    "Check says the existing evidence can clear the "
                    "Mapping-owned omission."
                ),
            ),
            telemetry=_telemetry(),
            transport_normalization="none",
        )
        expected = grounding_callbacks._ObservationResult(
            runtime=runtime,
            service_status="atomic_regrounding_started",
            observation=_observation_result(runtime, check).observation,
            control_status="succeeded",
        )
        run_draft = AsyncMock(return_value=expected)
        token = grounding_callbacks._bind_turn_message(
            state["task_id"],
            "a-interact",
            QUERY,
        )
        try:
            with (
                patch.object(
                    grounding_callbacks,
                    "_official_p2_follow_up",
                    return_value=FOLLOW_UP,
                ),
                patch.object(
                    grounding_callbacks,
                    "_propose_final_regrounding_gate",
                    AsyncMock(return_value=gate_result),
                ),
                patch.object(
                    grounding_callbacks,
                    "_run_atomic_regrounding_draft",
                    run_draft,
                ),
            ):
                result = await grounding_callbacks._resolve_final_regrounding_gate(
                    state,
                    result=_observation_result(runtime, check),
                    synchronization=(
                        grounding_callbacks._TaskGroundingSynchronization(
                            lock=asyncio.Lock()
                        )
                    ),
                    grounding_input=None,
                )
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertIs(result, expected)
        run_draft.assert_awaited_once()
        audit = state[grounding_callbacks.FINAL_REGROUNDING_GATE_AUDITS_KEY][-1]
        self.assertEqual(audit["decision"], "REGROUND_MAPPING")
        self.assertRegex(audit["mapping_owner_transition_sha256"], r"^[0-9a-f]{64}$")

    def test_gate_prompt_keeps_semantic_judgment_in_check(self) -> None:
        for fragment in (
            "Mapping owner transition",
            "不得把 semantic completeness 当成 NO_REGROUND",
            "只有 omission 非空本身不构成 re-Grounding 理由",
            "Gate 不重新验证公式或选择 operands",
            "由 runtime bounded terminal",
        ):
            self.assertIn(fragment, FINAL_REGROUNDING_GATE_PROMPT)


if __name__ == "__main__":
    unittest.main()
