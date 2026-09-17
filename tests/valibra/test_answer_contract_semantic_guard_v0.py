from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingRuntime,
    SQLGroundingState,
    UserClarificationRecord,
)
from valibra_agent.sql_grounding.semantic_guard_v0 import evaluate_candidate_sql


QUERY = "What is the average CO2 impact for shipments that are very late?"
FOLLOW_UP = "Order the same result by average CO2 impact descending."
GENERIC_FAIL = (
    "SQL failed Phase 1. Your SQL is not correct.\n"
    "Budget remaining: 9.0 bird-coins"
)


def invariant(kind: str, payload: dict) -> dict:
    return {"invariant_id": f"inv-{kind}", "kind": kind, "payload": payload}


def contract(*invariants: dict) -> dict:
    return {"verified_invariants": list(invariants)}


def frozen_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("reviewsandimprovements", "shipments"),
        join_keys=(
            "shipments.reckey = reviewsandimprovements.reckeyrev",
        ),
        column_mapping=(
            ColumnMapping(
                phrase="CO2 impact",
                targets=("reviewsandimprovements.carbonkg",),
            ),
            ColumnMapping(
                phrase="very late",
                targets=(
                    "shipments.actual_duration_hrs",
                    "shipments.planned_eta_hrs",
                ),
            ),
        ),
        domain_knowledge=(),
    )


def clarification() -> UserClarificationRecord:
    return UserClarificationRecord(
        phase=1,
        phrase="very late",
        kind="missing_knowledge",
        question="What rule defines very late?",
        answer=(
            "The difference between actual duration and planned time exceeds "
            "24 hours."
        ),
    )


def session_state(*, phase: int = 1, revision: int = 3) -> dict:
    runtime = GroundingRuntime(
        grounding_revision=revision,
        stage="SQL_ATTEMPT" if phase == 1 else "P2_INCREMENTAL",
        focus_dimension="none",
        grounding_state=frozen_state(),
    )
    return {
        "task_id": "answer-contract-guard-v0-test",
        "current_phase": phase,
        "phase1_completed": phase == 2,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 9.0,
        "initial_budget": 18.0,
        "tool_trajectory": [
            {
                "type": "tool",
                "tool": "get_all_knowledge_definitions",
                "phase": 1,
                "args": {},
                "result": json.dumps([]),
            }
        ],
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY: [
            clarification().model_dump(mode="json")
        ],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
    }


def emit_contract(state: dict, *, phase: int = 1, query: str = QUERY) -> None:
    runtime = GroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )
    grounding_input = {
        "query": query,
        "current_state": frozen_state().model_dump(mode="json"),
    }
    if phase == 2:
        grounding_input["follow_up"] = FOLLOW_UP
    grounding_callbacks._emit_answer_contract_shadow(
        state,
        runtime=runtime,
        phase=phase,
        grounding_input=grounding_input,
    )


class SemanticGuardDetectorTests(unittest.TestCase):
    def test_predicate_contradiction_and_unknown_equivalent(self) -> None:
        item = invariant(
            "predicate",
            {
                "target_ast": {
                    "op": "sub",
                    "args": [
                        {"field": "shipments.actual_duration_hrs"},
                        {"field": "shipments.planned_eta_hrs"},
                    ],
                },
                "operator": ">",
                "literal": {"value": "24"},
            },
        )
        matching = evaluate_candidate_sql(
            contract(item),
            "SELECT * FROM shipments WHERE "
            "shipments.actual_duration_hrs - shipments.planned_eta_hrs > 24",
        )
        conflict = evaluate_candidate_sql(
            contract(item),
            "SELECT * FROM shipments WHERE "
            "shipments.actual_duration_hrs - shipments.planned_eta_hrs >= 24",
        )
        equivalent_but_complex = evaluate_candidate_sql(
            contract(item),
            "SELECT * FROM shipments WHERE NOT "
            "(shipments.actual_duration_hrs - shipments.planned_eta_hrs <= 24)",
        )
        missing = evaluate_candidate_sql(contract(item), "SELECT 1")
        self.assertEqual(matching.status, "NO_CONTRADICTION")
        self.assertEqual(conflict.status, "CONTRADICTION")
        self.assertEqual(equivalent_but_complex.status, "UNKNOWN")
        self.assertEqual(missing.status, "UNKNOWN")

    def test_formula_aggregation_ordering_and_band_are_bounded(self) -> None:
        formula = invariant(
            "formula",
            {
                "expression_ast": {
                    "op": "div",
                    "args": [{"field": "t.cost"}, {"field": "t.revenue"}],
                }
            },
        )
        aggregation = invariant(
            "aggregation",
            {"function": "AVG", "target": {"field": "t.score"}},
        )
        ordering = invariant(
            "ordering",
            {"target": {"field": "t.score"}, "direction": "DESC"},
        )
        band = invariant(
            "predicate_band_set",
            {
                "target": {"field": "t.score"},
                "bands": [
                    {"operator": "<", "literal": "0.5"},
                    {"lower": "0.5", "upper": "0.8"},
                    {"operator": ">", "literal": "0.8"},
                ],
            },
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(formula), "SELECT t.revenue / t.cost FROM t"
            ).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(aggregation), "SELECT SUM(t.score) FROM t"
            ).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(aggregation),
                "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.score) FROM t",
            ).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(ordering), "SELECT t.score FROM t ORDER BY t.score ASC"
            ).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(band),
                "SELECT * FROM t WHERE t.score <= 0.5 OR t.score > 0.8",
            ).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(ordering), "SELECT t.score FROM t GROUP BY t.score"
            ).status,
            "UNKNOWN",
        )

    def test_real_identifier_shapes_preserve_bounded_detection(self) -> None:
        cold_chain = invariant(
            "predicate",
            {
                "target_ast": {
                    "op": "sub",
                    "args": [
                        {
                            "field": "shipments.shipment_overview "
                            "-> 'timing_performance' ->> 'actual_duration_hrs'"
                        },
                        {
                            "field": "shipments.shipment_overview "
                            "-> 'timing_performance' ->> 'planned_eta_hrs'"
                        },
                    ],
                },
                "operator": ">",
                "literal": {"value": "24"},
            },
        )
        cold_chain_sql = (
            "SELECT AVG(r.carbonkg) FROM shipments "
            "JOIN reviewsandimprovements r ON shipments.reckey = r.reckeyrev "
            "WHERE (shipment_overview -> 'timing_performance' "
            "->> 'actual_duration_hrs')::numeric - "
            "(shipment_overview -> 'timing_performance' "
            "->> 'planned_eta_hrs')::numeric >= 24"
        )
        band = invariant(
            "predicate_band_set",
            {
                "target": {"field": "Equipment.RELIAB_IDX"},
                "bands": [
                    {"operator": "<", "literal": "0.5"},
                    {"lower": "0.5", "upper": "0.8"},
                    {"operator": ">", "literal": "0.8"},
                ],
            },
        )
        band_sql = (
            'SELECT e."RELIAB_IDX" FROM "Equipment" e '
            'WHERE e."RELIAB_IDX" < 50 OR e."RELIAB_IDX" > 80'
        )
        ordering = invariant(
            "ordering",
            {
                "target": {
                    "field": "performance_and_safety.effectivenessindexval"
                },
                "direction": "ASC",
            },
        )
        ordering_sql = (
            "SELECT effectivenessindexval FROM performance_and_safety "
            "ORDER BY effectivenessindexval DESC"
        )
        self.assertEqual(
            evaluate_candidate_sql(contract(cold_chain), cold_chain_sql).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(contract(band), band_sql).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(contract(ordering), ordering_sql).status,
            "CONTRADICTION",
        )

    def test_json_extraction_syntax_is_not_itself_guarded(self) -> None:
        predicate = invariant(
            "predicate",
            {
                "target": {
                    "field": "EquipmentType.type_indices -> 'safety_idx'"
                },
                "operator": ">",
                "literal": {"value": "0.75"},
            },
        )
        matching_with_scalar_extraction = (
            'SELECT e."EQUIP_CODE" FROM "Equipment" e '
            'JOIN "EquipmentType" et ON e."EquipType" = et."EquipType" '
            "WHERE (et.type_indices ->> 'safety_idx')::numeric > 0.75"
        )
        conflicting_literal = matching_with_scalar_extraction.replace(
            "> 0.75", "> 75"
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(predicate), matching_with_scalar_extraction
            ).status,
            "NO_CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(contract(predicate), conflicting_literal).status,
            "CONTRADICTION",
        )

    def test_formula_direction_survives_sql_only_operand_wrappers(self) -> None:
        formula = invariant(
            "formula",
            {
                "expression_ast": {
                    "op": "div",
                    "args": [
                        {"field": "operational_metrics.maintcost"},
                        {"field": "operational_metrics.revloss"},
                    ],
                }
            },
        )
        reversed_sql = (
            "SELECT CAST(regexp_replace(om.revloss, '[^0-9.]', '', 'g') "
            "AS numeric) / CAST(om.maintcost AS numeric) "
            "FROM operational_metrics om"
        )
        same_direction_with_null_handling = (
            "SELECT CAST(om.maintcost AS numeric) / "
            "NULLIF(CAST(regexp_replace(om.revloss, '[^0-9.]', '', 'g') "
            "AS numeric), 0) FROM operational_metrics om"
        )
        self.assertEqual(
            evaluate_candidate_sql(contract(formula), reversed_sql).status,
            "CONTRADICTION",
        )
        self.assertEqual(
            evaluate_candidate_sql(
                contract(formula), same_direction_with_null_handling
            ).status,
            "NO_CONTRADICTION",
        )


class SemanticGuardRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            os.environ,
            {
                "VALIBRA_ANSWER_CONTRACT_GUARD_V0": "1",
                "VALIBRA_ANSWER_CONTRACT_SHADOW": "1",
            },
            clear=True,
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def _prepared_state(self, *, phase: int = 1) -> dict:
        state = session_state(phase=phase)
        emit_contract(state, phase=phase)
        return state

    def _arm(self, state: dict, *, phase: int = 1, response: str = GENERIC_FAIL) -> None:
        grounding_callbacks._update_answer_contract_guard_episode_after_submit(
            state,
            phase=phase,
            event="official_submit_failed",
            tool_response=response,
        )

    @staticmethod
    def _context(state: dict, call_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            state=state,
            function_call_id=call_id,
            invocation_id=f"inv-{call_id}",
        )

    async def test_exact_generic_failure_rejects_once_without_baseline_or_charge(self) -> None:
        state = self._prepared_state()
        self._arm(state)
        args = {
            "sql": "SELECT AVG(r.carbonkg) FROM reviewsandimprovements r "
            "JOIN shipments s ON s.reckey = r.reckeyrev WHERE "
            "s.actual_duration_hrs - s.planned_eta_hrs >= 24"
        }
        tool = SimpleNamespace(name="execute_sql")
        context = self._context(state, "guard-reject-1")
        with (
            patch(
                "system_agent.callbacks.before_tool_callback",
                new=AsyncMock(return_value=None),
            ) as baseline_before,
            patch(
                "system_agent.callbacks.after_tool_callback",
                new=AsyncMock(return_value=None),
            ) as baseline_after,
            patch.object(grounding_callbacks, "is_leaderboard_profile", return_value=True),
        ):
            before = await grounding_callbacks.before_tool_callback(tool, args, context)
            self.assertIn("VALIBRA_SEMANTIC_GUARD_REJECTED", before)
            baseline_before.assert_not_awaited()
            after = await grounding_callbacks.after_tool_callback(
                tool, args, context, before
            )
            self.assertEqual(after, before)
            baseline_after.assert_not_awaited()
        self.assertEqual(state["budget_remaining"], 9.0)
        audits = state[grounding_callbacks.ANSWER_CONTRACT_GUARD_AUDITS_KEY]
        self.assertEqual(audits[-1]["status"], "CONTRADICTION_REJECTED_NO_CHARGE")

        second_context = self._context(state, "guard-repeat-2")
        with patch(
            "system_agent.callbacks.before_tool_callback",
            new=AsyncMock(return_value=None),
        ) as baseline_before:
            second = await grounding_callbacks.before_tool_callback(
                tool, args, second_context
            )
            self.assertIsNone(second)
            baseline_before.assert_awaited_once()
        self.assertEqual(
            state[grounding_callbacks.ANSWER_CONTRACT_GUARD_AUDITS_KEY][-1]["status"],
            "REPEATED_VIOLATION_PASS_THROUGH",
        )

    async def test_first_sql_legal_correction_and_omission_pass_through(self) -> None:
        tool = SimpleNamespace(name="execute_sql")
        legal_args = {
            "sql": "SELECT s.reckey, AVG(r.carbonkg) "
            "FROM reviewsandimprovements r JOIN shipments s "
            "ON s.reckey = r.reckeyrev WHERE "
            "s.actual_duration_hrs - s.planned_eta_hrs > 24 GROUP BY s.reckey"
        }
        state = self._prepared_state()
        with patch(
            "system_agent.callbacks.before_tool_callback",
            new=AsyncMock(return_value=None),
        ) as baseline_before:
            first = await grounding_callbacks.before_tool_callback(
                tool, legal_args, self._context(state, "first-sql")
            )
            self.assertIsNone(first)
            baseline_before.assert_awaited_once()

        self._arm(state)
        with patch(
            "system_agent.callbacks.before_tool_callback",
            new=AsyncMock(return_value=None),
        ) as baseline_before:
            correction = await grounding_callbacks.before_tool_callback(
                tool, legal_args, self._context(state, "legal-correction")
            )
            self.assertIsNone(correction)
            baseline_before.assert_awaited_once()

        omitted = session_state()
        omitted[grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY] = []
        emit_contract(omitted, query="Show shipment records.")
        self._arm(omitted)
        with patch(
            "system_agent.callbacks.before_tool_callback",
            new=AsyncMock(return_value=None),
        ) as baseline_before:
            result = await grounding_callbacks.before_tool_callback(
                tool, legal_args, self._context(omitted, "omitted-contract")
            )
            self.assertIsNone(result)
            baseline_before.assert_awaited_once()
        self.assertEqual(
            omitted[grounding_callbacks.ANSWER_CONTRACT_GUARD_AUDITS_KEY][-1]["status"],
            "NO_CONTRACT_PASS_THROUGH",
        )

    async def test_nonexact_failure_stale_identity_and_p2_original_query_are_not_guarded(self) -> None:
        state = self._prepared_state()
        self._arm(state, response=f"{GENERIC_FAIL}\n[SYSTEM NOTE]")
        self.assertIsNone(grounding_callbacks._current_answer_contract_guard_episode(state))

        self._arm(state)
        records = state[grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY]
        records[0]["answer"] = "The difference exceeds 48 hours."
        args = {
            "sql": "SELECT AVG(r.carbonkg) FROM reviewsandimprovements r "
            "JOIN shipments s ON s.reckey = r.reckeyrev WHERE "
            "s.actual_duration_hrs - s.planned_eta_hrs >= 24"
        }
        with patch(
            "system_agent.callbacks.before_tool_callback",
            new=AsyncMock(return_value=None),
        ) as baseline_before:
            result = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="execute_sql"),
                args,
                self._context(state, "stale-contract"),
            )
            self.assertIsNone(result)
            baseline_before.assert_awaited_once()

        p2_state = self._prepared_state(phase=2)
        self._arm(
            p2_state,
            phase=2,
            response=(
                "SQL failed Phase 2. Your SQL is not correct.\n"
                "Budget remaining: 3.5 bird-coins"
            ),
        )
        with patch(
            "system_agent.callbacks.before_tool_callback",
            new=AsyncMock(return_value=None),
        ) as baseline_before:
            result = await grounding_callbacks.before_tool_callback(
                SimpleNamespace(name="execute_sql"),
                args,
                self._context(p2_state, "p2-no-followup-provenance"),
            )
            self.assertIsNone(result)
            baseline_before.assert_awaited_once()
        self.assertEqual(
            p2_state[grounding_callbacks.ANSWER_CONTRACT_GUARD_AUDITS_KEY][-1]["reason"],
            "P2_NO_FOLLOW_UP_PROVENANCE",
        )

    def test_flags_are_exact_and_default_off(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(grounding_callbacks._answer_contract_guard_v0_enabled())
            self.assertFalse(grounding_callbacks._answer_contract_runtime_enabled())
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_GUARD_V0": "true"},
            clear=True,
        ):
            self.assertFalse(grounding_callbacks._answer_contract_guard_v0_enabled())
        with patch.dict(
            os.environ,
            {"VALIBRA_ANSWER_CONTRACT_GUARD_V0": "1"},
            clear=True,
        ):
            self.assertTrue(grounding_callbacks._answer_contract_guard_v0_enabled())
            self.assertTrue(grounding_callbacks._answer_contract_runtime_enabled())


if __name__ == "__main__":
    unittest.main()
