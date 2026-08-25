from __future__ import annotations

import copy
import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from orchestrator import ainteract
from valibra_agent import evaluation, grounding_callbacks
from valibra_agent.evaluation import export_valibra_result
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingRuntime,
    SQLGroundingState,
    sql_grounding_state_sha256,
)


def _runtime() -> GroundingRuntime:
    state = SQLGroundingState(
        tables=("orders",),
        join_keys=(),
        column_mapping=(
            ColumnMapping(phrase="order total", targets=("orders.total",)),
        ),
        domain_knowledge=(),
    )
    return GroundingRuntime(
        grounding_revision=1,
        stage="SQL_ATTEMPT",
        focus_dimension="none",
        grounding_state=state,
    )


def _state() -> dict[str, object]:
    runtime = _runtime()
    update_audit = {
        "observation_id": "observation-1",
        "observation_type": "user_query",
        "stage": "INITIAL_GROUNDING",
        "service_status": "processed",
        "state_sha256": sql_grounding_state_sha256(runtime.grounding_state),
        "provider_attempted": True,
        "provider_status": "succeeded",
        "provider_usage": {
            "input_tokens": 100,
            "output_tokens": 40,
            "reasoning_tokens": 20,
            "total_tokens": 140,
        },
        "provider_latency_ms": 1250.0,
        "provider_reported_cost": 0.25,
    }
    return {
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY: 1,
        grounding_callbacks.GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY: {
            "1": 1,
            "2": 0,
        },
        grounding_callbacks.GROUNDING_PHASE_OUTCOMES_KEY: {
            "1": {
                "phase": 1,
                "status": "succeeded",
                "observation_id": "observation-1",
                "error_type": None,
                "provider_attempted": True,
                "grounding_revision": 1,
                "state_sha256": sql_grounding_state_sha256(
                    runtime.grounding_state
                ),
            }
        },
        grounding_callbacks.GROUNDING_CLARIFICATIONS_KEY: [
            {
                "phase": 1,
                "phrase": "manufacturer",
                "kind": "user_intent",
                "question": "Which manufacturer should be analyzed?",
                "answer": "Acme",
            }
        ],
        "system_agent_token_usage": {
            "model_calls": 2,
            "input_tokens": 200,
            "output_tokens": 80,
            "reasoning_tokens": 30,
            "total_tokens": 280,
        },
        "system_agent_llm_calls": [
            {
                "call_index": 1,
                "timestamp": "2026-08-25T00:00:00+00:00",
                "completed_at": "2026-08-25T00:00:00.100000+00:00",
                "usage": {"raw": {"cost": 0.10}},
                grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY: update_audit,
                grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY: {
                    "effective_mode": "active",
                    "injected": True,
                    "view_sha256": "a" * 64,
                },
            },
            {
                "call_index": 2,
                "timestamp": "2026-08-25T00:00:01+00:00",
                "completed_at": "2026-08-25T00:00:01.200000+00:00",
                "usage": {"raw": {"response_cost": 0.20}},
            },
        ],
        "tool_trajectory": [
            {
                "tool": "submit_sql",
                "phase": 1,
                "args": {"sql": "SELECT orders.total FROM orders"},
                "result": "FAIL",
                "cost": 3.0,
            }
        ],
        "dialogue_history": [{"role": "user", "text": "bounded"}],
        "adk_events": [{"final": True}],
        "initial_budget": 20.0,
        "budget_remaining": 9.5,
    }


class SQLGroundingEvaluationTests(unittest.TestCase):
    def test_current_runtime_and_telemetry_export(self) -> None:
        state = _state()
        result = export_valibra_result(
            state,
            user_simulator_audit={"llm_calls": [], "token_usage": {}},
        )

        self.assertEqual(result["export_status"], "succeeded")
        restored = GroundingRuntime.model_validate(result["runtime"])
        self.assertEqual(restored, _runtime())
        summary = result["grounding_summary"]
        self.assertEqual(summary["grounding_revision"], 1)
        self.assertEqual(summary["stage"], "SQL_ATTEMPT")
        self.assertEqual(summary["focus_dimension"], "none")
        self.assertEqual(
            summary["state_sha256"],
            sql_grounding_state_sha256(_runtime().grounding_state),
        )
        self.assertEqual(
            summary["dimensions"],
            {
                "tables": 1,
                "join_keys": 0,
                "column_mapping": 1,
                "domain_knowledge": 0,
            },
        )
        self.assertEqual(summary["clarifications"], {"total": 1, "answered": 1})
        self.assertEqual(summary["provider_calls"], 1)
        self.assertEqual(summary["provider_audits_captured"], 1)
        self.assertEqual(summary["total_tokens"], 140)
        self.assertEqual(summary["reasoning_tokens"], 20)
        self.assertEqual(summary["latency_ms"], 1250.0)
        self.assertEqual(summary["cost"], 0.25)
        usage = result["model_usage"]
        self.assertEqual(usage["main_agent"]["total_tokens"], 280)
        self.assertEqual(usage["grounding"]["total_tokens"], 140)
        self.assertEqual(usage["total_model_tokens"], 420)
        self.assertAlmostEqual(usage["total_model_cost"], 0.55)
        self.assertEqual(
            result["trajectory_manifest"]["locations"]["final_state"],
            {
                "collection": "grounding_runtime",
                "json_pointer": "/grounding_state",
            },
        )

    def test_old_requirement_payload_is_rejected_without_compatibility(self) -> None:
        old_payload = {
            "schema_version": "1.1",
            "grounding_revision": 1,
            "requirement_revision": 1,
            "grounding_state": {
                "requirement_frame": {
                    "value_slots": [],
                    "schema_slots": [],
                    "operation_slots": [],
                }
            },
        }
        result = export_valibra_result(
            {grounding_callbacks.GROUNDING_RUNTIME_KEY: old_payload}
        )
        self.assertEqual(result["export_status"], "failed")
        self.assertEqual(result["error_type"], "ValidationError")

    def test_missing_or_invalid_runtime_fails_open_with_bounded_type(self) -> None:
        self.assertEqual(
            export_valibra_result({}),
            {
                "export_status": "failed",
                "error_type": "MissingGroundingRuntime",
            },
        )
        result = export_valibra_result(
            {grounding_callbacks.GROUNDING_RUNTIME_KEY: {"secret": "hidden"}}
        )
        self.assertEqual(result["export_status"], "failed")
        self.assertNotIn("secret", repr(result))

    def test_manifest_is_stable_and_source_has_no_legacy_or_hidden_access(self) -> None:
        first = export_valibra_result(_state())["trajectory_manifest"]
        second = export_valibra_result(copy.deepcopy(_state()))[
            "trajectory_manifest"
        ]
        self.assertEqual(first, second)
        source = inspect.getsource(evaluation)
        for forbidden in (
            "task_data",
            "sol_sql",
            "test_cases",
            "httpx",
            "requests.",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class OrchestratorExportProfileTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _task() -> dict[str, object]:
        return {
            "instance_id": "sql-grounding-export-task",
            "selected_database": "orders_db",
            "amb_user_query": "List order totals.",
            "user_query_ambiguity": {"critical_ambiguity": []},
            "knowledge_ambiguity": [],
        }

    async def _run(self, profile: str) -> dict[str, object]:
        run_state = {
            **_state(),
            "phase1_completed": True,
            "phase2_completed": False,
            "total_reward": 1.0,
        }

        async def fake_post(url: str, payload: dict, timeout: float = 120.0):
            del payload, timeout
            if url.endswith("/run_session"):
                return {"state": copy.deepcopy(run_state), "response": "done"}
            return {"status": "ok"}

        async def fake_get(url: str, timeout: float = 120.0):
            del url, timeout
            return {"llm_calls": [], "token_usage": {}}

        clock = iter((100.0, 105.0))
        with (
            patch.dict(os.environ, {"VALIBRA_EXECUTION_PROFILE": profile}),
            patch.object(ainteract, "_post", side_effect=fake_post),
            patch.object(ainteract, "_get", side_effect=fake_get),
            patch.object(
                ainteract,
                "time",
                SimpleNamespace(time=lambda: next(clock)),
            ),
        ):
            return await ainteract.run_single_task(copy.deepcopy(self._task()))

    async def test_research_profile_exports_current_runtime(self) -> None:
        result = await self._run("research")
        self.assertEqual(result["valibra"]["export_status"], "succeeded")
        GroundingRuntime.model_validate(result["valibra"]["runtime"])

    async def test_leaderboard_profile_keeps_official_result_schema_clean(self) -> None:
        with patch.object(
            ainteract,
            "export_valibra_result",
            side_effect=AssertionError("leaderboard must not call exporter"),
        ):
            result = await self._run("leaderboard")
        self.assertNotIn("valibra", result)
        self.assertEqual(result["total_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
