"""Offline contract tests for the frozen SG7 evaluation-only pipeline."""

from __future__ import annotations

import asyncio
import inspect
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

if importlib.util.find_spec("scripts.analyze_p7_paired_results") is None:
    raise unittest.SkipTest(
        "Historical SG7 evaluation requires the retired P7 scripts and private "
        "fixtures; it is not the Full600 runtime release suite."
    )

from scripts import analyze_sg7_paired_results as analyzer
from scripts import build_sg7_manifest as builder
from scripts import run_sg7_paired_evaluation as runner
from scripts.build_p7_manifest import canonical_json_bytes, load_json, sha256_file
from valibra_agent.sql_grounding.models import GroundingRuntime
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
PRE_R1_SQL_GROUNDING_PROMPT_SHA256 = (
    "312a5c019c68d09aaf3c54e3991ef381d4dc2ded7564fdb344bd7113305ac594"
)
PRE_R1_SQL_GROUNDING_CONFIGURATION_SHA256 = (
    "489a7185cb711429b4c5346481ae639851f02ad41893554cbedb6ca703d2ba9e"
)


def _cleanup() -> dict:
    return {"verified": True, "residual_task_databases": [], "pending_and_blocked_zero": True}


def _sqlg_export() -> dict:
    return {
        "schema_version": "sg7-evaluation-1.0",
        "export_status": "succeeded",
        "variant": "sql_grounding_v1",
        "runtime": {
            "grounding_revision": 1,
            "stage": "SQL_ATTEMPT",
            "focus_dimension": "none",
            "state_sha256": "1" * 64,
            "tables": {"status": "nonempty", "count": 1},
            "join_keys": {"status": "empty", "count": 0},
            "column_mapping": {"status": "nonempty", "count": 1},
            "domain_knowledge": {"status": "empty", "count": 0},
        },
        "state_evolution": [
            {
                "observation_id": "obs-1",
                "observation_type": "schema",
                "service_status": "accepted",
                "revision_after": 1,
                "changed_dimensions": ["tables", "column_mapping"],
                "focus_before": "tables",
                "focus_after": "none",
                "stage": "SQL_ATTEMPT",
                "new_state_sha256": "1" * 64,
                "provider_attempted": True,
                "provider_status": "succeeded",
                "provider_usage": {"input_tokens": 10, "output_tokens": 4, "reasoning_tokens": 2, "total_tokens": 14},
                "provider_latency_ms": 25.0,
                "provider_reported_cost": None,
                "provider_error_type": None,
                "provider_request_sha256": "2" * 64,
                "provider_response_sha256": "3" * 64,
                "private_audit_ref": "research-runtime/private/call.json",
            }
        ],
        "stage_events": [{"event": "grounding_completed", "stage_before": "INITIAL_GROUNDING", "stage_after": "SQL_ATTEMPT"}],
        "gate": {"evaluated": 1, "hard_blocked": 0, "open": 1, "budget_liveness_bypass": 0, "failed_open": 0, "denial_count": 0, "repeated_premature_submit_count": 0, "blocked_submit_actual_tool_count": 0, "blocked_submit_official_trajectory_count": 0},
        "view": {"model_calls_audited": 1, "injection_success": 1, "injection_failed_open": 0, "max_chars": 100, "max_tokens": 30, "sha_count": 1},
        "hint": {"injection_success": 1, "injection_failed_open": 0, "focus_distribution": {"none": 1}},
        "errors": {"bounded_error_audit_count": 0, "provider_failures": 0, "provider_timeouts": 0},
        "stores": {"pending_count": 0, "blocked_count": 0},
    }


def _result(task_id: str, variant: str, reward: float = 0.5) -> dict:
    export = runner.export_b0_result({}) if variant == "b0" else _sqlg_export()
    return {
        "task_id": task_id,
        "instance_id": task_id,
        "total_reward": reward,
        "phase1_passed": reward > 0,
        "phase2_passed": False,
        "elapsed_seconds": 1.0,
        "initial_budget": 10.0,
        "budget_used": 1.0,
        "budget_remaining": 9.0,
        "prompt_flow": [{"timestamp": "2026-08-19T00:00:00Z", "completed_at": "2026-08-19T00:00:00.100000Z", "usage": {"input_tokens": 5, "output_tokens": 3, "reasoning_tokens": 1, "total_tokens": 8, "raw": {}}}],
        "tool_trajectory": [{"tool": "get_schema"}, {"tool": "submit_sql"}],
        "adk_events": [],
        "token_usage": {},
        "user_simulator_audit": {"llm_calls": [], "token_usage": {}},
        "valibra": export,
    }


class FrozenProtocolTests(unittest.TestCase):
    def test_manifest_rebuilds_byte_for_byte_and_preserves_order(self):
        source = load_json(builder.SOURCE_MANIFEST)
        expected = builder.build_manifest(source)
        actual_path = ROOT / "configs/sg7/sg7_full_0001_0030_manifest_v1.json"
        self.assertEqual(actual_path.read_bytes(), canonical_json_bytes(expected))
        self.assertEqual(sha256_file(actual_path), runner.EXPECTED_MANIFEST_SHA256)
        self.assertEqual(sum(pair["run_order"][0] == "b0" for pair in expected["pairs"]), 15)
        self.assertEqual(sum(pair["run_order"][0] == "sql_grounding_v1" for pair in expected["pairs"]), 15)
        self.assertNotIn("valibra_llm", actual_path.read_text())

    def test_protocol_and_diagnostics_are_frozen_before_network(self):
        protocol, manifest, diagnostics, tasks = runner.load_frozen_inputs()
        self.assertEqual(protocol["protocol_sha256"], runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(len(manifest["pairs"]), 30)
        self.assertEqual([item["task_id"] for item in diagnostics["tasks"]], ["cross_border_15"])
        self.assertEqual(len(tasks), 600)
        self.assertEqual(
            protocol["sql_grounding_v1"]["prompt_sha256"],
            PRE_R1_SQL_GROUNDING_PROMPT_SHA256,
        )
        self.assertNotEqual(
            PRE_R1_SQL_GROUNDING_PROMPT_SHA256,
            SQL_GROUNDING_PROMPT_SHA256,
        )
        self.assertEqual(protocol["sql_grounding_v1"]["form_schema_sha256"], SQL_GROUNDING_FORM_SCHEMA_SHA256)
        # SG7 is a frozen historical protocol. SG7-R1 intentionally changes the
        # current timeout/configuration contract without rewriting that record.
        self.assertEqual(
            protocol["sql_grounding_v1"]["configuration_sha256"],
            PRE_R1_SQL_GROUNDING_CONFIGURATION_SHA256,
        )
        self.assertNotEqual(
            PRE_R1_SQL_GROUNDING_CONFIGURATION_SHA256,
            SQL_GROUNDING_CONFIGURATION_SHA256,
        )

    def test_formal_b0_and_start_identity_match(self):
        self.assertEqual(runner.validate_source_identity()["formal_b0_fingerprint"], runner.EXPECTED_B0)

    def test_runtime_environment_requires_three_distinct_file_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name in ("main", "grounding", "user"):
                path = root / name
                path.write_text("offline-placeholder")
                files.append(path)
            env = {
                "MODEL_PRESET": "glm52_high_32768", "DATASET": "full", "PROMPT_VERSION": "v2", "PATIENCE": "3",
                "SYSTEM_AGENT_API_BASE": "https://open.bigmodel.cn/api/paas/v4",
                "USER_SIM_MODEL": "anthropic/claude-haiku-4-5-20251001", "USER_SIM_API_BASE": "https://api.anthropic.com", "USER_SIM_DISABLE_THINKING": "true", "USER_SIM_PROTOCOL_POLICY": "official", "USER_SIM_PROTOCOL_MAX_ATTEMPTS": "1",
                "GROUNDING_UPDATER_MODE": "llm", "GROUNDING_MODEL_PRESET": "glm52_high_32768", "GROUNDING_API_BASE": "https://open.bigmodel.cn/api/paas/v4", "GROUNDING_TIMEOUT_SECONDS": "30", "GROUNDING_MAX_TOKENS": "32768", "GROUNDING_MAX_CALLS_PER_TASK": "2", "GROUNDING_PROMPT_SHA256": SQL_GROUNDING_PROMPT_SHA256,
                "SYSTEM_AGENT_API_KEY_FILE": str(files[0]), "GROUNDING_API_KEY_FILE": str(files[1]), "USER_SIM_API_KEY_FILE": str(files[2]),
                "SYSTEM_AGENT_API_KEY": "", "GROUNDING_API_KEY": "", "USER_SIM_API_KEY": "",
                "PG_HOST": "127.0.0.1", "PG_PORT": "6433", "SYSTEM_AGENT_PORT": "6100", "USER_SIM_PORT": "6101", "DB_ENV_PORT": "6102",
            }
            self.assertTrue(runner.validate_runtime_environment(env)["validated"])
            env["GROUNDING_API_KEY_FILE"] = env["SYSTEM_AGENT_API_KEY_FILE"]
            with self.assertRaises(runner.SG7RunnerError):
                runner.validate_runtime_environment(env)


class ExportTests(unittest.TestCase):
    def test_b0_is_explicitly_not_applicable(self):
        self.assertEqual(runner.export_b0_result({})["export_status"], "not_applicable")

    def test_sqlg_export_is_bounded_and_omits_four_dimension_content(self):
        runtime = GroundingRuntime.model_validate({
            "grounding_revision": 1,
            "stage": "SQL_ATTEMPT",
            "focus_dimension": "none",
            "grounding_state": {
                "tables": ["secret_table"],
                "join_keys": [],
                "column_mapping": [{"phrase": "private phrase", "targets": ["secret_table.private_column"]}],
                "domain_knowledge": [],
            },
        })
        state = {
            runner.SQL_GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
            runner.PENDING_KEY: {}, runner.BLOCKED_KEY: {}, runner.GATE_AUDITS_KEY: [],
            "tool_trajectory": [], "system_agent_llm_calls": [],
        }
        exported = runner.export_sql_grounding_result(state)
        body = json.dumps(exported, sort_keys=True)
        self.assertNotIn("secret_table", body)
        self.assertNotIn("private phrase", body)
        self.assertNotIn("private_column", body)
        self.assertEqual(exported["runtime"]["tables"], {"status": "nonempty", "count": 1})
        self.assertEqual(exported["stores"], {"pending_count": 0, "blocked_count": 0})

    def test_evaluation_sources_do_not_score_sql_or_modify_production(self):
        source = inspect.getsource(runner) + inspect.getsource(analyzer)
        for token in ("evaluate_sql", "test_cases", "sol_sql", "ground_truth", "execute_sql("):
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        status = __import__("subprocess").check_output(
            [
                "git", "diff", "--name-only", runner.EXPECTED_START_HEAD, "--",
                "system_agent", "orchestrator", "valibra_agent/sql_grounding",
            ],
            cwd=ROOT,
            text=True,
        )
        self.assertEqual(
            set(status.splitlines()),
            {"valibra_agent/sql_grounding/updater.py"},
        )


class LifecycleAndAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_pair_adjacency_fsync_cleanup_and_no_rerun(self):
        manifest = load_json(runner.DEFAULT_MANIFEST)
        pairs = manifest["pairs"][:2]
        tasks = {pair["task_id"]: {"instance_id": pair["task_id"]} for pair in pairs}
        calls = []
        cleanups = []

        async def task_runner(variant, task):
            calls.append((task["instance_id"], variant))
            return _result(task["instance_id"], variant)

        async def cleanup_runner(variant, pair):
            cleanups.append((pair["task_id"], variant))
            return _cleanup()

        with tempfile.TemporaryDirectory() as directory:
            ledger = await runner.run_pairs(runtime_dir=Path(directory) / "run", pairs=pairs, tasks=tasks, task_runner=task_runner, cleanup_runner=cleanup_runner, expected_rows=4, label="test")
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        expected = [(pair["task_id"], variant) for pair in pairs for variant in pair["run_order"]]
        self.assertEqual(calls, expected)
        self.assertEqual(cleanups, expected)
        self.assertEqual(len(rows), 4)

    async def test_failure_stops_without_retry(self):
        pair = load_json(runner.DEFAULT_MANIFEST)["pairs"][0]
        calls = 0

        async def task_runner(variant, task):
            nonlocal calls
            calls += 1
            raise RuntimeError("offline synthetic failure")

        async def cleanup_runner(variant, pair):
            return _cleanup()

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                await runner.run_pairs(runtime_dir=Path(directory) / "run", pairs=[pair], tasks={pair["task_id"]: {"instance_id": pair["task_id"]}}, task_runner=task_runner, cleanup_runner=cleanup_runner, expected_rows=2, label="test")
        self.assertEqual(calls, 1)

    def test_primary_analysis_go_and_hold_rules_are_deterministic(self):
        protocol = load_json(runner.DEFAULT_PROTOCOL)
        manifest = load_json(runner.DEFAULT_MANIFEST)
        rows = []
        for pair in manifest["pairs"]:
            for variant in pair["run_order"]:
                reward = 0.25 if variant == "b0" else 0.75
                rows.append(runner.build_ledger_record(_result(pair["task_id"], variant, reward), pair=pair, variant=variant, cleanup=_cleanup()))
        result = analyzer.analyze(protocol, manifest, rows)
        self.assertEqual(result["execution_status"], "PASS")
        self.assertEqual(result["switch_decision"], "GO")
        self.assertEqual(result["complete_pairs"], 30)
        self.assertEqual((result["primary"]["wins"], result["primary"]["ties"], result["primary"]["losses"]), (30, 0, 0))
        self.assertEqual(result["primary"]["sql_grounding_v1"]["models"]["grounding"]["total_tokens"], 30 * 14)
        for row in rows:
            if row["variant"] == "sql_grounding_v1":
                row["reward"] = 0.25
        self.assertEqual(analyzer.analyze(protocol, manifest, rows)["switch_decision"], "HOLD")

    def test_blocked_official_result_is_integrity_failure(self):
        pair = load_json(runner.DEFAULT_MANIFEST)["pairs"][0]
        result = _result(pair["task_id"], "sql_grounding_v1")
        result["valibra"]["gate"]["blocked_submit_official_trajectory_count"] = 1
        row = runner.build_ledger_record(result, pair=pair, variant="sql_grounding_v1", cleanup=_cleanup())
        self.assertEqual(row["integrity"]["blocked_submit_official_trajectory_count"], 1)


class ServiceHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_requires_independent_b0_and_active_sg6b(self):
        protocol = load_json(runner.DEFAULT_PROTOCOL)
        request = {"thinking": {"type": "enabled", "clear_thinking": False}, "reasoning_effort": "high", "max_tokens": 32768, "temperature": 0.0, "tool_choice": "auto"}
        reports = [
            {"status": "healthy", "service": "system_agent", "model": "openai/glm-5.2", "generation_parameters": request},
            {"status": "healthy", "service": "valibra_agent", "variant": "SQL-Grounding-V1-SG6b-Active-Gate", "configuration_summary": {"model": "openai/glm-5.2", "generation_parameters": request, "grounding_updater": "llm", "grounding_provider_configuration_valid": True, "grounding_provider_enabled": True, "grounding_prompt_view_effective_mode": "active", "attempt_gate_mode": "active_first_submit", "attempt_gate_blocking_enabled": True, "attempt_gate_budget_liveness_bypass": True}},
            {"status": "healthy", "service": "user_simulator"},
            {"status": "healthy", "service": "db_environment"},
        ]
        with patch.object(runner.p7, "_get_json", AsyncMock(side_effect=reports)):
            self.assertTrue((await runner.validate_service_health(protocol))["validated"])


if __name__ == "__main__":
    unittest.main()
