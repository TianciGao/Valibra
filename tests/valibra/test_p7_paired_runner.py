"""Offline tests for the frozen P7 paired execution wrapper."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from scripts import run_p7_paired_evaluation as runner
from scripts.analyze_p7_paired_results import PairedAnalysisError, analyze_paired_results
from scripts.build_p7_manifest import canonical_json_bytes, load_json


ROOT = Path(__file__).resolve().parents[2]


def _fake_result(task_id: str, initial_budget: float, variant: str) -> dict:
    result = {
        "task_id": task_id,
        "instance_id": task_id,
        "total_reward": 0.75,
        "phase1_passed": True,
        "phase2_passed": False,
        "elapsed_seconds": 1.25,
        "initial_budget": initial_budget,
        "budget_used": 2.0,
        "budget_remaining": initial_budget - 2.0,
        "prompt_flow": [
            {
                "timestamp": "2026-08-09T00:00:00Z",
                "completed_at": "2026-08-09T00:00:00.100000Z",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "reasoning_tokens": 2,
                    "total_tokens": 15,
                    "raw": {},
                },
                "raw_response": {"bounded": True},
            },
            {"timestamp": "2026-08-09T00:00:00.200000Z"},
        ],
        "tool_trajectory": [],
        "adk_events": [],
        "dialogue_history": [],
        "token_usage": {},
        "user_simulator_audit": {
            "llm_calls": [],
            "token_usage": {},
            "dialogue_history": [],
        },
    }
    if variant == "valibra_llm":
        result["valibra"] = {
            "export_status": "succeeded",
            "grounding_summary": {"pending_count": 0},
        }
    return result


def _cleanup() -> dict:
    return {
        "verified": True,
        "agent_session_removed": True,
        "user_state_removed": True,
        "residual_task_databases": [],
    }


def _environment(temp: Path) -> dict[str, str]:
    files = {}
    for name in ("main", "grounding", "user"):
        path = temp / f"{name}.key"
        path.write_text("private-placeholder", encoding="utf-8")
        files[name] = str(path)
    return {
        "DATASET": "full",
        "MODEL_PRESET": "glm52_high_32768",
        "USER_SIM_PROFILE": "claude_haiku_4_5_official",
        "USER_SIM_MODEL": "anthropic/claude-haiku-4-5-20251001",
        "USER_SIM_API_BASE": "https://api.anthropic.com",
        "USER_SIM_DISABLE_THINKING": "true",
        "USER_SIM_PROTOCOL_POLICY": "official",
        "USER_SIM_PROTOCOL_MAX_ATTEMPTS": "1",
        "PROMPT_VERSION": "v2",
        "PATIENCE": "3",
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_PROMPT_VIEW_MODE": "active",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "300",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "2",
        "GROUNDING_PROMPT_SHA256": "5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d",
        "GROUNDING_API_BASE": "https://provider.invalid/v1",
        "SYSTEM_AGENT_API_BASE": "https://provider.invalid/v1",
        "SYSTEM_AGENT_API_KEY": "",
        "GROUNDING_API_KEY": "",
        "USER_SIM_API_KEY": "",
        "SYSTEM_AGENT_API_KEY_FILE": files["main"],
        "GROUNDING_API_KEY_FILE": files["grounding"],
        "USER_SIM_API_KEY_FILE": files["user"],
        "PG_HOST": "127.0.0.1",
        "PG_PORT": "6433",
        "SYSTEM_AGENT_PORT": "6100",
        "USER_SIM_PORT": "6101",
        "DB_ENV_PORT": "6102",
    }


class FrozenInputTests(unittest.TestCase):
    def test_frozen_inputs_and_all_hashes_validate(self):
        protocol, manifest, tasks = runner.load_frozen_run_inputs()
        self.assertEqual(protocol["protocol_sha256"], runner.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(len(manifest["pairs"]), 30)
        self.assertEqual(len(tasks), 600)

    def test_runtime_environment_is_exact_and_separates_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = runner.validate_runtime_environment(
                _environment(Path(directory))
            )
        self.assertTrue(summary["validated"])
        self.assertEqual(
            summary["credential_sources"],
            {
                "main_agent": "file",
                "grounding": "file",
                "user_simulator": "file",
            },
        )
        self.assertFalse(summary["credential_values_recorded"])

    def test_invalid_mode_or_direct_key_fails_before_run(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = _environment(Path(directory))
            environment["GROUNDING_UPDATER_MODE"] = "rule"
            environment["SYSTEM_AGENT_API_KEY"] = "must-not-be-used"
            with self.assertRaises(runner.P7RunnerError):
                runner.validate_runtime_environment(environment)

    def test_same_main_and_grounding_key_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = _environment(Path(directory))
            environment["GROUNDING_API_KEY_FILE"] = environment[
                "SYSTEM_AGENT_API_KEY_FILE"
            ]
            with self.assertRaises(runner.P7RunnerError):
                runner.validate_runtime_environment(environment)

    def test_formal_b0_fingerprint_matches_frozen_protocol(self):
        self.assertEqual(
            runner.formal_b0_fingerprint(),
            {
                "files": 68,
                "sha256": "66fa2d56eb3a4004d186c79ee5b1a160be85594ad20c0c4624b0a469abbe70e0",
            },
        )


class ServiceHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_proves_shared_configuration_and_active_llm(self):
        protocol = load_json(ROOT / "configs/p7/p7_protocol_v1.json")
        main = protocol["configurations"]["common"]["configuration"]["main_agent"]
        reports = [
            {
                "status": "healthy",
                "service": "system_agent",
                "model": main["model"],
                "generation_parameters": main["request"],
                "model_preset": {
                    "name": main["preset"],
                    "normalized_sha256": main["preset_sha256"],
                },
                "adk_available": True,
            },
            {
                "status": "healthy",
                "service": "valibra_agent",
                "adk_available": True,
                "configuration_summary": {
                    "model": main["model"],
                    "generation_parameters": main["request"],
                    "model_preset": main["preset"],
                    "normalized_model_sha256": main["preset_sha256"],
                    "grounding_updater": "llm",
                    "grounding_prompt_view_effective_mode": "active",
                    "prompt_view_injection_enabled": True,
                    "grounding_configuration_valid": True,
                    "grounding_prompt_view_configuration_valid": True,
                },
            },
            {"status": "healthy", "service": "user_simulator"},
            {"status": "healthy", "service": "db_environment"},
        ]
        with patch.object(
            runner, "_get_json", AsyncMock(side_effect=reports)
        ):
            result = await runner.validate_service_health(protocol)
        self.assertTrue(result["validated"])
        self.assertFalse(result["b0_has_grounding"])
        self.assertEqual(result["valibra_updater"], "llm")
        self.assertEqual(result["valibra_view"], "active")

    async def test_health_mismatch_fails_before_provider(self):
        protocol = load_json(ROOT / "configs/p7/p7_protocol_v1.json")
        reports = [
            {"status": "healthy", "service": "wrong"},
            {"status": "healthy", "service": "valibra_agent"},
            {"status": "healthy", "service": "user_simulator"},
            {"status": "healthy", "service": "db_environment"},
        ]
        with patch.object(
            runner, "_get_json", AsyncMock(side_effect=reports)
        ):
            with self.assertRaises(runner.P7RunnerError):
                await runner.validate_service_health(protocol)


class OfficialScoreTests(unittest.TestCase):
    def test_scores_are_copied_without_recomputation(self):
        original = {
            "total_reward": 0.375,
            "phase1_passed": False,
            "phase2_passed": True,
        }
        copied = runner._official_scores(original)
        self.assertEqual(
            copied,
            {"reward": 0.375, "p1_passed": False, "p2_passed": True},
        )
        self.assertEqual(original["total_reward"], copied["reward"])

    def test_missing_or_non_boolean_official_fields_are_rejected(self):
        with self.assertRaises(runner.P7RunnerError):
            runner._official_scores({"total_reward": 0, "phase1_passed": False})
        with self.assertRaises(runner.P7RunnerError):
            runner._official_scores(
                {
                    "total_reward": 0,
                    "phase1_passed": 1,
                    "phase2_passed": False,
                }
            )

    def test_provenance_uses_raw_sha_and_fixed_json_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            raw = runtime / "official_raw_results" / "01.json"
            raw.parent.mkdir()
            result = {
                "total_reward": 0.5,
                "phase1_passed": True,
                "phase2_passed": False,
            }
            raw.write_bytes(canonical_json_bytes(result))
            provenance = runner.score_provenance(
                result,
                pair_index=1,
                task_id="task-1",
                variant="b0",
                raw_path=raw,
                runtime_dir=runtime,
            )
        self.assertEqual(provenance["score_source"], "bird_interact_official")
        self.assertEqual(provenance["reward_json_pointer"], "/total_reward")
        self.assertEqual(provenance["p1_json_pointer"], "/phase1_passed")
        self.assertEqual(provenance["p2_json_pointer"], "/phase2_passed")
        self.assertEqual(len(provenance["official_raw_result_sha256"]), 64)

    def test_runner_source_has_no_sql_rescoring_or_reward_formula(self):
        source = inspect.getsource(runner)
        forbidden = (
            "execute_sql(",
            "evaluate_sql",
            "reward_formula",
            "phase1_passed +",
            "phase2_passed +",
            "sol_sql",
            "test_cases",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_json(ROOT / "configs/p7/p7_manifest_v1.json")
        self.pair = self.manifest["pairs"][0]

    def test_b0_ledger_has_no_grounding_calls(self):
        result = _fake_result(
            self.pair["task_id"], self.pair["initial_bird_coin"], "b0"
        )
        record = runner.build_ledger_record(
            result,
            pair=self.pair,
            variant="b0",
            cleanup=_cleanup(),
        )
        self.assertEqual(record["models"]["grounding"]["calls"], [])
        self.assertEqual(record["reward"], result["total_reward"])
        self.assertEqual(record["p1_passed"], result["phase1_passed"])
        self.assertEqual(record["p2_passed"], result["phase2_passed"])
        self.assertTrue(record["integrity"]["cleanup_complete"])

    def test_main_provider_and_local_call_are_kept_separate(self):
        result = _fake_result(
            self.pair["task_id"], self.pair["initial_bird_coin"], "b0"
        )
        record = runner.build_ledger_record(
            result,
            pair=self.pair,
            variant="b0",
            cleanup=_cleanup(),
        )
        calls = record["models"]["main_agent"]["calls"]
        self.assertEqual([call["kind"] for call in calls], ["provider", "local"])
        self.assertEqual(calls[0]["total_tokens"], 15)
        self.assertEqual(calls[0]["reasoning_tokens"], 2)
        self.assertIsNone(calls[1]["total_tokens"])

    def test_grounding_audit_uses_measured_latency_without_token_double_count(self):
        result = _fake_result(
            self.pair["task_id"],
            self.pair["initial_bird_coin"],
            "valibra_llm",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "research-runtime" / "grounding-llm" / "call.json"
            audit_path.parent.mkdir(parents=True)
            audit_path.write_text("{}", encoding="utf-8")
            result["prompt_flow"][0]["valibra_grounding_update"] = {
                "llm": {
                    "attempted": True,
                    "status": "succeeded",
                    "input_tokens": 7,
                    "output_tokens": 5,
                    "reasoning_tokens": 3,
                    "total_tokens": 12,
                    "latency_ms": 50.0,
                    "cost": None,
                    "raw_audit_ref": audit_path.relative_to(root).as_posix(),
                }
            }
            record = runner.build_ledger_record(
                result,
                pair=self.pair,
                variant="valibra_llm",
                cleanup=_cleanup(),
                project_root=root,
            )
        call = record["models"]["grounding"]["calls"][0]
        self.assertEqual(call["total_tokens"], 12)
        self.assertEqual(call["reasoning_tokens"], 3)
        self.assertIsNotNone(call["started_at"])
        self.assertIsNotNone(call["completed_at"])

    def test_cleanup_failure_remains_a_hard_gate(self):
        result = _fake_result(
            self.pair["task_id"], self.pair["initial_bird_coin"], "b0"
        )
        record = runner.build_ledger_record(
            result,
            pair=self.pair,
            variant="b0",
            cleanup={"verified": False},
        )
        self.assertFalse(record["integrity"]["cleanup_complete"])

    def test_failed_grounding_attempt_has_unreliable_usage(self):
        result = _fake_result(
            self.pair["task_id"],
            self.pair["initial_bird_coin"],
            "valibra_llm",
        )
        result["prompt_flow"][0]["valibra_grounding_update"] = {
            "llm": {
                "attempted": True,
                "status": "failed",
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 25.0,
                "cost": None,
                "raw_audit_ref": "",
            }
        }
        record = runner.build_ledger_record(
            result,
            pair=self.pair,
            variant="valibra_llm",
            cleanup=_cleanup(),
        )
        call = record["models"]["grounding"]["calls"][0]
        self.assertFalse(call["usage_reliable"])
        self.assertIsNone(call["total_tokens"])


class PairedLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.protocol = load_json(ROOT / "configs/p7/p7_protocol_v1.json")
        self.manifest = load_json(ROOT / "configs/p7/p7_manifest_v1.json")
        self.tasks = {
            pair["task_id"]: {
                "instance_id": pair["task_id"],
                "selected_database": pair["selected_database"],
            }
            for pair in self.manifest["pairs"]
        }

    async def test_exact_60_run_order_and_analysis_only_after_completion(self):
        calls = []
        analyzed_at = []

        async def task_runner(variant, task):
            calls.append((task["instance_id"], variant))
            pair = next(
                item
                for item in self.manifest["pairs"]
                if item["task_id"] == task["instance_id"]
            )
            return _fake_result(task["instance_id"], pair["initial_bird_coin"], variant)

        async def cleanup_runner(variant, pair):
            return _cleanup()

        def analyzer(ledger, output):
            rows = [line for line in ledger.read_text().splitlines() if line]
            analyzed_at.append(len(rows))
            output.write_text('{"status":"offline"}\n', encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "run"
            await runner.run_paired_evaluation(
                runtime_dir=runtime,
                protocol=self.protocol,
                manifest=self.manifest,
                tasks_by_id=self.tasks,
                task_runner=task_runner,
                cleanup_runner=cleanup_runner,
                analyzer_runner=analyzer,
                environment_summary={"validated": True},
            )
            ledger = [
                json.loads(line)
                for line in (runtime / "p7_ledger.jsonl").read_text().splitlines()
            ]
            provenance = [
                json.loads(line)
                for line in (runtime / "score_provenance.jsonl").read_text().splitlines()
            ]
            completion = json.loads((runtime / "completion.json").read_text())
        expected = [
            (pair["task_id"], variant)
            for pair in self.manifest["pairs"]
            for variant in pair["run_order"]
        ]
        self.assertEqual(calls, expected)
        self.assertEqual(len(ledger), 60)
        self.assertEqual(len(provenance), 60)
        self.assertEqual(analyzed_at, [60])
        self.assertTrue(completion["effect_summary_computed_only_after_60_rows"])
        self.assertEqual(sum(pair["run_order"][0] == "b0" for pair in self.manifest["pairs"]), 15)
        self.assertEqual(sum(pair["run_order"][0] == "valibra_llm" for pair in self.manifest["pairs"]), 15)

    async def test_provider_stage_failure_is_not_retried_or_analyzed(self):
        calls = 0
        analyzed = False

        async def task_runner(variant, task):
            nonlocal calls
            calls += 1
            raise RuntimeError("synthetic provider-stage failure")

        async def cleanup_runner(variant, pair):
            return _cleanup()

        def analyzer(ledger, output):
            nonlocal analyzed
            analyzed = True

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                await runner.run_paired_evaluation(
                    runtime_dir=Path(directory) / "run",
                    protocol=self.protocol,
                    manifest=self.manifest,
                    tasks_by_id=self.tasks,
                    task_runner=task_runner,
                    cleanup_runner=cleanup_runner,
                    analyzer_runner=analyzer,
                    environment_summary={"validated": True},
                )
        self.assertEqual(calls, 1)
        self.assertFalse(analyzed)

    async def test_cleanup_failure_stops_pair_and_blocks_analysis(self):
        calls = 0
        analyzed = False

        async def task_runner(variant, task):
            nonlocal calls
            calls += 1
            pair = self.manifest["pairs"][0]
            return _fake_result(task["instance_id"], pair["initial_bird_coin"], variant)

        async def cleanup_runner(variant, pair):
            raise runner.P7RunnerError("synthetic cleanup failure")

        def analyzer(ledger, output):
            nonlocal analyzed
            analyzed = True

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(runner.P7RunnerError):
                await runner.run_paired_evaluation(
                    runtime_dir=Path(directory) / "run",
                    protocol=self.protocol,
                    manifest=self.manifest,
                    tasks_by_id=self.tasks,
                    task_runner=task_runner,
                    cleanup_runner=cleanup_runner,
                    analyzer_runner=analyzer,
                    environment_summary={"validated": True},
                )
        self.assertEqual(calls, 1)
        self.assertFalse(analyzed)

    async def test_nonempty_runtime_and_task_substitution_are_rejected(self):
        async def no_task(variant, task):
            self.fail("task runner must not be called")

        async def no_cleanup(variant, pair):
            self.fail("cleanup must not be called")

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "run"
            runtime.mkdir()
            (runtime / "existing").write_text("x")
            with self.assertRaises(runner.P7RunnerError):
                await runner.run_paired_evaluation(
                    runtime_dir=runtime,
                    protocol=self.protocol,
                    manifest=self.manifest,
                    tasks_by_id=self.tasks,
                    task_runner=no_task,
                    cleanup_runner=no_cleanup,
                    analyzer_runner=lambda a, b: None,
                    environment_summary={},
                )
        substituted = copy.deepcopy(self.tasks)
        first = self.manifest["pairs"][0]["task_id"]
        substituted[first]["instance_id"] = "replacement"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(runner.P7RunnerError):
                await runner.run_paired_evaluation(
                    runtime_dir=Path(directory) / "run",
                    protocol=self.protocol,
                    manifest=self.manifest,
                    tasks_by_id=substituted,
                    task_runner=no_task,
                    cleanup_runner=no_cleanup,
                    analyzer_runner=lambda a, b: None,
                    environment_summary={},
                )

    def test_frozen_analyzer_refuses_incomplete_pair(self):
        with self.assertRaises(PairedAnalysisError):
            analyze_paired_results(self.protocol, self.manifest, [])


class OfflineCliTests(unittest.TestCase):
    def test_validate_only_makes_no_network_call(self):
        with patch.object(httpx.AsyncClient, "get", side_effect=AssertionError("network")), patch.object(
            httpx.AsyncClient, "post", side_effect=AssertionError("network")
        ):
            completed = subprocess.run(
                [
                    str(ROOT / ".venv-research" / "bin" / "python"),
                    str(ROOT / "scripts" / "run_p7_paired_evaluation.py"),
                    "--validate-only",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
            )
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["variant_runs"], 60)
        self.assertEqual(report["external_calls"], 0)


if __name__ == "__main__":
    unittest.main()
