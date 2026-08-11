"""Offline tests for the frozen 48-pair P7 supplemental experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_p7_supplement_results as analyzer
from scripts import build_p7_supplement_manifest as builder
from scripts import run_p7_paired_evaluation as base_runner
from scripts import run_p7_supplement_evaluation as runner
from scripts.build_p7_manifest import canonical_json_bytes, load_json, load_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _provider_call() -> dict:
    return {
        "kind": "provider",
        "usage_reliable": True,
        "input_tokens": 10,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "total_tokens": 15,
        "started_at": "2026-08-11T00:00:00+00:00",
        "completed_at": "2026-08-11T00:00:01+00:00",
        "cost": None,
    }


def _ledger_record(base_protocol: dict, pair: dict, variant: str) -> dict:
    return {
        "pair_index": pair["pair_index"],
        "task_id": pair["task_id"],
        "variant": variant,
        "configuration_sha256": base_protocol["configurations"][variant]["configuration_sha256"],
        "reward": 1.0 if variant == "valibra_llm" else 0.0,
        "p1_passed": variant == "valibra_llm",
        "p2_passed": False,
        "models": {
            "main_agent": {"calls": [_provider_call()]},
            "grounding": {"calls": [_provider_call()] if variant == "valibra_llm" else []},
            "user_simulator": {"calls": [_provider_call()]},
        },
        "latency": {"task_wall_ms": 1000.0},
        "bird_coin": {"initial": pair["initial_bird_coin"], "used": 1.0, "remaining": pair["initial_bird_coin"] - 1.0},
        "tools": {"total": 2, "ask_user": 1, "submit_sql": 1},
        "restart": {"pair_startup_restart_count": 0, "restart_only_before_provider_calls_and_results": True},
        "integrity": {
            "configuration_match": True,
            "official_contract_match": True,
            "credential_leak_absent": True,
            "ground_truth_leak_absent": True,
            "trajectory_complete": True,
            "trajectory_reconstructable": True,
            "export_complete": True,
            "pending_count": 0,
            "cleanup_complete": True,
            "task_substitution_absent": True,
            "post_provider_rerun_absent": True,
            "infrastructure_error": False,
        },
    }


def _official_result(task_id: str, initial_budget: float, variant: str) -> dict:
    result = {
        "task_id": task_id,
        "instance_id": task_id,
        "total_reward": 0.5,
        "phase1_passed": True,
        "phase2_passed": False,
        "elapsed_seconds": 0.01,
        "initial_budget": initial_budget,
        "budget_used": 1.0,
        "budget_remaining": initial_budget - 1.0,
        "prompt_flow": [],
        "tool_trajectory": [],
        "adk_events": [],
        "dialogue_history": [],
        "token_usage": {},
        "user_simulator_audit": {"llm_calls": [], "token_usage": {}, "dialogue_history": []},
    }
    if variant == "valibra_llm":
        result["valibra"] = {"export_status": "succeeded", "grounding_summary": {"pending_count": 0}}
    return result


class P7SupplementFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_protocol = load_json(ROOT / "configs/p7/p7_protocol_v1.json")
        cls.protocol = builder.load_supplement_protocol()
        cls.manifest = load_json(builder.DEFAULT_SUPPLEMENT_MANIFEST)
        cls.records = load_jsonl(base_runner.DEFAULT_DATASET)

    def test_manifest_reproduces_and_is_balanced(self):
        rebuilt = builder.build_supplement_manifest(self.records, self.protocol)
        self.assertEqual(canonical_json_bytes(rebuilt), builder.DEFAULT_SUPPLEMENT_MANIFEST.read_bytes())
        summary = builder.manifest_summary(rebuilt)
        self.assertEqual(summary["pairs"], 48)
        self.assertEqual(summary["unique_tasks"], 48)
        self.assertEqual(summary["per_block"], {str(index): 8 for index in range(1, 7)})
        self.assertEqual((summary["b0_first"], summary["valibra_first"]), (24, 24))
        self.assertEqual(summary["manifest_sha256"], runner.EXPECTED_MANIFEST_SHA256)

    def test_manifest_excludes_original_p7_and_development_task(self):
        original = load_json(ROOT / "configs/p7/p7_manifest_v1.json")
        original_ids = {pair["task_id"] for pair in original["pairs"]}
        supplement_ids = {pair["task_id"] for pair in self.manifest["pairs"]}
        self.assertFalse(original_ids & supplement_ids)
        self.assertNotIn("cross_border_15", supplement_ids)

    def test_protocol_preserves_original_decision_scope(self):
        self.assertFalse(self.protocol["interpretation"]["overwrites_original_p7"])
        self.assertFalse(self.protocol["interpretation"]["changes_original_p7_no_go"])
        self.assertFalse(self.protocol["interpretation"]["authorizes_p8"])
        self.assertEqual(self.protocol["base_p7"]["result_status"], "NO-GO retained unchanged")

    def test_complete_analyzer_is_supplemental_only(self):
        rows = [
            _ledger_record(self.base_protocol, pair, variant)
            for pair in self.manifest["pairs"]
            for variant in pair["run_order"]
        ]
        result = analyzer.analyze_supplement_results(
            self.base_protocol, self.protocol, self.manifest, rows
        )
        self.assertEqual(result["complete_pairs"], 48)
        self.assertEqual(result["original_p7_status"], "NO-GO retained unchanged")
        self.assertFalse(result["p8_authorized"])
        self.assertEqual(result["decision_scope"], "supplemental difficulty calibration only")
        self.assertEqual(result["supplemental_signal"], "GO")

    def test_analyzer_rejects_incomplete_ledger(self):
        with self.assertRaises(analyzer.base.PairedAnalysisError):
            analyzer.analyze_supplement_results(
                self.base_protocol, self.protocol, self.manifest, []
            )


class P7SupplementLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_96_adjacent_runs_and_analysis_after_completion(self):
        base_protocol, protocol, manifest, tasks = runner.load_frozen_supplement_inputs()
        calls = []
        analyzed_at = []

        async def fake_task(variant, task):
            calls.append((task["instance_id"], variant))
            pair = next(item for item in manifest["pairs"] if item["task_id"] == task["instance_id"])
            return _official_result(task["instance_id"], pair["initial_bird_coin"], variant)

        async def fake_cleanup(variant, pair):
            return {"verified": True, "agent_session_removed": True, "user_state_removed": True, "residual_task_databases": []}

        def fake_analyzer(ledger, output):
            analyzed_at.append(len([line for line in ledger.read_text().splitlines() if line]))
            output.write_text('{"status":"offline"}\n', encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            base_runner, "production_task_runner", side_effect=fake_task
        ), patch.object(
            base_runner, "production_cleanup", side_effect=fake_cleanup
        ), patch.object(runner, "production_analyzer", side_effect=fake_analyzer):
            runtime = Path(directory) / "run"
            await runner.run_supplement_evaluation(
                runtime_dir=runtime,
                base_protocol=base_protocol,
                supplement_protocol=protocol,
                manifest=manifest,
                tasks_by_id=tasks,
                environment_summary={"validated": True},
            )
            completion = json.loads((runtime / "completion.json").read_text())
            ledger_count = len((runtime / "p7_supplement_ledger.jsonl").read_text().splitlines())
        expected = [(pair["task_id"], variant) for pair in manifest["pairs"] for variant in pair["run_order"]]
        self.assertEqual(calls, expected)
        self.assertEqual(ledger_count, 96)
        self.assertEqual(analyzed_at, [96])
        self.assertTrue(completion["effect_summary_computed_only_after_96_rows"])
        self.assertEqual(completion["original_p7_status"], "NO-GO retained unchanged")
        self.assertFalse(completion["p8_authorized"])


class P7SupplementCliTests(unittest.TestCase):
    def test_validate_only_has_no_external_calls(self):
        completed = subprocess.run(
            [str(ROOT / ".venv-research/bin/python"), str(ROOT / "scripts/run_p7_supplement_evaluation.py"), "--validate-only"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["pairs"], 48)
        self.assertEqual(result["variant_runs"], 96)
        self.assertEqual(result["external_calls"], 0)
        self.assertFalse(result["p8_authorized"])


if __name__ == "__main__":
    unittest.main()
