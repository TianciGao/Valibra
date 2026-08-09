from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from scripts.analyze_p7_paired_results import (
    PairedAnalysisError,
    _assert_safe_summary,
    analyze_paired_results,
    bootstrap_mean_ci,
    preregistered_decision,
)
from scripts.build_p7_manifest import (
    DEFAULT_DATASET,
    DEFAULT_EXCLUSIONS,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    EXPECTED_FULL_SHA256,
    EXPECTED_P6_TARGET,
    build_manifest,
    canonical_json_bytes,
    load_excluded_task_ids,
    load_json,
    load_jsonl,
    manifest_summary,
    protocol_digest,
    selection_seed,
    validate_manifest,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[2]


def _provider_call(
    *,
    input_tokens=10,
    output_tokens=5,
    reasoning_tokens=2,
    cost=0.1,
    usage_reliable=True,
    started_at="2026-08-09T00:00:00+00:00",
    completed_at="2026-08-09T00:00:01+00:00",
):
    return {
        "kind": "provider",
        "usage_reliable": usage_reliable,
        "input_tokens": input_tokens if usage_reliable else None,
        "output_tokens": output_tokens if usage_reliable else None,
        "reasoning_tokens": reasoning_tokens if usage_reliable else None,
        "total_tokens": input_tokens + output_tokens if usage_reliable else None,
        "started_at": started_at,
        "completed_at": completed_at,
        "cost": cost,
    }


def _local_call():
    return {
        "kind": "local",
        "usage_reliable": False,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "started_at": "2026-08-09T00:00:01+00:00",
        "completed_at": "2026-08-09T00:00:02+00:00",
        "cost": None,
    }


def _record(protocol, pair, variant, reward):
    config_sha = protocol["configurations"][variant]["configuration_sha256"]
    main_calls = [_provider_call(), _local_call()]
    grounding_calls = (
        [
            _provider_call(
                input_tokens=4,
                output_tokens=6,
                reasoning_tokens=3,
                cost=0.02,
                completed_at="2026-08-09T00:00:00.500000+00:00",
            )
        ]
        if variant == "valibra_llm"
        else []
    )
    return {
        "pair_index": pair["pair_index"],
        "task_id": pair["task_id"],
        "variant": variant,
        "configuration_sha256": config_sha,
        "reward": reward,
        "p1_passed": reward > 0,
        "p2_passed": reward > 1,
        "models": {
            "main_agent": {"calls": main_calls},
            "grounding": {"calls": grounding_calls},
            "user_simulator": {
                "calls": [
                    _provider_call(
                        input_tokens=3,
                        output_tokens=2,
                        reasoning_tokens=0,
                        cost=0.01,
                        completed_at="2026-08-09T00:00:00.250000+00:00",
                    )
                ]
            },
        },
        "latency": {"task_wall_ms": 2000.0},
        "bird_coin": {
            "initial": pair["initial_bird_coin"],
            "used": pair["initial_bird_coin"],
            "remaining": 0.0,
        },
        "tools": {"total": 3, "ask_user": 1, "submit_sql": 1},
        "restart": {
            "pair_startup_restart_count": 0,
            "restart_only_before_provider_calls_and_results": True,
        },
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


def _records(protocol, manifest, *, b0_reward=0.0, valibra_reward=1.0):
    records = []
    for pair in manifest["pairs"]:
        for variant in pair["run_order"]:
            records.append(
                _record(
                    protocol,
                    pair,
                    variant,
                    valibra_reward if variant == "valibra_llm" else b0_reward,
                )
            )
    return records


class P7FrozenManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(DEFAULT_PROTOCOL)
        cls.manifest = load_json(DEFAULT_MANIFEST)
        cls.exclusions_document = load_json(DEFAULT_EXCLUSIONS)
        cls.excluded = load_excluded_task_ids(cls.exclusions_document)
        cls.records = load_jsonl(DEFAULT_DATASET)

    def test_protocol_and_manifest_reproduce_byte_for_byte(self):
        self.assertEqual(validate_protocol(self.protocol), self.protocol["protocol_sha256"])
        rebuilt = build_manifest(self.records, self.excluded)
        self.assertEqual(canonical_json_bytes(rebuilt), DEFAULT_MANIFEST.read_bytes())
        self.assertEqual(
            manifest_summary(rebuilt)["manifest_sha256"],
            "62d17c20097c81fa14351cd750f51babf567c6acd240aa51ff285a4a46fc8c2a",
        )

    def test_selection_seed_is_stable_and_input_sha_changes_manifest(self):
        first = selection_seed(EXPECTED_FULL_SHA256, EXPECTED_P6_TARGET)
        second = selection_seed(EXPECTED_FULL_SHA256, EXPECTED_P6_TARGET)
        changed = selection_seed("0" * 64, EXPECTED_P6_TARGET)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        normal = build_manifest(self.records, self.excluded)
        changed_manifest = build_manifest(
            self.records,
            self.excluded,
            full_sha256="0" * 64,
            enforce_diversity=False,
        )
        self.assertNotEqual(normal, changed_manifest)

    def test_only_committed_real_development_task_is_excluded(self):
        full_ids = {record["instance_id"] for record in self.records}
        found = set()
        tracked = subprocess.run(
            ["git", "ls-files", "evidence"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for name in tracked:
            path = ROOT / name
            if path.suffix != ".json":
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            def walk(item):
                if isinstance(item, dict):
                    for key, child in item.items():
                        if (
                            key in {"task_id", "instance_id", "selected_task_id"}
                            and isinstance(child, str)
                            and child in full_ids
                        ):
                            found.add(child)
                        walk(child)
                elif isinstance(item, list):
                    for child in item:
                        walk(child)

            walk(value)
        self.assertEqual(found, {"cross_border_15"})
        self.assertEqual(self.excluded, found)
        selected = {pair["task_id"] for pair in self.manifest["pairs"]}
        self.assertTrue(selected.isdisjoint(found))

    def test_manifest_has_30_unique_tasks_and_balanced_order(self):
        validate_manifest(self.manifest, excluded_task_ids=self.excluded)
        pairs = self.manifest["pairs"]
        self.assertEqual(len(pairs), 30)
        self.assertEqual(len({pair["task_id"] for pair in pairs}), 30)
        self.assertEqual(sum(pair["run_order"][0] == "b0" for pair in pairs), 15)
        self.assertEqual(
            sum(pair["run_order"][0] == "valibra_llm" for pair in pairs), 15
        )

    def test_manifest_diversity_gates_pass_without_reselection(self):
        summary = manifest_summary(self.manifest)
        self.assertGreaterEqual(summary["database_count"], 10)
        self.assertGreater(summary["follow_up_tasks"], 0)
        self.assertGreater(summary["query_ambiguity_tasks"], 0)
        self.assertGreater(summary["knowledge_ambiguity_tasks"], 0)

    def test_manifest_contains_no_forbidden_task_or_result_fields(self):
        text = DEFAULT_MANIFEST.read_text(encoding="utf-8")
        for forbidden in (
            "amb_user_query",
            '"follow_up":',
            "sol_sql",
            "test_cases",
            "external_knowledge",
            "reward",
            "score",
            "prompt",
            "response",
        ):
            self.assertNotIn(forbidden, text)

    def test_committed_configs_have_no_credential_values_or_key_paths(self):
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (DEFAULT_PROTOCOL, DEFAULT_MANIFEST, DEFAULT_EXCLUSIONS)
        )
        self.assertNotIn("/home/user/", text)
        self.assertNotRegex(text, r"sk-[A-Za-z0-9_-]{8,}")
        self.assertNotIn("Authorization:", text)
        self.assertIn('"credential_source_type": "file"', text)

    def test_protocol_configuration_hashes_are_frozen(self):
        self.assertEqual(
            protocol_digest(self.protocol), self.protocol["protocol_sha256"]
        )
        common = self.protocol["configurations"]["common"]
        common_sha = hashlib.sha256(
            canonical_json_bytes(common["configuration"])
        ).hexdigest()
        self.assertEqual(common_sha, common["configuration_sha256"])
        for variant in ("b0", "valibra_llm"):
            payload = {
                "common_configuration_sha256": common_sha,
                "variant": self.protocol["configurations"][variant]["variant"],
            }
            self.assertEqual(
                hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
                self.protocol["configurations"][variant]["configuration_sha256"],
            )


class P7PairedAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(DEFAULT_PROTOCOL)
        cls.manifest = load_json(DEFAULT_MANIFEST)

    def test_complete_pairs_calculate_primary_secondary_and_ledgers(self):
        result = analyze_paired_results(
            self.protocol,
            self.manifest,
            _records(self.protocol, self.manifest, b0_reward=0.0, valibra_reward=2.0),
        )
        self.assertEqual(result["complete_pairs"], 30)
        self.assertEqual(result["primary"]["mean_delta"], 2.0)
        self.assertEqual(result["win_tie_loss"], {"wins": 30, "ties": 0, "losses": 0})
        self.assertEqual(result["decision"], "GO")
        self.assertEqual(result["variants"]["b0"]["p1_rate"], 0.0)
        self.assertEqual(result["variants"]["valibra_llm"]["p2_rate"], 1.0)
        self.assertEqual(
            result["variants"]["b0"]["main_agent"]["tokens"]["total_tokens"],
            30 * 15,
        )
        self.assertEqual(result["variants"]["b0"]["total_model_tokens"], 30 * 15)
        self.assertEqual(
            result["variants"]["valibra_llm"]["total_model_tokens"],
            30 * (15 + 10),
        )
        self.assertEqual(
            result["variants"]["b0"]["main_agent"]["provider_latency_ms"],
            30000.0,
        )
        self.assertEqual(
            result["variants"]["b0"]["main_agent"]["provider_latency_included_calls"],
            30,
        )
        self.assertEqual(result["variants"]["b0"]["main_agent"]["provider_latency_excluded_calls"], 0)

    def test_reasoning_tokens_are_not_double_counted(self):
        result = analyze_paired_results(
            self.protocol, self.manifest, _records(self.protocol, self.manifest)
        )
        main = result["variants"]["b0"]["main_agent"]["tokens"]
        self.assertEqual(main["output_tokens"], 30 * 5)
        self.assertEqual(main["reasoning_tokens"], 30 * 2)
        self.assertEqual(main["total_tokens"], 30 * (10 + 5))

    def test_incomplete_cost_stays_null(self):
        records = _records(self.protocol, self.manifest)
        records[0]["models"]["main_agent"]["calls"][0]["cost"] = None
        result = analyze_paired_results(self.protocol, self.manifest, records)
        self.assertIsNone(result["variants"]["b0"]["main_agent"]["cost"])
        self.assertIsNone(result["variants"]["b0"]["total_model_cost"])
        self.assertTrue(result["hard_gates"]["passed"])

    def test_unreliable_usage_is_null_and_not_used_for_provider_latency(self):
        records = _records(self.protocol, self.manifest)
        call = records[0]["models"]["main_agent"]["calls"][0]
        call.update(
            {
                "usage_reliable": False,
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
            }
        )
        result = analyze_paired_results(self.protocol, self.manifest, records)
        b0 = result["variants"]["b0"]
        self.assertIsNone(b0["main_agent"]["tokens"])
        self.assertEqual(b0["main_agent"]["provider_latency_excluded_calls"], 1)
        self.assertEqual(b0["main_agent"]["provider_latency_included_calls"], 29)

    def test_bootstrap_is_deterministic_and_has_10000_iterations(self):
        first = bootstrap_mean_ci(
            [1.0, -1.0, 0.5] * 10,
            protocol_sha256=self.protocol["protocol_sha256"],
        )
        second = bootstrap_mean_ci(
            [1.0, -1.0, 0.5] * 10,
            protocol_sha256=self.protocol["protocol_sha256"],
        )
        self.assertEqual(first, second)
        self.assertEqual(first["iterations"], 10000)

    def test_preregistered_go_hold_no_go(self):
        self.assertEqual(preregistered_decision(0.1, 12, 8, []), "GO")
        self.assertEqual(preregistered_decision(-0.1, 8, 12, []), "NO-GO")
        self.assertEqual(preregistered_decision(0.0, 10, 10, []), "HOLD")
        self.assertEqual(preregistered_decision(0.1, 12, 8, ["gate"]), "NO-GO")

    def test_incomplete_missing_duplicate_and_nonadjacent_rows_are_rejected(self):
        records = _records(self.protocol, self.manifest)
        with self.assertRaises(PairedAnalysisError):
            analyze_paired_results(self.protocol, self.manifest, records[:-1])
        duplicated = copy.deepcopy(records)
        duplicated[1] = copy.deepcopy(duplicated[0])
        with self.assertRaises(PairedAnalysisError):
            analyze_paired_results(self.protocol, self.manifest, duplicated)
        nonadjacent = copy.deepcopy(records)
        nonadjacent[1], nonadjacent[2] = nonadjacent[2], nonadjacent[1]
        with self.assertRaises(PairedAnalysisError):
            analyze_paired_results(self.protocol, self.manifest, nonadjacent)

    def test_configuration_mismatch_is_rejected(self):
        records = _records(self.protocol, self.manifest)
        records[0]["configuration_sha256"] = "0" * 64
        with self.assertRaisesRegex(PairedAnalysisError, "configuration SHA mismatch"):
            analyze_paired_results(self.protocol, self.manifest, records)

    def test_unknown_or_sensitive_result_field_is_rejected(self):
        records = _records(self.protocol, self.manifest)
        records[0]["prompt"] = "not allowed"
        with self.assertRaises(PairedAnalysisError):
            analyze_paired_results(self.protocol, self.manifest, records)

    def test_hard_integrity_failure_and_b0_grounding_call_are_no_go(self):
        records = _records(self.protocol, self.manifest)
        records[0]["integrity"]["trajectory_complete"] = False
        records[0]["models"]["grounding"]["calls"] = [_provider_call()]
        result = analyze_paired_results(self.protocol, self.manifest, records)
        self.assertEqual(result["decision"], "NO-GO")
        self.assertFalse(result["hard_gates"]["passed"])
        self.assertTrue(
            any("grounding_provider_called" in item for item in result["hard_gates"]["failures"])
        )

    def test_forbidden_restart_is_no_go(self):
        records = _records(self.protocol, self.manifest)
        records[0]["restart"]["pair_startup_restart_count"] = 1
        records[0]["restart"]["restart_only_before_provider_calls_and_results"] = False
        result = analyze_paired_results(self.protocol, self.manifest, records)
        self.assertEqual(result["decision"], "NO-GO")
        self.assertTrue(
            any("forbidden_pair_restart" in item for item in result["hard_gates"]["failures"])
        )

    def test_safe_summary_contains_no_prompts_sql_or_hidden_data(self):
        result = analyze_paired_results(
            self.protocol, self.manifest, _records(self.protocol, self.manifest)
        )
        _assert_safe_summary(result)
        text = json.dumps(result, sort_keys=True)
        for forbidden in ("sol_sql", "test_cases", "ground_truth", "follow_up"):
            self.assertNotIn(forbidden, text)

    def test_offline_paths_make_zero_external_calls(self):
        # These pure functions receive all data as arguments and cannot start a
        # service or Provider. Successful completion is the external-call gate.
        rebuilt = build_manifest(
            load_jsonl(DEFAULT_DATASET),
            load_excluded_task_ids(load_json(DEFAULT_EXCLUSIONS)),
        )
        result = analyze_paired_results(
            self.protocol, rebuilt, _records(self.protocol, rebuilt)
        )
        self.assertEqual(result["complete_pairs"], 30)
        self.assertTrue(result["raw_results_mutated"] is False)


if __name__ == "__main__":
    unittest.main()
