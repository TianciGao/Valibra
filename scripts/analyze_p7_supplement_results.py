#!/usr/bin/env python3
"""Analyze the complete frozen 48-pair P7 supplemental ledger.

This analyzer reuses the original P7 record validator and metric functions.
Its output is explicitly a difficulty-calibration supplement: it cannot
replace the original P7 NO-GO or authorize P8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_p7_paired_results as base
from scripts import build_p7_supplement_manifest as supplement
from scripts.build_p7_manifest import canonical_json_bytes, load_json, sha256_file, validate_protocol


EXPECTED_PAIRS = supplement.EXPECTED_PAIRS
EXPECTED_ROWS = EXPECTED_PAIRS * 2
VARIANTS = base.VARIANTS


def analyze_supplement_results(
    base_protocol: Mapping[str, Any],
    supplement_protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_protocol_sha = validate_protocol(base_protocol)
    if base_protocol_sha != supplement_protocol["base_p7"]["protocol_sha256"]:
        raise base.PairedAnalysisError("base P7 protocol SHA mismatch")
    supplement_sha = supplement._semantic_sha256(supplement_protocol)
    if supplement_protocol.get("protocol_sha256") != supplement_sha:
        raise base.PairedAnalysisError("supplement protocol SHA mismatch")
    configuration_hashes = base._validate_config_hashes(base_protocol)
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIRS:
        raise base.PairedAnalysisError("analysis requires the complete 48-pair manifest")
    if len(raw_records) != EXPECTED_ROWS:
        raise base.PairedAnalysisError(
            f"analysis requires exactly {EXPECTED_ROWS} adjacent rows, got {len(raw_records)}"
        )

    expected_sequence: list[tuple[Mapping[str, Any], str]] = []
    for expected_index, pair in enumerate(pairs, 1):
        if not isinstance(pair, Mapping) or pair.get("pair_index") != expected_index:
            raise base.PairedAnalysisError("manifest pair order is invalid")
        order = pair.get("run_order")
        if order not in (["b0", "valibra_llm"], ["valibra_llm", "b0"]):
            raise base.PairedAnalysisError("manifest run_order is invalid")
        expected_sequence.extend((pair, variant) for variant in order)

    normalized = [
        base._validate_record(
            raw,
            pair=pair,
            expected_variant=variant,
            configuration_sha256=configuration_hashes[variant],
            row_index=index,
        )
        for index, (raw, (pair, variant)) in enumerate(
            zip(raw_records, expected_sequence), 1
        )
    ]
    by_variant = {
        variant: [record for record in normalized if record["variant"] == variant]
        for variant in VARIANTS
    }
    if any(len(records) != EXPECTED_PAIRS for records in by_variant.values()):
        raise base.PairedAnalysisError("each variant needs exactly 48 records")

    pair_summaries: list[dict[str, Any]] = []
    hard_gate_failures: list[str] = []
    for pair in pairs:
        pair_records = [
            record
            for record in normalized
            if record["pair_index"] == pair["pair_index"]
        ]
        if len(pair_records) != 2:
            raise base.PairedAnalysisError("incomplete pair after normalization")
        variants = {record["variant"]: record for record in pair_records}
        b0 = variants["b0"]
        valibra = variants["valibra_llm"]
        if b0["restart_count"] != valibra["restart_count"]:
            hard_gate_failures.append(
                f"{pair['task_id']}:inconsistent_pair_restart_count"
            )
        hard_gate_failures.extend(b0["hard_gate_failures"])
        hard_gate_failures.extend(valibra["hard_gate_failures"])
        delta = valibra["reward"] - b0["reward"]
        pair_summaries.append(
            {
                "pair_index": pair["pair_index"],
                "task_id": pair["task_id"],
                "run_order": pair["run_order"],
                "b0_reward": b0["reward"],
                "valibra_llm_reward": valibra["reward"],
                "reward_delta": delta,
                "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
            }
        )

    deltas = [float(pair["reward_delta"]) for pair in pair_summaries]
    wins = sum(pair["outcome"] == "win" for pair in pair_summaries)
    ties = sum(pair["outcome"] == "tie" for pair in pair_summaries)
    losses = sum(pair["outcome"] == "loss" for pair in pair_summaries)
    mean_delta = base._mean(deltas)
    hard_gate_failures = sorted(set(hard_gate_failures))
    supplemental_signal = base.preregistered_decision(
        mean_delta, wins, losses, hard_gate_failures
    )
    result = {
        "schema_version": "1.0",
        "protocol_id": supplement_protocol["protocol_id"],
        "protocol_sha256": supplement_sha,
        "base_p7_protocol_sha256": base_protocol_sha,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "complete_pairs": EXPECTED_PAIRS,
        "strict_pairing_valid": True,
        "primary": {
            "metric": "paired_mean_reward_delta_valibra_minus_b0",
            "mean_delta": mean_delta,
            "bootstrap_95_ci": base.bootstrap_mean_ci(
                deltas, protocol_sha256=supplement_sha
            ),
        },
        "win_tie_loss": {"wins": wins, "ties": ties, "losses": losses},
        "variants": {
            variant: base._variant_summary(records)
            for variant, records in by_variant.items()
        },
        "pairs": pair_summaries,
        "hard_gates": {
            "passed": not hard_gate_failures,
            "failures": hard_gate_failures,
        },
        "supplemental_signal": supplemental_signal,
        "decision_scope": "supplemental difficulty calibration only",
        "original_p7_status": "NO-GO retained unchanged",
        "p8_authorized": False,
        "raw_results_mutated": False,
    }
    base._assert_safe_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-protocol", type=Path, default=PROJECT_ROOT / "configs/p7/p7_protocol_v1.json")
    parser.add_argument("--protocol", type=Path, default=supplement.DEFAULT_SUPPLEMENT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=supplement.DEFAULT_SUPPLEMENT_MANIFEST)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.results.resolve() == args.output.resolve():
        raise base.PairedAnalysisError("output cannot overwrite the raw result ledger")
    result = analyze_supplement_results(
        load_json(args.base_protocol),
        load_json(args.protocol),
        load_json(args.manifest),
        base._load_result_jsonl(args.results),
    )
    result["raw_results_sha256"] = sha256_file(args.results)
    base._assert_safe_summary(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))
    print(
        json.dumps(
            {
                "status": "PASS",
                "supplemental_signal": result["supplemental_signal"],
                "complete_pairs": result["complete_pairs"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
