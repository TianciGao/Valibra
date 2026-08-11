#!/usr/bin/env python3
"""Analyze the complete Full original-order 1--30 paired ledger."""

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
from scripts import build_p7_easy_0001_0030_manifest as easy
from scripts.build_p7_manifest import canonical_json_bytes, load_json, sha256_file


def analyze_results(
    base_protocol: Mapping[str, Any],
    easy_protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocol_sha = easy.semantic_sha256(easy_protocol)
    if easy_protocol.get("protocol_sha256") != protocol_sha:
        raise base.PairedAnalysisError("early-index protocol SHA mismatch")
    if easy_protocol["base_p7"]["protocol_sha256"] != base_protocol.get("protocol_sha256"):
        raise base.PairedAnalysisError("base P7 protocol SHA mismatch")
    result = base.analyze_paired_results(base_protocol, manifest, raw_records)
    deltas = [float(pair["reward_delta"]) for pair in result["pairs"]]
    signal = result.pop("decision")
    result.update(
        {
            "protocol_id": easy_protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "base_p7_protocol_sha256": base_protocol["protocol_sha256"],
            "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
            "diagnostic_signal": signal,
            "decision_scope": "Full original-order 1-30 paired diagnostic only",
            "original_p7_status": "NO-GO retained unchanged",
            "p8_authorized": False,
        }
    )
    result["primary"]["bootstrap_95_ci"] = base.bootstrap_mean_ci(
        deltas, protocol_sha256=protocol_sha
    )
    base._assert_safe_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-protocol", type=Path, default=PROJECT_ROOT / "configs/p7/p7_protocol_v1.json")
    parser.add_argument("--protocol", type=Path, default=easy.DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=easy.DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.results.resolve() == args.output.resolve():
        raise base.PairedAnalysisError("output cannot overwrite the raw result ledger")
    result = analyze_results(
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
                "diagnostic_signal": result["diagnostic_signal"],
                "complete_pairs": result["complete_pairs"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
