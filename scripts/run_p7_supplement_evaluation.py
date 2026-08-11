#!/usr/bin/env python3
"""Run the frozen 48-pair P7 difficulty-calibration supplement.

The production task execution, score provenance, cleanup and ledger records
are delegated to the already verified P7 runner.  This wrapper only supplies
the independent 48-task manifest and prevents its result from overwriting the
original 30-pair P7 decision.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_p7_supplement_manifest as supplement
from scripts import run_p7_paired_evaluation as base
from scripts.build_p7_manifest import canonical_json_bytes, load_json, load_jsonl, sha256_file


EXPECTED_MANIFEST_SHA256 = "fa62cbbb5e64a9e16c074335ded7b3de25ca60836a03a1361e69686191d7a70e"
EXPECTED_SUPPLEMENT_PROTOCOL_SHA256 = "2f362e5b6e665696892d6327a77c7d0bc8c0fc64defa95d669c9364fa0620c55"
EXPECTED_PAIRS = supplement.EXPECTED_PAIRS
EXPECTED_ROWS = EXPECTED_PAIRS * 2


def load_frozen_supplement_inputs(
    *,
    base_protocol_path: Path = base.DEFAULT_PROTOCOL,
    base_manifest_path: Path = base.DEFAULT_MANIFEST,
    exclusions_path: Path = base.DEFAULT_EXCLUSIONS,
    supplement_protocol_path: Path = supplement.DEFAULT_SUPPLEMENT_PROTOCOL,
    supplement_manifest_path: Path = supplement.DEFAULT_SUPPLEMENT_MANIFEST,
    data_path: Path = base.DEFAULT_DATASET,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    base_protocol, _, tasks_by_id = base.load_frozen_run_inputs(
        protocol_path=base_protocol_path,
        manifest_path=base_manifest_path,
        exclusions_path=exclusions_path,
        data_path=data_path,
    )
    protocol = supplement.load_supplement_protocol(supplement_protocol_path)
    if protocol["protocol_sha256"] != EXPECTED_SUPPLEMENT_PROTOCOL_SHA256:
        raise base.P7RunnerError("supplement protocol differs from frozen value")
    records = load_jsonl(data_path)
    manifest = load_json(supplement_manifest_path)
    supplement.validate_supplement_manifest(
        manifest, records=records, protocol=protocol
    )
    rebuilt = supplement.build_supplement_manifest(records, protocol)
    if canonical_json_bytes(rebuilt) != supplement_manifest_path.read_bytes():
        raise base.P7RunnerError("supplement manifest does not reproduce byte-for-byte")
    if sha256_file(supplement_manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise base.P7RunnerError("supplement manifest SHA256 mismatch")
    return base_protocol, protocol, manifest, tasks_by_id


def production_analyzer(
    ledger_path: Path,
    output_path: Path,
    *,
    base_protocol_path: Path = base.DEFAULT_PROTOCOL,
    supplement_protocol_path: Path = supplement.DEFAULT_SUPPLEMENT_PROTOCOL,
    supplement_manifest_path: Path = supplement.DEFAULT_SUPPLEMENT_MANIFEST,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/analyze_p7_supplement_results.py"),
            "--base-protocol",
            str(base_protocol_path),
            "--protocol",
            str(supplement_protocol_path),
            "--manifest",
            str(supplement_manifest_path),
            "--results",
            str(ledger_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


async def run_supplement_evaluation(
    *,
    runtime_dir: Path,
    base_protocol: Mapping[str, Any],
    supplement_protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    environment_summary: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if supplement._semantic_sha256(supplement_protocol) != EXPECTED_SUPPLEMENT_PROTOCOL_SHA256:
        raise base.P7RunnerError("runtime supplement protocol differs from frozen value")
    if hashlib.sha256(canonical_json_bytes(manifest)).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise base.P7RunnerError("runtime supplement manifest differs from frozen value")
    if runtime_dir.exists() and any(runtime_dir.iterdir()):
        raise base.P7RunnerError("runtime directory must be new and empty")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = runtime_dir / "p7_supplement_ledger.jsonl"
    provenance_path = runtime_dir / "score_provenance.jsonl"
    lifecycle_path = runtime_dir / "lifecycle.jsonl"
    raw_dir = runtime_dir / "official_raw_results"
    raw_dir.mkdir(parents=True, exist_ok=False)
    base._atomic_json(
        runtime_dir / "configuration_summary.json",
        {
            "created_at": base._utc_now(),
            "base_p7_protocol_sha256": base.EXPECTED_PROTOCOL_SHA256,
            "supplement_protocol_sha256": EXPECTED_SUPPLEMENT_PROTOCOL_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "common_configuration_sha256": base.EXPECTED_COMMON_CONFIG_SHA256,
            "variant_configuration_sha256": base.EXPECTED_VARIANT_CONFIG_SHA256,
            "environment": dict(environment_summary),
            "effect_summary_before_completion": False,
            "original_p7_status": "NO-GO retained unchanged",
            "p8_authorized": False,
        },
    )

    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIRS:
        raise base.P7RunnerError("frozen supplement must have exactly 48 pairs")
    rows = 0
    for pair in pairs:
        task_id = str(pair["task_id"])
        task = tasks_by_id.get(task_id)
        if task is None or task.get("instance_id") != task_id:
            raise base.P7RunnerError(f"manifest task substitution detected: {task_id}")
        for variant in pair["run_order"]:
            rows += 1
            base._progress(pair, variant, "starting", "pending")
            base._append_jsonl(
                lifecycle_path,
                {
                    "row_index": rows,
                    "pair_index": pair["pair_index"],
                    "task_id": task_id,
                    "variant": variant,
                    "lifecycle": "started",
                    "cleanup": "pending",
                    "timestamp": base._utc_now(),
                },
            )
            started = time.perf_counter()
            try:
                result = await base.production_task_runner(variant, task)
            except Exception as exc:
                cleanup_status = "failed"
                cleanup_error_type = None
                try:
                    await base.production_cleanup(variant, pair)
                    cleanup_status = "verified"
                except Exception as cleanup_exc:
                    cleanup_error_type = type(cleanup_exc).__name__[:128]
                base._append_jsonl(
                    lifecycle_path,
                    {
                        "row_index": rows,
                        "pair_index": pair["pair_index"],
                        "task_id": task_id,
                        "variant": variant,
                        "lifecycle": "failed",
                        "cleanup": cleanup_status,
                        "timestamp": base._utc_now(),
                        "error_type": type(exc).__name__[:128],
                        "cleanup_error_type": cleanup_error_type,
                    },
                )
                raise
            wall_ms = (time.perf_counter() - started) * 1000.0
            if result.get("task_id", result.get("instance_id")) != task_id:
                raise base.P7RunnerError("official result task_id mismatch")
            result = dict(result)
            result["elapsed_seconds"] = result.get("elapsed_seconds", wall_ms / 1000.0)
            raw_path = raw_dir / f"{rows:03d}_{int(pair['pair_index']):02d}_{variant}.json"
            base._atomic_json(raw_path, result)
            provenance = base.score_provenance(
                result,
                pair_index=int(pair["pair_index"]),
                task_id=task_id,
                variant=variant,
                raw_path=raw_path,
                runtime_dir=runtime_dir,
            )
            base._append_jsonl(provenance_path, provenance)
            cleanup = await base.production_cleanup(variant, pair)
            ledger = base.build_ledger_record(
                result,
                pair=pair,
                variant=variant,
                cleanup=cleanup,
                project_root=project_root,
            )
            base._append_jsonl(ledger_path, ledger)
            base._append_jsonl(
                lifecycle_path,
                {
                    "row_index": rows,
                    "pair_index": pair["pair_index"],
                    "task_id": task_id,
                    "variant": variant,
                    "lifecycle": "completed",
                    "cleanup": "verified",
                    "timestamp": base._utc_now(),
                    "official_raw_result_sha256": provenance["official_raw_result_sha256"],
                },
            )
            base._progress(pair, variant, "completed", "verified")

    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    provenance_rows = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if rows != EXPECTED_ROWS or len(ledger_rows) != EXPECTED_ROWS:
        raise base.P7RunnerError("supplement ledger is not exactly 96 rows")
    if len(provenance_rows) != EXPECTED_ROWS:
        raise base.P7RunnerError("supplement score provenance is not exactly 96 rows")
    expected = [
        (pair["pair_index"], pair["task_id"], variant)
        for pair in pairs
        for variant in pair["run_order"]
    ]
    actual = [
        (row["pair_index"], row["task_id"], row["variant"])
        for row in ledger_rows
    ]
    if actual != expected:
        raise base.P7RunnerError("final supplement ledger differs from frozen manifest")
    if any(row["score_source"] != "bird_interact_official" for row in provenance_rows):
        raise base.P7RunnerError("official score provenance is incomplete")
    analysis_path = runtime_dir / "paired_analysis.json"
    production_analyzer(ledger_path, analysis_path)
    if not analysis_path.is_file():
        raise base.P7RunnerError("supplement analyzer did not produce output")
    base._atomic_json(
        runtime_dir / "completion.json",
        {
            "status": "complete",
            "completed_at": base._utc_now(),
            "pairs": EXPECTED_PAIRS,
            "variant_runs": EXPECTED_ROWS,
            "ledger_sha256": sha256_file(ledger_path),
            "score_provenance_sha256": sha256_file(provenance_path),
            "analysis_sha256": sha256_file(analysis_path),
            "effect_summary_computed_only_after_96_rows": True,
            "original_p7_status": "NO-GO retained unchanged",
            "p8_authorized": False,
        },
    )
    return analysis_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--base-protocol", type=Path, default=base.DEFAULT_PROTOCOL)
    parser.add_argument("--base-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--exclusions", type=Path, default=base.DEFAULT_EXCLUSIONS)
    parser.add_argument("--protocol", type=Path, default=supplement.DEFAULT_SUPPLEMENT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=supplement.DEFAULT_SUPPLEMENT_MANIFEST)
    parser.add_argument("--data", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    base_protocol, protocol, manifest, tasks = load_frozen_supplement_inputs(
        base_protocol_path=args.base_protocol,
        base_manifest_path=args.base_manifest,
        exclusions_path=args.exclusions,
        supplement_protocol_path=args.protocol,
        supplement_manifest_path=args.manifest,
        data_path=args.data,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "pairs": EXPECTED_PAIRS,
                    "variant_runs": EXPECTED_ROWS,
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "supplement_protocol_sha256": EXPECTED_SUPPLEMENT_PROTOCOL_SHA256,
                    "base_p7_protocol_sha256": base.EXPECTED_PROTOCOL_SHA256,
                    "external_calls": 0,
                    "formal_b0_fingerprint": base.formal_b0_fingerprint(),
                    "original_p7_status": "NO-GO retained unchanged",
                    "p8_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.runtime_dir is None:
        raise base.P7RunnerError("--runtime-dir is required for a real run")
    environment_summary = base.validate_runtime_environment(os.environ)
    environment_summary["formal_b0_fingerprint"] = base.formal_b0_fingerprint()
    environment_summary["service_health"] = asyncio.run(
        base.validate_service_health(base_protocol)
    )
    asyncio.run(
        run_supplement_evaluation(
            runtime_dir=args.runtime_dir,
            base_protocol=base_protocol,
            supplement_protocol=protocol,
            manifest=manifest,
            tasks_by_id=tasks,
            environment_summary=environment_summary,
        )
    )


if __name__ == "__main__":
    main()
