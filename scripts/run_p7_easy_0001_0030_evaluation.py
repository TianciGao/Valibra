#!/usr/bin/env python3
"""Run the frozen Full original-order 1--30 B0/Valibra paired diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_p7_easy_0001_0030_manifest as easy
from scripts import run_p7_paired_evaluation as base
from scripts.build_p7_manifest import canonical_json_bytes, load_json, load_jsonl, sha256_file


EXPECTED_PROTOCOL_SHA256 = "790c2e80e5907fe9faec3c60b45feaab76a87c98f98f9850a8e800ab5e4e506c"
EXPECTED_MANIFEST_SHA256 = "b00b51773a8c4236a6d53eca12afd601ffac819c8772d800d6d4480214e9ec9f"
EXPECTED_PAIRS = 30
EXPECTED_ROWS = 60


def load_frozen_inputs(
    *,
    base_protocol_path: Path = base.DEFAULT_PROTOCOL,
    base_manifest_path: Path = base.DEFAULT_MANIFEST,
    exclusions_path: Path = base.DEFAULT_EXCLUSIONS,
    protocol_path: Path = easy.DEFAULT_PROTOCOL,
    manifest_path: Path = easy.DEFAULT_MANIFEST,
    data_path: Path = base.DEFAULT_DATASET,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    base_protocol, _, tasks_by_id = base.load_frozen_run_inputs(
        protocol_path=base_protocol_path,
        manifest_path=base_manifest_path,
        exclusions_path=exclusions_path,
        data_path=data_path,
    )
    protocol = easy.load_protocol(protocol_path)
    if protocol["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise base.P7RunnerError("early-index protocol differs from frozen value")
    records = load_jsonl(data_path)
    manifest = load_json(manifest_path)
    easy.validate_manifest(manifest, records=records, protocol=protocol)
    if canonical_json_bytes(easy.build_manifest(records, protocol)) != manifest_path.read_bytes():
        raise base.P7RunnerError("early-index manifest does not reproduce byte-for-byte")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise base.P7RunnerError("early-index manifest SHA256 mismatch")
    return base_protocol, protocol, manifest, tasks_by_id


def production_analyzer(ledger_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/analyze_p7_easy_0001_0030_results.py"),
            "--results",
            str(ledger_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


async def run_evaluation(
    *,
    runtime_dir: Path,
    base_protocol: Mapping[str, Any],
    easy_protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    environment_summary: Mapping[str, Any],
    task_runner: base.TaskRunner,
    cleanup_runner: base.CleanupRunner,
    analyzer_runner: base.AnalyzerRunner,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if easy.semantic_sha256(easy_protocol) != EXPECTED_PROTOCOL_SHA256:
        raise base.P7RunnerError("runtime early-index protocol differs from frozen value")
    if hashlib.sha256(canonical_json_bytes(manifest)).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise base.P7RunnerError("runtime early-index manifest differs from frozen value")
    summary = dict(environment_summary)
    summary.update(
        {
            "early_index_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "early_index_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "selection": "Full records[0:30] / global_index 1-30",
            "original_p7_status": "NO-GO retained unchanged",
            "p8_authorized": False,
        }
    )
    analysis_path = await base.run_paired_evaluation(
        runtime_dir=runtime_dir,
        protocol=base_protocol,
        manifest=manifest,
        tasks_by_id=tasks_by_id,
        task_runner=task_runner,
        cleanup_runner=cleanup_runner,
        analyzer_runner=analyzer_runner,
        environment_summary=summary,
        project_root=project_root,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
    )
    completion_path = runtime_dir / "completion.json"
    completion = load_json(completion_path)
    completion.update(
        {
            "early_index_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "original_p7_status": "NO-GO retained unchanged",
            "p8_authorized": False,
        }
    )
    base._atomic_json(completion_path, completion)
    return analysis_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--base-protocol", type=Path, default=base.DEFAULT_PROTOCOL)
    parser.add_argument("--base-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--exclusions", type=Path, default=base.DEFAULT_EXCLUSIONS)
    parser.add_argument("--protocol", type=Path, default=easy.DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=easy.DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    base_protocol, protocol, manifest, tasks = load_frozen_inputs(
        base_protocol_path=args.base_protocol,
        base_manifest_path=args.base_manifest,
        exclusions_path=args.exclusions,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        data_path=args.data,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "pairs": EXPECTED_PAIRS,
                    "variant_runs": EXPECTED_ROWS,
                    "global_index_range": [1, 30],
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "early_index_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
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
        run_evaluation(
            runtime_dir=args.runtime_dir,
            base_protocol=base_protocol,
            easy_protocol=protocol,
            manifest=manifest,
            tasks_by_id=tasks,
            environment_summary=environment_summary,
            task_runner=base.production_task_runner,
            cleanup_runner=base.production_cleanup,
            analyzer_runner=production_analyzer,
        )
    )


if __name__ == "__main__":
    main()
