#!/usr/bin/env python3
"""Build or verify the frozen 48-pair P7 supplemental manifest.

Selection is blind to task text, SQL, GT, test cases, follow-up text and
historical scores.  Each 100-row Full block contributes exactly eight tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_p7_manifest as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPLEMENT_PROTOCOL = (
    PROJECT_ROOT / "configs" / "p7" / "p7_supplement_48_protocol_v1.json"
)
DEFAULT_SUPPLEMENT_MANIFEST = (
    PROJECT_ROOT / "configs" / "p7" / "p7_supplement_48_manifest_v1.json"
)
PROTOCOL_ID = "valibra-p7-supplement-48-v1"
EXPECTED_PAIRS = 48
TASKS_PER_BLOCK = 8
BLOCK_SIZE = 100
BLOCK_COUNT = 6
PAIR_KEYS = {
    "pair_index",
    "global_index",
    "block_index",
    "block_range",
    "task_id",
    "selected_database",
    "initial_bird_coin",
    "official_ambiguity_count_summary",
    "has_follow_up",
    "selection_hash",
    "run_order",
}
MANIFEST_KEYS = {"schema_version", "protocol_id", "selection_seed", "pairs"}


class SupplementManifestError(ValueError):
    """Supplement protocol or manifest violates its frozen contract."""


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("protocol_sha256", None)
    return hashlib.sha256(base.canonical_json_bytes(body)).hexdigest()


def load_supplement_protocol(path: Path = DEFAULT_SUPPLEMENT_PROTOCOL) -> dict[str, Any]:
    value = base.load_json(path)
    if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL_ID:
        raise SupplementManifestError("unexpected supplement protocol")
    actual = _semantic_sha256(value)
    if value.get("protocol_sha256") != actual:
        raise SupplementManifestError("supplement protocol SHA256 mismatch")
    frozen = value.get("frozen_inputs", {})
    if frozen.get("full_input_rows") != base.EXPECTED_FULL_ROWS:
        raise SupplementManifestError("supplement Full row count is not frozen")
    if frozen.get("full_input_sha256") != base.EXPECTED_FULL_SHA256:
        raise SupplementManifestError("supplement Full SHA256 is not frozen")
    selection = value.get("selection", {})
    expected_ranges = [
        [start, start + BLOCK_SIZE - 1]
        for start in range(1, base.EXPECTED_FULL_ROWS + 1, BLOCK_SIZE)
    ]
    if selection.get("block_ranges_1_based") != expected_ranges:
        raise SupplementManifestError("supplement block ranges changed")
    if selection.get("tasks_per_block") != TASKS_PER_BLOCK:
        raise SupplementManifestError("supplement tasks_per_block changed")
    if selection.get("selected_count") != EXPECTED_PAIRS:
        raise SupplementManifestError("supplement selected_count changed")
    return value


def _selection_hash(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest()


def _order_hash(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0order\0{task_id}".encode()).hexdigest()


def _excluded_ids(protocol: Mapping[str, Any]) -> set[str]:
    base_manifest = base.load_json(PROJECT_ROOT / protocol["base_p7"]["manifest_path"])
    original = {str(pair["task_id"]) for pair in base_manifest["pairs"]}
    development = {
        str(task_id) for task_id in protocol["selection"]["exclude_development_tasks"]
    }
    return original | development


def build_supplement_manifest(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if len(records) != base.EXPECTED_FULL_ROWS:
        raise SupplementManifestError("Full input must contain exactly 600 rows")
    seed = str(protocol["selection"]["selection_seed"])
    excluded = _excluded_ids(protocol)
    candidates: dict[int, list[dict[str, Any]]] = {
        block: [] for block in range(1, BLOCK_COUNT + 1)
    }
    seen: set[str] = set()
    for global_index, record in enumerate(records, 1):
        projected = base._project_record(record)
        task_id = projected["task_id"]
        if task_id in seen:
            raise SupplementManifestError("Full task IDs must be unique")
        seen.add(task_id)
        if task_id in excluded:
            continue
        block_index = (global_index - 1) // BLOCK_SIZE + 1
        candidates[block_index].append(
            {
                **projected,
                "global_index": global_index,
                "block_index": block_index,
                "block_range": [
                    (block_index - 1) * BLOCK_SIZE + 1,
                    block_index * BLOCK_SIZE,
                ],
                "selection_hash": _selection_hash(seed, task_id),
            }
        )
    selected: list[dict[str, Any]] = []
    for block_index in range(1, BLOCK_COUNT + 1):
        ordered = sorted(
            candidates[block_index],
            key=lambda item: (item["selection_hash"], item["task_id"]),
        )
        selected.extend(ordered[:TASKS_PER_BLOCK])
    selected.sort(key=lambda item: (_order_hash(seed, item["task_id"]), item["task_id"]))
    pairs = []
    for pair_index, item in enumerate(selected, 1):
        run_order = (
            ["b0", "valibra_llm"]
            if pair_index <= EXPECTED_PAIRS // 2
            else ["valibra_llm", "b0"]
        )
        pair = {**item, "pair_index": pair_index, "run_order": run_order}
        if set(pair) != PAIR_KEYS:
            raise SupplementManifestError("supplement manifest pair fields changed")
        pairs.append(pair)
    manifest = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "selection_seed": seed,
        "pairs": pairs,
    }
    validate_supplement_manifest(manifest, records=records, protocol=protocol)
    return manifest


def validate_supplement_manifest(
    manifest: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    if set(manifest) != MANIFEST_KEYS:
        raise SupplementManifestError("supplement manifest fields changed")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise SupplementManifestError("supplement manifest protocol mismatch")
    if manifest.get("selection_seed") != protocol["selection"]["selection_seed"]:
        raise SupplementManifestError("supplement manifest seed mismatch")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIRS:
        raise SupplementManifestError("supplement manifest must contain 48 pairs")
    ids = [pair.get("task_id") for pair in pairs]
    if len(set(ids)) != EXPECTED_PAIRS:
        raise SupplementManifestError("supplement task IDs must be unique")
    if set(ids).intersection(_excluded_ids(protocol)):
        raise SupplementManifestError("frozen exclusion entered supplement manifest")
    records_by_index = {index: record for index, record in enumerate(records, 1)}
    block_counts = {block: 0 for block in range(1, BLOCK_COUNT + 1)}
    b0_first = 0
    valibra_first = 0
    for expected_index, pair in enumerate(pairs, 1):
        if not isinstance(pair, Mapping) or set(pair) != PAIR_KEYS:
            raise SupplementManifestError("supplement pair shape mismatch")
        if pair["pair_index"] != expected_index:
            raise SupplementManifestError("supplement pair order is not continuous")
        global_index = pair["global_index"]
        if not isinstance(global_index, int) or not 1 <= global_index <= 600:
            raise SupplementManifestError("supplement global_index out of range")
        if records_by_index[global_index].get("instance_id") != pair["task_id"]:
            raise SupplementManifestError("supplement task/global_index mismatch")
        expected_block = (global_index - 1) // BLOCK_SIZE + 1
        if pair["block_index"] != expected_block:
            raise SupplementManifestError("supplement block_index mismatch")
        block_counts[expected_block] += 1
        if pair["run_order"] == ["b0", "valibra_llm"]:
            b0_first += 1
        elif pair["run_order"] == ["valibra_llm", "b0"]:
            valibra_first += 1
        else:
            raise SupplementManifestError("supplement run_order is invalid")
    if set(block_counts.values()) != {TASKS_PER_BLOCK}:
        raise SupplementManifestError("each Full block must contribute exactly 8 tasks")
    if (b0_first, valibra_first) != (24, 24):
        raise SupplementManifestError("supplement run order must be balanced 24/24")


def manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    pairs = manifest["pairs"]
    return {
        "pairs": len(pairs),
        "unique_tasks": len({pair["task_id"] for pair in pairs}),
        "per_block": {
            str(block): sum(pair["block_index"] == block for pair in pairs)
            for block in range(1, BLOCK_COUNT + 1)
        },
        "database_count": len({pair["selected_database"] for pair in pairs}),
        "b0_first": sum(pair["run_order"][0] == "b0" for pair in pairs),
        "valibra_first": sum(
            pair["run_order"][0] == "valibra_llm" for pair in pairs
        ),
        "manifest_sha256": hashlib.sha256(base.canonical_json_bytes(manifest)).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_SUPPLEMENT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SUPPLEMENT_MANIFEST)
    parser.add_argument("--data", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    protocol = load_supplement_protocol(args.protocol)
    if base.sha256_file(args.data) != base.EXPECTED_FULL_SHA256:
        raise SupplementManifestError("Full input SHA256 mismatch")
    records = base.load_jsonl(args.data)
    expected = build_supplement_manifest(records, protocol)
    if args.emit:
        print(base.canonical_json_bytes(expected).decode(), end="")
        return
    if not args.check:
        raise SupplementManifestError("choose --emit or --check; this script never writes")
    if not args.manifest.is_file():
        raise SupplementManifestError("supplement manifest is missing")
    if args.manifest.read_bytes() != base.canonical_json_bytes(expected):
        raise SupplementManifestError("supplement manifest does not reproduce byte-for-byte")
    print(json.dumps({"status": "PASS", **manifest_summary(expected)}, sort_keys=True))


if __name__ == "__main__":
    main()
