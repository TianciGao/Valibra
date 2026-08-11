#!/usr/bin/env python3
"""Build or verify the frozen Full original-order 1--30 paired manifest.

Task selection is exactly ``records[0:30]``.  The task order remains the
original Full order; only which variant runs first is deterministically
balanced without using task text, answers, SQL, or historical scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_p7_manifest as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "p7" / "p7_easy_0001_0030_protocol_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "p7" / "p7_easy_0001_0030_manifest_v1.json"
PROTOCOL_ID = "valibra-p7-easy-0001-0030-v1"
EXPECTED_PAIRS = 30
FIRST_GLOBAL_INDEX = 1
LAST_GLOBAL_INDEX = 30
PAIR_KEYS = {
    "pair_index",
    "global_index",
    "task_id",
    "selected_database",
    "initial_bird_coin",
    "official_ambiguity_count_summary",
    "has_follow_up",
    "selection_hash",
    "run_order",
}
MANIFEST_KEYS = {"schema_version", "protocol_id", "selection_seed", "pairs"}


class EasyManifestError(ValueError):
    """The early-index protocol or manifest violates its frozen contract."""


def semantic_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("protocol_sha256", None)
    return hashlib.sha256(base.canonical_json_bytes(body)).hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    value = base.load_json(path)
    if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL_ID:
        raise EasyManifestError("unexpected early-index protocol")
    if value.get("protocol_sha256") != semantic_sha256(value):
        raise EasyManifestError("early-index protocol SHA256 mismatch")
    frozen = value.get("frozen_inputs", {})
    if frozen.get("full_input_rows") != base.EXPECTED_FULL_ROWS:
        raise EasyManifestError("Full row count changed")
    if frozen.get("full_input_sha256") != base.EXPECTED_FULL_SHA256:
        raise EasyManifestError("Full input SHA256 changed")
    selection = value.get("selection", {})
    if selection.get("python_slice") != "records[0:30]":
        raise EasyManifestError("early-index Python slice changed")
    if selection.get("global_index_range_1_based") != [1, 30]:
        raise EasyManifestError("early-index range changed")
    if selection.get("selected_count") != EXPECTED_PAIRS:
        raise EasyManifestError("early-index task count changed")
    return value


def _selection_hash(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest()


def _order_hash(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0order\0{task_id}".encode()).hexdigest()


def build_manifest(
    records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if len(records) != base.EXPECTED_FULL_ROWS:
        raise EasyManifestError("Full input must contain exactly 600 rows")
    ids = [record.get("instance_id") for record in records]
    if len(set(ids)) != base.EXPECTED_FULL_ROWS:
        raise EasyManifestError("Full task IDs must be unique")
    seed = str(protocol["selection"]["selection_seed"])
    projected = []
    for global_index, record in enumerate(records[:EXPECTED_PAIRS], 1):
        item = base._project_record(record)
        projected.append(
            {
                **item,
                "global_index": global_index,
                "selection_hash": _selection_hash(seed, item["task_id"]),
            }
        )
    ranked = sorted(
        projected,
        key=lambda item: (_order_hash(seed, item["task_id"]), item["task_id"]),
    )
    b0_first_ids = {item["task_id"] for item in ranked[: EXPECTED_PAIRS // 2]}
    pairs = []
    for pair_index, item in enumerate(projected, 1):
        run_order = (
            ["b0", "valibra_llm"]
            if item["task_id"] in b0_first_ids
            else ["valibra_llm", "b0"]
        )
        pair = {**item, "pair_index": pair_index, "run_order": run_order}
        if set(pair) != PAIR_KEYS:
            raise EasyManifestError("early-index manifest pair fields changed")
        pairs.append(pair)
    manifest = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "selection_seed": seed,
        "pairs": pairs,
    }
    validate_manifest(manifest, records=records, protocol=protocol)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    if set(manifest) != MANIFEST_KEYS:
        raise EasyManifestError("early-index manifest fields changed")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise EasyManifestError("early-index manifest protocol mismatch")
    if manifest.get("selection_seed") != protocol["selection"]["selection_seed"]:
        raise EasyManifestError("early-index manifest seed mismatch")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIRS:
        raise EasyManifestError("early-index manifest must contain 30 pairs")
    ids = [pair.get("task_id") for pair in pairs]
    if len(set(ids)) != EXPECTED_PAIRS:
        raise EasyManifestError("early-index task IDs must be unique")
    b0_first = valibra_first = 0
    for expected_index, pair in enumerate(pairs, 1):
        if not isinstance(pair, Mapping) or set(pair) != PAIR_KEYS:
            raise EasyManifestError("early-index pair shape mismatch")
        if pair["pair_index"] != expected_index or pair["global_index"] != expected_index:
            raise EasyManifestError("early-index order is not original Full order")
        if records[expected_index - 1].get("instance_id") != pair["task_id"]:
            raise EasyManifestError("early-index task/global_index mismatch")
        if pair["run_order"] == ["b0", "valibra_llm"]:
            b0_first += 1
        elif pair["run_order"] == ["valibra_llm", "b0"]:
            valibra_first += 1
        else:
            raise EasyManifestError("early-index run_order is invalid")
    if (b0_first, valibra_first) != (15, 15):
        raise EasyManifestError("early-index run order must be balanced 15/15")


def manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    pairs = manifest["pairs"]
    return {
        "pairs": len(pairs),
        "unique_tasks": len({pair["task_id"] for pair in pairs}),
        "global_index_first": pairs[0]["global_index"],
        "global_index_last": pairs[-1]["global_index"],
        "database_count": len({pair["selected_database"] for pair in pairs}),
        "b0_first": sum(pair["run_order"][0] == "b0" for pair in pairs),
        "valibra_first": sum(pair["run_order"][0] == "valibra_llm" for pair in pairs),
        "manifest_sha256": hashlib.sha256(base.canonical_json_bytes(manifest)).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if base.sha256_file(args.data) != base.EXPECTED_FULL_SHA256:
        raise EasyManifestError("Full input SHA256 mismatch")
    records = base.load_jsonl(args.data)
    expected = build_manifest(records, protocol)
    if args.emit:
        print(base.canonical_json_bytes(expected).decode(), end="")
        return
    if not args.check:
        raise EasyManifestError("choose --emit or --check; this script never writes")
    if not args.manifest.is_file():
        raise EasyManifestError("early-index manifest is missing")
    if args.manifest.read_bytes() != base.canonical_json_bytes(expected):
        raise EasyManifestError("early-index manifest does not reproduce byte-for-byte")
    print(json.dumps({"status": "PASS", **manifest_summary(expected)}, sort_keys=True))


if __name__ == "__main__":
    main()
