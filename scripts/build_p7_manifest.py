#!/usr/bin/env python3
"""Build or verify the frozen P7 paired-evaluation task manifest.

Selection is intentionally blind to task text, SQL, test cases, ground truth,
and historical benchmark results.  Only the explicitly allowed task metadata
is projected from the frozen 600-row Full input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "bird-interact-full" / "bird_interact_data.jsonl"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "p7" / "p7_protocol_v1.json"
DEFAULT_EXCLUSIONS = (
    PROJECT_ROOT / "configs" / "p7" / "p7_development_exclusions_v1.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "p7" / "p7_manifest_v1.json"

PROTOCOL_ID = "valibra-p7-v1"
EXPECTED_FULL_SHA256 = (
    "a051f7a78462d6c17e840c048ea15c4be65b9f8eed61aad3d2df1370561b10c0"
)
EXPECTED_FULL_ROWS = 600
EXPECTED_P6_TARGET = "9bdcbf27112fe35679f46b2470c3afa8ac3ec330"
EXPECTED_TASK_COUNT = 30
PATIENCE = 3

MANIFEST_PAIR_KEYS = {
    "task_id",
    "selected_database",
    "initial_bird_coin",
    "official_ambiguity_count_summary",
    "has_follow_up",
    "selection_hash",
    "pair_index",
    "run_order",
}
AMBIGUITY_KEYS = {
    "query_critical",
    "query_non_critical",
    "knowledge",
    "budget_relevant_total",
}
FORBIDDEN_MANIFEST_KEYS = {
    "amb_user_query",
    "user_query",
    "follow_up",
    "sol_sql",
    "sql",
    "test_cases",
    "ground_truth",
    "gt",
    "external_knowledge",
    "prompt",
    "response",
    "reward",
    "score",
}


class ManifestContractError(ValueError):
    """The frozen input or generated manifest violates the P7 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ManifestContractError(
                    f"Full row {line_number} must be a JSON object"
                )
            rows.append(value)
    return rows


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    body = dict(protocol)
    claimed = body.pop("protocol_sha256", None)
    if claimed is not None and not isinstance(claimed, str):
        raise ManifestContractError("protocol_sha256 must be a string")
    return sha256_bytes(canonical_json_bytes(body))


def validate_protocol(protocol: Mapping[str, Any]) -> str:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ManifestContractError("unexpected protocol_id")
    protocol_sha = protocol_digest(protocol)
    if protocol.get("protocol_sha256") != protocol_sha:
        raise ManifestContractError(
            f"protocol SHA mismatch: expected {protocol_sha}, "
            f"got {protocol.get('protocol_sha256')}"
        )
    frozen = protocol.get("frozen_inputs")
    if not isinstance(frozen, Mapping):
        raise ManifestContractError("missing frozen_inputs")
    expected = {
        "full_input_rows": EXPECTED_FULL_ROWS,
        "full_input_sha256": EXPECTED_FULL_SHA256,
        "p6_pass_target": EXPECTED_P6_TARGET,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ManifestContractError(f"frozen input mismatch for {key}")
    configurations = protocol.get("configurations")
    if not isinstance(configurations, Mapping):
        raise ManifestContractError("missing configurations")
    common = configurations.get("common")
    if not isinstance(common, Mapping):
        raise ManifestContractError("missing common configuration")
    common_sha = sha256_bytes(canonical_json_bytes(common.get("configuration")))
    if common.get("configuration_sha256") != common_sha:
        raise ManifestContractError("common configuration SHA mismatch")
    for variant in ("b0", "valibra_llm"):
        entry = configurations.get(variant)
        if not isinstance(entry, Mapping):
            raise ManifestContractError(f"missing {variant} configuration")
        variant_sha = sha256_bytes(
            canonical_json_bytes(
                {
                    "common_configuration_sha256": common_sha,
                    "variant": entry.get("variant"),
                }
            )
        )
        if entry.get("configuration_sha256") != variant_sha:
            raise ManifestContractError(f"{variant} configuration SHA mismatch")
    return protocol_sha


def selection_seed(
    full_sha256: str = EXPECTED_FULL_SHA256,
    p6_target: str = EXPECTED_P6_TARGET,
) -> str:
    material = f"{PROTOCOL_ID}{full_sha256}{p6_target}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def selection_hash(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest()


def order_hash(seed: str, task_id: str) -> str:
    return hashlib.sha256(
        f"{seed}\0order\0{task_id}".encode("utf-8")
    ).hexdigest()


def _list_count(value: Any, field: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, list):
        raise ManifestContractError(f"{field} must be a list")
    return len(value)


def _project_record(record: Mapping[str, Any]) -> dict[str, Any]:
    task_id = record.get("instance_id")
    database = record.get("selected_database")
    if not isinstance(task_id, str) or not task_id:
        raise ManifestContractError("every Full row needs a non-empty instance_id")
    if not isinstance(database, str) or not database:
        raise ManifestContractError(
            f"task {task_id} needs a non-empty selected_database"
        )

    query = record.get("user_query_ambiguity") or {}
    if not isinstance(query, Mapping):
        raise ManifestContractError(
            f"task {task_id} user_query_ambiguity must be an object"
        )
    critical = _list_count(
        query.get("critical_ambiguity"),
        f"{task_id}.user_query_ambiguity.critical_ambiguity",
    )
    non_critical = _list_count(
        query.get("non_critical_ambiguity"),
        f"{task_id}.user_query_ambiguity.non_critical_ambiguity",
    )
    knowledge = _list_count(
        record.get("knowledge_ambiguity"),
        f"{task_id}.knowledge_ambiguity",
    )
    budget_relevant = critical + knowledge
    ambiguity_summary = {
        "query_critical": critical,
        "query_non_critical": non_critical,
        "knowledge": knowledge,
        "budget_relevant_total": budget_relevant,
    }
    return {
        "task_id": task_id,
        "selected_database": database,
        "initial_bird_coin": float(6 + 2 * budget_relevant + 2 * PATIENCE),
        "official_ambiguity_count_summary": ambiguity_summary,
        # Presence only.  No follow-up body is copied or hashed.
        "has_follow_up": bool(record.get("follow_up")),
    }


def load_excluded_task_ids(exclusion_document: Mapping[str, Any]) -> set[str]:
    entries = exclusion_document.get("exclusions")
    if not isinstance(entries, list):
        raise ManifestContractError("exclusions must be a list")
    result: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ManifestContractError("every exclusion must be an object")
        task_id = entry.get("task_id")
        refs = entry.get("committed_evidence_refs")
        if not isinstance(task_id, str) or not task_id:
            raise ManifestContractError("every exclusion needs a task_id")
        if task_id in result:
            raise ManifestContractError(f"duplicate exclusion: {task_id}")
        if not isinstance(refs, list) or not refs or not all(
            isinstance(ref, str) and ref.startswith("evidence/") for ref in refs
        ):
            raise ManifestContractError(
                f"exclusion {task_id} needs committed evidence refs"
            )
        result.add(task_id)
    if "cross_border_15" not in result:
        raise ManifestContractError("cross_border_15 must be excluded")
    return result


def build_manifest(
    records: Sequence[Mapping[str, Any]],
    excluded_task_ids: set[str],
    *,
    full_sha256: str = EXPECTED_FULL_SHA256,
    p6_target: str = EXPECTED_P6_TARGET,
    task_count: int = EXPECTED_TASK_COUNT,
    enforce_diversity: bool = True,
) -> dict[str, Any]:
    if len(records) != EXPECTED_FULL_ROWS:
        raise ManifestContractError(
            f"Full input must contain {EXPECTED_FULL_ROWS} rows, got {len(records)}"
        )
    projected = [_project_record(record) for record in records]
    task_ids = [item["task_id"] for item in projected]
    if len(set(task_ids)) != EXPECTED_FULL_ROWS:
        raise ManifestContractError("Full instance_id values must be 600 unique values")
    unknown_exclusions = excluded_task_ids.difference(task_ids)
    if unknown_exclusions:
        raise ManifestContractError(
            f"exclusions not present in Full: {sorted(unknown_exclusions)}"
        )
    seed = selection_seed(full_sha256, p6_target)
    candidates = [
        {
            **item,
            "selection_hash": selection_hash(seed, item["task_id"]),
        }
        for item in projected
        if item["task_id"] not in excluded_task_ids
    ]
    selected = sorted(
        candidates,
        key=lambda item: (item["selection_hash"], item["task_id"]),
    )[:task_count]
    selected.sort(key=lambda item: (order_hash(seed, item["task_id"]), item["task_id"]))

    pairs: list[dict[str, Any]] = []
    for index, item in enumerate(selected, 1):
        run_order = (
            ["b0", "valibra_llm"]
            if index <= task_count // 2
            else ["valibra_llm", "b0"]
        )
        pair = {**item, "pair_index": index, "run_order": run_order}
        if set(pair) != MANIFEST_PAIR_KEYS:
            raise ManifestContractError("generated manifest field set changed")
        pairs.append(pair)

    manifest = {"pairs": pairs}
    validate_manifest(
        manifest,
        excluded_task_ids=excluded_task_ids,
        expected_task_count=task_count,
        enforce_diversity=enforce_diversity,
    )
    return manifest


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    excluded_task_ids: set[str],
    expected_task_count: int = EXPECTED_TASK_COUNT,
    enforce_diversity: bool = True,
) -> None:
    if set(manifest) != {"pairs"}:
        raise ManifestContractError("manifest top level must contain only pairs")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != expected_task_count:
        raise ManifestContractError(
            f"manifest needs exactly {expected_task_count} pairs"
        )
    task_ids: list[str] = []
    b0_first = 0
    valibra_first = 0
    for expected_index, pair in enumerate(pairs, 1):
        if not isinstance(pair, Mapping) or set(pair) != MANIFEST_PAIR_KEYS:
            raise ManifestContractError("manifest pair field set is not frozen")
        if pair.get("pair_index") != expected_index:
            raise ManifestContractError("pair_index must be continuous in file order")
        task_id = pair.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ManifestContractError("manifest task_id must be non-empty")
        task_ids.append(task_id)
        run_order = pair.get("run_order")
        if run_order == ["b0", "valibra_llm"]:
            b0_first += 1
        elif run_order == ["valibra_llm", "b0"]:
            valibra_first += 1
        else:
            raise ManifestContractError(f"invalid run_order for {task_id}")
        summary = pair.get("official_ambiguity_count_summary")
        if not isinstance(summary, Mapping) or set(summary) != AMBIGUITY_KEYS:
            raise ManifestContractError("ambiguity summary field set changed")
    if len(set(task_ids)) != expected_task_count:
        raise ManifestContractError("manifest task_id values must be unique")
    if set(task_ids).intersection(excluded_task_ids):
        raise ManifestContractError("development exclusion entered the manifest")
    expected_half = expected_task_count // 2
    if (b0_first, valibra_first) != (expected_half, expected_half):
        raise ManifestContractError("run-order balance must be 15/15")
    forbidden = {key.lower() for key in _walk_keys(manifest)}.intersection(
        FORBIDDEN_MANIFEST_KEYS
    )
    if forbidden:
        raise ManifestContractError(
            f"forbidden manifest fields present: {sorted(forbidden)}"
        )
    if enforce_diversity:
        databases = {pair["selected_database"] for pair in pairs}
        has_follow_up = any(pair["has_follow_up"] for pair in pairs)
        has_query_ambiguity = any(
            pair["official_ambiguity_count_summary"]["query_critical"] > 0
            or pair["official_ambiguity_count_summary"]["query_non_critical"] > 0
            for pair in pairs
        )
        has_knowledge_ambiguity = any(
            pair["official_ambiguity_count_summary"]["knowledge"] > 0
            for pair in pairs
        )
        failures = []
        if len(databases) < 10:
            failures.append(f"database coverage is {len(databases)}, expected >=10")
        if not has_follow_up:
            failures.append("no follow-up task selected")
        if not has_query_ambiguity:
            failures.append("no query ambiguity selected")
        if not has_knowledge_ambiguity:
            failures.append("no knowledge ambiguity selected")
        if failures:
            raise ManifestContractError("; ".join(failures))


def manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    pairs = manifest["pairs"]
    return {
        "task_count": len(pairs),
        "unique_task_ids": len({pair["task_id"] for pair in pairs}),
        "database_count": len({pair["selected_database"] for pair in pairs}),
        "b0_first": sum(pair["run_order"][0] == "b0" for pair in pairs),
        "valibra_first": sum(
            pair["run_order"][0] == "valibra_llm" for pair in pairs
        ),
        "follow_up_tasks": sum(bool(pair["has_follow_up"]) for pair in pairs),
        "query_ambiguity_tasks": sum(
            pair["official_ambiguity_count_summary"]["query_critical"] > 0
            or pair["official_ambiguity_count_summary"]["query_non_critical"] > 0
            for pair in pairs
        ),
        "knowledge_ambiguity_tasks": sum(
            pair["official_ambiguity_count_summary"]["knowledge"] > 0
            for pair in pairs
        ),
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed manifest; this script never writes files",
    )
    args = parser.parse_args()
    if not args.check:
        raise SystemExit(
            "refusing to write a frozen manifest; use --check against the "
            "reviewed committed file"
        )

    protocol = load_json(args.protocol)
    protocol_sha = validate_protocol(protocol)
    data_sha = sha256_file(args.data)
    if data_sha != EXPECTED_FULL_SHA256:
        raise ManifestContractError(
            f"Full SHA mismatch: expected {EXPECTED_FULL_SHA256}, got {data_sha}"
        )
    exclusions_document = load_json(args.exclusions)
    excluded = load_excluded_task_ids(exclusions_document)
    records = load_jsonl(args.data)
    expected = build_manifest(records, excluded)
    committed_bytes = args.manifest.read_bytes()
    expected_bytes = canonical_json_bytes(expected)
    if committed_bytes != expected_bytes:
        raise ManifestContractError("committed manifest bytes are not reproducible")
    summary = {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "full_input_sha256": data_sha,
        "selection_seed": selection_seed(),
        "development_exclusions": sorted(excluded),
        **manifest_summary(expected),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
