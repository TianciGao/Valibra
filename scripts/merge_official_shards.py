"""Validate and merge completed official shard directories in global-index order."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

from shared.config import PROJECT_ROOT, settings
from shared.llm import _read_api_key_file


PREDICTION_KEYS = (
    "instance_id",
    "subtask_1_predicted_sql",
    "subtask_2_predicted_sql",
)
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "tool_prompt_tokens",
)
FORBIDDEN_PUBLIC_MARKERS = (
    '"sol_sql"',
    '"test_cases"',
    '"prompt_flow"',
    '"reasoning_content"',
    '"ground_truth"',
    '"api_key"',
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_source(source: Path) -> Dict[str, Any]:
    summary = _json(source / "summary.json")
    checkpoint = _json(source / "checkpoint.json")
    manifest = _jsonl(source / "shard_manifest.jsonl")
    predictions = _jsonl(source / "submission_predictions.jsonl")
    if summary.get("status") != "complete" or checkpoint.get("status") != "complete":
        raise RuntimeError(f"Source shard is not complete: {source}")
    if len(manifest) != len(predictions):
        raise RuntimeError(f"Manifest/prediction count mismatch: {source}")
    if len({record["instance_id"] for record in manifest}) != len(manifest):
        raise RuntimeError(f"Duplicate source instance_id: {source}")
    if [record["global_index"] for record in manifest] != sorted(
        record["global_index"] for record in manifest
    ):
        raise RuntimeError(f"Source manifest is not in global-index order: {source}")
    if [record["instance_id"] for record in manifest] != [
        record["instance_id"] for record in predictions
    ]:
        raise RuntimeError(f"Source prediction order mismatch: {source}")
    for prediction in predictions:
        if tuple(prediction) != PREDICTION_KEYS:
            raise RuntimeError(f"Unexpected prediction fields in {source}")
    if _sha256_file(source / "submission_predictions.jsonl") != summary.get(
        "prediction_file_sha256"
    ):
        raise RuntimeError(f"Source prediction SHA256 mismatch: {source}")
    if checkpoint.get("all_completed_cleanup_verified") is not True:
        raise RuntimeError(f"Source cleanup is not fully verified: {source}")
    return {
        "path": source,
        "summary": summary,
        "checkpoint": checkpoint,
        "manifest": manifest,
        "predictions": predictions,
    }


def _task_envelope(source: Path, global_index: int) -> Dict[str, Any]:
    path = source / "logs" / "task_results" / f"{global_index:04d}.json"
    envelope = _json(path)
    if envelope.get("cleanup", {}).get("verified") is not True:
        raise RuntimeError(f"Task cleanup is not verified: {path}")
    if envelope["cleanup"].get("residual_after") != []:
        raise RuntimeError(f"Task has residual databases: {path}")
    return envelope


def _calculate_metrics(task_refs: list[dict]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    reward = 0.0
    phase1 = 0
    phase2 = 0
    elapsed = 0.0
    tokens = {field: 0 for field in TOKEN_FIELDS}
    reward_distribution: Counter[str] = Counter()
    tool_calls: Counter[str] = Counter()
    system_llm_calls = 0
    user_llm_calls = 0
    dialogue_turns = 0
    budget_used = 0.0
    with_subtask1 = 0
    with_subtask2 = 0
    without_submit = 0

    for ref in task_refs:
        result = _task_envelope(ref["source"], ref["global_index"])["result"]
        task_reward = float(result.get("total_reward", 0) or 0)
        reward += task_reward
        reward_distribution[f"{task_reward:g}"] += 1
        phase1 += int(bool(result.get("phase1_passed")))
        phase2 += int(bool(result.get("phase2_passed")))
        elapsed += float(result.get("elapsed_seconds", 0) or 0)
        budget_used += float(result.get("budget_used", 0) or 0)
        combined = (result.get("token_usage") or {}).get("combined") or {}
        for field in TOKEN_FIELDS:
            tokens[field] += int(combined.get(field, 0) or 0)
        trajectory = result.get("tool_trajectory") or []
        for event in trajectory:
            tool_calls[str(event.get("tool") or "unknown")] += 1
        if not any(event.get("tool") == "submit_sql" for event in trajectory):
            without_submit += 1
        system_llm_calls += len(result.get("prompt_flow") or [])
        user_llm_calls += len(
            (result.get("user_simulator_audit") or {}).get("llm_calls") or []
        )
        dialogue_turns += len(result.get("dialogue_history") or [])
        with_subtask1 += int(bool(result.get("subtask_1_predicted_sql")))
        with_subtask2 += int(bool(result.get("subtask_2_predicted_sql")))

    total = len(task_refs)
    metrics = {
        "completed_tasks": total,
        "total_reward": reward,
        "average_reward": reward / total if total else 0.0,
        "phase1_count": phase1,
        "phase1_rate": phase1 / total if total else 0.0,
        "phase2_count": phase2,
        "phase2_rate": phase2 / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "token_usage": tokens,
    }
    internal = {
        "reward_distribution": dict(sorted(reward_distribution.items())),
        "system_agent_llm_calls": system_llm_calls,
        "user_simulator_llm_calls": user_llm_calls,
        "dialogue_turns": dialogue_turns,
        "tool_calls_total": sum(tool_calls.values()),
        "tool_calls_by_name": dict(sorted(tool_calls.items())),
        "tasks_with_subtask_1_prediction": with_subtask1,
        "tasks_with_subtask_2_prediction": with_subtask2,
        "tasks_without_submit_sql": without_submit,
        "total_budget_used": budget_used,
        "average_budget_used": budget_used / total if total else 0.0,
    }
    return metrics, internal


def _write_private(path: Path, task_refs: list[dict], metrics: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write('{"mode":"a-interact","leaderboard_mode":"stress","metrics":')
        json.dump(metrics, handle, ensure_ascii=False, default=str)
        handle.write(',"results":[')
        for position, ref in enumerate(task_refs):
            if position:
                handle.write(",")
            envelope = _task_envelope(ref["source"], ref["global_index"])
            private_result = {
                "global_index": envelope["global_index"],
                "instance_id": envelope["instance_id"],
                "selected_database": envelope["selected_database"],
                "model_completed_at": envelope["model_completed_at"],
                "cleanup": envelope["cleanup"],
                **envelope["result"],
            }
            json.dump(
                private_result,
                handle,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
        handle.write("]}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _credential_values() -> list[str]:
    values = [
        value
        for value in (
            settings.system_agent_api_key,
            settings.user_sim_api_key,
            settings.litellm_api_key,
        )
        if value
    ]
    for configured in (
        settings.system_agent_api_key_file,
        settings.user_sim_api_key_file,
    ):
        if configured:
            values.append(_read_api_key_file(configured))
    return values


def _assert_public_safe(paths: Iterable[Path]) -> None:
    credentials = _credential_values()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for marker in FORBIDDEN_PUBLIC_MARKERS:
            if marker in lowered:
                raise RuntimeError(f"Forbidden private marker {marker} in {path}")
        for credential in credentials:
            if credential and credential in text:
                raise RuntimeError(f"Credential material detected in {path}")


def merge(
    sources: list[Path],
    output: Path,
    input_path: Path,
    expected_start: int,
    expected_end: int,
) -> None:
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing merge directory: {output}")
    validated = [_validate_source(source.resolve()) for source in sources]
    input_sha = _sha256_file(input_path)
    input_hashes = {item["summary"]["full_input"]["sha256"] for item in validated}
    config_hashes = {item["summary"]["configuration_sha256"] for item in validated}
    if input_hashes != {input_sha}:
        raise RuntimeError(f"Source full-input SHA256 mismatch: {input_hashes}")
    if len(config_hashes) != 1:
        raise RuntimeError(f"Source configuration SHA256 mismatch: {config_hashes}")
    config_sha = next(iter(config_hashes))
    configuration = validated[0]["summary"]["configuration"]
    recomputed_config_sha = hashlib.sha256(
        _canonical_json(configuration).encode("utf-8")
    ).hexdigest()
    if recomputed_config_sha != config_sha:
        raise RuntimeError("Source configuration summary does not match its SHA256")
    if any(item["summary"]["configuration"] != configuration for item in validated):
        raise RuntimeError("Source configuration summaries differ")

    records: list[dict] = []
    for item in validated:
        predictions_by_id = {
            prediction["instance_id"]: prediction
            for prediction in item["predictions"]
        }
        for manifest_record in item["manifest"]:
            records.append({
                **manifest_record,
                "prediction": predictions_by_id[manifest_record["instance_id"]],
                "source": item["path"],
            })
    records.sort(key=lambda record: record["global_index"])
    expected_indices = list(range(expected_start, expected_end + 1))
    indices = [record["global_index"] for record in records]
    identifiers = [record["instance_id"] for record in records]
    if indices != expected_indices:
        raise RuntimeError(
            f"Merged global indices are not continuous {expected_start}-{expected_end}"
        )
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Merged instance_id values are not unique")

    output.mkdir(parents=True)
    (output / "logs").mkdir()
    manifest = [
        {
            "global_index": record["global_index"],
            "instance_id": record["instance_id"],
            "selected_database": record["selected_database"],
        }
        for record in records
    ]
    predictions = [record["prediction"] for record in records]
    manifest_path = output / "shard_manifest.jsonl"
    prediction_path = output / "submission_predictions.jsonl"
    _atomic_text(
        manifest_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in manifest
        ),
    )
    _atomic_text(
        prediction_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in predictions
        ),
    )

    task_refs = [
        {"source": record["source"], "global_index": record["global_index"]}
        for record in records
    ]
    metrics, internal = _calculate_metrics(task_refs)
    _write_private(output / "result_private.json", task_refs, metrics)
    prediction_sha = _sha256_file(prediction_path)
    source_summaries = [
        {
            "path": str(item["path"]),
            "global_index_range": item["summary"]["global_index_range"],
            "completed_task_count": item["summary"]["completed_task_count"],
            "prediction_file_sha256": item["summary"]["prediction_file_sha256"],
        }
        for item in validated
    ]
    checkpoint = {
        "status": "complete",
        "full_input_sha256": input_sha,
        "configuration_sha256": config_sha,
        "global_index_range": [expected_start, expected_end],
        "expected_count": len(expected_indices),
        "completed_count": len(records),
        "completed_global_indices": indices,
        "completed_instance_ids": identifiers,
        "remaining_global_indices": [],
        "next_global_index": None,
        "in_progress": None,
        "all_completed_cleanup_verified": True,
        "prediction_file_sha256": prediction_sha,
        "source_shards": source_summaries,
    }
    summary = {
        "status": "complete",
        "full_input": {
            "path": str(input_path),
            "record_count": 600,
            "sha256": input_sha,
        },
        "configuration": configuration,
        "configuration_sha256": config_sha,
        "global_index_range": [expected_start, expected_end],
        "slice": f"records[{expected_start - 1}:{expected_end}]",
        "expected_task_count": len(expected_indices),
        "completed_task_count": len(records),
        "instance_ids": identifiers,
        "prediction_file": "submission_predictions.jsonl",
        "prediction_file_sha256": prediction_sha,
        "metrics": metrics,
        "internal_execution": internal,
        "all_completed_cleanup_verified": True,
        "source_shards": source_summaries,
        "merge_validation": {
            "identical_full_input_sha256": True,
            "identical_configuration_sha256": True,
            "sorted_by_global_index": True,
            "unique_instance_id": True,
        },
    }
    _atomic_json(output / "checkpoint.json", checkpoint)
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "logs" / "merge_validation.json",
        {
            "status": "passed",
            "source_shards": source_summaries,
            "record_count": len(records),
            "unique_instance_ids": len(set(identifiers)),
            "global_index_range": [expected_start, expected_end],
            "full_input_sha256": input_sha,
            "configuration_sha256": config_sha,
            "prediction_file_sha256": prediction_sha,
        },
    )
    _assert_public_safe((
        manifest_path,
        prediction_path,
        output / "checkpoint.json",
        output / "summary.json",
        output / "logs" / "merge_validation.json",
    ))
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "task_count": len(records),
        "global_index_range": [expected_start, expected_end],
        "full_input_sha256": input_sha,
        "configuration_sha256": config_sha,
        "prediction_file_sha256": prediction_sha,
        "metrics": metrics,
        "internal_execution": internal,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "bird-interact-full" / "bird_interact_data.jsonl"),
    )
    parser.add_argument("--expected-start", type=int, default=501)
    parser.add_argument("--expected-end", type=int, default=600)
    args = parser.parse_args()
    merge(
        [Path(source) for source in args.source],
        Path(args.output).resolve(),
        Path(args.input).resolve(),
        args.expected_start,
        args.expected_end,
    )


if __name__ == "__main__":
    main()
