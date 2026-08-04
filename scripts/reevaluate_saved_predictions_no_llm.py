#!/usr/bin/env python3
"""Re-evaluate frozen Full predictions using only the DB Environment.

This program never contacts the System Agent or User Simulator.  It validates
the frozen prediction artifact against the original Full input, submits the
saved SQL to the DB Environment, and writes an atomic checkpoint after every
completed task.  Ground truth and predicted SQL are deliberately omitted from
the re-evaluation result; their source artifacts are identified by SHA256.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx


PREDICTION_KEYS = {
    "instance_id",
    "subtask_1_predicted_sql",
    "subtask_2_predicted_sql",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Expected JSON objects in {path}")
    return rows


def _validated_sql_list(value: Any, field: str, instance_id: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{instance_id}: {field} must be a list")
    if not all(isinstance(sql, str) for sql in value):
        raise RuntimeError(f"{instance_id}: {field} contains a non-string value")
    return value


def _combine_sql(sqls: list[str]) -> str:
    statements = []
    for sql in sqls:
        if sql.strip():
            statements.append(sql.rstrip().rstrip(";") + ";")
    return "\n".join(statements)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_historical_outcomes(summary_path: Path) -> dict[str, dict[str, bool]]:
    summary = _load_json(summary_path)
    outcomes: dict[str, dict[str, bool]] = {}
    for shard in summary.get("source_shards", []):
        shard_path = Path(shard["path"])
        index_path = shard_path / "result_private_index.json"
        index = _load_json(index_path)
        for item in index.get("completed", []):
            instance_id = item.get("instance_id")
            result = item.get("result_summary") or {}
            if not instance_id or instance_id in outcomes:
                raise RuntimeError(
                    f"Missing or duplicate historical instance_id: {instance_id!r}"
                )
            outcomes[instance_id] = {
                "p1_passed": bool(result.get("phase1_passed")),
                "p2_passed": bool(result.get("phase2_passed")),
            }
    return outcomes


def _transition(old: bool, new: bool) -> str:
    if old and new:
        return "unchanged_pass"
    if old and not new:
        return "lost"
    if not old and new:
        return "gained"
    return "unchanged_fail"


def _aggregate(
    ordered: list[dict[str, Any]],
    *,
    data_path: Path,
    prediction_path: Path,
    historical_summary_path: Path,
    evaluator_files: list[Path],
) -> dict[str, Any]:
    p1 = sum(bool(row["new_p1_passed"]) for row in ordered)
    p2 = sum(bool(row["new_p2_passed"]) for row in ordered)
    old_p1 = sum(bool(row["old_p1_passed"]) for row in ordered)
    old_p2 = sum(bool(row["old_p2_passed"]) for row in ordered)
    transitions = {
        phase: {
            state: sum(row[f"{phase}_transition"] == state for row in ordered)
            for state in ("gained", "lost", "unchanged_pass", "unchanged_fail")
        }
        for phase in ("p1", "p2")
    }
    return {
        "check": "saved_glm47_predictions_db_only_reevaluation_no_llm",
        "model_calls": 0,
        "user_simulator_calls": 0,
        "data_path": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "historical_summary_path": str(historical_summary_path.resolve()),
        "historical_summary_sha256": _sha256(historical_summary_path),
        "evaluator_file_sha256": {
            str(path): _sha256(path) for path in evaluator_files
        },
        "expected_tasks": 600,
        "completed_tasks": len(ordered),
        "historical": {
            "p1_passed": old_p1,
            "p2_passed": old_p2,
            "reward": old_p2 + 0.7 * (old_p1 - old_p2),
        },
        "reevaluated": {
            "p1_passed": p1,
            "p2_passed": p2,
            "reward": p2 + 0.7 * (p1 - p2),
        },
        "delta": {
            "p1_passed": p1 - old_p1,
            "p2_passed": p2 - old_p2,
            "reward": (p2 + 0.7 * (p1 - p2))
            - (old_p2 + 0.7 * (old_p1 - old_p2)),
        },
        "transitions": transitions,
        "all_cleanup_completed": bool(ordered)
        and all(bool(row["cleanup_completed"]) for row in ordered),
        "protocol_error_count": sum("protocol_error" in row for row in ordered),
        "cleanup_error_count": sum("cleanup_error" in row for row in ordered),
        "results": ordered,
    }


async def _post(
    client: httpx.AsyncClient,
    db_env: str,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(f"{db_env}{endpoint}", json=payload)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected response from {endpoint}")
    return value


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--historical-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--db-env", default="http://127.0.0.1:6102")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    if args.concurrency < 1:
        raise RuntimeError("concurrency must be at least 1")

    records = _load_jsonl(args.data)
    predictions = _load_jsonl(args.predictions)
    historical_summary = _load_json(args.historical_summary)
    historical = _load_historical_outcomes(args.historical_summary)
    if len(records) != 600 or len(predictions) != 600:
        raise RuntimeError(
            f"Expected 600 Full records and predictions, found "
            f"{len(records)} and {len(predictions)}"
        )

    instance_ids = [record.get("instance_id") for record in records]
    prediction_ids = [prediction.get("instance_id") for prediction in predictions]
    if len(set(instance_ids)) != 600 or None in instance_ids:
        raise RuntimeError("Full input instance_id values are missing or duplicated")
    if prediction_ids != instance_ids:
        raise RuntimeError("Predictions are not in exact Full source order")
    if set(historical) != set(instance_ids):
        raise RuntimeError("Historical compact results do not cover the same 600 tasks")
    if historical_summary.get("full_input", {}).get("sha256") != _sha256(args.data):
        raise RuntimeError("Historical run and Full input SHA256 do not match")
    if historical_summary.get("prediction_file_sha256") != _sha256(args.predictions):
        raise RuntimeError("Historical summary and prediction SHA256 do not match")

    normalized_predictions: dict[str, dict[str, list[str]]] = {}
    for prediction in predictions:
        instance_id = prediction["instance_id"]
        if set(prediction) != PREDICTION_KEYS:
            raise RuntimeError(
                f"{instance_id}: unexpected prediction keys {sorted(set(prediction))}"
            )
        normalized_predictions[instance_id] = {
            "p1": _validated_sql_list(
                prediction["subtask_1_predicted_sql"],
                "subtask_1_predicted_sql",
                instance_id,
            ),
            "p2": _validated_sql_list(
                prediction["subtask_2_predicted_sql"],
                "subtask_2_predicted_sql",
                instance_id,
            ),
        }

    project_root = Path(__file__).resolve().parents[1]
    evaluator_files = [
        project_root / "shared" / "db_utils.py",
        project_root / "db_environment" / "server.py",
        project_root / "db_environment" / "fixture_compat.py",
    ]
    results: dict[str, dict[str, Any]] = {}
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(600.0)

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        health = await client.get(f"{args.db_env}/health")
        health.raise_for_status()

        async def run_one(global_index: int, source: dict[str, Any]) -> None:
            async with semaphore:
                task_id = source["instance_id"]
                saved = normalized_predictions[task_id]
                old = historical[task_id]
                record = copy.deepcopy(source)
                record["_interact_mode"] = "a-interact"
                item: dict[str, Any] = {
                    "global_index": global_index,
                    "instance_id": task_id,
                    "selected_database": source.get("selected_database"),
                    "p1_prediction_count": len(saved["p1"]),
                    "p2_prediction_count": len(saved["p2"]),
                    "old_p1_passed": old["p1_passed"],
                    "old_p2_passed": old["p2_passed"],
                    "new_p1_passed": False,
                    "new_p2_passed": False,
                    "cleanup_completed": False,
                }
                try:
                    await _post(
                        client,
                        args.db_env,
                        "/init_task",
                        {"task_id": task_id, "task_data": record},
                    )
                    if saved["p1"] and _combine_sql(saved["p1"]):
                        phase1 = await _post(
                            client,
                            args.db_env,
                            "/submit",
                            {"task_id": task_id, "sql": _combine_sql(saved["p1"])},
                        )
                        item["new_p1_passed"] = bool(phase1.get("passed"))
                        item["p1_phase_completed"] = phase1.get("phase_completed")
                        item["p1_has_follow_up"] = bool(phase1.get("has_follow_up"))
                    else:
                        item["p1_no_prediction"] = True

                    follow_up = record.get("follow_up") or {}
                    if item["new_p1_passed"] and follow_up.get("sol_sql"):
                        if saved["p2"] and _combine_sql(saved["p2"]):
                            phase2 = await _post(
                                client,
                                args.db_env,
                                "/submit",
                                {"task_id": task_id, "sql": _combine_sql(saved["p2"])},
                            )
                            item["new_p2_passed"] = bool(phase2.get("passed"))
                            item["p2_phase_completed"] = phase2.get("phase_completed")
                        else:
                            item["p2_no_prediction"] = True
                    elif item["new_p1_passed"] and not follow_up.get("sol_sql"):
                        item["new_p2_passed"] = True
                        item["p2_not_applicable"] = True
                except Exception as exc:
                    item["protocol_error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    try:
                        await _post(
                            client,
                            args.db_env,
                            "/cleanup_task",
                            {"task_id": task_id},
                        )
                        item["cleanup_completed"] = True
                    except Exception as exc:
                        item["cleanup_error"] = f"{type(exc).__name__}: {exc}"

                item["p1_transition"] = _transition(
                    item["old_p1_passed"], item["new_p1_passed"]
                )
                item["p2_transition"] = _transition(
                    item["old_p2_passed"], item["new_p2_passed"]
                )
                async with write_lock:
                    results[task_id] = item
                    ordered = [results[i] for i in instance_ids if i in results]
                    payload = _aggregate(
                        ordered,
                        data_path=args.data,
                        prediction_path=args.predictions,
                        historical_summary_path=args.historical_summary,
                        evaluator_files=evaluator_files,
                    )
                    _atomic_write(args.output, payload)
                    print(
                        f"[{len(ordered):03d}/600] "
                        f"old_p1={payload['historical']['p1_passed']} "
                        f"new_p1={payload['reevaluated']['p1_passed']} "
                        f"old_p2={payload['historical']['p2_passed']} "
                        f"new_p2={payload['reevaluated']['p2_passed']} "
                        f"last={task_id}",
                        flush=True,
                    )

        await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(records, start=1))
        )


if __name__ == "__main__":
    asyncio.run(main())
