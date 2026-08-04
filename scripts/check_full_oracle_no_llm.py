#!/usr/bin/env python3
"""Run a DB-only Full oracle/protocol check without invoking either LLM service.

This checker deliberately talks only to the DB Environment.  It submits the
complete ground-truth SQL sequence for each phase, because a number of Full
tasks contain more than one SQL statement.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _combine_sql(sqls: Any) -> str:
    """Encode a SQL sequence for the service's string-only submit contract."""
    if isinstance(sqls, str):
        return sqls
    if not isinstance(sqls, list):
        return ""
    statements = []
    for sql in sqls:
        if isinstance(sql, str) and sql.strip():
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


async def _post(
    client: httpx.AsyncClient,
    db_env: str,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(f"{db_env}{endpoint}", json=payload)
    response.raise_for_status()
    return response.json()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--db-env", default="http://127.0.0.1:6102")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    records = _load_jsonl(args.data)
    instance_ids = [record.get("instance_id") for record in records]
    if len(records) != 600:
        raise RuntimeError(f"Expected 600 Full tasks, found {len(records)}")
    if len(set(instance_ids)) != 600 or None in instance_ids:
        raise RuntimeError("Full input instance_id values are missing or duplicated")

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
                record = copy.deepcopy(source)
                record["_interact_mode"] = "a-interact"
                item: dict[str, Any] = {
                    "global_index": global_index,
                    "instance_id": task_id,
                    "selected_database": source.get("selected_database"),
                    "p1_gt_statement_count": len(source.get("sol_sql", [])),
                    "p2_gt_statement_count": len(
                        (source.get("follow_up") or {}).get("sol_sql", [])
                    ),
                    "p1_passed": False,
                    "p2_passed": False,
                    "cleanup_completed": False,
                }
                try:
                    await _post(
                        client,
                        args.db_env,
                        "/init_task",
                        {"task_id": task_id, "task_data": record},
                    )
                    phase1 = await _post(
                        client,
                        args.db_env,
                        "/submit",
                        {"task_id": task_id, "sql": _combine_sql(record.get("sol_sql"))},
                    )
                    item["p1_passed"] = bool(phase1.get("passed"))
                    item["p1_phase_completed"] = phase1.get("phase_completed")
                    item["p1_has_follow_up"] = bool(phase1.get("has_follow_up"))

                    follow_up = record.get("follow_up") or {}
                    if item["p1_passed"] and follow_up.get("sol_sql"):
                        phase2 = await _post(
                            client,
                            args.db_env,
                            "/submit",
                            {
                                "task_id": task_id,
                                "sql": _combine_sql(follow_up.get("sol_sql")),
                            },
                        )
                        item["p2_passed"] = bool(phase2.get("passed"))
                        item["p2_phase_completed"] = phase2.get("phase_completed")
                    elif not follow_up.get("sol_sql"):
                        item["p2_passed"] = True
                        item["p2_not_applicable"] = True
                except Exception as exc:  # Preserve the check failure, then clean up.
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

                async with write_lock:
                    results[task_id] = item
                    ordered = [results[i] for i in instance_ids if i in results]
                    p1_passed = sum(bool(row["p1_passed"]) for row in ordered)
                    p2_passed = sum(bool(row["p2_passed"]) for row in ordered)
                    payload = {
                        "check": "full_db_only_oracle_no_llm",
                        "model_calls": 0,
                        "data_path": str(args.data.resolve()),
                        "data_sha256": _sha256(args.data),
                        "expected_tasks": 600,
                        "completed_tasks": len(ordered),
                        "p1_passed": p1_passed,
                        "p2_passed": p2_passed,
                        "all_cleanup_completed": all(
                            bool(row["cleanup_completed"]) for row in ordered
                        ),
                        "results": ordered,
                    }
                    _atomic_write(args.output, payload)
                    print(
                        f"[{len(ordered):03d}/600] p1={p1_passed} p2={p2_passed} "
                        f"last={task_id}",
                        flush=True,
                    )

        await asyncio.gather(
            *(run_one(index, record) for index, record in enumerate(records, start=1))
        )


if __name__ == "__main__":
    asyncio.run(main())
