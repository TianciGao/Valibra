"""Migrate a generic runner checkpoint into per-task resumable envelopes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    task_manifest = Path(args.task_manifest).resolve()
    run_dir = Path(args.run_dir).resolve()
    manifest_data = json.loads(task_manifest.read_text(encoding="utf-8"))
    manifest = manifest_data["tasks"]
    by_id = {record["instance_id"]: record for record in manifest}
    source_data = json.loads(source.read_text(encoding="utf-8"))
    results = source_data.get("results") or []
    if not results:
        raise RuntimeError("Seed checkpoint has no results")

    migrated = []
    seen: set[str] = set()
    for result in results:
        instance_id = result.get("instance_id") or result.get("task_id")
        if instance_id in seen:
            raise RuntimeError(f"Duplicate seed instance_id: {instance_id}")
        seen.add(instance_id)
        record = by_id.get(instance_id)
        if record is None:
            raise RuntimeError(f"Seed task is outside frozen range: {instance_id}")
        global_index = int(record["global_index"])
        path = run_dir / "logs" / "task_results" / f"{global_index:04d}.json"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing envelope: {path}")

        completed_at = None
        for call in reversed(result.get("prompt_flow") or []):
            completed_at = call.get("completed_at")
            if completed_at:
                break
        envelope = {
            "global_index": global_index,
            "instance_id": instance_id,
            "selected_database": record["selected_database"],
            "model_completed_at": completed_at or _utc_now(),
            "cleanup": {
                "verified": False,
                "migration_note": (
                    "DB cleanup returned 200 in the original runner; all service "
                    "processes were stopped. Official cleanup will be reverified "
                    "before this result is checkpointed."
                ),
            },
            "result": result,
        }
        _atomic_json(path, envelope)
        migrated.append(
            {
                "global_index": global_index,
                "instance_id": instance_id,
                "selected_database": record["selected_database"],
                "envelope_path": str(path),
                "envelope_sha256": _sha256_file(path),
            }
        )

    migrated.sort(key=lambda item: item["global_index"])
    indices = [item["global_index"] for item in migrated]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise RuntimeError(f"Seed indices are not continuous: {indices}")
    report = {
        "schema_version": 1,
        "migrated_at": _utc_now(),
        "source": {
            "path": str(source),
            "sha256": _sha256_file(source),
            "result_count": len(results),
        },
        "task_manifest": {
            "path": str(task_manifest),
            "sha256": _sha256_file(task_manifest),
        },
        "migrated_count": len(migrated),
        "global_indices": indices,
        "tasks": migrated,
        "model_outcomes_rerun": False,
        "cleanup_reverification_pending": True,
    }
    _atomic_json(run_dir / "logs" / "seed_migration.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
