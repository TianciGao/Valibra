"""Prepare and verify a contiguous range from the 600-record Full input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SOURCE_COUNT = 600


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(
    input_path: Path,
    output_dir: Path,
    global_start: int,
    global_end: int,
) -> dict[str, Any]:
    if global_start < 1 or global_end > EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Range must be within 1-{EXPECTED_SOURCE_COUNT}"
        )
    if global_start > global_end:
        raise RuntimeError("Range start must not exceed range end")

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    raw_input = input_path.read_bytes()
    decoded_lines = [
        line for line in raw_input.decode("utf-8").splitlines() if line.strip()
    ]
    if len(decoded_lines) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_COUNT} source records, "
            f"got {len(decoded_lines)}"
        )

    records = [json.loads(line) for line in decoded_lines]
    slice_start = global_start - 1
    slice_end = global_end
    selected_records = records[slice_start:slice_end]
    selected_lines = decoded_lines[slice_start:slice_end]
    expected_count = global_end - global_start + 1
    if len(selected_records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} selected records, "
            f"got {len(selected_records)}"
        )

    tasks = []
    for global_index, record in enumerate(selected_records, global_start):
        instance_id = record.get("instance_id")
        selected_database = record.get("selected_database")
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError(f"Task {global_index} has invalid instance_id")
        if not isinstance(selected_database, str) or not selected_database:
            raise RuntimeError(
                f"Task {global_index} has invalid selected_database"
            )
        tasks.append(
            {
                "global_index": global_index,
                "instance_id": instance_id,
                "selected_database": selected_database,
            }
        )

    indices = [task["global_index"] for task in tasks]
    identifiers = [task["instance_id"] for task in tasks]
    if indices != list(range(global_start, global_end + 1)):
        raise RuntimeError("Selected global_index values are not continuous")
    if len(set(identifiers)) != expected_count:
        raise RuntimeError("Selected instance_id values are not unique")

    range_label = f"{global_start:04d}_{global_end:04d}"
    task_bytes = ("\n".join(selected_lines) + "\n").encode("utf-8")
    manifest_jsonl_bytes = "".join(
        json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n"
        for task in tasks
    ).encode("utf-8")
    task_path = output_dir / f"tasks_{range_label}.jsonl"
    manifest_jsonl_path = output_dir / f"task_manifest_{range_label}.jsonl"
    metadata_path = output_dir / "manifest.json"
    _atomic_write(task_path, task_bytes)
    _atomic_write(manifest_jsonl_path, manifest_jsonl_bytes)

    metadata = {
        "schema_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path),
            "record_count": len(records),
            "sha256": _sha256_bytes(raw_input),
        },
        "selection": {
            "global_index_range": [global_start, global_end],
            "python_slice": f"records[{slice_start}:{slice_end}]",
            "record_count": expected_count,
            "unique_instance_id_count": len(set(identifiers)),
        },
        "task_jsonl": {
            "path": str(task_path),
            "sha256": _sha256_bytes(task_bytes),
        },
        "task_manifest_jsonl": {
            "path": str(manifest_jsonl_path),
            "sha256": _sha256_bytes(manifest_jsonl_bytes),
        },
        "tasks": tasks,
    }
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    if _sha256_file(task_path) != metadata["task_jsonl"]["sha256"]:
        raise RuntimeError("Task JSONL SHA256 verification failed")
    if (
        _sha256_file(manifest_jsonl_path)
        != metadata["task_manifest_jsonl"]["sha256"]
    ):
        raise RuntimeError("Task manifest SHA256 verification failed")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument(
        "--input",
        default=str(
            PROJECT_ROOT / "bird-interact-full" / "bird_interact_data.jsonl"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metadata = prepare(
        Path(args.input),
        Path(args.output_dir),
        args.start,
        args.end,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
