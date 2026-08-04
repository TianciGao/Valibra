"""Prepare the frozen Full-dataset task slice records[500:600]."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_START = 501
GLOBAL_END = 600
SLICE_START = GLOBAL_START - 1
SLICE_END = GLOBAL_END
EXPECTED_SOURCE_COUNT = 600
EXPECTED_TASK_COUNT = 100


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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def prepare(input_path: Path, output_dir: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    raw_input = input_path.read_bytes()
    decoded_lines = [
        line
        for line in raw_input.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(decoded_lines) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_SOURCE_COUNT} source records, "
            f"got {len(decoded_lines)}"
        )

    records = [json.loads(line) for line in decoded_lines]
    selected_records = records[SLICE_START:SLICE_END]
    selected_lines = decoded_lines[SLICE_START:SLICE_END]
    if len(selected_records) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_TASK_COUNT} selected records, "
            f"got {len(selected_records)}"
        )

    manifest_records = []
    for global_index, record in enumerate(selected_records, GLOBAL_START):
        instance_id = record.get("instance_id")
        selected_database = record.get("selected_database")
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError(f"Task {global_index} has invalid instance_id")
        if not isinstance(selected_database, str) or not selected_database:
            raise RuntimeError(
                f"Task {global_index} has invalid selected_database"
            )
        manifest_records.append(
            {
                "global_index": global_index,
                "instance_id": instance_id,
                "selected_database": selected_database,
            }
        )

    indices = [record["global_index"] for record in manifest_records]
    identifiers = [record["instance_id"] for record in manifest_records]
    if indices != list(range(GLOBAL_START, GLOBAL_END + 1)):
        raise RuntimeError("Selected global_index values are not continuous")
    if len(set(identifiers)) != EXPECTED_TASK_COUNT:
        raise RuntimeError("Selected instance_id values are not unique")

    task_bytes = (
        "\n".join(selected_lines) + "\n"
    ).encode("utf-8")
    manifest_jsonl_bytes = (
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for record in manifest_records
        )
    ).encode("utf-8")

    task_path = output_dir / "tasks_0501_0600.jsonl"
    manifest_jsonl_path = output_dir / "task_manifest_0501_0600.jsonl"
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
            "global_index_range": [GLOBAL_START, GLOBAL_END],
            "python_slice": "records[500:600]",
            "record_count": len(selected_records),
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
        "tasks": manifest_records,
    }
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write(metadata_path, metadata_bytes)

    # Verify the bytes that reached disk, not only the in-memory payload.
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
    parser.add_argument(
        "--input",
        default=str(
            PROJECT_ROOT
            / "bird-interact-full"
            / "bird_interact_data.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "prepared_tasks" / "full_0501_0600"),
    )
    args = parser.parse_args()
    metadata = prepare(Path(args.input), Path(args.output_dir))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
