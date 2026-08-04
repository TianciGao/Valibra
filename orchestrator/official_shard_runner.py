"""Resumable official Full 50/100-record range runner.

This runner deliberately separates:
- public predictions: SQL only;
- private audit data: prompts, reasoning, tool traces, and evaluator outcomes.

Each completed model run is first committed to an individual task envelope.
Cleanup is then verified before the task is added to the public checkpoint.
On restart, model-complete envelopes are never rerun; pending cleanup is retried.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import traceback
from typing import Any, Dict, Iterable

import httpx
import psycopg2

from orchestrator.ainteract import run_single_task
from shared.config import (
    PROJECT_ROOT,
    active_model_preset_report,
    normalized_system_agent_config,
    settings,
)
from shared.db_utils import drop_task_db, task_database_names
from system_agent.callbacks import TOOL_COSTS


logger = logging.getLogger(__name__)

SHARD_START = 551
SHARD_END = 600
SLICE_START = SHARD_START - 1
SLICE_END = SHARD_END
EXPECTED_INPUT_COUNT = 600
EXPECTED_SHARD_COUNT = 50
MODE = "a-interact"
LEADERBOARD_MODE = "stress"
CONCURRENCY = 1
USER_SIM_PROFILE = os.environ.get(
    "USER_SIM_PROFILE", "claude_haiku_4_5_gptsapi"
)
USER_SIM_PROFILES = {
    "claude_haiku_4_5_gptsapi": {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "prompt_version": "v2",
        "provider_hidden_thinking_disabled": True,
        "api_base": None,
        "use_bearer_for_custom_base": True,
        "protocol_policy": "official",
        "protocol_max_attempts": 1,
    },
    "claude_haiku_4_5_official": {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "prompt_version": "v2",
        "provider_hidden_thinking_disabled": True,
        "api_base": "https://api.anthropic.com",
        "use_bearer_for_custom_base": False,
        "protocol_policy": "official",
        "protocol_max_attempts": 1,
    },
    "gpt4o_gptsapi": {
        # LiteLLM needs the provider prefix; the outgoing OpenAI-compatible
        # request uses model=gpt-4o.
        "model": "openai/gpt-4o",
        "prompt_version": "v2",
        "provider_hidden_thinking_disabled": False,
        "api_base": "https://api.gptsapi.net",
        "use_bearer_for_custom_base": False,
        "protocol_policy": "strict_retry",
        "protocol_max_attempts": 3,
    },
}
PUBLIC_PREDICTION_KEYS = (
    "instance_id",
    "subtask_1_predicted_sql",
    "subtask_2_predicted_sql",
)
FORBIDDEN_PUBLIC_MARKERS = (
    '"sol_sql"',
    '"test_cases"',
    '"prompt_flow"',
    '"reasoning_content"',
    '"ground_truth"',
    '"api_key"',
)
IMMUTABLE_RUN_FILES = (
    "shared/config.py",
    "shared/llm.py",
    "shared/db_utils.py",
    "system_agent/agent.py",
    "system_agent/tools.py",
    "system_agent/callbacks.py",
    "system_agent/adk_runtime.py",
    "system_agent/server.py",
    "user_simulator/prompts.py",
    "user_simulator/sql_parser.py",
    "user_simulator/server.py",
    "db_environment/server.py",
    "orchestrator/ainteract.py",
    "orchestrator/official_shard_runner.py",
)


def _configure_shard(start: int, end: int) -> None:
    global SHARD_START, SHARD_END, SLICE_START, SLICE_END
    global EXPECTED_SHARD_COUNT

    if start < 1 or end > EXPECTED_INPUT_COUNT or start > end:
        raise ValueError(
            f"Shard range must be within 1-{EXPECTED_INPUT_COUNT}: {start}-{end}"
        )
    count = end - start + 1
    if count not in (50, 100):
        raise ValueError(
            "Official range runner requires exactly 50 or 100 records, "
            f"got {count}"
        )
    SHARD_START = start
    SHARD_END = end
    SLICE_START = start - 1
    SLICE_END = end
    EXPECTED_SHARD_COUNT = count


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _atomic_json(path: Path, value: Any, *, indent: int = 2) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=indent, default=str) + "\n",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_input(input_path: Path) -> tuple[list[dict], list[dict]]:
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_INPUT_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_INPUT_COUNT} input records, got {len(records)}"
        )
    shard = records[SLICE_START:SLICE_END]
    if len(shard) != EXPECTED_SHARD_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_SHARD_COUNT} shard records, got {len(shard)}"
        )
    identifiers = [record["instance_id"] for record in shard]
    if len(set(identifiers)) != EXPECTED_SHARD_COUNT:
        raise RuntimeError("Shard instance_id values are not unique")
    return records, shard


def _shard_manifest(shard: list[dict]) -> list[dict]:
    return [
        {
            "global_index": global_index,
            "instance_id": record["instance_id"],
            "selected_database": record["selected_database"],
        }
        for global_index, record in enumerate(shard, SHARD_START)
    ]


def _manifest_text(manifest: list[dict]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in manifest
    )


def _immutable_file_hashes() -> Dict[str, str]:
    return {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in IMMUTABLE_RUN_FILES
    }


def _configuration_summary() -> Dict[str, Any]:
    preset = active_model_preset_report()
    return {
        "dataset": "full",
        "framework": "BIRD-Interact-ADK",
        "interaction_mode": MODE,
        "leaderboard_mode": LEADERBOARD_MODE,
        "concurrency": CONCURRENCY,
        "system_agent": normalized_system_agent_config(),
        "model_preset": (
            {
                "name": preset["name"],
                "normalized_sha256": preset["normalized_sha256"],
                "file_sha256": preset["file_sha256"],
            }
            if preset is not None
            else None
        ),
        "user_simulator": {
            "profile": USER_SIM_PROFILE,
            "model": settings.user_sim_model,
            "prompt_version": settings.prompt_version,
            "provider_hidden_thinking_disabled": settings.user_sim_disable_thinking,
            "api_base": settings.user_sim_api_base.rstrip("/"),
            "use_bearer_for_custom_base": (
                settings.user_sim_use_bearer_for_custom_base
            ),
            "provider_failure_policy": "fail_fast_uncheckpointed",
            "protocol_policy": settings.user_sim_protocol_policy,
            "protocol_max_attempts": settings.user_sim_protocol_max_attempts,
        },
        "patience": settings.patience,
        "budget_formula": "6 + 2 * ambiguity_count + 2 * patience",
        "tool_costs": TOOL_COSTS,
        "immutable_run_file_sha256": _immutable_file_hashes(),
    }


def _validate_configuration(config: Dict[str, Any]) -> None:
    preset = active_model_preset_report()
    if preset is None:
        raise RuntimeError("MODEL_PRESET is required for an official frozen run")
    if USER_SIM_PROFILE not in USER_SIM_PROFILES:
        raise RuntimeError(f"Unknown USER_SIM_PROFILE: {USER_SIM_PROFILE}")
    user_sim_expected = dict(USER_SIM_PROFILES[USER_SIM_PROFILE])
    if user_sim_expected["api_base"] is None:
        # The historical Claude profile supported a user-owned compatible
        # gateway.  New named profiles freeze their endpoint explicitly.
        user_sim_expected["api_base"] = settings.user_sim_api_base.rstrip("/")
    user_sim_expected.update(
        {
            "profile": USER_SIM_PROFILE,
            "provider_failure_policy": "fail_fast_uncheckpointed",
            "protocol_policy": settings.user_sim_protocol_policy,
            "protocol_max_attempts": settings.user_sim_protocol_max_attempts,
        }
    )
    expected = {
        "dataset": "full",
        "framework": "BIRD-Interact-ADK",
        "interaction_mode": "a-interact",
        "leaderboard_mode": "stress",
        "concurrency": 1,
        "system_agent": preset["normalized_config"],
        "model_preset": {
            "name": preset["name"],
            "normalized_sha256": preset["normalized_sha256"],
            "file_sha256": preset["file_sha256"],
        },
        "user_simulator": user_sim_expected,
        "patience": 3,
    }
    for key in (
        "dataset",
        "framework",
        "interaction_mode",
        "leaderboard_mode",
        "concurrency",
        "system_agent",
        "model_preset",
        "user_simulator",
        "patience",
    ):
        if config.get(key) != expected[key]:
            raise RuntimeError(
                f"Fixed configuration mismatch for {key}: "
                f"expected {expected[key]!r}, got {config.get(key)!r}"
            )


def _task_result_path(run_dir: Path, global_index: int) -> Path:
    return run_dir / "logs" / "task_results" / f"{global_index:04d}.json"


def _compact_envelope(path: Path, envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only indexing/cleanup metadata in memory; the full trace stays on disk."""
    result = envelope.get("result") or {}
    return {
        "_path": str(path),
        "global_index": envelope["global_index"],
        "instance_id": envelope["instance_id"],
        "selected_database": envelope["selected_database"],
        "model_completed_at": envelope["model_completed_at"],
        "cleanup": envelope.get("cleanup", {"verified": False}),
        "result_summary": {
            "total_reward": result.get("total_reward", 0),
            "phase1_passed": result.get("phase1_passed", False),
            "phase2_passed": result.get("phase2_passed", False),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
            "token_usage": (result.get("token_usage") or {}).get(
                "combined", {}
            ),
        },
        "prediction": {
            "subtask_1_predicted_sql": result.get(
                "subtask_1_predicted_sql", []
            ),
            "subtask_2_predicted_sql": result.get(
                "subtask_2_predicted_sql", []
            ),
        },
    }


def _load_full_envelope(envelope: Dict[str, Any]) -> Dict[str, Any]:
    if "result" in envelope:
        return envelope
    path = envelope.get("_path")
    if not path:
        raise RuntimeError("Compact task envelope has no backing path")
    return _load_json(Path(path))


def _read_envelopes(
    run_dir: Path, manifest: list[dict]
) -> Dict[int, Dict[str, Any]]:
    envelopes: Dict[int, Dict[str, Any]] = {}
    expected = {record["global_index"]: record for record in manifest}
    task_dir = run_dir / "logs" / "task_results"
    if not task_dir.exists():
        return envelopes
    for path in sorted(task_dir.glob("*.json")):
        envelope = _load_json(path)
        global_index = int(envelope["global_index"])
        if global_index not in expected:
            raise RuntimeError(f"Unexpected task result outside shard: {path}")
        reference = expected[global_index]
        if (
            envelope.get("instance_id") != reference["instance_id"]
            or envelope.get("selected_database") != reference["selected_database"]
        ):
            raise RuntimeError(f"Task envelope identity mismatch: {path}")
        if global_index in envelopes:
            raise RuntimeError(f"Duplicate task envelope for index {global_index}")
        envelopes[global_index] = _compact_envelope(path, envelope)
    return envelopes


async def _post_cleanup(instance_id: str) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{settings.db_env_port}/cleanup_task"
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.post(url, json={"task_id": instance_id})
        response.raise_for_status()
        return response.json()


async def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def _residual_task_databases(selected_database: str, instance_id: str) -> list[str]:
    candidates = list(task_database_names(selected_database, instance_id).values())
    connection = psycopg2.connect(
        dbname="postgres",
        user=settings.pg_user,
        password=settings.pg_password,
        host=settings.pg_host,
        port=settings.pg_port,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT datname FROM pg_database WHERE datname = ANY(%s)",
                (candidates,),
            )
            names = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()
    return sorted(names)


async def _ensure_cleanup(
    selected_database: str, instance_id: str
) -> Dict[str, Any]:
    system_agent_response = await _post_json(
        f"http://127.0.0.1:{settings.system_agent_port}/cleanup_session",
        {"task_id": instance_id, "mode": MODE},
    )
    user_simulator_response = await _post_json(
        f"http://127.0.0.1:{settings.user_sim_port}/cleanup_task",
        {"task_id": instance_id},
    )
    endpoint_response = await _post_cleanup(instance_id)
    residual_before = _residual_task_databases(selected_database, instance_id)
    removed_directly: list[str] = []
    if residual_before:
        for database_name in residual_before:
            await asyncio.to_thread(drop_task_db, database_name)
            removed_directly.append(database_name)
    residual_after = _residual_task_databases(selected_database, instance_id)
    if residual_after:
        raise RuntimeError(
            f"Cleanup verification failed for {instance_id}: {residual_after}"
        )
    return {
        "verified": True,
        "verified_at": _utc_now(),
        "system_agent": system_agent_response,
        "user_simulator": user_simulator_response,
        "db_environment": endpoint_response,
        "residual_before_direct_cleanup": residual_before,
        "removed_directly": removed_directly,
        "residual_after": residual_after,
    }


def _result_metrics(envelopes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    reward = 0.0
    phase1 = 0
    phase2 = 0
    elapsed = 0.0
    token_fields = {
        field: 0
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "tool_prompt_tokens",
        )
    }
    for envelope in envelopes:
        result = envelope.get("result_summary")
        if result is None:
            result = _load_full_envelope(envelope)["result"]
        total += 1
        reward += float(result.get("total_reward", 0) or 0)
        phase1 += int(bool(result.get("phase1_passed")))
        phase2 += int(bool(result.get("phase2_passed")))
        elapsed += float(result.get("elapsed_seconds", 0) or 0)
        usage = result.get("token_usage") or {}
        combined = usage.get("combined") or usage
        for field in token_fields:
            token_fields[field] += int(combined.get(field, 0) or 0)
    return {
        "completed_tasks": total,
        "total_reward": reward,
        "average_reward": reward / total if total else 0.0,
        "phase1_count": phase1,
        "phase1_rate": phase1 / total if total else 0.0,
        "phase2_count": phase2,
        "phase2_rate": phase2 / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "token_usage": token_fields,
    }


def _ordered_complete_envelopes(
    envelopes: Dict[int, Dict[str, Any]]
) -> list[Dict[str, Any]]:
    return [
        envelopes[index]
        for index in sorted(envelopes)
        if envelopes[index].get("cleanup", {}).get("verified") is True
    ]


def _write_private_result(
    path: Path, envelopes: list[Dict[str, Any]]
) -> None:
    metrics = _result_metrics(envelopes)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write('{"mode":"a-interact","leaderboard_mode":"stress","metrics":')
        json.dump(metrics, handle, ensure_ascii=False, default=str)
        handle.write(',"results":[')
        for position, envelope in enumerate(envelopes):
            if position:
                handle.write(",")
            full_envelope = _load_full_envelope(envelope)
            private_result = {
                "global_index": full_envelope["global_index"],
                "instance_id": full_envelope["instance_id"],
                "selected_database": full_envelope["selected_database"],
                "model_completed_at": full_envelope["model_completed_at"],
                "cleanup": full_envelope["cleanup"],
                **full_envelope["result"],
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


def _write_predictions(
    path: Path, envelopes: list[Dict[str, Any]]
) -> None:
    records = []
    for envelope in envelopes:
        prediction = envelope.get("prediction")
        if prediction is None:
            result = _load_full_envelope(envelope)["result"]
            prediction = {
                "subtask_1_predicted_sql": result.get(
                    "subtask_1_predicted_sql", []
                ),
                "subtask_2_predicted_sql": result.get(
                    "subtask_2_predicted_sql", []
                ),
            }
        records.append({
            "instance_id": envelope["instance_id"],
            **prediction,
        })
    _atomic_text(
        path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )


def _assert_public_safe(paths: Iterable[Path]) -> None:
    from shared.llm import _read_api_key_file

    key_values = [
        value
        for value in (
            settings.system_agent_api_key,
            settings.user_sim_api_key,
            settings.litellm_api_key,
        )
        if value
    ]
    for configured_path in (
        settings.system_agent_api_key_file,
        settings.user_sim_api_key_file,
    ):
        if not configured_path:
            continue
        key_path = Path(configured_path).expanduser()
        if not key_path.is_absolute():
            key_path = PROJECT_ROOT / key_path
        if key_path.exists():
            key_values.append(_read_api_key_file(configured_path))

    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for marker in FORBIDDEN_PUBLIC_MARKERS:
            if marker in lowered:
                raise RuntimeError(f"Forbidden private marker {marker} in {path}")
        for value in key_values:
            if value and value in text:
                raise RuntimeError(f"Credential material detected in {path}")


def _validate_predictions(
    path: Path,
    complete_envelopes: list[Dict[str, Any]],
    *,
    require_complete: bool,
) -> None:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != len(complete_envelopes):
        raise RuntimeError("Prediction count differs from completed result count")
    if require_complete and len(lines) != EXPECTED_SHARD_COUNT:
        raise RuntimeError(
            f"Final prediction file must have exactly {EXPECTED_SHARD_COUNT} lines"
        )
    records = [json.loads(line) for line in lines]
    for record in records:
        if tuple(record.keys()) != PUBLIC_PREDICTION_KEYS:
            raise RuntimeError(
                f"Prediction record has unexpected fields: {tuple(record.keys())}"
            )
    identifiers = [record["instance_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Prediction instance_id values are not unique")
    expected_ids = [envelope["instance_id"] for envelope in complete_envelopes]
    if identifiers != expected_ids:
        raise RuntimeError("Prediction order differs from global_index order")


def _write_state(
    *,
    run_dir: Path,
    input_path: Path,
    input_sha256: str,
    manifest: list[dict],
    config: Dict[str, Any],
    config_sha256: str,
    envelopes: Dict[int, Dict[str, Any]],
    started_at: str,
    in_progress: Dict[str, Any] | None,
    last_infrastructure_error: Dict[str, Any] | None = None,
) -> None:
    complete = _ordered_complete_envelopes(envelopes)
    prediction_path = run_dir / "submission_predictions.jsonl"
    private_path = run_dir / "result_private.json"
    _write_predictions(prediction_path, complete)
    _validate_predictions(
        prediction_path,
        complete,
        require_complete=len(complete) == EXPECTED_SHARD_COUNT,
    )
    prediction_sha256 = _sha256_file(prediction_path)
    completed_indices = [envelope["global_index"] for envelope in complete]
    completed_ids = [envelope["instance_id"] for envelope in complete]
    status = (
        "complete"
        if len(complete) == EXPECTED_SHARD_COUNT and in_progress is None
        else "in_progress"
    )
    if status == "complete":
        _write_private_result(private_path, complete)
    private_index = {
        "mode": MODE,
        "leaderboard_mode": LEADERBOARD_MODE,
        "status": status,
        "metrics": _result_metrics(complete),
        "storage": "logs/task_results/{global_index:04d}.json",
        "completed": [
            {
                key: envelope[key]
                for key in (
                    "global_index",
                    "instance_id",
                    "selected_database",
                    "model_completed_at",
                    "cleanup",
                    "result_summary",
                    "prediction",
                )
            }
            for envelope in complete
        ],
    }
    _atomic_json(run_dir / "result_private_index.json", private_index)
    remaining = [
        record["global_index"]
        for record in manifest
        if record["global_index"] not in completed_indices
    ]
    checkpoint = {
        "status": status,
        "updated_at": _utc_now(),
        "started_at": started_at,
        "full_input_sha256": input_sha256,
        "configuration_sha256": config_sha256,
        "global_index_range": [SHARD_START, SHARD_END],
        "expected_count": EXPECTED_SHARD_COUNT,
        "completed_count": len(complete),
        "completed_global_indices": completed_indices,
        "completed_instance_ids": completed_ids,
        "remaining_global_indices": remaining,
        "next_global_index": remaining[0] if remaining else None,
        "in_progress": in_progress,
        "all_completed_cleanup_verified": all(
            envelope.get("cleanup", {}).get("verified") is True
            for envelope in complete
        ),
        "prediction_file_sha256": prediction_sha256,
        "last_infrastructure_error": last_infrastructure_error,
    }
    _atomic_json(run_dir / "checkpoint.json", checkpoint)

    metrics = _result_metrics(complete)
    summary = {
        "status": status,
        "started_at": started_at,
        "updated_at": checkpoint["updated_at"],
        "completed_at": _utc_now() if status == "complete" else None,
        "full_input": {
            "path": str(input_path),
            "record_count": EXPECTED_INPUT_COUNT,
            "sha256": input_sha256,
        },
        "configuration": config,
        "configuration_sha256": config_sha256,
        "global_index_range": [SHARD_START, SHARD_END],
        "slice": f"records[{SLICE_START}:{SLICE_END}]",
        "expected_task_count": EXPECTED_SHARD_COUNT,
        "completed_task_count": len(complete),
        "instance_ids": [record["instance_id"] for record in manifest],
        "completed_instance_ids": completed_ids,
        "prediction_file": "submission_predictions.jsonl",
        "prediction_file_sha256": prediction_sha256,
        "metrics": metrics,
        "all_completed_cleanup_verified": checkpoint[
            "all_completed_cleanup_verified"
        ],
        "merge_requirements": {
            "require_identical_full_input_sha256": input_sha256,
            "require_identical_configuration_sha256": config_sha256,
            "sort_by": "global_index",
            "require_unique_instance_id": True,
        },
    }
    _atomic_json(run_dir / "summary.json", summary)
    _assert_public_safe((
        run_dir / "shard_manifest.jsonl",
        run_dir / "checkpoint.json",
        run_dir / "submission_predictions.jsonl",
        run_dir / "summary.json",
    ))


def _prepare_run(
    run_dir: Path, input_path: Path
) -> tuple[list[dict], list[dict], str, Dict[str, Any], str, str]:
    _, shard = _load_input(input_path)
    manifest = _shard_manifest(shard)
    input_sha256 = _sha256_file(input_path)
    config = _configuration_summary()
    _validate_configuration(config)
    config_sha256 = _sha256_bytes(_canonical_json(config).encode("utf-8"))
    manifest_path = run_dir / "shard_manifest.jsonl"
    expected_manifest_text = _manifest_text(manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "task_results").mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != expected_manifest_text:
            raise RuntimeError(
                "Existing shard_manifest.jsonl does not match "
                f"records[{SLICE_START}:{SLICE_END}]"
            )
    else:
        _atomic_text(manifest_path, expected_manifest_text)

    checkpoint_path = run_dir / "checkpoint.json"
    started_at = _utc_now()
    if checkpoint_path.exists():
        checkpoint = _load_json(checkpoint_path)
        started_at = checkpoint.get("started_at") or started_at
        if checkpoint.get("full_input_sha256") != input_sha256:
            raise RuntimeError("Full input SHA256 differs from existing checkpoint")
        if checkpoint.get("configuration_sha256") != config_sha256:
            raise RuntimeError("Configuration SHA256 differs from existing checkpoint")
    return shard, manifest, input_sha256, config, config_sha256, started_at


async def _recover_pending_cleanup(
    run_dir: Path,
    envelopes: Dict[int, Dict[str, Any]],
) -> None:
    for global_index in sorted(envelopes):
        envelope_ref = envelopes[global_index]
        envelope = _load_full_envelope(envelope_ref)
        if envelope.get("cleanup", {}).get("verified") is True:
            continue
        logger.info(
            "Recovering cleanup without rerunning model: %d %s",
            global_index,
            envelope["instance_id"],
        )
        envelope["cleanup"] = await _ensure_cleanup(
            envelope["selected_database"],
            envelope["instance_id"],
        )
        path = _task_result_path(run_dir, global_index)
        _atomic_json(path, envelope)
        envelopes[global_index] = _compact_envelope(path, envelope)


async def run(
    run_dir: Path, input_path: Path, *, prepare_only: bool = False
) -> None:
    (
        shard,
        manifest,
        input_sha256,
        config,
        config_sha256,
        started_at,
    ) = _prepare_run(run_dir, input_path)
    envelopes = _read_envelopes(run_dir, manifest)
    await _recover_pending_cleanup(run_dir, envelopes)
    _write_state(
        run_dir=run_dir,
        input_path=input_path,
        input_sha256=input_sha256,
        manifest=manifest,
        config=config,
        config_sha256=config_sha256,
        envelopes=envelopes,
        started_at=started_at,
        in_progress=None,
    )
    if prepare_only:
        logger.info("Prepared shard only: %s", run_dir)
        return

    by_index = {
        record["global_index"]: (record, task)
        for record, task in zip(manifest, shard)
    }
    completed = {
        envelope["global_index"]
        for envelope in _ordered_complete_envelopes(envelopes)
    }
    for global_index in range(SHARD_START, SHARD_END + 1):
        manifest_record, task = by_index[global_index]
        if global_index in completed:
            logger.info(
                "Skipping checkpointed task %d/%d: %s",
                global_index,
                SHARD_END,
                manifest_record["instance_id"],
            )
            continue

        in_progress = {
            "global_index": global_index,
            "instance_id": manifest_record["instance_id"],
            "selected_database": manifest_record["selected_database"],
            "started_at": _utc_now(),
        }
        _write_state(
            run_dir=run_dir,
            input_path=input_path,
            input_sha256=input_sha256,
            manifest=manifest,
            config=config,
            config_sha256=config_sha256,
            envelopes=envelopes,
            started_at=started_at,
            in_progress=in_progress,
        )
        logger.info(
            "=== Global task %d/%d: %s (%s) ===",
            global_index,
            SHARD_END,
            manifest_record["instance_id"],
            manifest_record["selected_database"],
        )

        try:
            result = await run_single_task(task)
        except Exception as exc:
            cleanup_error = None
            try:
                await _ensure_cleanup(
                    manifest_record["selected_database"],
                    manifest_record["instance_id"],
                )
            except Exception as cleanup_exc:
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            failure = {
                **in_progress,
                "failed_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "cleanup_error": cleanup_error,
            }
            public_failure = {
                "global_index": global_index,
                "instance_id": manifest_record["instance_id"],
                "failed_at": failure["failed_at"],
                "error_type": type(exc).__name__,
                "cleanup_pending": cleanup_error is not None,
            }
            failure_path = run_dir / "logs" / "infrastructure_failures.jsonl"
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            _write_state(
                run_dir=run_dir,
                input_path=input_path,
                input_sha256=input_sha256,
                manifest=manifest,
                config=config,
                config_sha256=config_sha256,
                envelopes=envelopes,
                started_at=started_at,
                in_progress=in_progress,
                last_infrastructure_error=public_failure,
            )
            logger.error(
                "Infrastructure failure at %d %s; task is not checkpointed",
                global_index,
                manifest_record["instance_id"],
            )
            traceback.print_exc()
            raise

        envelope = {
            "global_index": global_index,
            "instance_id": manifest_record["instance_id"],
            "selected_database": manifest_record["selected_database"],
            "model_completed_at": _utc_now(),
            "cleanup": {"verified": False},
            "result": result,
        }
        # Commit the model outcome before cleanup. A restart will never rerun it.
        envelope_path = _task_result_path(run_dir, global_index)
        _atomic_json(envelope_path, envelope)
        envelope["cleanup"] = await _ensure_cleanup(
            manifest_record["selected_database"],
            manifest_record["instance_id"],
        )
        _atomic_json(envelope_path, envelope)
        envelopes[global_index] = _compact_envelope(envelope_path, envelope)
        completed.add(global_index)
        _write_state(
            run_dir=run_dir,
            input_path=input_path,
            input_sha256=input_sha256,
            manifest=manifest,
            config=config,
            config_sha256=config_sha256,
            envelopes=envelopes,
            started_at=started_at,
            in_progress=None,
        )
        logger.info(
            "Checkpoint committed: %d/%d (%s), reward=%.2f",
            len(completed),
            EXPECTED_SHARD_COUNT,
            manifest_record["instance_id"],
            float(result.get("total_reward", 0) or 0),
        )

    complete = _ordered_complete_envelopes(envelopes)
    if len(complete) != EXPECTED_SHARD_COUNT:
        raise RuntimeError(
            f"Shard stopped with {len(complete)}/{EXPECTED_SHARD_COUNT} results"
        )
    expected_indices = list(range(SHARD_START, SHARD_END + 1))
    actual_indices = [envelope["global_index"] for envelope in complete]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"Completed global indices are not continuous {SHARD_START}-{SHARD_END}"
        )
    _validate_predictions(
        run_dir / "submission_predictions.jsonl",
        complete,
        require_complete=True,
    )
    logger.info(
        "Shard complete: %d/%d, last global_index=%d",
        EXPECTED_SHARD_COUNT,
        EXPECTED_SHARD_COUNT,
        SHARD_END,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--start", type=int, default=551)
    parser.add_argument("--end", type=int, default=600)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "bird-interact-full" / "bird_interact_data.jsonl"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    _configure_shard(args.start, args.end)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(
        run(
            Path(args.run_dir).resolve(),
            Path(args.input).resolve(),
            prepare_only=args.prepare_only,
        )
    )


if __name__ == "__main__":
    main()
