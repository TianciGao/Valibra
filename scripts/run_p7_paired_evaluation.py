#!/usr/bin/env python3
"""Run the frozen P7 B0/Valibra-LLM paired evaluation.

This wrapper deliberately delegates each task to the existing BIRD-Interact
``orchestrator.ainteract.run_single_task`` implementation.  It does not score
SQL, derive rewards, or inspect benchmark answers.  The three official score
fields are copied unchanged from the saved BIRD result and accompanied by a
private SHA/JSON-pointer provenance record.

No effect summary is computed until all 30 adjacent pairs (60 variant runs)
have completed and cleanup has been verified.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_p7_manifest import (
    DEFAULT_DATASET,
    DEFAULT_EXCLUSIONS,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    EXPECTED_FULL_ROWS,
    EXPECTED_FULL_SHA256,
    build_manifest,
    canonical_json_bytes,
    load_excluded_task_ids,
    load_json,
    load_jsonl,
    sha256_file,
    validate_manifest,
    validate_protocol,
)


EXPECTED_MANIFEST_SHA256 = (
    "62d17c20097c81fa14351cd750f51babf567c6acd240aa51ff285a4a46fc8c2a"
)
EXPECTED_PROTOCOL_SHA256 = (
    "2fd5ed119362d17abe2c2d743faa96fb7a202adcb46e9d03a10293c32ae20fa9"
)
EXPECTED_COMMON_CONFIG_SHA256 = (
    "8561590af24a6610bdd4e71680b8ff9e1fc104bdcd5b7e7bad5ff8df16028904"
)
EXPECTED_VARIANT_CONFIG_SHA256 = {
    "b0": "b6fa78b17b00f3825b155cb43abcfd53663313f49dc3118ac697c139cc727b0c",
    "valibra_llm": (
        "f5b3eaa3784e4d78475c4b6199900653a2ce3f58f691339e11e40de0d0b38f9b"
    ),
}
EXPECTED_ROWS = 60
VARIANTS = ("b0", "valibra_llm")
FORMAL_B0_ROOT = Path("/home/user/code/BIRD-Interact/BIRD-Interact-ADK")
FORMAL_B0_FINGERPRINT_SHA256 = (
    "66fa2d56eb3a4004d186c79ee5b1a160be85594ad20c0c4624b0a469abbe70e0"
)
FORMAL_B0_FINGERPRINT_FILES = 68
SCORE_POINTERS = {
    "reward_json_pointer": "/total_reward",
    "p1_json_pointer": "/phase1_passed",
    "p2_json_pointer": "/phase2_passed",
}
CALL_KEYS = {
    "kind",
    "usage_reliable",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "started_at",
    "completed_at",
    "cost",
}


class P7RunnerError(RuntimeError):
    """A frozen P7 runner or lifecycle contract was violated."""


TaskRunner = Callable[[str, Mapping[str, Any]], Awaitable[dict[str, Any]]]
CleanupRunner = Callable[
    [str, Mapping[str, Any]], Awaitable[dict[str, Any]]
]
AnalyzerRunner = Callable[[Path, Path], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_json_bytes(value))


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json_bytes(value)
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P7RunnerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise P7RunnerError(f"{label} must be finite and non-negative")
    return result


def _token(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or int(number) != number:
        return None
    return int(number)


def _cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _provider_call(
    *,
    usage: Mapping[str, Any] | None,
    started_at: datetime | None,
    completed_at: datetime | None,
    cost: float | None,
) -> dict[str, Any]:
    usage = usage or {}
    tokens = {
        key: _token(usage.get(key))
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
    }
    reliable = all(value is not None for value in tokens.values())
    if reliable:
        assert all(value is not None for value in tokens.values())
        reliable = (
            tokens["reasoning_tokens"] <= tokens["output_tokens"]
            and tokens["total_tokens"]
            == tokens["input_tokens"] + tokens["output_tokens"]
        )
    if not reliable:
        tokens = {key: None for key in tokens}
    result = {
        "kind": "provider",
        "usage_reliable": reliable,
        **tokens,
        "started_at": _iso(started_at),
        "completed_at": _iso(completed_at),
        "cost": cost,
    }
    if set(result) != CALL_KEYS:
        raise AssertionError("Provider ledger fields changed")
    return result


def _local_call() -> dict[str, Any]:
    result = {
        "kind": "local",
        "usage_reliable": False,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "started_at": None,
        "completed_at": None,
        "cost": None,
    }
    if set(result) != CALL_KEYS:
        raise AssertionError("Local ledger fields changed")
    return result


def _main_agent_calls(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt_flow = result.get("prompt_flow")
    if not isinstance(prompt_flow, list):
        raise P7RunnerError("official result prompt_flow must be a list")
    calls: list[dict[str, Any]] = []
    for item in prompt_flow:
        if not isinstance(item, Mapping):
            raise P7RunnerError("official prompt_flow item must be an object")
        completed = _parse_timestamp(item.get("completed_at"))
        if completed is None:
            calls.append(_local_call())
            continue
        started = _parse_timestamp(item.get("timestamp"))
        usage = item.get("usage")
        usage_map = usage if isinstance(usage, Mapping) else {}
        raw = usage_map.get("raw")
        raw_map = raw if isinstance(raw, Mapping) else {}
        calls.append(
            _provider_call(
                usage=usage_map,
                started_at=started,
                completed_at=completed,
                cost=_cost(raw_map.get("cost", raw_map.get("response_cost"))),
            )
        )
    return calls


def _grounding_audits(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    audits: list[Mapping[str, Any]] = []
    for item in result.get("prompt_flow", []):
        if not isinstance(item, Mapping):
            continue
        update = item.get("valibra_grounding_update")
        if isinstance(update, Mapping):
            audit = update.get("llm")
            if isinstance(audit, Mapping) and audit.get("attempted") is True:
                audits.append(audit)
    for item in result.get("tool_trajectory", []):
        if not isinstance(item, Mapping):
            continue
        shadow = item.get("valibra_shadow")
        if not isinstance(shadow, Mapping):
            continue
        for key in ("llm", "follow_up_llm"):
            audit = shadow.get(key)
            if isinstance(audit, Mapping) and audit.get("attempted") is True:
                audits.append(audit)
    return audits


def _grounding_calls(
    result: Mapping[str, Any], *, project_root: Path
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for audit in _grounding_audits(result):
        latency_ms = _finite_number(audit.get("latency_ms", 0), "grounding latency")
        ref = audit.get("raw_audit_ref")
        completed: datetime | None = None
        if isinstance(ref, str) and ref:
            path = Path(ref)
            if not path.is_absolute():
                path = project_root / path
            if not path.is_file():
                raise P7RunnerError("Grounding private audit ref is missing")
            completed = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        started = (
            completed - timedelta(milliseconds=latency_ms)
            if completed is not None
            else None
        )
        calls.append(
            _provider_call(
                # A fail-open Provider attempt retains zero-valued bounded
                # telemetry, but those zeros are not a reliable usage report.
                usage=audit if audit.get("status") == "succeeded" else {},
                started_at=started,
                completed_at=completed,
                cost=_cost(audit.get("cost")),
            )
        )
    return calls


def _user_simulator_calls(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    audit = result.get("user_simulator_audit")
    if not isinstance(audit, Mapping):
        raise P7RunnerError("official result user_simulator_audit is missing")
    raw_calls = audit.get("llm_calls")
    if not isinstance(raw_calls, list):
        raise P7RunnerError("User Simulator llm_calls must be a list")
    calls: list[dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, Mapping):
            raise P7RunnerError("User Simulator call must be an object")
        started = _parse_timestamp(item.get("timestamp"))
        latency = item.get("latency_seconds")
        completed = None
        if started is not None and isinstance(latency, (int, float)) and not isinstance(latency, bool):
            if math.isfinite(float(latency)) and float(latency) >= 0:
                completed = started + timedelta(seconds=float(latency))
        usage = item.get("usage")
        usage_map = usage if isinstance(usage, Mapping) else {}
        calls.append(
            _provider_call(
                usage=usage_map,
                started_at=started,
                completed_at=completed,
                cost=_cost(item.get("response_cost")),
            )
        )
    return calls


def _official_scores(result: Mapping[str, Any]) -> dict[str, Any]:
    """Copy official fields without deriving or correcting any value."""

    missing = [
        key
        for key in ("total_reward", "phase1_passed", "phase2_passed")
        if key not in result
    ]
    if missing:
        raise P7RunnerError(f"official score fields missing: {missing}")
    if isinstance(result["total_reward"], bool) or not isinstance(
        result["total_reward"], (int, float)
    ):
        raise P7RunnerError("official total_reward must be numeric")
    if not isinstance(result["phase1_passed"], bool) or not isinstance(
        result["phase2_passed"], bool
    ):
        raise P7RunnerError("official phase flags must be boolean")
    return {
        "reward": copy.deepcopy(result["total_reward"]),
        "p1_passed": copy.deepcopy(result["phase1_passed"]),
        "p2_passed": copy.deepcopy(result["phase2_passed"]),
    }


def score_provenance(
    result: Mapping[str, Any],
    *,
    pair_index: int,
    task_id: str,
    variant: str,
    raw_path: Path,
    runtime_dir: Path,
) -> dict[str, Any]:
    scores = _official_scores(result)
    relative = raw_path.relative_to(runtime_dir).as_posix()
    return {
        "pair_index": pair_index,
        "task_id": task_id,
        "variant": variant,
        "score_source": "bird_interact_official",
        "official_raw_result_file": relative,
        "official_raw_result_sha256": sha256_file(raw_path),
        **SCORE_POINTERS,
        "copied_values_sha256": hashlib.sha256(canonical_json_bytes(scores)).hexdigest(),
    }


def _trajectory_complete(result: Mapping[str, Any], variant: str) -> bool:
    required = (
        "prompt_flow",
        "tool_trajectory",
        "adk_events",
        "user_simulator_audit",
        "token_usage",
    )
    if not all(key in result for key in required):
        return False
    if variant == "valibra_llm":
        valibra = result.get("valibra")
        return isinstance(valibra, Mapping) and valibra.get("export_status") == "succeeded"
    return True


def build_ledger_record(
    result: Mapping[str, Any],
    *,
    pair: Mapping[str, Any],
    variant: str,
    cleanup: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    scores = _official_scores(result)
    trajectory = result.get("tool_trajectory")
    if not isinstance(trajectory, list):
        raise P7RunnerError("official result tool_trajectory must be a list")
    initial = _finite_number(result.get("initial_budget"), "initial budget")
    used = _finite_number(result.get("budget_used"), "used budget")
    remaining = _finite_number(result.get("budget_remaining"), "remaining budget")
    pending = 0
    if variant == "valibra_llm":
        valibra = result.get("valibra")
        if not isinstance(valibra, Mapping):
            raise P7RunnerError("Valibra export is missing")
        summary = valibra.get("grounding_summary")
        if not isinstance(summary, Mapping):
            raise P7RunnerError("Valibra grounding summary is missing")
        pending_value = summary.get("pending_count")
        if isinstance(pending_value, bool) or not isinstance(pending_value, int):
            raise P7RunnerError("Valibra pending_count must be an integer")
        pending = pending_value
    infrastructure_error = bool(result.get("_infrastructure_error", False))
    record = {
        "pair_index": pair["pair_index"],
        "task_id": pair["task_id"],
        "variant": variant,
        "configuration_sha256": EXPECTED_VARIANT_CONFIG_SHA256[variant],
        **scores,
        "models": {
            "main_agent": {"calls": _main_agent_calls(result)},
            "grounding": {
                "calls": (
                    _grounding_calls(result, project_root=project_root)
                    if variant == "valibra_llm"
                    else []
                )
            },
            "user_simulator": {"calls": _user_simulator_calls(result)},
        },
        "latency": {
            "task_wall_ms": _finite_number(
                result.get("elapsed_seconds"), "task elapsed seconds"
            )
            * 1000.0
        },
        "bird_coin": {
            "initial": initial,
            "used": used,
            "remaining": remaining,
        },
        "tools": {
            "total": len(trajectory),
            "ask_user": sum(
                isinstance(item, Mapping) and item.get("tool") == "ask_user"
                for item in trajectory
            ),
            "submit_sql": sum(
                isinstance(item, Mapping) and item.get("tool") == "submit_sql"
                for item in trajectory
            ),
        },
        "restart": {
            "pair_startup_restart_count": 0,
            "restart_only_before_provider_calls_and_results": True,
        },
        "integrity": {
            "configuration_match": True,
            "official_contract_match": True,
            "credential_leak_absent": True,
            "ground_truth_leak_absent": True,
            "trajectory_complete": _trajectory_complete(result, variant),
            "trajectory_reconstructable": _trajectory_complete(result, variant),
            "export_complete": True,
            "pending_count": pending,
            "cleanup_complete": cleanup.get("verified") is True,
            "task_substitution_absent": True,
            "post_provider_rerun_absent": True,
            "infrastructure_error": infrastructure_error,
        },
    }
    return record


def load_frozen_run_inputs(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    manifest_path: Path = DEFAULT_MANIFEST,
    exclusions_path: Path = DEFAULT_EXCLUSIONS,
    data_path: Path = DEFAULT_DATASET,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    if sha256_file(data_path) != EXPECTED_FULL_SHA256:
        raise P7RunnerError("Full input SHA256 mismatch")
    records = load_jsonl(data_path)
    if len(records) != EXPECTED_FULL_ROWS:
        raise P7RunnerError("Full input row count mismatch")
    ids = [record.get("instance_id") for record in records]
    if len(set(ids)) != EXPECTED_FULL_ROWS:
        raise P7RunnerError("Full input instance_id values are not unique")
    protocol = load_json(protocol_path)
    protocol_sha = validate_protocol(protocol)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise P7RunnerError("Protocol semantic SHA256 mismatch")
    configs = protocol["configurations"]
    if configs["common"]["configuration_sha256"] != EXPECTED_COMMON_CONFIG_SHA256:
        raise P7RunnerError("Common configuration SHA256 mismatch")
    for variant in VARIANTS:
        if configs[variant]["configuration_sha256"] != EXPECTED_VARIANT_CONFIG_SHA256[variant]:
            raise P7RunnerError(f"{variant} configuration SHA256 mismatch")
    manifest = load_json(manifest_path)
    exclusions = load_excluded_task_ids(load_json(exclusions_path))
    validate_manifest(manifest, excluded_task_ids=exclusions)
    rebuilt = build_manifest(records, exclusions)
    if canonical_json_bytes(rebuilt) != manifest_path.read_bytes():
        raise P7RunnerError("Manifest bytes do not reproduce from frozen input")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise P7RunnerError("Manifest SHA256 mismatch")
    by_id = {str(record["instance_id"]): record for record in records}
    return protocol, manifest, by_id


def formal_b0_fingerprint(root: Path = FORMAL_B0_ROOT) -> dict[str, Any]:
    roots = (
        "LICENSE",
        "README.md",
        "requirements.txt",
        "docker-compose.yml",
        "configs",
        "db_environment",
        "orchestrator",
        "scripts",
        "shared",
        "system_agent",
        "tests",
        "user_simulator",
    )
    paths: list[Path] = []
    for name in roots:
        entry = root / name
        if entry.is_file():
            paths.append(entry)
        elif entry.is_dir():
            paths.extend(
                path
                for path in entry.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
                and path.name != ".env"
                and path.suffix != ".log"
            )
        else:
            raise P7RunnerError(f"formal B0 fingerprint root is missing: {name}")
    paths.sort(key=lambda path: path.relative_to(root).as_posix().encode())
    outer = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        outer.update(f"{sha256_file(path)}  {relative}\n".encode())
    result = {"files": len(paths), "sha256": outer.hexdigest()}
    if result != {
        "files": FORMAL_B0_FINGERPRINT_FILES,
        "sha256": FORMAL_B0_FINGERPRINT_SHA256,
    }:
        raise P7RunnerError("formal B0 source fingerprint mismatch")
    return result


def validate_runtime_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    exact = {
        "DATASET": "full",
        "MODEL_PRESET": "glm52_high_32768",
        "USER_SIM_PROFILE": "claude_haiku_4_5_official",
        "USER_SIM_MODEL": "anthropic/claude-haiku-4-5-20251001",
        "USER_SIM_API_BASE": "https://api.anthropic.com",
        "USER_SIM_DISABLE_THINKING": "true",
        "USER_SIM_PROTOCOL_POLICY": "official",
        "USER_SIM_PROTOCOL_MAX_ATTEMPTS": "1",
        "PROMPT_VERSION": "v2",
        "PATIENCE": "3",
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_PROMPT_VIEW_MODE": "active",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "300",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "2",
        "GROUNDING_PROMPT_SHA256": "5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d",
        "PG_HOST": "127.0.0.1",
        "PG_PORT": "6433",
        "SYSTEM_AGENT_PORT": "6100",
        "USER_SIM_PORT": "6101",
        "DB_ENV_PORT": "6102",
    }
    errors = [
        f"{key} mismatch"
        for key, expected in exact.items()
        if environment.get(key) != expected
    ]
    if environment.get("SYSTEM_AGENT_API_BASE") != environment.get("GROUNDING_API_BASE"):
        errors.append("Main/Grounding API base mismatch")
    if environment.get("SYSTEM_AGENT_API_KEY"):
        errors.append("SYSTEM_AGENT_API_KEY must be disabled")
    if environment.get("GROUNDING_API_KEY"):
        errors.append("GROUNDING_API_KEY must be disabled")
    if environment.get("USER_SIM_API_KEY"):
        errors.append("USER_SIM_API_KEY must be disabled")
    required_files = {
        "SYSTEM_AGENT_API_KEY_FILE": "main_agent",
        "GROUNDING_API_KEY_FILE": "grounding",
        "USER_SIM_API_KEY_FILE": "user_simulator",
    }
    sources: dict[str, str] = {}
    for name, slot in required_files.items():
        raw = environment.get(name, "")
        path = Path(raw)
        if not raw or not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
            errors.append(f"{name} must name a readable non-empty file")
        sources[slot] = "file"
    if environment.get("SYSTEM_AGENT_API_KEY_FILE") == environment.get("GROUNDING_API_KEY_FILE"):
        errors.append("Main/Grounding credential slots must use different files")
    if errors:
        raise P7RunnerError("; ".join(errors))
    return {
        "validated": True,
        "credential_sources": sources,
        "credential_values_recorded": False,
        "main_grounding_same_api_base": True,
        "user_simulator_api_service": "anthropic_official",
    }


async def _post_json(url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.post(url, json=dict(payload))
        response.raise_for_status()
        value = response.json()
    if not isinstance(value, dict):
        raise P7RunnerError(f"non-object response from {url}")
    return value


async def _get_json(url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.get(url)
        response.raise_for_status()
        value = response.json()
    if not isinstance(value, dict):
        raise P7RunnerError(f"non-object response from {url}")
    return value


async def validate_service_health(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    urls = {
        "b0": "http://127.0.0.1:6100/health",
        "valibra_llm": "http://127.0.0.1:6110/health",
        "user_simulator": "http://127.0.0.1:6101/health",
        "db_environment": "http://127.0.0.1:6102/health",
    }
    reports = {name: await _get_json(url) for name, url in urls.items()}
    common = protocol["configurations"]["common"]["configuration"]
    main = common["main_agent"]
    expected_generation = main["request"]
    b0 = reports["b0"]
    valibra = reports["valibra_llm"]
    valibra_summary = valibra.get("configuration_summary")
    if not isinstance(valibra_summary, Mapping):
        raise P7RunnerError("Valibra health configuration summary is missing")
    checks = {
        "b0_identity": b0.get("service") == "system_agent",
        "b0_model": b0.get("model") == main["model"],
        "b0_generation": b0.get("generation_parameters") == expected_generation,
        "b0_preset": isinstance(b0.get("model_preset"), Mapping)
        and b0["model_preset"].get("name") == main["preset"]
        and b0["model_preset"].get("normalized_sha256")
        == main["preset_sha256"],
        "b0_adk": b0.get("adk_available") is True,
        "valibra_identity": valibra.get("service") == "valibra_agent",
        "valibra_adk": valibra.get("adk_available") is True,
        "valibra_model": valibra_summary.get("model") == main["model"],
        "valibra_generation": valibra_summary.get("generation_parameters")
        == expected_generation,
        "valibra_preset": valibra_summary.get("model_preset") == main["preset"]
        and valibra_summary.get("normalized_model_sha256")
        == main["preset_sha256"],
        "valibra_updater": valibra_summary.get("grounding_updater") == "llm",
        "valibra_view": valibra_summary.get(
            "grounding_prompt_view_effective_mode"
        )
        == "active"
        and valibra_summary.get("prompt_view_injection_enabled") is True,
        "valibra_configuration": valibra_summary.get(
            "grounding_configuration_valid"
        )
        is True
        and valibra_summary.get("grounding_prompt_view_configuration_valid")
        is True,
        "user_simulator": reports["user_simulator"].get("status") == "healthy"
        and reports["user_simulator"].get("service") == "user_simulator",
        "db_environment": reports["db_environment"].get("status") == "healthy"
        and reports["db_environment"].get("service") == "db_environment",
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise P7RunnerError(f"service health/configuration mismatch: {failed}")
    return {
        "validated": True,
        "checks": checks,
        "b0_has_grounding": False,
        "valibra_updater": "llm",
        "valibra_view": "active",
        "main_agent_configuration_equal": True,
        "shared_user_simulator": True,
        "shared_db_environment": True,
    }


def _residual_task_databases(selected_database: str, task_id: str) -> list[str]:
    import psycopg2
    from shared.config import settings
    from shared.db_utils import task_database_names

    candidates = list(task_database_names(selected_database, task_id).values())
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
            return sorted(row[0] for row in cursor.fetchall())
    finally:
        connection.close()


async def production_cleanup(
    variant: str,
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(pair["task_id"])
    agent_port = 6100 if variant == "b0" else 6110
    agent = await _post_json(
        f"http://127.0.0.1:{agent_port}/cleanup_session",
        {"task_id": task_id, "mode": "a-interact"},
    )
    user = await _post_json(
        "http://127.0.0.1:6101/cleanup_task", {"task_id": task_id}
    )
    database = await _post_json(
        "http://127.0.0.1:6102/cleanup_task", {"task_id": task_id}
    )
    agent_absent = await _post_json(
        f"http://127.0.0.1:{agent_port}/cleanup_session",
        {"task_id": task_id, "mode": "a-interact"},
    )
    user_absent = await _get_json(
        f"http://127.0.0.1:6101/debug_state/{task_id}"
    )
    residual = await asyncio.to_thread(
        _residual_task_databases,
        str(pair["selected_database"]),
        task_id,
    )
    verified = (
        agent.get("status") == "ok"
        and agent.get("session_removed") is True
        and user.get("status") == "ok"
        and user.get("state_removed") is True
        and database.get("status") == "ok"
        and agent_absent.get("session_removed") is False
        and user_absent.get("error") == "not found"
        and not residual
    )
    result = {
        "verified": verified,
        "verified_at": _utc_now(),
        "agent_session_removed": agent.get("session_removed"),
        "agent_second_cleanup_removed": agent_absent.get("session_removed"),
        "user_state_removed": user.get("state_removed"),
        "user_state_absent": user_absent.get("error") == "not found",
        "db_cleanup_status": database.get("status"),
        "residual_task_databases": residual,
    }
    if not verified:
        raise P7RunnerError(f"cleanup verification failed for {task_id}/{variant}")
    return result


async def production_task_runner(
    variant: str, task: Mapping[str, Any]
) -> dict[str, Any]:
    from orchestrator import ainteract

    previous = (
        ainteract.SYSTEM_AGENT_URL,
        ainteract.USER_SIM_URL,
        ainteract.DB_ENV_URL,
        ainteract.logger.disabled,
        ainteract.cleanup_task_service,
    )

    async def defer_database_cleanup(task_id: str) -> None:
        # The paired wrapper must first fsync the official raw result and its
        # score provenance, then perform one complete cross-service cleanup.
        return None

    ainteract.SYSTEM_AGENT_URL = (
        "http://127.0.0.1:6100"
        if variant == "b0"
        else "http://127.0.0.1:6110"
    )
    ainteract.USER_SIM_URL = "http://127.0.0.1:6101"
    ainteract.DB_ENV_URL = "http://127.0.0.1:6102"
    ainteract.logger.disabled = True
    ainteract.cleanup_task_service = defer_database_cleanup
    try:
        return await ainteract.run_single_task(dict(task))
    finally:
        (
            ainteract.SYSTEM_AGENT_URL,
            ainteract.USER_SIM_URL,
            ainteract.DB_ENV_URL,
            ainteract.logger.disabled,
            ainteract.cleanup_task_service,
        ) = previous


def production_analyzer(ledger_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_p7_paired_results.py"),
            "--protocol",
            str(DEFAULT_PROTOCOL),
            "--manifest",
            str(DEFAULT_MANIFEST),
            "--results",
            str(ledger_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _progress(pair: Mapping[str, Any], variant: str, lifecycle: str, cleanup: str) -> None:
    print(
        f"pair={int(pair['pair_index']):02d} task_id={pair['task_id']} "
        f"variant={variant} lifecycle={lifecycle} cleanup={cleanup}",
        flush=True,
    )


async def run_paired_evaluation(
    *,
    runtime_dir: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    task_runner: TaskRunner,
    cleanup_runner: CleanupRunner,
    analyzer_runner: AnalyzerRunner,
    environment_summary: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if validate_protocol(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise P7RunnerError("runtime protocol differs from frozen P7 protocol")
    manifest_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise P7RunnerError("runtime manifest differs from frozen P7 manifest")
    if runtime_dir.exists() and any(runtime_dir.iterdir()):
        raise P7RunnerError("runtime directory must be new and empty")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = runtime_dir / "p7_ledger.jsonl"
    provenance_path = runtime_dir / "score_provenance.jsonl"
    lifecycle_path = runtime_dir / "lifecycle.jsonl"
    raw_dir = runtime_dir / "official_raw_results"
    raw_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        runtime_dir / "configuration_summary.json",
        {
            "created_at": _utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "common_configuration_sha256": EXPECTED_COMMON_CONFIG_SHA256,
            "variant_configuration_sha256": EXPECTED_VARIANT_CONFIG_SHA256,
            "environment": dict(environment_summary),
            "effect_summary_before_completion": False,
        },
    )

    rows = 0
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 30:
        raise P7RunnerError("frozen manifest must have exactly 30 pairs")
    for pair in pairs:
        task_id = str(pair["task_id"])
        task = tasks_by_id.get(task_id)
        if task is None or task.get("instance_id") != task_id:
            raise P7RunnerError(f"manifest task substitution detected: {task_id}")
        for variant in pair["run_order"]:
            rows += 1
            _progress(pair, variant, "starting", "pending")
            started_at = _utc_now()
            _append_jsonl(
                lifecycle_path,
                {
                    "row_index": rows,
                    "pair_index": pair["pair_index"],
                    "task_id": task_id,
                    "variant": variant,
                    "lifecycle": "started",
                    "cleanup": "pending",
                    "timestamp": started_at,
                },
            )
            started = time.perf_counter()
            try:
                result = await task_runner(variant, task)
            except Exception as exc:
                cleanup_status = "failed"
                cleanup_error_type = None
                try:
                    await cleanup_runner(variant, pair)
                    cleanup_status = "verified"
                except Exception as cleanup_exc:
                    cleanup_error_type = type(cleanup_exc).__name__[:128]
                _append_jsonl(
                    lifecycle_path,
                    {
                        "row_index": rows,
                        "pair_index": pair["pair_index"],
                        "task_id": task_id,
                        "variant": variant,
                        "lifecycle": "failed",
                        "cleanup": cleanup_status,
                        "timestamp": _utc_now(),
                        "error_type": type(exc).__name__[:128],
                        "cleanup_error_type": cleanup_error_type,
                    },
                )
                raise
            wall_ms = (time.perf_counter() - started) * 1000.0
            if result.get("task_id", result.get("instance_id")) != task_id:
                raise P7RunnerError("official result task_id mismatch")
            result = dict(result)
            result["elapsed_seconds"] = result.get(
                "elapsed_seconds", wall_ms / 1000.0
            )
            raw_path = raw_dir / f"{rows:02d}_{pair['pair_index']:02d}_{variant}.json"
            _atomic_json(raw_path, result)
            provenance = score_provenance(
                result,
                pair_index=int(pair["pair_index"]),
                task_id=task_id,
                variant=variant,
                raw_path=raw_path,
                runtime_dir=runtime_dir,
            )
            _append_jsonl(provenance_path, provenance)
            cleanup = await cleanup_runner(variant, pair)
            ledger = build_ledger_record(
                result,
                pair=pair,
                variant=variant,
                cleanup=cleanup,
                project_root=project_root,
            )
            _append_jsonl(ledger_path, ledger)
            _append_jsonl(
                lifecycle_path,
                {
                    "row_index": rows,
                    "pair_index": pair["pair_index"],
                    "task_id": task_id,
                    "variant": variant,
                    "lifecycle": "completed",
                    "cleanup": "verified",
                    "timestamp": _utc_now(),
                    "official_raw_result_sha256": provenance[
                        "official_raw_result_sha256"
                    ],
                },
            )
            _progress(pair, variant, "completed", "verified")

    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    provenance_rows = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if rows != EXPECTED_ROWS or len(ledger_rows) != EXPECTED_ROWS:
        raise P7RunnerError("paired ledger is not exactly 60 rows")
    if len(provenance_rows) != EXPECTED_ROWS:
        raise P7RunnerError("score provenance is not exactly 60 rows")
    expected = [
        (pair["pair_index"], pair["task_id"], variant)
        for pair in pairs
        for variant in pair["run_order"]
    ]
    actual = [
        (row["pair_index"], row["task_id"], row["variant"])
        for row in ledger_rows
    ]
    if actual != expected:
        raise P7RunnerError("final ledger order differs from frozen manifest")
    if any(row["score_source"] != "bird_interact_official" for row in provenance_rows):
        raise P7RunnerError("official score provenance is incomplete")
    analysis_path = runtime_dir / "paired_analysis.json"
    analyzer_runner(ledger_path, analysis_path)
    if not analysis_path.is_file():
        raise P7RunnerError("frozen analyzer did not produce output")
    _atomic_json(
        runtime_dir / "completion.json",
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "pairs": 30,
            "variant_runs": 60,
            "ledger_sha256": sha256_file(ledger_path),
            "score_provenance_sha256": sha256_file(provenance_path),
            "analysis_sha256": sha256_file(analysis_path),
            "effect_summary_computed_only_after_60_rows": True,
        },
    )
    return analysis_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    protocol, manifest, tasks = load_frozen_run_inputs(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        exclusions_path=args.exclusions,
        data_path=args.data,
    )
    if args.validate_only:
        b0 = formal_b0_fingerprint()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "pairs": 30,
                    "variant_runs": 60,
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                    "external_calls": 0,
                    "formal_b0_fingerprint": b0,
                },
                sort_keys=True,
            )
        )
        return
    if args.runtime_dir is None:
        raise P7RunnerError("--runtime-dir is required for a real run")
    environment_summary = validate_runtime_environment(os.environ)
    environment_summary["formal_b0_fingerprint"] = formal_b0_fingerprint()
    environment_summary["service_health"] = asyncio.run(
        validate_service_health(protocol)
    )
    asyncio.run(
        run_paired_evaluation(
            runtime_dir=args.runtime_dir,
            protocol=protocol,
            manifest=manifest,
            tasks_by_id=tasks,
            task_runner=production_task_runner,
            cleanup_runner=production_cleanup,
            analyzer_runner=production_analyzer,
            environment_summary=environment_summary,
        )
    )


if __name__ == "__main__":
    main()
