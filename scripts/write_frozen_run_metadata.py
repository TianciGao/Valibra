"""Write reproducibility metadata for a frozen contiguous-range run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    active_model_preset_report,
    normalized_system_agent_config,
    settings,
)
from system_agent.agent import AINTERACT_INSTRUCTION
from system_agent.callbacks import MAX_MODEL_TURNS, TOOL_COSTS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _git_metadata() -> dict[str, Any]:
    root = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--verify", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "repository_root": (
            root.stdout.strip() if root.returncode == 0 else None
        ),
        "commit_available": commit.returncode == 0,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "commit_error": (
            None
            if commit.returncode == 0
            else (commit.stderr.strip() or "repository has no HEAD commit")
        ),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--dependency-freeze", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise RuntimeError("concurrency must be positive")
    preset = active_model_preset_report()
    if preset is None:
        raise RuntimeError("MODEL_PRESET is required")

    run_dir = Path(args.run_dir).resolve()
    task_manifest_path = Path(args.task_manifest).resolve()
    dependency_freeze = Path(args.dependency_freeze).resolve()
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    selection = task_manifest["selection"]
    global_start, global_end = selection["global_index_range"]
    expected_count = global_end - global_start + 1
    if (
        global_start < 1
        or global_end > 600
        or selection["record_count"] != expected_count
        or selection["unique_instance_id_count"] != expected_count
    ):
        raise RuntimeError("Frozen task manifest is not a valid contiguous range")

    provider_files = {"shared/llm.py": _sha256_file(PROJECT_ROOT / "shared/llm.py")}
    local_provider = PROJECT_ROOT / "shared" / "_local_provider.py"
    if local_provider.is_file():
        provider_files["shared/_local_provider.py"] = _sha256_file(local_provider)

    dependency_versions = {
        name: _package_version(name)
        for name in (
            "google-adk",
            "google-genai",
            "litellm",
            "pydantic",
            "pydantic-settings",
            "httpx",
            "fastapi",
            "uvicorn",
            "psycopg2-binary",
            "sqlglot",
            "tiktoken",
        )
    }
    metadata = {
        "schema_version": 1,
        "status": "dry-run-prepared" if args.dry_run else "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": args.started_at,
        "result_directory": str(run_dir),
        "benchmark": {
            "dataset": "full",
            "interaction_mode": "a-interact",
            "leaderboard_mode": "stress",
            "global_index_range": [global_start, global_end],
            "task_count": expected_count,
            "concurrency": args.concurrency,
        },
        "model_preset": preset,
        "normalized_system_agent_config": normalized_system_agent_config(),
        "git": _git_metadata(),
        "provider": {
            "files_sha256": provider_files,
            "combined_sha256": _canonical_sha256(provider_files),
        },
        "dependencies": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": dependency_versions,
            "freeze_path": str(dependency_freeze),
            "freeze_sha256": _sha256_file(dependency_freeze),
            "requirements_sha256": _sha256_file(
                PROJECT_ROOT / "requirements.txt"
            ),
        },
        "tasks": task_manifest,
        "frozen_ainteract": {
            "prompt_sha256": hashlib.sha256(
                AINTERACT_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "source_sha256": {
                relative: _sha256_file(PROJECT_ROOT / relative)
                for relative in (
                    "system_agent/agent.py",
                    "system_agent/tools.py",
                    "system_agent/callbacks.py",
                )
            },
            "tool_names": list(TOOL_COSTS),
            "tool_costs": TOOL_COSTS,
            "callbacks": [
                "before_model_callback",
                "after_model_callback",
                "before_tool_callback",
                "after_tool_callback",
            ],
            "max_model_turns": MAX_MODEL_TURNS,
            "user_simulator": {
                "profile": os.environ.get("USER_SIM_PROFILE", ""),
                "model": settings.user_sim_model,
                "prompt_version": settings.prompt_version,
                "provider_hidden_thinking_disabled": (
                    settings.user_sim_disable_thinking
                ),
                "api_base": settings.user_sim_api_base.rstrip("/"),
                "use_bearer_for_custom_base": (
                    settings.user_sim_use_bearer_for_custom_base
                ),
                "credential_configured": bool(
                    settings.user_sim_api_key or settings.user_sim_api_key_file
                ),
                "provider_failure_policy": "fail_fast_uncheckpointed",
                "protocol_policy": settings.user_sim_protocol_policy,
                "protocol_max_attempts": (
                    settings.user_sim_protocol_max_attempts
                ),
                "prompts_file_sha256": _sha256_file(
                    PROJECT_ROOT / "user_simulator" / "prompts.py"
                ),
                "server_file_sha256": _sha256_file(
                    PROJECT_ROOT / "user_simulator" / "server.py"
                ),
            },
            "patience": settings.patience,
            "budget_formula": "6 + 2 * ambiguity_count + 2 * patience",
        },
        "expected_outputs": {
            "private_result": str(run_dir / "result_private.json"),
            "submission_predictions": str(
                run_dir / "submission_predictions.jsonl"
            ),
            "logs": str(run_dir / "logs"),
        },
    }
    output = run_dir / "run_manifest.json"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
