"""Run the official text-ReAct (non-ADK) evaluator one task at a time.

The one-task supervisor keeps the upstream prompt/action/evaluator loop intact
while adding task-level checkpoints, frozen provider parameters, complete raw
model-call retention, and infrastructure-error quarantine.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as distribution_version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


ADK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OFFICIAL_ROOT = ADK_ROOT.parent / "bird_interact_agent"
EXPECTED_SOURCE_COUNT = 600
TOOL_COSTS = {
    "ask": 2,
    "submit": 3,
    "execute": 1,
    "get_schema": 1,
    "get_all_column_meanings": 1,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1,
}
P1_GATE_DEFAULT_COUNT = 100
P1_GATE_DEFAULT_THRESHOLD = 0.15
OFFICIAL_BASELINE_PATH = ADK_ROOT / "configs" / "official_nonadk_baseline.json"
SUBMISSION_MODEL_NAME = "my_model"
SUBMISSION_TOKEN_ENCODING = "cl100k_base"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_official_baseline(official_root: Path) -> dict[str, Any]:
    baseline = json.loads(OFFICIAL_BASELINE_PATH.read_text(encoding="utf-8"))
    required_versions = baseline["official_environment"]
    installed_versions = {
        package: distribution_version(package)
        for package in required_versions
    }
    mismatched_versions = {
        package: {"expected": expected, "actual": installed_versions[package]}
        for package, expected in required_versions.items()
        if installed_versions[package] != expected
    }
    if mismatched_versions:
        raise RuntimeError(
            "Official dependency mismatch: "
            + json.dumps(mismatched_versions, ensure_ascii=False)
        )

    patched = set(baseline["locally_patched_files"])
    mismatched_sources = {}
    for relative, expected in baseline["bird_interact_agent_sha256"].items():
        if relative in patched:
            continue
        actual = sha256_file(official_root / relative)
        if actual != expected:
            mismatched_sources[relative] = {"expected": expected, "actual": actual}
    if mismatched_sources:
        raise RuntimeError(
            "Unapproved deviation from official source baseline: "
            + json.dumps(mismatched_sources, ensure_ascii=False)
        )
    return {
        "manifest": str(OFFICIAL_BASELINE_PATH),
        "manifest_sha256": sha256_file(OFFICIAL_BASELINE_PATH),
        "repository": baseline["repository"],
        "commit": baseline["commit"],
        "required_dependency_versions": required_versions,
        "installed_dependency_versions": installed_versions,
        "verified_unchanged_source_count": (
            len(baseline["bird_interact_agent_sha256"]) - len(patched & set(baseline["bird_interact_agent_sha256"]))
        ),
        "locally_patched_files": baseline["locally_patched_files"],
        "patch_scope": baseline["patch_scope"],
    }


def token_count(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding(SUBMISSION_TOKEN_ENCODING).encode(text or ""))


def numbered_raw_agent_calls(task_dir: Path) -> dict[int, dict[str, Any]]:
    calls = {}
    pattern = re.compile(r"\.agent_raw_turn_(\d+)\.jsonl$")
    for path in task_dir.glob("results.jsonl.agent_raw_turn_*.jsonl"):
        match = pattern.search(path.name)
        if match is None:
            continue
        rows = load_jsonl(path)
        if len(rows) != 1:
            raise RuntimeError(f"Expected one system call in {path}, got {len(rows)}")
        calls[int(match.group(1))] = rows[0]
    return calls


def turns_with_logical_phase(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Repair the upstream history label written after a P1 transition.

    The upstream runner changes ``current_phase`` to 2 before it appends the
    successful P1 submit turn.  Its evaluator result is correct, but that one
    history label is not.  Reconstructing the phase from the success reward is
    required to export final P1/P2 SQL into the official submission fields.
    """

    logical_phase = 1
    result = []
    for original in status.get("interaction_history", []):
        turn = dict(original)
        turn["logical_phase"] = logical_phase
        result.append(turn)
        action = str(turn.get("action") or "")
        observation = str(turn.get("observation") or "")
        reward = float(turn.get("reward", 0.0) or 0.0)
        if (
            logical_phase == 1
            and action.startswith("submit(")
            and (reward >= 0.7 or "Phase 1 SQL Correct!" in observation)
        ):
            logical_phase = 2
    return result


def build_submission_prompt_flow(
    status: dict[str, Any],
    task_dir: Path,
    *,
    system_model: str,
    user_model: str,
) -> list[dict[str, Any]]:
    """Build the sanitized per-system-turn trace requested by BIRD."""

    raw_calls = numbered_raw_agent_calls(task_dir)
    prompt_flow = []
    previous_remaining = float(status.get("total_budget", 0.0) or 0.0)
    for turn in turns_with_logical_phase(status):
        turn_number = int(turn.get("turn", 0) or 0)
        raw = raw_calls.get(turn_number)
        if raw is None:
            raise RuntimeError(f"Missing raw system call for turn {turn_number}")
        budget = turn.get("budget_after_action") or {}
        remaining = float(budget.get("remaining_budget", previous_remaining))
        action = str(turn.get("action") or "")
        action_name = action.split("(", 1)[0] if "(" in action else action
        action_input = (
            parse_action_arg(action, action_name)
            if action_name in TOOL_COSTS and action
            else ""
        )
        observation = str(turn.get("observation") or "")
        response_audit = raw.get("response_audit") or {}
        action_executed = not (
            response_audit.get("terminal_model_failure")
            or "Budget depleted. Agent failed to submit" in observation
        )
        prompt_flow.append({
            "step": turn_number,
            "phase": int(turn["logical_phase"]),
            "model": SUBMISSION_MODEL_NAME,
            "model_display_name": system_model,
            "user_simulator": user_model,
            "prompt": str(raw.get("prompt") or ""),
            "response": str(raw.get("response") or ""),
            "provider_visible_response": str(
                response_audit.get("provider_visible_content", raw.get("response") or "")
            ),
            "reasoning_content": str(raw.get("reasoning_content") or ""),
            "control_response_source": response_audit.get(
                "control_response_source", "content"
            ),
            "action": action,
            "action_executed": action_executed,
            "observation": observation,
            "remaining_budget": remaining,
            "action_input_tokens": token_count(action_input),
            "action_output_tokens": token_count(observation),
            "action_cost": round(previous_remaining - remaining, 10),
            "model_token_usage": raw.get("token_usage") or {},
            "finish_reason": response_audit.get("finish_reason"),
            "outcome_classification": response_audit.get(
                "outcome_classification", "completed_response"
            ),
        })
        previous_remaining = remaining
    if len(prompt_flow) != len(status.get("interaction_history", [])):
        raise RuntimeError("prompt_flow length does not match interaction history")
    return prompt_flow


def git_metadata(path: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    commit = run("rev-parse", "--verify", "HEAD")
    root = run("rev-parse", "--show-toplevel")
    status = run("status", "--porcelain")
    return {
        "root": root.stdout.strip() if root.returncode == 0 else None,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def parse_action_arg(action: str, prefix: str) -> str:
    match = re.search(rf"{re.escape(prefix)}\((.*)\)", action, re.DOTALL)
    raw = match.group(1).strip() if match else ""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, str):
            return parsed
    except Exception:
        pass
    for quote in ('"""', "'''", '"', "'"):
        if raw.startswith(quote) and raw.endswith(quote):
            return raw[len(quote):-len(quote)]
    return raw


def summarize_raw_calls(task_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = {
        "model_calls": 0,
        "system_agent_calls": 0,
        "user_simulator_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    infrastructure_errors: list[dict[str, Any]] = []
    for path in sorted(task_dir.glob("results.jsonl.*_raw_turn_*.jsonl")):
        role = "system_agent" if ".agent_raw_" in path.name else "user_simulator"
        for row_index, row in enumerate(load_jsonl(path), 1):
            totals["model_calls"] += 1
            totals[f"{role}_calls"] += 1
            usage = row.get("token_usage") or {}
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_tokens",
                "reasoning_tokens",
            ):
                totals[key] += int(usage.get(key, 0) or 0)
            if not row.get("response_audit"):
                infrastructure_errors.append({
                    "file": str(path),
                    "row": row_index,
                    "role": role,
                    "response": str(row.get("response", ""))[:1000],
                    "reason": "frozen provider call has no response_audit",
                })
    return totals, infrastructure_errors


def reset_database(database: str, host: str, port: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PGPASSWORD"] = "123123"
    commands = [
        [
            "psql", "-h", host, "-p", str(port), "-U", "root", "-d", "postgres",
            "-c",
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{database}' AND pid <> pg_backend_pid();",
        ],
        ["dropdb", "--if-exists", "-h", host, "-p", str(port), "-U", "root", database],
        [
            "createdb", "-h", host, "-p", str(port), "-U", "root", database,
            "--template", f"{database}_template",
        ],
    ]
    for command in commands:
        subprocess.run(
            command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=True,
        )
    return {"verified": True, "database": database, "completed_at": utc_now()}


def task_envelope(
    global_index: int,
    record: dict[str, Any],
    status: dict[str, Any],
    task_dir: Path,
    cleanup: dict[str, Any],
    system_model: str,
    user_model: str,
) -> dict[str, Any]:
    phase_submissions: dict[int, list[str]] = {1: [], 2: []}
    for turn in turns_with_logical_phase(status):
        action = str(turn.get("action") or "")
        phase = int(turn["logical_phase"])
        if action.startswith("submit(") and phase in phase_submissions:
            phase_submissions[phase].append(parse_action_arg(action, "submit"))

    final_phase_submissions = {
        phase: [submissions[-1]] if submissions else []
        for phase, submissions in phase_submissions.items()
    }

    token_usage, infrastructure_errors = summarize_raw_calls(task_dir)
    if infrastructure_errors:
        raise RuntimeError(
            "Provider infrastructure error detected: "
            + json.dumps(infrastructure_errors[0], ensure_ascii=False)
        )

    prompt_flow = build_submission_prompt_flow(
        status,
        task_dir,
        system_model=system_model,
        user_model=user_model,
    )
    return {
        "global_index": global_index,
        "instance_id": record["instance_id"],
        "selected_database": record["selected_database"],
        "model_completed_at": utc_now(),
        "cleanup": cleanup,
        "result": {
            "framework": "official-text-react-non-adk",
            "phase1_passed": bool(status.get("phase1_completed")),
            "phase2_passed": bool(status.get("phase2_completed")),
            "total_reward": (
                0.7 * int(bool(status.get("phase1_completed")))
                + 0.3 * int(bool(status.get("phase2_completed")))
            ),
            "subtask_1_predicted_sql": final_phase_submissions[1],
            "subtask_2_predicted_sql": final_phase_submissions[2],
            "all_submitted_sql": {
                "phase1": phase_submissions[1],
                "phase2": phase_submissions[2],
            },
            "token_usage": token_usage,
            "prompt_flow": prompt_flow,
            "official_status": status,
            "raw_call_directory": str(task_dir),
        },
    }


def load_envelopes(run_dir: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((run_dir / "logs" / "task_results").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        result[int(value["global_index"])] = value
    return result


def enforce_p1_gate(
    run_dir: Path,
    manifest: list[dict[str, Any]],
    envelopes: dict[int, dict[str, Any]],
    *,
    gate_count: int,
    threshold: float,
) -> bool:
    """Return True when the run must pause before the next task.

    The gate is deliberately outside the frozen model configuration SHA: it
    controls only whether another task starts and does not alter any evaluated
    request, prompt, tool, budget, GT, or evaluator behavior.
    """

    if gate_count <= 0 or len(envelopes) < gate_count:
        return False
    expected_indices = [item["global_index"] for item in manifest[:gate_count]]
    if not all(index in envelopes for index in expected_indices):
        raise RuntimeError("P1 gate cannot evaluate a non-contiguous prefix")

    p1_count = sum(
        int(bool(envelopes[index]["result"]["phase1_passed"]))
        for index in expected_indices
    )
    rate = p1_count / gate_count
    should_pause = rate < threshold
    override = os.environ.get("ALLOW_LOW_P1_CONTINUE") == "1"
    report = {
        "evaluated_at": utc_now(),
        "global_index_range": [expected_indices[0], expected_indices[-1]],
        "completed_count": gate_count,
        "phase1_count": p1_count,
        "phase1_rate": rate,
        "threshold": threshold,
        "comparison": "phase1_rate < threshold",
        "decision": (
            "continue_override"
            if should_pause and override
            else "pause_and_investigate"
            if should_pause
            else "continue"
        ),
        "override_environment": override,
    }
    atomic_json(run_dir / "p1_gate_after_0100.json", report)
    if not should_pause or override:
        return False

    for name in ("checkpoint.json", "summary.json"):
        path = run_dir / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "paused_low_p1"
        value["updated_at"] = report["evaluated_at"]
        value["p1_gate"] = report
        if name == "checkpoint.json":
            value["in_progress"] = None
        atomic_json(path, value)
    print(
        f"[{utc_now()}] P1 gate paused run after {gate_count} tasks: "
        f"{p1_count}/{gate_count}={rate:.2%} < {threshold:.2%}",
        flush=True,
    )
    return True


def write_state(
    run_dir: Path,
    manifest: list[dict[str, Any]],
    envelopes: dict[int, dict[str, Any]],
    config: dict[str, Any],
    config_sha: str,
    input_path: Path,
    input_sha: str,
    started_at: str,
    in_progress: int | None,
    infrastructure_error: dict[str, Any] | None = None,
) -> None:
    ordered = [envelopes[i] for i in sorted(envelopes)]
    predictions = []
    official_submission = []
    totals = {
        "completed_tasks": len(ordered),
        "phase1_count": 0,
        "phase2_count": 0,
        "total_reward": 0.0,
        "token_usage": {
            key: 0
            for key in (
                "model_calls", "system_agent_calls", "user_simulator_calls",
                "input_tokens", "output_tokens", "total_tokens",
                "cached_tokens", "reasoning_tokens",
            )
        },
    }
    private_index = []
    for envelope in ordered:
        result = envelope["result"]
        predictions.append({
            "instance_id": envelope["instance_id"],
            "subtask_1_predicted_sql": result["subtask_1_predicted_sql"],
            "subtask_2_predicted_sql": result["subtask_2_predicted_sql"],
        })
        official_submission.append({
            "instance_id": envelope["instance_id"],
            "subtask_1_predicted_sql": result["subtask_1_predicted_sql"],
            "subtask_2_predicted_sql": result["subtask_2_predicted_sql"],
            "prompt_flow": result["prompt_flow"],
        })
        totals["phase1_count"] += int(result["phase1_passed"])
        totals["phase2_count"] += int(result["phase2_passed"])
        totals["total_reward"] += float(result["total_reward"])
        for key in totals["token_usage"]:
            totals["token_usage"][key] += int(result["token_usage"].get(key, 0) or 0)
        private_index.append({
            "global_index": envelope["global_index"],
            "instance_id": envelope["instance_id"],
            "selected_database": envelope["selected_database"],
            "phase1_passed": result["phase1_passed"],
            "phase2_passed": result["phase2_passed"],
            "token_usage": result["token_usage"],
            "task_result": f"logs/task_results/{envelope['global_index']:04d}.json",
        })
    count = len(ordered)
    totals["phase1_rate"] = totals["phase1_count"] / count if count else 0.0
    totals["phase2_rate"] = totals["phase2_count"] / count if count else 0.0
    totals["average_reward"] = totals["total_reward"] / count if count else 0.0

    atomic_text(
        run_dir / "submission_predictions.jsonl",
        "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in predictions),
    )
    atomic_text(
        run_dir / "submission_official.jsonl",
        "".join(
            json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n"
            for x in official_submission
        ),
    )
    atomic_json(run_dir / "result_private_index.json", {
        "storage": "logs/task_results/{global_index:04d}.json",
        "results": private_index,
        "metrics": totals,
    })
    completed_indices = [item["global_index"] for item in ordered]
    remaining = [item["global_index"] for item in manifest if item["global_index"] not in envelopes]
    status = "complete" if not remaining and in_progress is None else "in_progress"
    checkpoint = {
        "status": status,
        "updated_at": utc_now(),
        "started_at": started_at,
        "global_index_range": [manifest[0]["global_index"], manifest[-1]["global_index"]],
        "expected_count": len(manifest),
        "completed_count": count,
        "completed_global_indices": completed_indices,
        "completed_instance_ids": [item["instance_id"] for item in ordered],
        "remaining_global_indices": remaining,
        "next_global_index": remaining[0] if remaining else None,
        "in_progress": in_progress,
        "last_infrastructure_error": infrastructure_error,
        "full_input_sha256": input_sha,
        "configuration_sha256": config_sha,
    }
    atomic_json(run_dir / "checkpoint.json", checkpoint)
    prediction_sha = sha256_file(run_dir / "submission_predictions.jsonl")
    official_submission_sha = sha256_file(run_dir / "submission_official.jsonl")
    atomic_json(run_dir / "summary.json", {
        "status": status,
        "started_at": started_at,
        "updated_at": checkpoint["updated_at"],
        "full_input": {"path": str(input_path), "record_count": 600, "sha256": input_sha},
        "configuration": config,
        "configuration_sha256": config_sha,
        "global_index_range": checkpoint["global_index_range"],
        "expected_task_count": len(manifest),
        "completed_task_count": count,
        "instance_ids": [item["instance_id"] for item in manifest],
        "completed_instance_ids": checkpoint["completed_instance_ids"],
        "prediction_file_sha256": prediction_sha,
        "official_submission_file_sha256": official_submission_sha,
        "metrics": totals,
    })
    if status == "complete":
        atomic_json(run_dir / "result_private.json", {
            "mode": "a-interact",
            "leaderboard_mode": "stress",
            "framework": "official-text-react-non-adk",
            "metrics": totals,
            "results": ordered,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--start", type=int, default=301)
    parser.add_argument("--end", type=int, default=600)
    parser.add_argument(
        "--p1-gate-after",
        type=int,
        default=int(os.environ.get("P1_GATE_AFTER", P1_GATE_DEFAULT_COUNT)),
    )
    parser.add_argument(
        "--p1-gate-threshold",
        type=float,
        default=float(
            os.environ.get("P1_GATE_THRESHOLD", P1_GATE_DEFAULT_THRESHOLD)
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    input_path = Path(args.input).resolve()
    official_root = Path(args.official_root).resolve()
    if args.start < 1 or args.end > 600 or args.start > args.end:
        raise RuntimeError("Invalid global-index range")
    official_baseline = validate_official_baseline(official_root)

    from shared.config import (
        active_model_preset_report,
        normalized_system_agent_config,
        settings,
    )
    preset = active_model_preset_report()
    if preset is None or preset["name"] != "glm47_matched_32768":
        raise RuntimeError("MODEL_PRESET=glm47_matched_32768 is required")
    if settings.dataset != "full" or settings.user_sim_model != "anthropic/claude-haiku-4-5-20251001":
        raise RuntimeError("Frozen Full/User Simulator configuration mismatch")
    if not settings.user_sim_disable_thinking:
        raise RuntimeError("USER_SIM_DISABLE_THINKING=true is required")

    records = load_jsonl(input_path)
    if len(records) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(f"Expected 600 records, got {len(records)}")
    selected = records[args.start - 1:args.end]
    manifest = [
        {
            "global_index": index,
            "instance_id": record["instance_id"],
            "selected_database": record["selected_database"],
        }
        for index, record in enumerate(selected, args.start)
    ]
    if len(manifest) != args.end - args.start + 1:
        raise RuntimeError("Range length mismatch")
    if [item["global_index"] for item in manifest] != list(range(args.start, args.end + 1)):
        raise RuntimeError("Global indices are not continuous")
    if len({item["instance_id"] for item in manifest}) != len(manifest):
        raise RuntimeError("instance_id values are not unique")

    source_files = [
        "experiments/utils/prompts.py",
        "src/envs/user_simulator/prompts.py",
        "batch_run_bird_interact/main.py",
        "batch_run_bird_interact/prompt_utils.py",
        "batch_run_bird_interact/action_handler.py",
        "batch_run_bird_interact/sample_status.py",
        "src/envs/bird_interact_env/test_case_utils/db_utils.py",
        "src/envs/bird_interact_env/test_case_utils/test_utils.py",
        "src/llm_utils/call_api_batch.py",
        "src/llm_utils/frozen_provider.py",
    ]
    source_hashes = {name: sha256_file(official_root / name) for name in source_files}
    supervisor_files = [
        "shared/config.py",
        "shared/llm.py",
        "shared/model_presets.py",
        "scripts/preflight_official_nonadk.py",
        "scripts/run_official_nonadk_range.py",
        "scripts/validate_nonadk_submission.py",
        "scripts/run_official_nonadk_full_glm47.sh",
        "configs/model_presets/glm47_matched_32768.json",
        "configs/official_nonadk_baseline.json",
        "configs/official_nonadk_requirements.txt",
    ]
    supervisor_hashes = {
        name: sha256_file(ADK_ROOT / name) for name in supervisor_files
    }
    config = {
        "dataset": "full",
        "framework": "official-text-react-non-adk",
        "interaction_mode": "a-interact",
        "leaderboard_mode": "stress",
        "concurrency": 1,
        "system_agent": normalized_system_agent_config(),
        "model_preset": {
            "name": preset["name"],
            "normalized_sha256": preset["normalized_sha256"],
            "file_sha256": preset["file_sha256"],
        },
        "user_simulator": {
            "model": settings.user_sim_model,
            "mode": "encoder_decoder",
            "prompt_version": "v2",
            "provider_hidden_thinking_disabled": True,
            "encoder_max_tokens": 6000,
            "decoder_max_tokens": 6000,
        },
        "budget": {
            "user_patience_budget": 6,
            "formula": "6 + 2 * ambiguity_count + user_patience_budget",
            "average_full_stress_budget": 17.86,
            "tool_costs": TOOL_COSTS,
        },
        "max_turns": 60,
        "native_api_tools": False,
        "tool_choice_note": "Sent as frozen request field; nine tools use official text-ReAct actions.",
        "source_sha256": source_hashes,
        "supervisor_sha256": supervisor_hashes,
        "provider_endpoints": {
            "system_agent_api_base": settings.system_agent_api_base,
            "user_simulator_api_base": settings.user_sim_api_base,
        },
        "official_upstream_baseline": official_baseline,
        "submission_trace": {
            "file": "submission_official.jsonl",
            "model_identifier_per_guideline": SUBMISSION_MODEL_NAME,
            "action_token_encoding": SUBMISSION_TOKEN_ENCODING,
            "contains_user_simulator_private_prompts": False,
        },
        "git": {
            "adk": git_metadata(ADK_ROOT),
            "official_nonadk": git_metadata(official_root),
        },
    }
    config_sha = canonical_sha256(config)
    input_sha = sha256_file(input_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "tasks").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "task_results").mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in manifest
    )
    manifest_path = run_dir / "shard_manifest.jsonl"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise RuntimeError("Existing shard manifest mismatch")
    atomic_text(manifest_path, manifest_text)
    preset_source = ADK_ROOT / "configs" / "model_presets" / "glm47_matched_32768.json"
    preset_copy = run_dir / "model_preset.json"
    if preset_copy.exists() and preset_copy.read_bytes() != preset_source.read_bytes():
        raise RuntimeError("Existing run directory has a different frozen model preset")
    if not preset_copy.exists():
        shutil.copy2(preset_source, preset_copy)
    started_at_path = run_dir / ".started_at"
    if started_at_path.exists():
        started_at = started_at_path.read_text(encoding="utf-8").strip()
    else:
        started_at = utc_now()
        atomic_text(started_at_path, started_at + "\n")
    run_manifest = {
        "schema_version": 1,
        "created_at": started_at,
        "result_directory": str(run_dir),
        "full_input_sha256": input_sha,
        "configuration": config,
        "configuration_sha256": config_sha,
        "selection": {
            "global_index_range": [args.start, args.end],
            "python_slice": f"records[{args.start - 1}:{args.end}]",
            "record_count": len(manifest),
        },
    }
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        existing_manifest = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        immutable_fields = (
            "full_input_sha256",
            "configuration_sha256",
            "selection",
        )
        mismatches = {
            field: {
                "existing": existing_manifest.get(field),
                "current": run_manifest.get(field),
            }
            for field in immutable_fields
            if existing_manifest.get(field) != run_manifest.get(field)
        }
        if mismatches:
            raise RuntimeError(
                "Refusing to mix configurations in an existing run directory: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
    else:
        atomic_json(run_manifest_path, run_manifest)

    envelopes = load_envelopes(run_dir)
    write_state(
        run_dir, manifest, envelopes, config, config_sha, input_path, input_sha,
        started_at, None,
    )
    if enforce_p1_gate(
        run_dir,
        manifest,
        envelopes,
        gate_count=args.p1_gate_after,
        threshold=args.p1_gate_threshold,
    ):
        return
    # Keep the virtual-environment launcher path. Resolving its symlink would
    # invoke the system interpreter and lose the venv's site-packages.
    python = Path(sys.executable)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join((str(official_root), str(ADK_ROOT)))

    for manifest_item, record in zip(manifest, selected):
        global_index = manifest_item["global_index"]
        if global_index in envelopes:
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["instance_id"])
        task_dir = run_dir / "logs" / "tasks" / f"{global_index:04d}_{safe_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_dir / "results.jsonl"
        log_path = task_dir / "experiment.log"
        console_path = task_dir / "console.log"
        write_state(
            run_dir, manifest, envelopes, config, config_sha, input_path, input_sha,
            started_at, global_index,
        )
        command = [
            str(python), str(official_root / "batch_run_bird_interact" / "main.py"),
            "--data_path", str(input_path),
            "--output_path", str(result_path),
            "--agent_model", settings.system_agent_model,
            "--user_model", settings.user_sim_model,
            "--user_sim_mode", "encoder_decoder",
            "--user_sim_prompt_version", "v2",
            "--user_patience_budget", "6",
            "--max_turns", "60",
            "--num_threads", "1",
            "--user_num_threads", "1",
            "--start_index", str(global_index - 1),
            "--limit", "1",
            "--db_host", settings.pg_host,
            "--db_port", str(settings.pg_port),
            "--log_level", "DEBUG",
            "--log_file", str(log_path),
            "--verbose",
        ]
        if result_path.exists():
            command.append("--resume")
        atomic_json(task_dir / "command.json", {
            "argv": command,
            "started_at": utc_now(),
            "global_index": global_index,
            "instance_id": record["instance_id"],
        })
        print(f"[{utc_now()}] starting global_index={global_index} instance_id={record['instance_id']}", flush=True)
        with console_path.open("a", encoding="utf-8") as console:
            completed = subprocess.run(
                command,
                cwd=official_root,
                env=child_env,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            error = {
                "global_index": global_index,
                "instance_id": record["instance_id"],
                "returncode": completed.returncode,
                "console": str(console_path),
                "detected_at": utc_now(),
            }
            write_state(
                run_dir, manifest, envelopes, config, config_sha, input_path,
                input_sha, started_at, None, error,
            )
            raise RuntimeError(f"Official runner failed: {error}")
        statuses = load_jsonl(result_path)
        if len(statuses) != 1:
            raise RuntimeError(f"Expected one status for task {global_index}, got {len(statuses)}")
        try:
            cleanup = reset_database(
                record["selected_database"], settings.pg_host, settings.pg_port
            )
            envelope = task_envelope(
                global_index,
                record,
                statuses[0],
                task_dir,
                cleanup,
                settings.system_agent_model,
                settings.user_sim_model,
            )
        except Exception as exc:
            error = {
                "global_index": global_index,
                "instance_id": record["instance_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "detected_at": utc_now(),
            }
            write_state(
                run_dir, manifest, envelopes, config, config_sha, input_path,
                input_sha, started_at, None, error,
            )
            raise
        result_file = run_dir / "logs" / "task_results" / f"{global_index:04d}.json"
        atomic_json(result_file, envelope)
        envelopes[global_index] = envelope
        write_state(
            run_dir, manifest, envelopes, config, config_sha, input_path,
            input_sha, started_at, None,
        )
        print(
            f"[{utc_now()}] completed {global_index}: "
            f"p1={int(envelope['result']['phase1_passed'])} "
            f"p2={int(envelope['result']['phase2_passed'])}",
            flush=True,
        )
        if enforce_p1_gate(
            run_dir,
            manifest,
            envelopes,
            gate_count=args.p1_gate_after,
            threshold=args.p1_gate_threshold,
        ):
            return


if __name__ == "__main__":
    main()
