#!/usr/bin/env python3
"""Exercise the real two-stage User Simulator protocol on Full task 451."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shared.audit import summarize_usage, utc_now
from shared.config import PROJECT_ROOT, settings
from user_simulator.server import (
    PROTOCOL_MAX_ATTEMPTS,
    PROTOCOL_POLICY,
    TaskSimState,
    _generate_response,
    _parse_action,
)


TASK_INDEX = 451
QUESTION = (
    'What do you mean by "really difficult situations"? Should I use:\n'
    "1. High hazard levels (Severity-4 or Severity-5) from disaster events\n"
    "2. High emergency levels (Black or Red) from operations\n"
    "3. A combination of both high hazard and high emergency levels"
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = PROJECT_ROOT.parent / "Dataset" / "bird-interact-full" / "bird_interact_data.jsonl"
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 600:
        raise RuntimeError(f"Expected 600 Full records, found {len(records)}")
    task = records[TASK_INDEX - 1]
    if task.get("instance_id") != "disaster_relief_12":
        raise RuntimeError("Full task 451 identity changed")

    state = TaskSimState(task)
    schema_path = (
        PROJECT_ROOT.parent
        / "Dataset"
        / "bird-interact-full"
        / task["selected_database"]
        / f"{task['selected_database']}_schema.txt"
    )
    state.db_schema = schema_path.read_text(encoding="utf-8")

    action = _parse_action(state, QUESTION)
    answer = _generate_response(state, QUESTION, action)
    report = {
        "status": "passed",
        "timestamp": utc_now(),
        "task": {
            "global_index": TASK_INDEX,
            "instance_id": task["instance_id"],
            "selected_database": task["selected_database"],
        },
        "user_simulator": {
            "model": settings.user_sim_model,
            "api_base": settings.user_sim_api_base.rstrip("/"),
            "prompt_version": settings.prompt_version,
            "protocol_policy": PROTOCOL_POLICY,
            "protocol_max_attempts": PROTOCOL_MAX_ATTEMPTS,
        },
        "question": QUESTION,
        "action": action,
        "answer": answer,
        "llm_calls": state.llm_calls,
        "token_usage": summarize_usage(state.llm_calls),
        "all_protocol_attempts_valid_or_retried": all(
            "protocol" in call for call in state.llm_calls
        ),
        "final_protocol_valid": {
            stage: any(
                call.get("stage") == stage
                and call.get("protocol", {}).get("valid") is True
                for call in state.llm_calls
            )
            for stage in ("action_parser", "response_generator")
        },
    }
    if not all(report["final_protocol_valid"].values()):
        raise RuntimeError("A User Simulator protocol stage did not validate")
    _atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "task": report["task"],
                "user_simulator": report["user_simulator"],
                "action": action,
                "answer": answer,
                "call_count": len(state.llm_calls),
                "token_usage": report["token_usage"],
                "final_protocol_valid": report["final_protocol_valid"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
