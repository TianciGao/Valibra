"""Write a reproducibility manifest and the exact selected task snapshot."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path

from shared.config import (
    active_model_preset_report,
    normalized_system_agent_config,
    settings,
)
from system_agent.callbacks import TOOL_COSTS


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--started-at", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    data_path = Path(args.data).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_lines = []
    selected_tasks = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            selected_lines.append(line.rstrip("\n"))
            selected_tasks.append(json.loads(line))
            if len(selected_tasks) >= args.limit:
                break

    snapshot_path = run_dir / "selected_tasks_with_ground_truth.jsonl"
    snapshot_path.write_text(
        "\n".join(selected_lines) + ("\n" if selected_lines else ""),
        encoding="utf-8",
    )

    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = {
        "started_at": args.started_at,
        "benchmark": "BIRD-Interact",
        "dataset": "full",
        "interaction_mode": "a-interact",
        "leaderboard_mode": "stress",
        "task_limit": args.limit,
        "concurrency": args.concurrency,
        "task_ids": [task.get("instance_id") for task in selected_tasks],
        "data_path": str(data_path),
        "data_sha256": data_sha256,
        "task_snapshot": str(snapshot_path),
        "models": {
            "system_agent": {
                **normalized_system_agent_config(),
                "api_base": settings.system_agent_api_base,
                "model_preset": active_model_preset_report(),
                "transport": {
                    "adapter": "LiteLLM OpenAI-compatible",
                    "provider_specific_parameters_via": "extra_body",
                },
            },
            "user_simulator": {
                "model": settings.user_sim_model,
                "api_base": settings.user_sim_api_base,
                "prompt_version": settings.prompt_version,
                "temperature": 0.0,
                "action_parser_max_tokens": 500,
                "response_generator_max_tokens": 1024,
                "provider_hidden_thinking_disabled": settings.user_sim_disable_thinking,
            },
        },
        "budget": {
            "formula": "6 + 2 * ambiguity_count + 2 * patience",
            "patience": settings.patience,
            "equivalent_user_patience_budget": 2 * settings.patience,
            "tool_costs": TOOL_COSTS,
        },
        "services": {
            "system_agent_port": settings.system_agent_port,
            "user_simulator_port": settings.user_sim_port,
            "db_environment_port": settings.db_env_port,
            "postgresql_host": settings.pg_host,
            "postgresql_port": settings.pg_port,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "google-adk",
                    "google-genai",
                    "litellm",
                    "pydantic",
                    "pydantic-settings",
                    "tiktoken",
                    "httpx",
                    "fastapi",
                )
            },
        },
        "audit_contents": [
            "service logs",
            "orchestrator log",
            "exact selected task snapshot (contains ground truth; local analysis only)",
            "system-agent ADK requests and raw responses",
            "ADK events and provider-reported token usage",
            "user-simulator prompts, raw responses, dialogue, and token usage",
            "tool arguments, full observations, token counts, costs, and SQL",
            "leaderboard-format prediction JSONL",
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
