"""Fail-closed preflight for the frozen Full a-Interact Stress run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

from scripts.run_official_nonadk_range import (
    ADK_ROOT,
    DEFAULT_OFFICIAL_ROOT,
    TOOL_COSTS,
    load_jsonl,
    sha256_file,
    validate_official_baseline,
)


EXPECTED_INPUT_SHA256 = (
    "a051f7a78462d6c17e840c048ea15c4be65b9f8eed61aad3d2df1370561b10c0"
)
EXPECTED_DATABASE = {
    "databases": 22,
    "tables": 244,
    "tables_with_data": 244,
    "columns": 2011,
    "rows": 273571,
}
EXPECTED_MODEL_CONFIG = {
    "model": "openai/glm-4.7",
    "thinking": {"type": "enabled", "clear_thinking": False},
    "max_tokens": 32768,
    "temperature": 0.0,
    "top_p": 0.95,
    "tool_choice": "auto",
}


def load_db_checker(path: Path):
    spec = importlib.util.spec_from_file_location("bird_official_db_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official DB checker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_database(host: str, port: int) -> dict[str, Any]:
    checker_path = ADK_ROOT.parent / "env" / "check_db_metadata.py"
    checker = load_db_checker(checker_path)
    connection = checker.connect_to_database(host, port, "root", "123123")
    databases = checker.get_database_list(connection)
    connection.close()
    metadata = [
        checker.get_database_metadata(host, port, "root", "123123", database)
        for database in databases
    ]
    if any(item is None for item in metadata):
        raise RuntimeError("Official DB metadata checker failed for at least one database")
    table_details = [
        table
        for database in metadata
        for table in database["table_details"]
    ]
    actual = {
        "databases": len(metadata),
        "tables": sum(item["tables"] for item in metadata),
        "tables_with_data": sum(
            int(table["estimated_rows"] > 0) for table in table_details
        ),
        "columns": sum(item["columns"] for item in metadata),
        "rows": sum(item["total_rows"] for item in metadata),
        "size_mb": round(sum(item["size_mb"] for item in metadata), 2),
        "database_names": sorted(item["database"] for item in metadata),
        "official_checker_sha256": sha256_file(checker_path),
    }
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in EXPECTED_DATABASE.items()
        if actual[key] != expected
    }
    if mismatches:
        raise RuntimeError(
            "Full database metadata mismatch: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return actual


def inspect_container_log() -> dict[str, Any]:
    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            "bird_interact_postgresql_full",
            "--format",
            "{{json .}}",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    container = json.loads(inspect.stdout)
    logs = subprocess.run(
        ["docker", "logs", "bird_interact_postgresql_full"],
        text=True,
        capture_output=True,
        check=True,
    )
    combined_logs = logs.stdout + logs.stderr
    marker = "Errors occurred during import:"
    if marker in combined_logs:
        raise RuntimeError(f"Database import error marker found: {marker}")
    return {
        "container_running": bool(container["State"]["Running"]),
        "container_image": container["Config"]["Image"],
        "container_image_digest": container["Image"],
        "import_error_marker": marker,
        "import_error_marker_count": combined_logs.count(marker),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ADK_ROOT / "bird-interact-full" / "bird_interact_data.jsonl"),
    )
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--db-host", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=5433)
    parser.add_argument("--output")
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()

    from shared.config import (
        active_model_preset_report,
        normalized_system_agent_config,
        settings,
    )

    failures = []
    input_path = Path(args.input).resolve()
    official_root = Path(args.official_root).resolve()
    records = load_jsonl(input_path)
    input_sha = sha256_file(input_path)
    if len(records) != 600:
        failures.append(f"Full input has {len(records)} rows, expected 600")
    if len({item.get("instance_id") for item in records}) != 600:
        failures.append("Full input instance_id values are not 600 unique values")
    if input_sha != EXPECTED_INPUT_SHA256:
        failures.append(f"Full input SHA256 mismatch: {input_sha}")

    preset = active_model_preset_report()
    model_config = normalized_system_agent_config()
    if preset is None or preset["name"] != "glm47_matched_32768":
        failures.append("MODEL_PRESET must be glm47_matched_32768")
    if model_config != EXPECTED_MODEL_CONFIG:
        failures.append(
            "Normalized model config mismatch: "
            + json.dumps(model_config, ensure_ascii=False, sort_keys=True)
        )
    if settings.dataset != "full":
        failures.append(f"DATASET must be full, got {settings.dataset}")
    if settings.user_sim_model != "anthropic/claude-haiku-4-5-20251001":
        failures.append(f"Unexpected User Simulator: {settings.user_sim_model}")
    if not settings.user_sim_disable_thinking:
        failures.append("User Simulator provider hidden thinking must be disabled")

    baseline = validate_official_baseline(official_root)
    container = None
    database = None
    if not args.skip_db:
        container = inspect_container_log()
        if not container["container_running"]:
            failures.append("Full PostgreSQL container is not running")
        database = inspect_database(args.db_host, args.db_port)

    report = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "automated_ready": not failures,
        "failures": failures,
        "full_input": {
            "path": str(input_path),
            "record_count": len(records),
            "unique_instance_id_count": len(
                {item.get("instance_id") for item in records}
            ),
            "sha256": input_sha,
        },
        "model_preset": preset,
        "normalized_model_config": model_config,
        "user_simulator": {
            "model": settings.user_sim_model,
            "prompt_version": "v2",
            "mode": "encoder_decoder",
            "provider_hidden_thinking_disabled": settings.user_sim_disable_thinking,
            "encoder_max_tokens": 6000,
            "decoder_max_tokens": 6000,
        },
        "evaluation": {
            "dataset": settings.dataset,
            "mode": "a-interact",
            "leaderboard_mode": "stress",
            "concurrency": 1,
            "user_patience_budget": 6,
            "budget_formula": "6 + 2 * ambiguity_count + user_patience_budget",
            "tool_costs": TOOL_COSTS,
            "max_turns": 60,
        },
        "official_upstream_baseline": baseline,
        "database_container": container,
        "database_metadata": database,
        "thinking_transport_disclosure": {
            "clear_thinking_request_value": False,
            "runner_transport": "official stateless full-text ReAct prompt per turn",
            "provider_reasoning_history_forwarded_as_chat_blocks": False,
            "note": (
                "The request field is exact, but Z.AI preserved-thinking chat "
                "history is not active in the upstream text-ReAct transport."
            ),
        },
        "manual_submission_requirements": [
            "Provide temporary model/API access to the BIRD team for validation.",
            "Disclose the third-party User Simulator gateway and permit identity validation.",
            "Obtain BIRD expert acceptance of the customized User Simulator (rejection rate below 15%).",
            "Send only submission_official.jsonl; never send result_private files containing GT/test cases.",
            "For a verified badge, provide the complete source and frozen environment for pipeline evaluation.",
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
