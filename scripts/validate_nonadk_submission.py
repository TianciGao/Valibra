"""Validate a sanitized BIRD a-Interact submission before handoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


REQUIRED_TOP_LEVEL = {
    "instance_id",
    "subtask_1_predicted_sql",
    "subtask_2_predicted_sql",
    "prompt_flow",
}
REQUIRED_FLOW_FIELDS = {
    "model",
    "user_simulator",
    "prompt",
    "response",
    "action",
    "remaining_budget",
    "action_input_tokens",
    "action_output_tokens",
    "action_cost",
}
FORBIDDEN_KEYS = {
    "original_data",
    "sol_sql",
    "test_cases",
    "test_case",
    "preprocess_sql",
    "ground_truth",
    "gt_sql",
    "api_key",
    "authorization",
}
SECRET_PATTERNS = (
    re.compile(r"Authorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
OFFICIAL_ACTION_COSTS = {
    "ask": 2.0,
    "submit": 3.0,
    "execute": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_forbidden_keys(value: Any) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                found.add(str(key))
            found.update(nested_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(nested_forbidden_keys(child))
    return found


def validate_submission(
    submission_path: Path,
    manifest_path: Path,
    *,
    expected_count: int,
    expected_start: int,
    expected_end: int,
) -> dict[str, Any]:
    rows = load_jsonl(submission_path)
    manifest = load_jsonl(manifest_path)
    errors: list[str] = []

    if len(rows) != expected_count:
        errors.append(f"submission row count {len(rows)} != {expected_count}")
    if len(manifest) != expected_count:
        errors.append(f"manifest row count {len(manifest)} != {expected_count}")

    expected_indices = list(range(expected_start, expected_end + 1))
    actual_indices = [item.get("global_index") for item in manifest]
    if actual_indices != expected_indices:
        errors.append("manifest global_index range/order mismatch")

    ids = [str(row.get("instance_id") or "") for row in rows]
    manifest_ids = [str(row.get("instance_id") or "") for row in manifest]
    if ids != manifest_ids:
        errors.append("submission instance_id order does not match manifest")
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        errors.append("submission instance_id values are empty or not unique")

    for row_number, row in enumerate(rows, 1):
        if set(row) != REQUIRED_TOP_LEVEL:
            errors.append(
                f"row {row_number} top-level fields mismatch: {sorted(set(row))}"
            )
        for sql_field in ("subtask_1_predicted_sql", "subtask_2_predicted_sql"):
            sqls = row.get(sql_field)
            if not isinstance(sqls, list) or not all(isinstance(sql, str) for sql in sqls):
                errors.append(f"row {row_number} {sql_field} must be a list of strings")
        flow = row.get("prompt_flow")
        if not isinstance(flow, list) or not flow:
            errors.append(f"row {row_number} prompt_flow must be non-empty")
            continue
        for step_number, step in enumerate(flow, 1):
            missing = REQUIRED_FLOW_FIELDS - set(step)
            if missing:
                errors.append(
                    f"row {row_number} step {step_number} missing {sorted(missing)}"
                )
                continue
            if step.get("model") != "my_model":
                errors.append(f"row {row_number} step {step_number} model must be my_model")
            if not isinstance(step.get("prompt"), str) or not step["prompt"]:
                errors.append(f"row {row_number} step {step_number} prompt is empty")
            action = str(step.get("action") or "")
            action_name = action.split("(", 1)[0] if "(" in action else action
            actual_cost = float(step.get("action_cost", -1))
            expected_cost = OFFICIAL_ACTION_COSTS.get(action_name)
            observation = str(step.get("observation") or "")
            action_executed = bool(
                step.get(
                    "action_executed",
                    "Budget depleted. Agent failed to submit" not in observation
                    and not str(step.get("outcome_classification", "")).startswith(
                        ("context_length", "output_limit")
                    ),
                )
            )
            expected_actual_cost = expected_cost if action_executed else 0.0
            if (
                expected_actual_cost is not None
                and abs(actual_cost - expected_actual_cost) > 1e-9
            ):
                errors.append(
                    f"row {row_number} step {step_number} action_cost "
                    f"{actual_cost} != {expected_actual_cost} for {action_name} "
                    f"(executed={action_executed})"
                )
            for token_field in ("action_input_tokens", "action_output_tokens"):
                value = step.get(token_field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(
                        f"row {row_number} step {step_number} invalid {token_field}"
                    )

    forbidden = nested_forbidden_keys(rows)
    if forbidden:
        errors.append(f"forbidden/private keys found: {sorted(forbidden)}")
    raw_text = submission_path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        if pattern.search(raw_text):
            errors.append(f"possible secret matched pattern: {pattern.pattern}")

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "submission": str(submission_path.resolve()),
        "submission_sha256": sha256_file(submission_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_count": expected_count,
        "row_count": len(rows),
        "unique_instance_id_count": len(set(ids)),
        "global_index_range": [expected_start, expected_end],
        "secret_scan_passed": not any("secret" in error for error in errors),
        "private_key_scan_passed": not bool(forbidden),
        "valid": not errors,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=600)
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.end - args.start + 1 != args.expected_count:
        raise RuntimeError("expected-count does not match start/end range")
    report = validate_submission(
        Path(args.submission),
        Path(args.manifest),
        expected_count=args.expected_count,
        expected_start=args.start,
        expected_end=args.end,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
