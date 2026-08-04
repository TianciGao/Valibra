"""Generate a compact Markdown index for a full audit result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _code_block(value: Any, language: str = "") -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f"```{language}\n{text}\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    parser.add_argument("output_markdown")
    args = parser.parse_args()

    result_path = Path(args.result_json)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    results = data.get("results", [])

    total_usage = {
        field: sum(
            int(item.get("token_usage", {}).get("combined", {}).get(field, 0) or 0)
            for item in results
        )
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "tool_prompt_tokens",
        )
    }
    system_calls = sum(len(item.get("prompt_flow", [])) for item in results)
    user_calls = sum(
        len(item.get("user_simulator_audit", {}).get("llm_calls", []))
        for item in results
    )
    max_token_calls = []
    for item in results:
        for call in item.get("prompt_flow", []):
            raw = call.get("raw_response") or {}
            if isinstance(raw, dict) and raw.get("finish_reason") == "MAX_TOKENS":
                max_token_calls.append((item.get("task_id"), call.get("call_index")))

    lines = [
        "# BIRD-Interact GLM-5.2 Full a-Interact audit",
        "",
        "## Run summary",
        "",
        f"- Tasks: {len(results)}",
        f"- System-model request records: {system_calls}",
        f"- User-simulator model calls: {user_calls}",
        f"- Combined provider-reported tokens: {total_usage['total_tokens']}",
        f"- Input/output tokens: {total_usage['input_tokens']} / {total_usage['output_tokens']}",
        f"- Cached/reasoning tokens: {total_usage['cached_tokens']} / {total_usage['reasoning_tokens']}",
        f"- Calls ending at `MAX_TOKENS`: {max_token_calls or 'none'}",
        "",
        "Provider-reported input totals may already include cached tokens; cached tokens are shown separately and are not added again.",
        "",
        "## Per-task index",
        "",
        "| Task | Reward | P1 | P2 | Budget used | System calls | User calls | Tools | Total tokens | Final finish |",
        "|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---|",
    ]

    for item in results:
        finishes = []
        for call in item.get("prompt_flow", []):
            raw = call.get("raw_response") or {}
            if isinstance(raw, dict) and raw.get("finish_reason"):
                finishes.append(raw["finish_reason"])
        lines.append(
            "| {task} | {reward} | {p1} | {p2} | {used}/{initial} | {sc} | {uc} | {tools} | {tokens} | {finish} |".format(
                task=_cell(item.get("task_id")),
                reward=item.get("total_reward", 0),
                p1="Y" if item.get("phase1_passed") else "N",
                p2="Y" if item.get("phase2_passed") else "N",
                used=item.get("budget_used", 0),
                initial=item.get("initial_budget", 0),
                sc=len(item.get("prompt_flow", [])),
                uc=len(item.get("user_simulator_audit", {}).get("llm_calls", [])),
                tools=len(item.get("tool_trajectory", [])),
                tokens=item.get("token_usage", {}).get("combined", {}).get("total_tokens", 0),
                finish=finishes[-1] if finishes else "",
            )
        )

    for item in results:
        task_id = item.get("task_id", "?")
        lines.extend([
            "",
            f"## {task_id}",
            "",
            f"- Elapsed: {item.get('elapsed_seconds', 0):.1f}s",
            f"- Final response: {_cell(item.get('final_response', ''))[:500]}",
            "",
            "### Submitted SQL",
            "",
            _code_block(item.get("all_submitted_sql", {}), "json"),
            "",
            "### Actual agent/user dialogue",
            "",
            _code_block(item.get("dialogue_history", []), "json"),
            "",
            "### System-model calls",
            "",
            "| Call | Input | Output | Cached | Reasoning | Finish | Actions | Budget after |",
            "|---:|---:|---:|---:|---:|---|---|---:|",
        ])
        for call in item.get("prompt_flow", []):
            usage = call.get("usage", {})
            raw = call.get("raw_response") or {}
            finish = raw.get("finish_reason", "") if isinstance(raw, dict) else ""
            actions = ", ".join(
                action.get("tool", "?")
                for action in call.get("actions", [])
                if isinstance(action, dict)
            )
            lines.append(
                f"| {call.get('call_index', '')} | {usage.get('input_tokens', 0)} | "
                f"{usage.get('output_tokens', 0)} | {usage.get('cached_tokens', 0)} | "
                f"{usage.get('reasoning_tokens', 0)} | {_cell(finish)} | "
                f"{_cell(actions)} | {call.get('remaining_budget', '')} |"
            )

        lines.extend([
            "",
            "### User-simulator calls",
            "",
            "| Call | Stage | Input | Output | Total | Latency (s) |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for call in item.get("user_simulator_audit", {}).get("llm_calls", []):
            usage = call.get("usage", {})
            lines.append(
                f"| {call.get('call_index', '')} | {_cell(call.get('stage', ''))} | "
                f"{usage.get('input_tokens', 0)} | {usage.get('output_tokens', 0)} | "
                f"{usage.get('total_tokens', 0)} | {call.get('latency_seconds', 0):.3f} |"
            )

    output_path = Path(args.output_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
