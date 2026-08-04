"""Export the audit result to BIRD-Interact's recommended prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    parser.add_argument("output_jsonl")
    args = parser.parse_args()

    source = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for result in source.get("results", []):
        record = {
            "instance_id": result.get("instance_id") or result.get("task_id"),
            "subtask_1_predicted_sql": result.get("subtask_1_predicted_sql", []),
            "subtask_2_predicted_sql": result.get("subtask_2_predicted_sql", []),
            "prompt_flow": result.get("prompt_flow", []),
        }
        if result.get("error"):
            record["error"] = result["error"]
        lines.append(json.dumps(record, ensure_ascii=False, default=str))

    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
