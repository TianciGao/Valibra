#!/usr/bin/env python3
"""Make one secret-free User Simulator provider smoke call."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shared.config import settings
from shared.llm import call_llm_with_details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    details = call_llm_with_details(
        [
            {
                "role": "user",
                "content": "Reply with exactly: USER_SIM_PROVIDER_OK",
            }
        ],
        model_name=settings.user_sim_model,
        temperature=0,
        max_tokens=32,
    )
    content = str(details.get("content") or "").strip()
    if not content:
        raise RuntimeError("User Simulator provider returned empty content")

    report = {
        "status": "passed",
        "profile": os.environ.get("USER_SIM_PROFILE", ""),
        "model": settings.user_sim_model,
        "api_base": settings.user_sim_api_base.rstrip("/"),
        "credential_configured": bool(
            settings.user_sim_api_key or settings.user_sim_api_key_file
        ),
        "temperature": 0,
        "max_tokens": 32,
        "content": content,
        "usage": details.get("usage") or {},
        "latency_seconds": details.get("latency_seconds"),
        "provider": details.get("provider", ""),
        "timestamp": details.get("timestamp"),
        "raw_response": details.get("raw_response"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "raw_response"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
