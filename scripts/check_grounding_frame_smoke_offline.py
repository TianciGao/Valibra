#!/usr/bin/env python3
"""Replay retained Grounding responses through the local fixed-form boundary.

This checker never imports or invokes a Provider client. It prints only bounded
validation metadata, never the retained response body.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from valibra_agent.requirement_grounding.models import RequirementGroundingRuntime
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.service import process_observation_with_llm
from valibra_agent.requirement_grounding.updater import (
    GroundingLLMResponse,
    LLMUpdater,
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    load_grounding_llm_config,
)


class _RetainedResponseClient:
    """In-memory replay client with no network or credential path."""

    provider_may_continue_after_cancel = False
    provider_may_bill_after_cancel = False

    def __init__(self, content: str, response_sha256: str) -> None:
        self.content = content
        self.response_sha256 = response_sha256
        self.calls = 0

    async def complete(self, request: Any) -> GroundingLLMResponse:
        self.calls += 1
        return GroundingLLMResponse(
            content=self.content,
            usage={},
            model=str(request.preset_config.get("model", "")),
            provider="offline-retained-response",
            credential_source="",
            response_sha256=self.response_sha256,
        )


def _audit_content(audit: dict[str, Any]) -> str:
    try:
        content = audit["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("retained audit has no response text") from exc
    if not isinstance(content, str):
        raise ValueError("retained audit response text is not a string")
    return content


def _raw_operation_types(content: str) -> list[Any]:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []
    slots = payload.get("operation_slots", []) if isinstance(payload, dict) else []
    if not isinstance(slots, list):
        return []
    return [
        slot.get("operation_type")
        for slot in slots
        if isinstance(slot, dict)
    ]


async def _replay(path: Path, question: str) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    content = _audit_content(audit)
    response_sha256 = str(audit.get("response_sha256", ""))
    client = _RetainedResponseClient(content, response_sha256)
    config = load_grounding_llm_config(
        PROJECT_ROOT,
        {
            "GROUNDING_UPDATER_MODE": "llm",
            "GROUNDING_MODEL_PRESET": "glm52_high_32768",
            "GROUNDING_TIMEOUT_SECONDS": "30",
            "GROUNDING_MAX_TOKENS": "32768",
            "GROUNDING_MAX_CALLS_PER_TASK": "2",
            "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
        },
    )
    observation = build_observation(
        task_id="p4c-offline-retained-smoke",
        observation_type="user_query",
        phase=1,
        sequence=1,
        source="retained_provider_response",
        raw=question,
        summary=question,
    )
    result = await process_observation_with_llm(
        RequirementGroundingRuntime(),
        observation,
        updater=LLMUpdater(client, config),
    )
    frame = result.runtime.grounding_state.requirement_frame
    return {
        "audit_file": path.name,
        "response_sha256": response_sha256,
        "raw_operation_types": _raw_operation_types(content),
        "offline_replay_calls": client.calls,
        "status": result.status,
        "grounding_revision": result.runtime.grounding_revision,
        "requirement_revision": result.runtime.requirement_revision,
        "processed_observation_count": len(result.runtime.processed_observation_ids),
        "slot_count": len(frame.value_slots) + len(frame.schema_slots) + len(frame.operation_slots),
        "ambiguity_count": len(result.runtime.grounding_state.ambiguity_index),
        "prompt_sha256": config.prompt_sha256,
        "form_schema_sha256": LLM_FRAME_FORM_SCHEMA_SHA256,
        "provider_requests": 0,
    }


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--expect-status",
        choices=("failed", "processed"),
        required=True,
    )
    args = parser.parse_args()
    results = [await _replay(path.resolve(), args.question) for path in args.audit]
    output = {
        "schema_version": "1.0",
        "mode": "offline_retained_response_replay",
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(item["status"] == args.expect_status for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
