import asyncio
import copy
import inspect
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent import server as valibra_server
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.service import (
    process_observation_with_llm,
)
from valibra_agent.requirement_grounding.updater import (
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    LLMUpdater,
    load_grounding_llm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROMPT_SHA256 = (
    "ddaaa23fd3c824a704ea1e17a769b4c8b1af5c998949fccfe8d564b18f78f7c4"
)
EXPECTED_FORM_SHA256 = (
    "f7409a7267b3fddcb40d69574e6187320884951ad4e96980bb590ee0058c38d1"
)
PRE_P71C_PROMPT_SHA256 = (
    "5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d"
)
PRE_P71C_FORM_SHA256 = (
    "441a59c410a99ef0db53b8e974aeeeaea1bcd1735aabdc3cb51f59e0b6e069a2"
)


def _llm_config(*, timeout="0.05", max_calls="4"):
    return load_grounding_llm_config(
        PROJECT_ROOT,
        {
            "GROUNDING_UPDATER_MODE": "llm",
            "GROUNDING_MODEL_PRESET": "glm52_high_32768",
            "GROUNDING_TIMEOUT_SECONDS": timeout,
            "GROUNDING_MAX_TOKENS": "32768",
            "GROUNDING_MAX_CALLS_PER_TASK": max_calls,
            "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
        },
    )


def _content(
    *,
    value_mention=None,
    schema_mention=None,
    operation_mention=None,
    operation_type="order",
):
    value_slots = []
    if value_mention is not None:
        value_slots.append(
            {
                "slot_role": "time_constraint",
                "mention": value_mention,
                "interpretation": f"calendar year {value_mention}",
                "value_type": "time",
            }
        )
    schema_slots = []
    if schema_mention is not None:
        schema_slots.append(
            {
                "slot_role": "schema_candidate",
                "mention": schema_mention,
                "interpretation": f"candidate concept {schema_mention}",
            }
        )
    operation_slots = []
    if operation_mention is not None:
        operation_slots.append(
            {
                "slot_role": "ordering",
                "mention": operation_mention,
                "interpretation": "descending result order",
                "operation_type": operation_type,
                "parameters": {"direction": "desc"},
            }
        )
    return json.dumps(
        {
            "proposal_outcome": (
                "populated"
                if value_slots or schema_slots or operation_slots
                else "no_extractable_requirement"
            ),
            "value_slots": value_slots,
            "schema_slots": schema_slots,
            "operation_slots": operation_slots,
            "ambiguities": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


QUERY = "Show customer names from 2024 sorted by total descending."
QUERY_CONTENT = _content(
    value_mention="2024",
    schema_mention="customer names",
    operation_mention="sorted by total descending",
)
ANSWER = "Use 2023 instead."
ANSWER_CONTENT = _content(value_mention="2023")


class FakeClient:
    provider_may_continue_after_cancel = False
    provider_may_bill_after_cancel = False

    def __init__(self, *, content=QUERY_CONTENT, error=None, block=False):
        self.content = content
        self.error = error
        self.block = block
        self.calls = 0
        self.requests = []

    async def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        await asyncio.sleep(0)
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        content = self.content(request) if callable(self.content) else self.content
        return {
            "content": content,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 9,
                "reasoning_tokens": 4,
                "total_tokens": 21,
                "cost": None,
            },
            "model": "openai/glm-5.2",
            "provider": "offline-fake",
            "credential_source": "file",
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "raw_audit_ref": "private://grounding-llm/offline-fake.json",
        }


def _state(task_id="task-p4d", *, phase=1, budget=20.0):
    return {
        "task_id": task_id,
        "current_phase": phase,
        "budget_remaining": budget,
        "initial_budget": budget,
        "tool_trajectory": [],
        "system_agent_llm_calls": [{"actions": []}],
        "_active_llm_call_index": 0,
    }


def _context(state, call_id):
    return SimpleNamespace(
        state=state,
        function_call_id=call_id,
        invocation_id=f"inv-{call_id}",
    )


def _runtime(state):
    return RequirementGroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


async def _run_bound_query(state, message, request, *, model_calls=1):
    token = grounding_callbacks._bind_turn_message(
        state["task_id"],
        "a-interact",
        message,
    )
    try:
        result = None
        for _ in range(model_calls):
            result = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=state),
                request,
            )
        return result
    finally:
        grounding_callbacks._reset_turn_message(token)

if __name__ == "__main__":
    unittest.main()
