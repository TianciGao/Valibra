import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import unittest
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from shared.audit import to_jsonable
from system_agent import callbacks as baseline_callbacks
from system_agent.agent import AINTERACT_INSTRUCTION
from valibra_agent import agent as valibra_agent_module
from valibra_agent import grounding_callbacks
from valibra_agent import server as valibra_server
from valibra_agent.adk_runtime import AdkRuntime
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.updater import (
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    LLMUpdater,
)
from tests.valibra.test_p4d_llm_callback_shadow import (
    FakeClient,
    QUERY,
    QUERY_CONTENT,
    _llm_config,
)
from tests.valibra.test_p4h_agent_requirement_view_shadow import _semantic_state


EXPECTED_PROMPT_SHA256 = (
    "ddaaa23fd3c824a704ea1e17a769b4c8b1af5c998949fccfe8d564b18f78f7c4"
)
EXPECTED_FORM_SHA256 = (
    "f7409a7267b3fddcb40d69574e6187320884951ad4e96980bb590ee0058c38d1"
)
EXPECTED_CONFIG_SHA256 = (
    "2ec2accb786a1f1e4d35027affe52c0402957861f59a93832583ac4094066dce"
)
PRE_P71C_PROMPT_SHA256 = (
    "5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d"
)
PRE_P71C_FORM_SHA256 = (
    "441a59c410a99ef0db53b8e974aeeeaea1bcd1735aabdc3cb51f59e0b6e069a2"
)
PRE_P71C_CONFIG_SHA256 = (
    "83ba93c060b110a0e48485f8d5083052d96a67c8a79892ab77024be3c4b5ccd9"
)
FIXED_RESPONSE = "LOCAL_P4I_MAIN_STUB_RESPONSE"


def _sha256_json(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _visible_request(request):
    raw = to_jsonable(request)
    return {
        key: raw[key]
        for key in (
            "model",
            "contents",
            "config",
            "live_connect_config",
            "cache_config",
            "previous_interaction_id",
        )
        if key in raw
    }


def _component_shas(request):
    config = request.get("config", {})
    generation = {
        key: value
        for key, value in config.items()
        if key not in {"system_instruction", "tools"}
    }
    return {
        "contents_sha256": _sha256_json(request.get("contents", [])),
        "tools_sha256": _sha256_json(config.get("tools", [])),
        "generation_config_sha256": _sha256_json(generation),
        "base_system_instruction_sha256": hashlib.sha256(
            grounding_callbacks._strip_requirement_view_blocks(
                config.get("system_instruction", "")
            ).encode("utf-8")
        ).hexdigest(),
    }


def _request(system_instruction="ORIGINAL SYSTEM INSTRUCTION"):
    return LlmRequest(
        model="local-p4i-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="USER MESSAGE MUST NOT CHANGE")],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="frozen_test_tool",
                            description="offline test declaration",
                        )
                    ]
                )
            ],
        ),
    )


def _state():
    semantic = _semantic_state()
    frame = semantic.requirement_frame
    # The rendering fixture intentionally carries no Evidence objects.  Remove
    # the helper's evidence refs so the full Runtime also satisfies K0's
    # referential-integrity checks instead of being fail-open reset to empty.
    frame = frame.model_copy(
        update={
            "value_slots": tuple(
                slot.model_copy(update={"evidence_refs": ()})
                for slot in frame.value_slots
            ),
            "schema_slots": tuple(
                slot.model_copy(update={"evidence_refs": ()})
                for slot in frame.schema_slots
            ),
            "operation_slots": tuple(
                slot.model_copy(update={"evidence_refs": ()})
                for slot in frame.operation_slots
            ),
        }
    )
    semantic = semantic.model_copy(update={"requirement_frame": frame})
    runtime = RequirementGroundingRuntime(grounding_state=semantic)
    return {
        "task_id": "task-p4i-unit",
        "current_phase": 1,
        "initial_budget": 20.0,
        "budget_remaining": 20.0,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
    }

class AdkAppendInstructionsContractTests(unittest.TestCase):
    def test_local_adk_250_list_contract_only_appends_system_instruction(self):
        self.assertEqual(importlib.metadata.version("google-adk"), "2.5.0")
        self.assertTrue(callable(LlmRequest.append_instructions))
        self.assertEqual(
            str(inspect.signature(LlmRequest.append_instructions)),
            "(self, instructions: 'Union[list[str], types.Content]') -> 'list[types.Content]'",
        )
        request = _request()
        contents = copy.deepcopy(request.contents)
        tools = copy.deepcopy(request.config.tools)
        result = request.append_instructions(["ADDED INSTRUCTION"])
        self.assertEqual(result, [])
        self.assertEqual(request.contents, contents)
        self.assertEqual(request.config.tools, tools)
        self.assertEqual(
            request.config.system_instruction,
            "ORIGINAL SYSTEM INSTRUCTION\n\nADDED INSTRUCTION",
        )

if __name__ == "__main__":
    unittest.main()
