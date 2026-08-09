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
    "5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d"
)
EXPECTED_FORM_SHA256 = (
    "441a59c410a99ef0db53b8e974aeeeaea1bcd1735aabdc3cb51f59e0b6e069a2"
)
EXPECTED_CONFIG_SHA256 = (
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


class PromptViewModeTests(unittest.TestCase):
    def test_unset_empty_and_three_exact_modes(self):
        self.assertEqual(
            grounding_callbacks.grounding_prompt_view_status({})[
                "effective_mode"
            ],
            "shadow",
        )
        for raw, expected in (
            ("", "shadow"),
            ("off", "off"),
            ("shadow", "shadow"),
            ("active", "active"),
        ):
            with self.subTest(raw=raw):
                status = grounding_callbacks.grounding_prompt_view_status(
                    {"GROUNDING_PROMPT_VIEW_MODE": raw}
                )
                self.assertEqual(status["requested_mode"], expected)
                self.assertEqual(status["effective_mode"], expected)
                self.assertTrue(status["configuration_valid"])
                self.assertIsNone(status["error_type"])

    def test_invalid_values_never_normalize_or_fallback(self):
        for raw in ("OFF", "Active", " shadow", "active ", "enabled", "typo"):
            with self.subTest(raw=raw):
                status = grounding_callbacks.grounding_prompt_view_status(
                    {"GROUNDING_PROMPT_VIEW_MODE": raw}
                )
                self.assertEqual(status["requested_mode"], "invalid")
                self.assertEqual(status["effective_mode"], "invalid")
                self.assertFalse(status["configuration_valid"])
                self.assertEqual(status["error_type"], "InvalidPromptViewMode")

    def test_view_and_updater_modes_are_orthogonal(self):
        for updater in ("", "llm"):
            for view in ("off", "shadow", "active"):
                with self.subTest(updater=updater or "rule", view=view):
                    environment = {
                        "GROUNDING_UPDATER_MODE": updater,
                        "GROUNDING_PROMPT_VIEW_MODE": view,
                    }
                    self.assertEqual(
                        grounding_callbacks._requested_updater_mode(environment),
                        "rule" if updater == "" else "llm",
                    )
                    self.assertEqual(
                        grounding_callbacks.grounding_prompt_view_status(
                            environment
                        )["effective_mode"],
                        view,
                    )


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


class ActiveViewCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def _call(self, mode, request, *, renderer=None, delegate=None):
        state = _state()
        context = SimpleNamespace(state=state)
        original = baseline_callbacks.before_model_callback
        baseline = delegate or AsyncMock(side_effect=original)
        patches = [
            patch.dict(
                os.environ,
                {
                    "GROUNDING_UPDATER_MODE": "",
                    "GROUNDING_PROMPT_VIEW_MODE": mode,
                },
            ),
            patch.object(baseline_callbacks, "before_model_callback", baseline),
        ]
        if renderer is not None:
            patches.append(
                patch.object(
                    grounding_callbacks,
                    "render_prompt_view",
                    side_effect=renderer,
                )
            )
        entered = [item.start() for item in patches]
        try:
            result = await grounding_callbacks.before_model_callback(
                context, request
            )
        finally:
            for item in reversed(patches):
                item.stop()
        return result, state, baseline, entered

    async def test_off_does_not_render_record_or_mutate(self):
        request = _request()
        before = copy.deepcopy(request)
        renderer = RuntimeError("off must not call renderer")
        _, state, baseline, _ = await self._call(
            "off", request, renderer=renderer
        )
        baseline.assert_awaited_once_with(SimpleNamespace(state=state), request)
        self.assertEqual(request, before)
        audit = state["system_agent_llm_calls"][0][
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY
        ]
        self.assertEqual(audit["effective_mode"], "off")
        self.assertFalse(audit["injected"])
        self.assertNotIn("view", audit)
        self.assertEqual(
            audit["request_sha256_before"], audit["request_sha256_after"]
        )

    async def test_shadow_records_without_mutating(self):
        request = _request()
        before = copy.deepcopy(request)
        _, state, _, _ = await self._call("shadow", request)
        self.assertEqual(request, before)
        audit = state["system_agent_llm_calls"][0][
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY
        ]
        self.assertEqual(audit["effective_mode"], "shadow")
        self.assertTrue(audit["view"])
        self.assertFalse(audit["injected"])
        self.assertEqual(
            audit["request_sha256_before"], audit["request_sha256_after"]
        )

    async def test_active_changes_only_system_instruction_before_baseline_audit(self):
        request = _request()
        contents = copy.deepcopy(request.contents)
        tools = copy.deepcopy(request.config.tools)
        temperature = request.config.temperature
        original_system = request.config.system_instruction
        result, state, baseline, _ = await self._call("active", request)
        self.assertIsNone(result)
        baseline.assert_awaited_once()
        self.assertEqual(request.contents, contents)
        self.assertEqual(request.config.tools, tools)
        self.assertEqual(request.config.temperature, temperature)
        self.assertTrue(request.config.system_instruction.startswith(original_system))
        self.assertEqual(
            request.config.system_instruction.count(
                grounding_callbacks.REQUIREMENT_VIEW_BEGIN
            ),
            1,
        )
        self.assertEqual(
            request.config.system_instruction.count(
                grounding_callbacks.REQUIREMENT_VIEW_END
            ),
            1,
        )
        model_audit = state["system_agent_llm_calls"][0]
        audited_system = model_audit["request"]["config"]["system_instruction"]
        self.assertEqual(audited_system, request.config.system_instruction)
        view_audit = model_audit[
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY
        ]
        self.assertTrue(view_audit["injected"])
        self.assertNotEqual(
            view_audit["request_sha256_before"],
            view_audit["request_sha256_after"],
        )

    async def test_empty_view_does_not_inject(self):
        request = _request()
        before = copy.deepcopy(request)
        _, state, _, _ = await self._call(
            "active", request, renderer=lambda *_args, **_kwargs: ""
        )
        self.assertEqual(request, before)
        audit = state["system_agent_llm_calls"][0][
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY
        ]
        self.assertFalse(audit["injected"])
        self.assertEqual(audit["view"], "")

    async def test_invalid_mode_does_not_render_inject_or_fallback(self):
        request = _request()
        before = copy.deepcopy(request)
        state = _state()
        context = SimpleNamespace(state=state)
        delegate = AsyncMock(side_effect=baseline_callbacks.before_model_callback)
        with (
            patch.dict(
                os.environ,
                {
                    "GROUNDING_UPDATER_MODE": "",
                    "GROUNDING_PROMPT_VIEW_MODE": "ACTIVE",
                },
            ),
            patch.object(
                grounding_callbacks,
                "render_prompt_view",
                side_effect=AssertionError("invalid mode rendered a View"),
            ) as renderer,
            patch.object(baseline_callbacks, "before_model_callback", delegate),
        ):
            await grounding_callbacks.before_model_callback(context, request)
        renderer.assert_not_called()
        delegate.assert_awaited_once_with(context, request)
        self.assertEqual(request, before)
        audit = state["system_agent_llm_calls"][0][
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY
        ]
        self.assertEqual(audit["effective_mode"], "invalid")
        self.assertEqual(audit["error_type"], "InvalidPromptViewMode")
        self.assertFalse(audit["injected"])
        self.assertNotIn("view", audit)

    async def test_renderer_failure_is_fail_open_and_request_is_unchanged(self):
        request = _request()
        before = copy.deepcopy(request)
        sentinel = object()
        delegate = AsyncMock(return_value=sentinel)
        result, state, _, _ = await self._call(
            "active",
            request,
            renderer=RuntimeError("synthetic renderer failure"),
            delegate=delegate,
        )
        self.assertIs(result, sentinel)
        self.assertEqual(request, before)
        self.assertEqual(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]["grounding_revision"],
            0,
        )

    async def test_append_failure_restores_entire_request(self):
        request = _request()
        before = copy.deepcopy(request)
        original_append = LlmRequest.append_instructions

        def broken_append(self, instructions):
            self.config.system_instruction += "\n\nPARTIAL MUTATION"
            raise RuntimeError("synthetic append failure")

        with patch.object(LlmRequest, "append_instructions", new=broken_append):
            _, state, _, _ = await self._call("active", request)
        self.assertEqual(request, before)
        audit = state["system_agent_llm_calls"][0][
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY
        ]
        self.assertFalse(audit["injected"])
        self.assertEqual(audit["error_type"], "RuntimeError")
        self.assertEqual(
            audit["request_sha256_before"], audit["request_sha256_after"]
        )
        self.assertIs(LlmRequest.append_instructions, original_append)

    async def test_same_request_is_never_double_injected(self):
        request = _request()
        state = _state()
        context = SimpleNamespace(state=state)
        original = baseline_callbacks.before_model_callback
        delegate = AsyncMock(side_effect=original)
        with (
            patch.dict(os.environ, {"GROUNDING_PROMPT_VIEW_MODE": "active"}),
            patch.object(baseline_callbacks, "before_model_callback", delegate),
        ):
            await grounding_callbacks.before_model_callback(context, request)
            await grounding_callbacks.before_model_callback(context, request)
        self.assertEqual(delegate.await_count, 2)
        self.assertEqual(
            request.config.system_instruction.count(
                grounding_callbacks.REQUIREMENT_VIEW_BEGIN
            ),
            1,
        )

    async def test_new_model_round_does_not_accumulate_old_view(self):
        first = _request()
        second = _request()
        state = _state()
        context = SimpleNamespace(state=state)
        original = baseline_callbacks.before_model_callback
        with (
            patch.dict(os.environ, {"GROUNDING_PROMPT_VIEW_MODE": "active"}),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                AsyncMock(side_effect=original),
            ),
        ):
            await grounding_callbacks.before_model_callback(context, first)
            runtime = RequirementGroundingRuntime.model_validate(
                state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
            )
            frame = runtime.grounding_state.requirement_frame
            changed = frame.value_slots[0].model_copy(
                update={
                    "mention": "2025",
                    "current_interpretation": "2025",
                }
            )
            new_frame = frame.model_copy(
                update={"value_slots": (changed, frame.value_slots[1])}
            )
            new_state = runtime.grounding_state.model_copy(
                update={"requirement_frame": new_frame}
            )
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime.model_copy(
                update={"grounding_state": new_state}
            ).model_dump(mode="json")
            await grounding_callbacks.before_model_callback(context, second)
        self.assertIn('"year": "2025"', second.config.system_instruction)
        self.assertNotIn('"year": "2024"', second.config.system_instruction)
        self.assertEqual(
            second.config.system_instruction.count(
                grounding_callbacks.REQUIREMENT_VIEW_BEGIN
            ),
            1,
        )

    async def test_grounding_failure_never_becomes_user_visible_response(self):
        request = _request()
        before = copy.deepcopy(request)
        sentinel = object()
        delegate = AsyncMock(return_value=sentinel)
        with (
            patch.dict(os.environ, {"GROUNDING_PROMPT_VIEW_MODE": "active"}),
            patch.object(
                grounding_callbacks,
                "_consume_bound_user_message",
                AsyncMock(side_effect=RuntimeError("synthetic Grounding failure")),
            ),
            patch.object(baseline_callbacks, "before_model_callback", delegate),
        ):
            result = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=_state()), request
            )
        self.assertIs(result, sentinel)
        self.assertEqual(request, before)


class HealthModeTests(unittest.TestCase):
    def test_health_summary_and_variant_report_each_view_mode(self):
        for view, injection in (
            ("off", False),
            ("shadow", False),
            ("active", True),
        ):
            with self.subTest(view=view):
                with patch.dict(
                    os.environ,
                    {
                        "GROUNDING_UPDATER_MODE": "",
                        "GROUNDING_PROMPT_VIEW_MODE": view,
                    },
                ):
                    summary = valibra_server._configuration_summary()
                self.assertEqual(
                    summary["grounding_prompt_view_requested_mode"], view
                )
                self.assertEqual(
                    summary["grounding_prompt_view_effective_mode"], view
                )
                self.assertTrue(
                    summary["grounding_prompt_view_configuration_valid"]
                )
                self.assertEqual(
                    summary["prompt_view_injection_enabled"], injection
                )
                self.assertFalse(summary["prompt_view_injected"])
                self.assertEqual(
                    valibra_server._variant(summary),
                    f"P4.3c-Rule-{view.title()}",
                )

    def test_llm_and_invalid_view_are_reported_without_credentials(self):
        with (
            patch.dict(
                os.environ,
                {
                    "GROUNDING_UPDATER_MODE": "llm",
                    "GROUNDING_PROMPT_VIEW_MODE": "off",
                },
            ),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=object(),
            ),
        ):
            llm = valibra_server._configuration_summary()
        self.assertEqual(valibra_server._variant(llm), "P4.3c-LLM-Off")
        with patch.dict(
            os.environ,
            {
                "GROUNDING_UPDATER_MODE": "",
                "GROUNDING_PROMPT_VIEW_MODE": "bad",
            },
        ):
            invalid = valibra_server._configuration_summary()
        self.assertEqual(
            valibra_server._variant(invalid),
            "P4.3c-Rule-Invalid-View-Config",
        )
        self.assertFalse(
            invalid["grounding_prompt_view_configuration_valid"]
        )
        self.assertEqual(
            invalid["grounding_prompt_view_error_type"],
            "InvalidPromptViewMode",
        )
        self.assertFalse(invalid["prompt_view_injection_enabled"])


async def run_three_mode_lifecycle_smoke():
    captures = []

    class LocalMainModel(BaseLlm):
        @classmethod
        def supported_models(cls):
            return ["local-p4i-main-stub"]

        async def generate_content_async(
            self,
            llm_request,
            stream=False,
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            captures.append(_visible_request(llm_request))
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=FIXED_RESPONSE)],
                )
            )

    client = FakeClient(content=QUERY_CONTENT)
    updater = LLMUpdater(client, _llm_config(max_calls="4"))
    stub = LocalMainModel(model="local-p4i-main-stub")
    results = {}

    async def forbidden_provider(**_kwargs):
        raise AssertionError("real Provider call is forbidden")

    for mode in ("off", "shadow", "active"):
        runtime = AdkRuntime()
        self_error = runtime.error
        if not runtime.available:
            raise AssertionError(f"ADK runtime unavailable: {self_error}")
        with (
            patch.dict(
                os.environ,
                {
                    "GROUNDING_UPDATER_MODE": "llm",
                    "GROUNDING_PROMPT_VIEW_MODE": mode,
                },
            ),
            patch.object(valibra_agent_module, "_build_model", return_value=stub),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
            patch("litellm.acompletion", new=forbidden_provider),
        ):
            await runtime.init_session(
                task_id="p4i-same-task",
                mode="a-interact",
                state={
                    "task_id": "p4i-same-task",
                    "current_phase": 1,
                    "initial_budget": 20.0,
                    "budget_remaining": 20.0,
                },
                reset=True,
            )
            result = await runtime.run_turn(
                task_id="p4i-same-task",
                mode="a-interact",
                message=QUERY,
            )
            await runtime.cleanup_session("p4i-same-task", "a-interact")
        request = captures[-1]
        audit = result["state"]["system_agent_llm_calls"][0][
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY
        ]
        results[mode] = {
            "request": request,
            "request_sha256": _sha256_json(request),
            "components": _component_shas(request),
            "response_sha256": hashlib.sha256(
                result["response"].encode("utf-8")
            ).hexdigest(),
            "injected": audit["injected"],
            "view_sha256": audit.get("view_sha256"),
            "injection_block_sha256": audit.get("injection_block_sha256"),
            "tool_calls": len(result["state"].get("tool_trajectory", [])),
            "bird_coin_spent": 20.0 - result["state"]["budget_remaining"],
            "system_agent_tokens": result["state"].get(
                "system_agent_token_usage", {}
            ).get("total_tokens", 0),
        }

    off = results["off"]
    shadow = results["shadow"]
    active = results["active"]
    normalized_active = copy.deepcopy(active["request"])
    active_system = normalized_active["config"]["system_instruction"]
    normalized_active["config"]["system_instruction"] = (
        grounding_callbacks._strip_requirement_view_blocks(active_system)
    )
    normalized_sha = _sha256_json(normalized_active)

    if off["request_sha256"] != shadow["request_sha256"]:
        raise AssertionError("off and shadow requests differ")
    if active["request_sha256"] == off["request_sha256"]:
        raise AssertionError("active request did not change")
    if normalized_sha != off["request_sha256"]:
        raise AssertionError("normalized active request differs from off")
    for field in (
        "contents_sha256",
        "tools_sha256",
        "generation_config_sha256",
        "base_system_instruction_sha256",
    ):
        if len({results[mode]["components"][field] for mode in results}) != 1:
            raise AssertionError(f"request component differs: {field}")
    if len({results[mode]["response_sha256"] for mode in results}) != 1:
        raise AssertionError("deterministic main responses differ")
    if grounding_callbacks.REQUIREMENT_VIEW_BEGIN not in active_system:
        raise AssertionError("active request contains no View block")
    if active_system.count(grounding_callbacks.REQUIREMENT_VIEW_BEGIN) != 1:
        raise AssertionError("active request contains multiple View blocks")
    if [results[mode]["injected"] for mode in results] != [False, False, True]:
        raise AssertionError("mode injection flags are incorrect")
    if any(results[mode]["tool_calls"] for mode in results):
        raise AssertionError("smoke used a tool")
    if any(results[mode]["bird_coin_spent"] for mode in results):
        raise AssertionError("smoke spent bird-coin")
    if any(results[mode]["system_agent_tokens"] for mode in results):
        raise AssertionError("local main stub reported Provider tokens")
    if client.calls != 3:
        raise AssertionError("fake Grounding client was not called once per session")

    return {
        "status": "PASS",
        "google_adk_version": importlib.metadata.version("google-adk"),
        "append_instructions_signature": str(
            inspect.signature(LlmRequest.append_instructions)
        ),
        "off_request_sha256": off["request_sha256"],
        "shadow_request_sha256": shadow["request_sha256"],
        "active_request_sha256": active["request_sha256"],
        "normalized_active_request_sha256": normalized_sha,
        "normalized_active_equals_off": normalized_sha == off["request_sha256"],
        "contents_sha256": off["components"]["contents_sha256"],
        "tools_sha256": off["components"]["tools_sha256"],
        "generation_config_sha256": off["components"][
            "generation_config_sha256"
        ],
        "static_instruction_sha256": off["components"][
            "base_system_instruction_sha256"
        ],
        "view_sha256": active["view_sha256"],
        "injection_block_sha256": active["injection_block_sha256"],
        "injected_flags": {
            mode: results[mode]["injected"] for mode in results
        },
        "active_block_count": active_system.count(
            grounding_callbacks.REQUIREMENT_VIEW_BEGIN
        ),
        "final_response_sha256": active["response_sha256"],
        "fake_grounding_calls": client.calls,
        "system_agent_provider_calls": 0,
        "grounding_provider_calls": 0,
        "database_calls": 0,
        "user_simulator_calls": 0,
        "benchmark_runs": 0,
        "tool_calls": 0,
        "bird_coin_spent": 0.0,
    }


class RealAdkLifecycleSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_fresh_sessions_off_shadow_active(self):
        summary = await run_three_mode_lifecycle_smoke()
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["normalized_active_equals_off"])
        self.assertEqual(summary["active_block_count"], 1)
        self.assertEqual(summary["system_agent_provider_calls"], 0)
        self.assertEqual(summary["grounding_provider_calls"], 0)


class FrozenBoundaryTests(unittest.TestCase):
    def test_frozen_contracts_and_static_instruction_are_unchanged(self):
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, EXPECTED_PROMPT_SHA256)
        self.assertEqual(LLM_FRAME_FORM_SCHEMA_SHA256, EXPECTED_FORM_SHA256)
        self.assertEqual(
            _llm_config(timeout="300", max_calls="2").configuration_sha256,
            EXPECTED_CONFIG_SHA256,
        )
        self.assertNotIn(
            grounding_callbacks.REQUIREMENT_VIEW_BEGIN,
            AINTERACT_INSTRUCTION,
        )


if __name__ == "__main__":
    unittest.main()
