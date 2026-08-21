from __future__ import annotations

import asyncio
import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from shared.audit import to_jsonable
from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks, server
from valibra_agent.sql_grounding.control import render_control_hint
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
)
from valibra_agent.sql_grounding.prompt_view import (
    DEFAULT_MAX_VIEW_CHARS,
    DEFAULT_MAX_VIEW_ITEMS,
    DEFAULT_MAX_VIEW_TOKENS,
    count_grounding_view_tokens,
    render_grounding_view,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
)


PROMPT_SHA = "812a189320a2f77efed13c99f5f4ba56538570542e341e46167d36f3b2a6f9d6"
FORM_SHA = "1f7e3c1f1ae86876f63de951bcade30fc1ba338e046416fe033331d447775d15"
CONFIG_SHA = "a507a6f3513e53d4c8c784589d15679569070b14251500e74ae77d4597dcf143"
QUERY = "Show the maintenance cost."
_ORIGINAL_MODE = os.environ.get("GROUNDING_UPDATER_MODE")


def setUpModule() -> None:
    os.environ.pop("GROUNDING_UPDATER_MODE", None)


def tearDownModule() -> None:
    if _ORIGINAL_MODE is None:
        os.environ.pop("GROUNDING_UPDATER_MODE", None)
    else:
        os.environ["GROUNDING_UPDATER_MODE"] = _ORIGINAL_MODE


def task_state(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "current_phase": 1,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 12.0,
        "initial_budget": 12.0,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
    }


def complete_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=(),
        join_keys=(),
        column_mapping=(),
        domain_knowledge=(),
    )


def request(system_instruction: str = "BASE SYSTEM INSTRUCTION") -> LlmRequest:
    return LlmRequest(
        model="local-sg6a-model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="USER CONTENT MUST STAY EXACT")],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            top_p=0.95,
        ),
    )


def config_without_system(value: LlmRequest) -> dict:
    payload = to_jsonable(value.config)
    payload.pop("system_instruction", None)
    return payload


class ScriptedUpdater:
    def __init__(self, response: GroundingLLMResponse) -> None:
        self.response = response
        self.calls = 0

    async def propose(self, runtime, observation, **kwargs):
        del runtime, observation, kwargs
        self.calls += 1
        return GroundingUpdaterResult(
            response=self.response,
            telemetry=GroundingLLMTelemetry(
                attempted=False,
                status="succeeded",
                request_sha256="",
                response_sha256="",
                prompt_sha256=PROMPT_SHA,
                form_schema_sha256=FORM_SHA,
                configuration_sha256=CONFIG_SHA,
            ),
            transport_normalization="none",
        )


class RaisingUpdater:
    async def propose(self, runtime, observation, **kwargs):
        del runtime, observation, kwargs
        raise RuntimeError("synthetic update failure")


async def run_before_model(
    current: dict,
    active_request: LlmRequest,
    *,
    query: str | None = None,
) -> tuple[object, list[dict]]:
    seen: list[dict] = []
    sentinel = object()

    async def baseline(callback_context, llm_request):
        seen.append(copy.deepcopy(to_jsonable(llm_request)))
        calls = callback_context.state.setdefault("system_agent_llm_calls", [])
        calls.append({"actions": []})
        callback_context.state["_active_llm_call_index"] = len(calls) - 1
        return sentinel

    token = None
    if query is not None:
        token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", query
        )
    try:
        with patch.object(
            baseline_callbacks,
            "before_model_callback",
            AsyncMock(side_effect=baseline),
        ):
            result = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=current), active_request
            )
    finally:
        if token is not None:
            grounding_callbacks._reset_turn_message(token)
    return result, seen


class SG6aActiveInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_basic_active_pair_is_ordered_and_only_system_instruction_changes(self):
        current = task_state("sg6a-basic")
        active_request = request()
        original = copy.deepcopy(active_request)
        result, seen = await run_before_model(current, active_request)

        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(result)
        self.assertEqual(active_request.contents, original.contents)
        self.assertEqual(config_without_system(active_request), config_without_system(original))
        instruction = active_request.config.system_instruction
        self.assertEqual(
            grounding_callbacks._strip_active_grounding_context(instruction),
            original.config.system_instruction,
        )
        for marker in grounding_callbacks._ACTIVE_CONTEXT_MARKERS:
            self.assertEqual(instruction.count(marker), 1)
        self.assertLess(
            instruction.index(grounding_callbacks.GROUNDING_VIEW_BEGIN),
            instruction.index(grounding_callbacks.CONTROL_HINT_BEGIN),
        )
        self.assertIn("[VALIBRA DATABASE GROUNDING]", instruction)
        self.assertIn("[VALIBRA CONTROL]", instruction)
        self.assertEqual(seen[0]["config"]["system_instruction"], instruction)

        call = current["system_agent_llm_calls"][0]
        view = call[grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY]
        control = call[grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY]
        self.assertEqual(view["mode"], "active")
        self.assertTrue(view["injected"])
        self.assertEqual(view["injection_status"], "succeeded")
        self.assertEqual(view["view_chars"], view["chars"])
        self.assertEqual(view["view_tokens"], view["tokens_cl100k"])
        self.assertEqual(
            view["view_block_sha256"], view["injection_block_sha256"]
        )
        self.assertEqual(control["mode"], "active_hint")
        self.assertTrue(control["control_hint_injected"])
        self.assertEqual(control["injection_status"], "succeeded")
        self.assertEqual(
            control["control_hint_tokens"],
            control["control_hint_tokens_cl100k"],
        )
        self.assertFalse(control["attempt_gate"]["blocked"])
        audit_text = json.dumps({"view": view, "control": control})
        for forbidden in (
            "[VALIBRA DATABASE GROUNDING]",
            "[VALIBRA CONTROL]",
        ):
            self.assertNotIn(forbidden, audit_text)
        self.assertNotIn("system_instruction", view)
        self.assertNotIn("system_instruction", control)

    async def test_same_request_is_idempotent_and_replaces_both_current_blocks(self):
        active_request = request("BASE\nWITH EXACT WHITESPACE  ")
        first_view = render_grounding_view(SQLGroundingState()).text
        first_hint = render_control_hint("tables").text
        grounding_callbacks._inject_active_grounding_context(
            active_request,
            view_text=first_view,
            control_hint_text=first_hint,
        )
        first_instruction = active_request.config.system_instruction
        grounding_callbacks._inject_active_grounding_context(
            active_request,
            view_text=first_view,
            control_hint_text=first_hint,
        )
        self.assertEqual(active_request.config.system_instruction, first_instruction)

        second_view = render_grounding_view(complete_state()).text
        second_hint = render_control_hint("none").text
        grounding_callbacks._inject_active_grounding_context(
            active_request,
            view_text=second_view,
            control_hint_text=second_hint,
        )
        instruction = active_request.config.system_instruction
        self.assertNotIn("Current grounding focus: tables.", instruction)
        self.assertIn("Current grounding focus: none.", instruction)
        self.assertNotIn("Tables:\n- null", instruction)
        self.assertIn("Tables:\n- []", instruction)
        self.assertEqual(
            grounding_callbacks._strip_active_grounding_context(instruction),
            "BASE\nWITH EXACT WHITESPACE  ",
        )
        for marker in grounding_callbacks._ACTIVE_CONTEXT_MARKERS:
            self.assertEqual(instruction.count(marker), 1)

    async def test_current_turn_user_query_does_not_change_state_or_focus(self):
        current = task_state("sg6a-current-turn")
        active_request = request()
        updater = ScriptedUpdater(
            GroundingLLMResponse(
                sql_grounding_state=SQLGroundingState(),
                user_clarification_requests=(),
                next_focus_dimension="column_mapping",
            )
        )
        with patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater):
            await run_before_model(current, active_request, query=QUERY)
        self.assertEqual(updater.calls, 0)
        runtime = GroundingRuntime.model_validate(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.stage, "INITIAL_GROUNDING")
        self.assertEqual(runtime.focus_dimension, "tables")
        instruction = active_request.config.system_instruction
        self.assertIn("Tables:\n- null", instruction)
        self.assertIn("Current grounding focus: tables.", instruction)
        audit = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        self.assertEqual(audit["grounding_revision"], 0)
        self.assertEqual(audit["stage"], "INITIAL_GROUNDING")
        self.assertEqual(audit["focus_dimension"], "tables")

    async def test_render_occurs_after_current_turn_runtime_is_stored(self):
        current = task_state("sg6a-current-state")
        active_request = request()
        current_runtime = GroundingRuntime(
            grounding_revision=1,
            stage="SQL_ATTEMPT",
            focus_dimension="none",
            grounding_state=SQLGroundingState(
                tables=("current_metrics",),
                join_keys=(),
                column_mapping=(),
                domain_knowledge=(),
            ),
        )

        async def consume(state, runtime):
            del runtime
            grounding_callbacks._store_runtime(state, current_runtime)
            return {"service_status": "accepted", "observation_type": "user_query"}

        with patch.object(
            grounding_callbacks,
            "_consume_bound_user_message",
            AsyncMock(side_effect=consume),
        ):
            await run_before_model(current, active_request)
        instruction = active_request.config.system_instruction
        self.assertIn('"current_metrics"', instruction)
        self.assertIn("Current grounding focus: none.", instruction)
        audit = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        self.assertEqual(audit["grounding_revision"], 1)
        self.assertEqual(audit["focus_dimension"], "none")

    async def test_corrupt_runtime_suppresses_both_blocks_for_that_turn(self):
        current = task_state("sg6a-corrupt")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = {"bad": "runtime"}
        active_request = request()
        original = copy.deepcopy(active_request)
        _, seen = await run_before_model(current, active_request)
        self.assertEqual(active_request, original)
        self.assertEqual(seen[0]["config"]["system_instruction"], "BASE SYSTEM INSTRUCTION")
        view = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        control = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
        ]
        self.assertTrue(view["runtime_degraded"])
        self.assertFalse(view["injected"])
        self.assertEqual(view["injection_status"], "failed_open")
        self.assertFalse(control["control_hint_injected"])
        self.assertEqual(control["injection_status"], "failed_open")

    async def test_user_query_skip_injects_previous_valid_runtime(self):
        current = task_state("sg6a-update-fail")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = GroundingRuntime(
            stage="SQL_ATTEMPT",
            focus_dimension="none",
            grounding_state=complete_state(),
        ).model_dump(mode="json")
        active_request = request()
        with patch.object(
            grounding_callbacks,
            "_SQL_GROUNDING_UPDATER",
            RaisingUpdater(),
        ):
            await run_before_model(current, active_request, query=QUERY)
        self.assertIn(grounding_callbacks.GROUNDING_VIEW_BEGIN, active_request.config.system_instruction)
        self.assertIn("Current grounding focus: none.", active_request.config.system_instruction)
        view = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        update = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY
        ]
        self.assertTrue(view["injected"])
        self.assertFalse(view["runtime_degraded"])
        self.assertEqual(
            update["service_status"], "skipped_provider_no_state_evidence"
        )

    async def test_malformed_previous_blocks_fail_open_without_partial_injection(self):
        malformed = (
            f"BASE\n\n{grounding_callbacks.GROUNDING_VIEW_BEGIN}\nbroken",
            f"BASE\n\n{grounding_callbacks.GROUNDING_VIEW_END}",
            (
                f"BASE\n\n{grounding_callbacks.GROUNDING_VIEW_BEGIN}\na\n"
                f"{grounding_callbacks.GROUNDING_VIEW_BEGIN}\nb\n"
                f"{grounding_callbacks.GROUNDING_VIEW_END}\n\n"
                f"{grounding_callbacks.CONTROL_HINT_BEGIN}\nc\n"
                f"{grounding_callbacks.CONTROL_HINT_END}"
            ),
            (
                f"BASE\n\n{grounding_callbacks.GROUNDING_VIEW_BEGIN}\na\n"
                f"{grounding_callbacks.CONTROL_HINT_BEGIN}\nb\n"
                f"{grounding_callbacks.GROUNDING_VIEW_END}\n\n"
                f"{grounding_callbacks.CONTROL_HINT_END}"
            ),
        )
        for index, instruction in enumerate(malformed):
            with self.subTest(index=index):
                current = task_state(f"sg6a-malformed-{index}")
                active_request = request(instruction)
                original = copy.deepcopy(active_request)
                await run_before_model(current, active_request)
                self.assertEqual(active_request, original)
                view = current["system_agent_llm_calls"][0][
                    grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
                ]
                control = current["system_agent_llm_calls"][0][
                    grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY
                ]
                self.assertFalse(view["injected"])
                self.assertFalse(control["control_hint_injected"])
                self.assertEqual(view["error_type"], "ValueError")

    async def test_marker_collision_and_append_exception_rollback_exactly(self):
        active_request = request()
        original = copy.deepcopy(active_request)
        with self.assertRaises(ValueError):
            grounding_callbacks._inject_active_grounding_context(
                active_request,
                view_text=(
                    "[VALIBRA DATABASE GROUNDING]\n"
                    + grounding_callbacks.CONTROL_HINT_BEGIN
                ),
                control_hint_text=render_control_hint("tables").text,
            )
        self.assertEqual(active_request, original)

        original_append = LlmRequest.append_instructions

        def broken_append(self, instructions):
            del instructions
            self.config.system_instruction = "PARTIAL"
            self.contents.append(types.Content(role="user", parts=[]))
            self.model = "mutated-model"
            raise RuntimeError("synthetic append failure")

        current = task_state("sg6a-append-failure")
        active_request = request()
        original = copy.deepcopy(active_request)
        with patch.object(LlmRequest, "append_instructions", new=broken_append):
            await run_before_model(current, active_request)
        self.assertIs(LlmRequest.append_instructions, original_append)
        self.assertEqual(active_request, original)
        view = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        self.assertFalse(view["injected"])
        self.assertEqual(view["injection_status"], "failed_open")
        self.assertEqual(view["error_type"], "RuntimeError")

    async def test_concurrent_tasks_inject_no_cross_contamination(self):
        first = task_state("sg6a-concurrent-a")
        second = task_state("sg6a-concurrent-b")
        second[grounding_callbacks.GROUNDING_RUNTIME_KEY] = GroundingRuntime(
            stage="P2_INCREMENTAL",
            focus_dimension="domain_knowledge",
            grounding_state=complete_state(),
        ).model_dump(mode="json")
        first_request = request("BASE-A")
        second_request = request("BASE-B")

        async def one(current, active_request):
            await asyncio.sleep(0)
            return await run_before_model(current, active_request)

        await asyncio.gather(
            one(first, first_request),
            one(second, second_request),
        )
        self.assertIn("Current grounding focus: tables.", first_request.config.system_instruction)
        self.assertNotIn("domain_knowledge", first_request.config.system_instruction)
        self.assertIn(
            "Current grounding focus: domain_knowledge.",
            second_request.config.system_instruction,
        )
        self.assertNotIn("BASE-B", first_request.config.system_instruction)
        self.assertNotIn("BASE-A", second_request.config.system_instruction)


class SG6aBoundsGateHealthTests(unittest.IsolatedAsyncioTestCase):
    def test_view_and_hint_bounds_and_no_session_semantic_leakage(self):
        tables = tuple(f"table_{index:02d}" for index in range(64))
        mappings = tuple(
            ColumnMapping(
                phrase=f"phrase {index:02d}",
                targets=(f"table_{index:02d}.value",),
            )
            for index in range(64)
        )
        knowledge = tuple(
            DomainKnowledge(kind="business_rule", content=f"rule {index:02d}")
            for index in range(64)
        )
        rendered = render_grounding_view(
            SQLGroundingState(
                tables=tables,
                join_keys=(),
                column_mapping=mappings,
                domain_knowledge=knowledge,
            )
        )
        self.assertLessEqual(rendered.char_count, DEFAULT_MAX_VIEW_CHARS)
        self.assertLessEqual(rendered.token_count, DEFAULT_MAX_VIEW_TOKENS)
        self.assertLessEqual(rendered.included_items, DEFAULT_MAX_VIEW_ITEMS)
        self.assertEqual(rendered.included_items + rendered.omitted_items, 193)
        self.assertGreater(rendered.omitted_items, 0)

        for focus in (
            "tables",
            "join_keys",
            "column_mapping",
            "domain_knowledge",
            "none",
        ):
            with self.subTest(focus=focus):
                hint = render_control_hint(focus)
                self.assertLessEqual(len(hint.text), 1_024)
                self.assertLessEqual(count_grounding_view_tokens(hint.text), 256)

        active_request = request()
        grounding_callbacks._inject_active_grounding_context(
            active_request,
            view_text=render_grounding_view(complete_state()).text,
            control_hint_text=render_control_hint("none").text,
        )
        instruction = active_request.config.system_instruction
        for forbidden in (
            "raw tool secret",
            "ValidationContext",
            "telemetry",
            "credential",
            "pending-call",
            "Bird-Coin",
            "SQL plan",
        ):
            self.assertNotIn(forbidden, instruction)

    async def test_active_view_is_unchanged_while_closed_submit_is_blocked(self):
        valibra = task_state("sg6a-gate-v")
        tool = SimpleNamespace(name="submit_sql")
        args = {"sql": "SELECT 1"}
        valibra_context = SimpleNamespace(
            state=valibra,
            function_call_id="gate-v",
            invocation_id="inv-gate-v",
        )
        valibra_before = await grounding_callbacks.before_tool_callback(
            tool, args, valibra_context
        )
        self.assertEqual(
            valibra_before["status"],
            "VALIBRA_FIRST_SUBMIT_BLOCKED",
        )
        self.assertEqual(valibra["budget_remaining"], 12.0)

        valibra_after = await grounding_callbacks.after_tool_callback(
            tool, args, valibra_context, valibra_before
        )
        self.assertEqual(valibra_after, valibra_before)
        self.assertEqual(valibra["tool_trajectory"], [])
        audit = valibra[grounding_callbacks.GROUNDING_GATE_AUDITS_KEY][0]
        self.assertEqual(audit["effective_gate_action"], "blocked")

    def test_health_and_frozen_contracts(self):
        summary = server._configuration_summary()
        self.assertEqual(summary["grounding_core"], "sql_grounding_v1")
        self.assertEqual(summary["grounding_prompt_view_effective_mode"], "active")
        self.assertTrue(summary["prompt_view_injection_enabled"])
        self.assertTrue(summary["prompt_view_injected"])
        self.assertEqual(summary["control_mode"], "active_hint")
        self.assertTrue(summary["control_hint_injection_enabled"])
        self.assertEqual(summary["attempt_gate_mode"], "active_first_submit")
        self.assertTrue(summary["attempt_gate_blocking_enabled"])
        self.assertTrue(summary["attempt_gate_budget_liveness_bypass"])
        self.assertTrue(summary["attempt_gate_enabled"])
        self.assertEqual(server._variant(summary), "SQL-Grounding-V1-SG6b-Active-Gate")
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)


if __name__ == "__main__":
    unittest.main()
