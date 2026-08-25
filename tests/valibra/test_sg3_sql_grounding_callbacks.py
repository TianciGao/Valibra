from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    ValidationContext,
    canonical_json,
)
from valibra_agent.sql_grounding.prompt_view import render_grounding_view
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
)

PROMPT_SHA = "db5a44e94a0e7fba92e78e6ca99e6b324affdee92aaed4dcd45010ff03876454"
FORM_SHA = "728fc43c6ed85e72e60c9ebf85b00487059a2871020a28764b9641c69b84ed81"
CONFIG_SHA = "31115d248a048c76a9ffd947aade7c4fa725915f82243809f06b8e14cbd98f0c"
QUERY = "Show the maintenance cost."
_ORIGINAL_GROUNDING_UPDATER_MODE = os.environ.get("GROUNDING_UPDATER_MODE")


def setUpModule():
    # SG3 regression is the explicit empty-mode passthrough control.  Do not
    # let a developer's local SG4 .env turn an offline test into Provider I/O.
    os.environ.pop("GROUNDING_UPDATER_MODE", None)


def tearDownModule():
    if _ORIGINAL_GROUNDING_UPDATER_MODE is None:
        os.environ.pop("GROUNDING_UPDATER_MODE", None)
    else:
        os.environ["GROUNDING_UPDATER_MODE"] = _ORIGINAL_GROUNDING_UPDATER_MODE


def state(task_id: str = "sg3-task", *, budget: float = 10.0) -> dict:
    return {
        "task_id": task_id,
        "current_phase": 1,
        "budget_remaining": budget,
        "initial_budget": budget,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
    }


def context(current_state: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=current_state,
        function_call_id=call_id,
        invocation_id=f"inv-{call_id}",
    )


def runtime(current_state: dict) -> GroundingRuntime:
    return GroundingRuntime.model_validate(
        current_state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


def raw_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


async def run_bound_query(current_state: dict, request: object, *, twice=False):
    token = grounding_callbacks._bind_turn_message(
        current_state["task_id"], "a-interact", QUERY
    )
    try:
        callback_context = SimpleNamespace(state=current_state)
        result = await grounding_callbacks.before_model_callback(
            callback_context, request
        )
        if twice:
            result = await grounding_callbacks.before_model_callback(
                callback_context, request
            )
        return result
    finally:
        grounding_callbacks._reset_turn_message(token)


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


class SG3ModelLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_before_model_initializes_new_runtime_and_never_changes_request(self):
        current = state("sg3-model")
        request = {"contents": [{"role": "user", "text": "unchanged"}]}
        original = copy.deepcopy(request)
        sentinel = object()
        async def fake_baseline(callback_context, llm_request):
            del llm_request
            calls = callback_context.state.setdefault("system_agent_llm_calls", [])
            calls.append({"actions": []})
            callback_context.state["_active_llm_call_index"] = len(calls) - 1
            return sentinel

        delegate = AsyncMock(side_effect=fake_baseline)
        with patch.object(baseline_callbacks, "before_model_callback", delegate):
            result = await run_bound_query(current, request, twice=True)

        self.assertIs(result, sentinel)
        self.assertEqual(delegate.await_count, 2)
        self.assertEqual(request, original)
        self.assertEqual(runtime(current), GroundingRuntime())
        self.assertNotIn("valibra:grounding_runtime", current)
        first = current["system_agent_llm_calls"][0]
        self.assertEqual(
            first[grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY][
                "observation_type"
            ],
            "user_query",
        )
        self.assertNotIn(
            grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY,
            current["system_agent_llm_calls"][1],
        )
        view = first[grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY]
        self.assertTrue(view["request_unchanged"])
        self.assertFalse(view["injected"])
        self.assertNotIn("view", view)

    async def test_after_model_delegates_once_and_returns_exact_object(self):
        response = SimpleNamespace(content="same")
        sentinel = object()
        delegate = AsyncMock(return_value=sentinel)
        callback_context = SimpleNamespace(state={})
        with patch.object(baseline_callbacks, "after_model_callback", delegate):
            result = await grounding_callbacks.after_model_callback(
                callback_context, response
            )
        self.assertIs(result, sentinel)
        self.assertEqual(response.content, "same")
        delegate.assert_awaited_once_with(callback_context, response)

    async def test_shadow_hash_or_audit_failure_cannot_replace_baseline_result(self):
        current = state("sg3-model-fail-open")
        sentinel = object()
        delegate = AsyncMock(return_value=sentinel)
        with (
            patch.object(baseline_callbacks, "before_model_callback", delegate),
            patch.object(
                grounding_callbacks,
                "_request_sha256",
                side_effect=RuntimeError("synthetic hash failure"),
            ),
        ):
            result = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=current), {"model_visible": "same"}
            )
        self.assertIs(result, sentinel)
        delegate.assert_awaited_once()
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_ERROR_AUDIT_KEY][-1][
                "service_status"
            ],
            "failed_open",
        )

        async def fake_baseline(callback_context, request):
            del request
            callback_context.state["system_agent_llm_calls"].append({"actions": []})
            callback_context.state["_active_llm_call_index"] = 0
            return sentinel

        current = state("sg3-audit-fail-open")
        with (
            patch.object(baseline_callbacks, "before_model_callback", fake_baseline),
            patch.object(
                grounding_callbacks,
                "_attach_model_call_audit",
                side_effect=RuntimeError("synthetic audit failure"),
            ),
        ):
            result = await run_bound_query(current, {"safe": True})
        self.assertIs(result, sentinel)
        self.assertEqual(
            current[grounding_callbacks.GROUNDING_ERROR_AUDIT_KEY][-1]["stage"],
            "before_model_audit",
        )

    async def test_corrupt_new_runtime_is_degraded_and_old_runtime_is_ignored(self):
        current = state("sg3-degraded")
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = {"bad": "shape"}
        current["valibra:grounding_runtime"] = {"secret_old_shape": True}
        request = {"prompt": "same"}
        async def fake_baseline(callback_context, llm_request):
            del llm_request
            calls = callback_context.state.setdefault("system_agent_llm_calls", [])
            calls.append({"actions": []})
            callback_context.state["_active_llm_call_index"] = len(calls) - 1
            return None

        with patch.object(
            baseline_callbacks,
            "before_model_callback",
            AsyncMock(side_effect=fake_baseline),
        ):
            await run_bound_query(current, request)
        self.assertEqual(runtime(current), GroundingRuntime())
        audit = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        self.assertTrue(audit["runtime_degraded"])
        self.assertEqual(current["valibra:grounding_runtime"], {"secret_old_shape": True})

    async def test_concurrent_contextvars_do_not_cross_queries_or_runtime(self):
        first = state("sg3-a")
        second = state("sg3-b")

        async def delegate(callback_context, request):
            del request
            await asyncio.sleep(0)
            calls = callback_context.state.setdefault("system_agent_llm_calls", [])
            calls.append({"actions": []})
            callback_context.state["_active_llm_call_index"] = len(calls) - 1
            return None

        async def one(current, query):
            token = grounding_callbacks._bind_turn_message(
                current["task_id"], "a-interact", query
            )
            try:
                await grounding_callbacks.before_model_callback(
                    SimpleNamespace(state=current), {"task": current["task_id"]}
                )
            finally:
                grounding_callbacks._reset_turn_message(token)

        with patch.object(baseline_callbacks, "before_model_callback", delegate):
            await asyncio.gather(one(first, "Query A"), one(second, "Query B"))
        a = first["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY
        ]
        b = second["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY
        ]
        self.assertNotEqual(a["observation_id"], b["observation_id"])
        self.assertEqual(runtime(first), GroundingRuntime())
        self.assertEqual(runtime(second), GroundingRuntime())


class SG3ToolLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.turn_token = None

    async def asyncTearDown(self):
        if self.turn_token is not None:
            grounding_callbacks._reset_turn_message(self.turn_token)

    def bind(self, current: dict) -> None:
        self.turn_token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", QUERY
        )

    async def test_baseline_budget_rejection_creates_no_pending_and_matches_b0(self):
        valibra_state = state("sg3-budget-v", budget=1.0)
        baseline_state = state("sg3-budget-b", budget=1.0)
        tool = SimpleNamespace(name="ask_user")
        args = {"question": "q"}
        valibra_result = await grounding_callbacks.before_tool_callback(
            tool, args, context(valibra_state, "call-v")
        )
        baseline_result = await baseline_callbacks.before_tool_callback(
            tool, args, context(baseline_state, "call-b")
        )
        self.assertEqual(valibra_result, baseline_result)
        self.assertEqual(
            valibra_state["budget_remaining"], baseline_state["budget_remaining"]
        )
        self.assertEqual(
            valibra_state.get(grounding_callbacks.GROUNDING_PENDING_KEY), None
        )

    async def test_pending_is_registered_only_after_baseline_and_outside_runtime(self):
        current = state("sg3-pending-order")
        tool = SimpleNamespace(name="execute_sql")
        call_context = context(current, "call-order")

        async def baseline_first(tool, args, tool_context):
            del tool, args
            self.assertNotIn(
                grounding_callbacks.GROUNDING_PENDING_KEY,
                tool_context.state,
            )
            return None

        with patch.object(
            baseline_callbacks, "before_tool_callback", baseline_first
        ):
            result = await grounding_callbacks.before_tool_callback(
                tool,
                {"sql": "SELECT 1"},
                call_context,
            )
        self.assertIsNone(result)
        pending = current[grounding_callbacks.GROUNDING_PENDING_KEY]["call-order"]
        self.assertEqual(pending["tool_name"], "execute_sql")
        self.assertRegex(pending["args_digest"], r"^[0-9a-f]{64}$")
        runtime_payload = GroundingRuntime().model_dump(mode="json")
        self.assertNotIn("pending", json.dumps(runtime_payload))
        self.assertNotIn("sequence", json.dumps(runtime_payload))

    async def test_original_response_not_baseline_override_builds_digest(self):
        current = state("sg3-raw")
        self.bind(current)
        tool = SimpleNamespace(name="execute_sql")
        args = {"sql": "SELECT 1"}
        call_context = context(current, "call-raw")
        original = {"rows": [{"value": 1}]}
        override = {"visible": "baseline override"}

        async def fake_after(tool, args, tool_context, tool_response):
            del tool, args
            tool_context.state["tool_trajectory"].append(
                {"type": "tool", "result": tool_response}
            )
            return override

        with (
            patch.object(
                baseline_callbacks, "before_tool_callback", AsyncMock(return_value=None)
            ) as before,
            patch.object(baseline_callbacks, "after_tool_callback", fake_after),
        ):
            await grounding_callbacks.before_tool_callback(
                tool, args, call_context
            )
            result = await grounding_callbacks.after_tool_callback(
                tool, args, call_context, original
            )
        self.assertIs(result, override)
        self.assertEqual(before.await_count, 1)
        audit = current["tool_trajectory"][0][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(audit["raw_digest"], raw_digest(original))
        self.assertNotEqual(audit["raw_digest"], raw_digest(override))
        self.assertEqual(
            audit["service_status"], "stored_official_evidence_only"
        )
        self.assertEqual(current[grounding_callbacks.GROUNDING_PENDING_KEY], {})

    async def test_bird_coin_and_official_trajectory_fields_match_baseline(self):
        valibra_state = state("sg3-parity-v", budget=10.0)
        baseline_state = state("sg3-parity-b", budget=10.0)
        self.bind(valibra_state)
        tool = SimpleNamespace(name="execute_sql")
        args = {"sql": "SELECT 1"}
        response = "one row"
        with patch.object(baseline_callbacks, "utc_now", return_value="fixed"):
            await grounding_callbacks.before_tool_callback(
                tool, args, context(valibra_state, "call-parity")
            )
            valibra_override = await grounding_callbacks.after_tool_callback(
                tool,
                args,
                context(valibra_state, "call-parity"),
                response,
            )
            await baseline_callbacks.before_tool_callback(
                tool, args, context(baseline_state, "call-baseline")
            )
            baseline_override = await baseline_callbacks.after_tool_callback(
                tool,
                args,
                context(baseline_state, "call-baseline"),
                response,
            )
        self.assertEqual(valibra_override, baseline_override)
        self.assertEqual(
            valibra_state["budget_remaining"], baseline_state["budget_remaining"]
        )
        valibra_event = dict(valibra_state["tool_trajectory"][0])
        self.assertIn(grounding_callbacks.SHADOW_AUDIT_KEY, valibra_event)
        valibra_event.pop(grounding_callbacks.SHADOW_AUDIT_KEY)
        valibra_event.pop(grounding_callbacks.GROUNDING_CONTROL_AUDIT_KEY)
        self.assertEqual(valibra_event, baseline_state["tool_trajectory"][0])

    async def test_same_name_calls_pair_out_of_order_by_exact_id(self):
        current = state("sg3-pair")
        self.bind(current)
        tool = SimpleNamespace(name="execute_sql")
        first = context(current, "same-a")
        second = context(current, "same-b")

        async def fake_after(tool, args, tool_context, tool_response):
            del tool, args
            tool_context.state["tool_trajectory"].append(
                {"result": tool_response}
            )
            return None

        with (
            patch.object(
                baseline_callbacks, "before_tool_callback", AsyncMock(return_value=None)
            ),
            patch.object(baseline_callbacks, "after_tool_callback", fake_after),
        ):
            await grounding_callbacks.before_tool_callback(
                tool, {"sql": "SELECT 1"}, first
            )
            await grounding_callbacks.before_tool_callback(
                tool, {"sql": "SELECT 2"}, second
            )
            self.assertEqual(
                set(current[grounding_callbacks.GROUNDING_PENDING_KEY]),
                {"same-a", "same-b"},
            )
            await grounding_callbacks.after_tool_callback(
                tool, {"sql": "SELECT 2"}, second, "second"
            )
            self.assertEqual(
                set(current[grounding_callbacks.GROUNDING_PENDING_KEY]), {"same-a"}
            )
            await grounding_callbacks.after_tool_callback(
                tool, {"sql": "SELECT 1"}, first, "first"
            )
        self.assertEqual(current[grounding_callbacks.GROUNDING_PENDING_KEY], {})
        ids = [
            item[grounding_callbacks.SHADOW_AUDIT_KEY]["function_call_id"]
            for item in current["tool_trajectory"]
        ]
        self.assertEqual(ids, ["same-b", "same-a"])

    async def test_official_error_and_exception_stay_model_invisible(self):
        current = state("sg3-errors")
        self.bind(current)
        tool = SimpleNamespace(name="execute_sql")
        official = context(current, "official-error")
        raised = context(current, "raised-error")

        async def fake_after(tool, args, tool_context, tool_response):
            del tool, args
            tool_context.state["tool_trajectory"].append({"result": tool_response})
            return "B0-visible"

        with (
            patch.object(
                baseline_callbacks, "before_tool_callback", AsyncMock(return_value=None)
            ),
            patch.object(baseline_callbacks, "after_tool_callback", fake_after),
        ):
            await grounding_callbacks.before_tool_callback(tool, {}, official)
            returned = await grounding_callbacks.after_tool_callback(
                tool,
                {},
                official,
                "Error calling DB environment: offline",
            )
            await grounding_callbacks.before_tool_callback(tool, {}, raised)
            error_returned = await grounding_callbacks.on_tool_error_callback(
                tool, {}, raised, RuntimeError("private outage details")
            )
        self.assertEqual(returned, "B0-visible")
        self.assertIsNone(error_returned)
        audit = current["tool_trajectory"][0][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertTrue(audit["official_error"])
        self.assertEqual(audit["observation_type"], "tool_error")
        self.assertEqual(current[grounding_callbacks.GROUNDING_PENDING_KEY], {})
        serialized = json.dumps(current)
        self.assertNotIn("private outage details", serialized)

    async def test_submit_follow_up_is_captured_with_official_control_transition(self):
        current = state("sg3-submit")
        current["tool_trajectory"].extend(
            [
                {
                    "type": "tool",
                    "tool": "get_schema",
                    "phase": 1,
                    "args": {},
                    "result": "CREATE TABLE users (\n  active BOOLEAN\n);",
                },
                {
                    "type": "tool",
                    "tool": "get_all_column_meanings",
                    "phase": 1,
                    "args": {},
                    "result": "{}",
                },
                {
                    "type": "tool",
                    "tool": "get_all_knowledge_definitions",
                    "phase": 1,
                    "args": {},
                    "result": "[]",
                },
            ]
        )
        current[grounding_callbacks.GROUNDING_RUNTIME_KEY] = GroundingRuntime(
            stage="SQL_ATTEMPT",
            focus_dimension="none",
            grounding_state=SQLGroundingState(
                tables=(),
                join_keys=(),
                column_mapping=(),
                domain_knowledge=(),
            ),
        ).model_dump(mode="json")
        self.bind(current)
        tool = SimpleNamespace(name="submit_sql")
        call_context = context(current, "call-submit")
        raw = (
            "Phase 1 passed\nFollow-up question: Now include active users."
            "\nBudget remaining: 4 bird-coins"
        )

        async def fake_after(tool, args, tool_context, tool_response):
            del tool, args
            tool_context.state["phase1_completed"] = True
            tool_context.state["current_phase"] = 2
            tool_context.state["tool_trajectory"].append({"result": tool_response})
            return "same override"

        with (
            patch.object(
                baseline_callbacks, "before_tool_callback", AsyncMock(return_value=None)
            ),
            patch.object(baseline_callbacks, "after_tool_callback", fake_after),
        ):
            await grounding_callbacks.before_tool_callback(
                tool, {"sql": "SELECT 1"}, call_context
            )
            returned = await grounding_callbacks.after_tool_callback(
                tool, {"sql": "SELECT 1"}, call_context, raw
            )
        self.assertEqual(returned, "same override")
        audit = current["tool_trajectory"][-1][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(audit["service_status"], "skipped_control_lifecycle_only")
        staged = audit["p2_follow_up"]["staged_grounding"]
        self.assertEqual(
            [item["service_status"] for item in staged],
            ["rejected", "rejected", "noop"],
        )
        self.assertEqual(
            [item["observation_type"] for item in staged],
            ["p2_follow_up"] * 3,
        )
        self.assertEqual(runtime(current).stage, "P2_INCREMENTAL")

    async def test_schema_result_is_evidence_only_after_stage3(self):
        current = state("sg3-scripted")
        self.bind(current)
        tool = SimpleNamespace(name="get_schema")
        call_context = context(current, "call-schema")
        target_state = SQLGroundingState(
            tables=("operational_metrics",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("operational_metrics.maintcost",),
                ),
            ),
            domain_knowledge=(),
        )
        updater = ScriptedUpdater(
            GroundingLLMResponse(
                sql_grounding_state=target_state,
                user_clarification_requests=(),
                next_focus_dimension="none",
            )
        )

        def validation_context(current_state, observation):
            del current_state
            return ValidationContext(
                current_query=QUERY,
                latest_observation_id=observation.observation_id,
                known_tables=frozenset({"operational_metrics"}),
                known_columns=frozenset({"operational_metrics.maintcost"}),
            )

        async def fake_after(tool, args, tool_context, tool_response):
            del tool, args
            tool_context.state["tool_trajectory"].append({"result": tool_response})
            return None

        with (
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
            patch.object(
                grounding_callbacks,
                "_build_validation_context",
                validation_context,
            ),
            patch.object(
                baseline_callbacks, "before_tool_callback", AsyncMock(return_value=None)
            ),
            patch.object(baseline_callbacks, "after_tool_callback", fake_after),
        ):
            await grounding_callbacks.before_tool_callback(tool, {}, call_context)
            await grounding_callbacks.after_tool_callback(
                tool, {}, call_context, {"tables": ["operational_metrics"]}
            )
        self.assertEqual(updater.calls, 0)
        updated = runtime(current)
        self.assertEqual(updated.grounding_revision, 0)
        self.assertEqual(updated.grounding_state, SQLGroundingState())
        audit = current["tool_trajectory"][0][grounding_callbacks.SHADOW_AUDIT_KEY]
        self.assertEqual(audit["service_status"], "stored_official_evidence_only")

        request = {"model_visible": "unchanged"}
        before = copy.deepcopy(request)
        async def fake_before(callback_context, llm_request):
            del llm_request
            calls = callback_context.state.setdefault("system_agent_llm_calls", [])
            calls.append({"actions": []})
            callback_context.state["_active_llm_call_index"] = len(calls) - 1
            return None

        with patch.object(baseline_callbacks, "before_model_callback", fake_before):
            await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=current), request
            )
        self.assertEqual(request, before)
        view_audit = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        self.assertEqual(
            view_audit["view_sha256"],
            render_grounding_view(updated.grounding_state).sha256,
        )
        self.assertFalse(view_audit["injected"])


class SG3SourceAndFreezeTests(unittest.TestCase):
    def test_all_official_tool_observation_mappings_are_exact(self):
        self.assertEqual(
            dict(grounding_callbacks._TOOL_OBSERVATION_TYPES),
            {
                "get_schema": "schema",
                "get_all_column_meanings": "metadata",
                "get_column_meaning": "metadata",
                "get_all_external_knowledge_names": "knowledge",
                "get_knowledge_definition": "knowledge",
                "get_all_knowledge_definitions": "knowledge",
                "execute_sql": "sql_execution",
                "ask_user": "user_answer",
                "submit_sql": "submission",
            },
        )

    def test_source_uses_only_new_core_and_frozen_gate_contract(self):
        source = inspect.getsource(grounding_callbacks)
        self.assertNotIn("requirement_grounding", source)
        self.assertNotIn('"valibra:grounding_runtime"', source)
        self.assertNotIn("LiteLLM", source)
        self.assertIn("append_instructions", source)
        self.assertIn("transition_grounding_stage", source)
        self.assertIn("evaluate_first_submit_gate", source)
        self.assertIn("render_control_hint", source)

    def test_frozen_hashes_and_production_passthrough_are_unchanged(self):
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)
        self.assertEqual(
            type(grounding_callbacks._SQL_GROUNDING_UPDATER).__name__,
            "_PassthroughSQLGroundingUpdater",
        )


if __name__ == "__main__":
    unittest.main()
