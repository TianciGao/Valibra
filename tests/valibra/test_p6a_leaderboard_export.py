import asyncio
import copy
import inspect
import json
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from tests.valibra.k0_fixtures import (
    graph_patch,
    make_ambiguity,
    make_candidate,
    make_observation,
)
from tests.valibra.test_p4d_llm_callback_shadow import (
    FakeClient,
    QUERY,
    QUERY_CONTENT,
    _llm_config,
)
from valibra_agent import evaluation, grounding_callbacks
from valibra_agent.evaluation import export_valibra_result
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
    RuntimeMetrics,
    ValibraError,
)
from valibra_agent.requirement_grounding.reducer import apply_patch
from valibra_agent.requirement_grounding.updater import (
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    LLMUpdater,
)


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
B0_ROOT = Path("/home/user/code/BIRD-Interact/BIRD-Interact-ADK")
EXPECTED_PROMPT_SHA256 = (
    "5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d"
)
EXPECTED_FORM_SHA256 = (
    "441a59c410a99ef0db53b8e974aeeeaea1bcd1735aabdc3cb51f59e0b6e069a2"
)
EXPECTED_CONFIG_SHA256 = (
    "83ba93c060b110a0e48485f8d5083052d96a67c8a79892ab77024be3c4b5ccd9"
)


SNAPSHOT_CODE = r'''
import importlib
import inspect
import json
import os

from orchestrator.ainteract import calculate_initial_budget
from system_agent.callbacks import TOOL_COSTS
from system_agent.tools import get_ainteract_tools, submit_sql

server = importlib.import_module(os.environ["SNAP_SERVER_MODULE"])
fixed = [
    {},
    {"user_query_ambiguity": {"critical_ambiguity": [{"x": 1}]}},
    {"knowledge_ambiguity": [{"x": 1}, {"x": 2}]},
    {
        "user_query_ambiguity": {"critical_ambiguity": [{}, {}, {}]},
        "knowledge_ambiguity": [{}, {}],
    },
]
routes = []
for route in server.app.routes:
    if route.path in {"/init_session", "/run_session", "/cleanup_session", "/health"}:
        routes.append({
            "path": route.path,
            "methods": sorted(route.methods or []),
            "endpoint": route.endpoint.__name__,
        })
payload = {
    "tools": [
        {"name": tool.name, "signature": str(inspect.signature(tool.func))}
        for tool in get_ainteract_tools()
    ],
    "tool_costs": dict(sorted(TOOL_COSTS.items())),
    "budgets": [calculate_initial_budget(item) for item in fixed],
    "submit_sql_signature": str(inspect.signature(submit_sql)),
    "routes": sorted(routes, key=lambda item: item["path"]),
    "schemas": {
        "init": server.SessionInitRequest.model_json_schema(mode="validation"),
        "run": server.SessionRunRequest.model_json_schema(mode="validation"),
        "cleanup": server.SessionCleanupRequest.model_json_schema(mode="validation"),
    },
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''


def _normalize_contract(value):
    if isinstance(value, list):
        return [_normalize_contract(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_contract(value[key])
            for key in sorted(value)
            if key != "description"
        }
    return value


def _contract_snapshot(root, python_path, server_module):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["SNAP_SERVER_MODULE"] = server_module
    completed = subprocess.run(
        [str(python_path), "-c", SNAPSHOT_CODE],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _populated_runtime(*, include_cost=False):
    runtime = RequirementGroundingRuntime()
    observation = make_observation()
    candidates = (
        make_candidate("cand-a", "A", "effect-a"),
        make_candidate("cand-b", "B", "effect-b"),
    )
    ambiguity = make_ambiguity(candidates=candidates, status="unresolved")
    runtime = apply_patch(
        runtime,
        graph_patch(runtime, observation, ambiguity=ambiguity),
    )
    metrics = {
        "llm_updater_calls": 2,
        "llm_updater_errors": 1,
        "llm_updater_timeouts": 1,
        "llm_updater_input_tokens": 100,
        "llm_updater_output_tokens": 40,
        "llm_updater_reasoning_tokens": 20,
        "llm_updater_total_tokens": 140,
        "llm_updater_latency_ms": 1250.0,
    }
    if include_cost:
        metrics["llm_updater_cost"] = 0.25
    return runtime.model_copy(
        update={
            "metrics": RuntimeMetrics.model_validate(metrics),
            "last_error": ValibraError(
                stage="updater",
                error_type="SyntheticBoundedError",
                message_preview="bounded",
                observation_id=observation.observation_id,
                sequence=2,
                timestamp="2026-08-09T00:00:00+00:00",
            ),
        }
    )


def _state(*, include_cost=False):
    runtime = _populated_runtime(include_cost=include_cost)
    prompt_flow = [
        {
            "call_index": 1,
            "timestamp": "2026-08-09T00:00:00+00:00",
            "completed_at": "2026-08-09T00:00:00.100000+00:00",
            "usage": {"raw": {"cost": 0.10}} if include_cost else {},
            grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY: {
                "observation_id": runtime.processed_observation_ids[0],
                "observation_type": "user_query",
                "phase": 1,
                "status": "processed",
                "grounding_revision": 1,
                "requirement_revision": 1,
                "llm": {
                    "attempted": True,
                    "cost": 0.10 if include_cost else None,
                },
            },
            grounding_callbacks.REQUIREMENT_VIEW_AUDIT_KEY: {
                "effective_mode": "active",
                "injected": True,
                "view_sha256": "a" * 64,
            },
        },
        {
            "call_index": 2,
            "timestamp": "2026-08-09T00:00:01+00:00",
            "completed_at": "2026-08-09T00:00:01.200000+00:00",
            "usage": {"raw": {"response_cost": 0.20}} if include_cost else {},
        },
    ]
    tool_trajectory = [
        {
            "tool": "ask_user",
            grounding_callbacks.SHADOW_AUDIT_KEY: {
                "observation_id": "b" * 64,
                "observation_type": "user_answer",
                "tool_name": "ask_user",
                "status": "processed",
                "llm": {
                    "attempted": True,
                    "cost": 0.15 if include_cost else None,
                },
            },
        },
        {
            "tool": "submit_sql",
            grounding_callbacks.SHADOW_AUDIT_KEY: {
                "observation_id": "c" * 64,
                "observation_type": "submission",
                "tool_name": "submit_sql",
                "status": "processed",
                "follow_up_observation_id": "d" * 64,
                "follow_up_status": "processed",
            },
        },
    ]
    return {
        grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
        "system_agent_token_usage": {
            "model_calls": 2,
            "input_tokens": 200,
            "output_tokens": 80,
            "reasoning_tokens": 30,
            "total_tokens": 280,
        },
        "system_agent_llm_calls": prompt_flow,
        "tool_trajectory": tool_trajectory,
        "dialogue_history": [{"role": "user", "text": "bounded"}],
        "adk_events": [{"type": "adk_event", "final": True}],
        "initial_budget": 20.0,
        "budget_remaining": 9.5,
    }


class FormalB0ContractTests(unittest.TestCase):
    def test_isolated_b0_and_research_runtime_contracts_match(self):
        b0 = _contract_snapshot(
            B0_ROOT,
            B0_ROOT / ".venv-adk/bin/python",
            "system_agent.server",
        )
        research = _contract_snapshot(
            RESEARCH_ROOT,
            RESEARCH_ROOT / ".venv-research/bin/python",
            "valibra_agent.server",
        )
        self.assertEqual(_normalize_contract(b0), _normalize_contract(research))
        self.assertEqual(len(b0["tools"]), 9)
        self.assertEqual(b0["budgets"], [12.0, 14.0, 16.0, 22.0])
        self.assertEqual(len(b0["routes"]), 4)


class EvaluationExportTests(unittest.TestCase):
    def test_complete_runtime_ledger_latency_and_locations_export(self):
        state = _state()
        result = export_valibra_result(
            state,
            user_simulator_audit={"llm_calls": [], "token_usage": {"total_tokens": 7}},
        )
        self.assertEqual(result["export_status"], "succeeded")
        restored = RequirementGroundingRuntime.model_validate(result["runtime"])
        self.assertEqual(restored.model_dump(mode="json"), result["runtime"])
        summary = result["grounding_summary"]
        self.assertEqual(summary["grounding_revision"], 1)
        self.assertEqual(summary["requirement_revision"], 1)
        self.assertRegex(summary["requirement_semantic_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(summary["slots"]["total"], 1)
        self.assertEqual(summary["evidence_count"], 1)
        self.assertEqual(summary["ambiguity_count"], 1)
        self.assertEqual(summary["processed_observation_count"], 1)
        self.assertEqual(summary["pending_count"], 0)
        self.assertEqual(summary["llm_calls"], 2)
        self.assertEqual(summary["llm_errors"], 1)
        self.assertEqual(summary["llm_timeouts"], 1)
        self.assertEqual(summary["latency_ms"], 1250.0)
        usage = result["model_usage"]
        self.assertEqual(usage["main_agent"]["total_tokens"], 280)
        self.assertEqual(usage["grounding"]["total_tokens"], 140)
        self.assertEqual(usage["grounding"]["reasoning_tokens"], 20)
        self.assertEqual(usage["total_model_tokens"], 420)
        self.assertIsNone(usage["main_agent"]["cost"])
        self.assertIsNone(usage["grounding"]["cost"])
        self.assertIsNone(usage["total_model_cost"])
        self.assertAlmostEqual(result["latency"]["main_agent_ms"], 300.0)
        self.assertEqual(result["latency"]["grounding_ms"], 1250.0)
        self.assertFalse(result["bird_coin"]["included_in_model_usage"])
        locations = result["trajectory_manifest"]["locations"]
        self.assertEqual(len(locations["initial_user_query_grounding"]), 1)
        self.assertEqual(len(locations["ask_user_answer_grounding"]), 1)
        self.assertEqual(len(locations["phase2_follow_up_grounding"]), 1)
        self.assertEqual(len(locations["tool_observations"]), 2)
        self.assertEqual(len(locations["requirement_view_model_calls"]), 1)

    def test_reasoning_is_not_double_counted_and_reported_costs_sum(self):
        result = export_valibra_result(_state(include_cost=True))
        usage = result["model_usage"]
        self.assertEqual(usage["total_model_tokens"], 280 + 140)
        self.assertAlmostEqual(usage["main_agent"]["cost"], 0.30)
        self.assertAlmostEqual(usage["grounding"]["cost"], 0.25)
        self.assertAlmostEqual(usage["total_model_cost"], 0.55)
        self.assertNotIn("bird_coin", usage)

    def test_partial_grounding_cost_is_not_reported_as_complete(self):
        state = _state(include_cost=True)
        shadow = state["tool_trajectory"][0][
            grounding_callbacks.SHADOW_AUDIT_KEY
        ]
        shadow["llm"]["cost"] = None
        result = export_valibra_result(state)
        usage = result["model_usage"]
        self.assertAlmostEqual(usage["main_agent"]["cost"], 0.30)
        self.assertIsNone(usage["grounding"]["cost"])
        self.assertIsNone(usage["total_model_cost"])

    def test_missing_grounding_call_audit_makes_cost_unknown(self):
        state = _state(include_cost=True)
        del state["tool_trajectory"][0][
            grounding_callbacks.SHADOW_AUDIT_KEY
        ]["llm"]
        result = export_valibra_result(state)
        self.assertIsNone(result["model_usage"]["grounding"]["cost"])
        self.assertIsNone(result["model_usage"]["total_model_cost"])

    def test_zero_grounding_calls_have_zero_cost(self):
        state = _state(include_cost=True)
        runtime = RequirementGroundingRuntime()
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY] = runtime.model_dump(
            mode="json"
        )
        for call in state["system_agent_llm_calls"]:
            call.pop(grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY, None)
        for event in state["tool_trajectory"]:
            shadow = event.get(grounding_callbacks.SHADOW_AUDIT_KEY, {})
            shadow.pop("llm", None)
            shadow.pop("follow_up_llm", None)
        result = export_valibra_result(state)
        self.assertEqual(result["model_usage"]["grounding"]["cost"], 0.0)
        self.assertAlmostEqual(result["model_usage"]["total_model_cost"], 0.30)

    def test_main_agent_cost_requires_every_call_to_report(self):
        state = _state(include_cost=True)
        state["system_agent_llm_calls"][1]["usage"] = {"raw": {}}
        result = export_valibra_result(state)
        usage = result["model_usage"]
        self.assertIsNone(usage["main_agent"]["cost"])
        self.assertAlmostEqual(usage["grounding"]["cost"], 0.25)
        self.assertIsNone(usage["total_model_cost"])

    def test_manifest_is_stable_and_changes_only_for_changed_collection(self):
        state = _state()
        first = export_valibra_result(state)["trajectory_manifest"]
        second = export_valibra_result(copy.deepcopy(state))["trajectory_manifest"]
        self.assertEqual(first, second)
        changed = copy.deepcopy(state)
        changed["dialogue_history"].append({"role": "model", "text": "new"})
        third = export_valibra_result(changed)["trajectory_manifest"]
        self.assertNotEqual(
            first["dialogue_history"]["sha256"],
            third["dialogue_history"]["sha256"],
        )
        for key in (
            "prompt_flow",
            "tool_trajectory",
            "adk_events",
            "user_simulator_audit",
            "grounding_runtime",
        ):
            self.assertEqual(first[key], third[key])

    def test_missing_and_invalid_runtime_fail_open_with_bounded_type(self):
        self.assertEqual(
            export_valibra_result({}),
            {
                "export_status": "failed",
                "error_type": "MissingGroundingRuntime",
            },
        )
        invalid = export_valibra_result(
            {grounding_callbacks.GROUNDING_RUNTIME_KEY: {"secret": "not exported"}}
        )
        self.assertEqual(invalid["export_status"], "failed")
        self.assertEqual(invalid["error_type"], "ValidationError")
        self.assertNotIn("secret", json.dumps(invalid))

    def test_main_latency_is_null_without_reliable_complete_timestamps(self):
        state = _state()
        del state["system_agent_llm_calls"][1]["completed_at"]
        result = export_valibra_result(state)
        self.assertIsNone(result["latency"]["main_agent_ms"])

    def test_adapter_has_no_hidden_task_or_network_access(self):
        source = inspect.getsource(evaluation)
        for forbidden in (
            "task_data",
            "sol_sql",
            "test_cases",
            "_last_submit_raw",
            "os.environ",
            "httpx",
            "requests.",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_frozen_grounding_contracts_are_unchanged(self):
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, EXPECTED_PROMPT_SHA256)
        self.assertEqual(LLM_FRAME_FORM_SCHEMA_SHA256, EXPECTED_FORM_SHA256)
        self.assertEqual(
            _llm_config(timeout="300", max_calls="2").configuration_sha256,
            EXPECTED_CONFIG_SHA256,
        )


class InitialGroundingAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_query_audit_attaches_to_exact_model_call_once(self):
        state = {
            "task_id": "p6a-audit",
            "current_phase": 1,
            "budget_remaining": 20.0,
            "initial_budget": 20.0,
            "tool_trajectory": [],
            "system_agent_llm_calls": [],
        }
        request = {"contents": [{"text": "main request remains unchanged"}]}
        before = copy.deepcopy(request)
        client = FakeClient(content=QUERY_CONTENT)
        updater = LLMUpdater(client, _llm_config())

        async def baseline_delegate(callback_context, llm_request):
            calls = list(callback_context.state["system_agent_llm_calls"])
            calls.append({"request": copy.deepcopy(llm_request), "actions": []})
            callback_context.state["system_agent_llm_calls"] = calls
            callback_context.state["_active_llm_call_index"] = len(calls) - 1
            return None

        token = grounding_callbacks._bind_turn_message(
            state["task_id"],
            "a-interact",
            QUERY,
        )
        try:
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
                    return_value=updater,
                ),
                patch.object(
                    baseline_callbacks,
                    "before_model_callback",
                    AsyncMock(side_effect=baseline_delegate),
                ),
            ):
                context = SimpleNamespace(state=state)
                await grounding_callbacks.before_model_callback(context, request)
                await grounding_callbacks.before_model_callback(context, request)
        finally:
            grounding_callbacks._reset_turn_message(token)

        self.assertEqual(request, before)
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(state["system_agent_llm_calls"]), 2)
        first = state["system_agent_llm_calls"][0]
        second = state["system_agent_llm_calls"][1]
        audit = first[grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY]
        self.assertEqual(audit["observation_type"], "user_query")
        self.assertEqual(audit["phase"], 1)
        self.assertEqual(audit["status"], "processed")
        self.assertGreater(audit["grounding_revision"], 0)
        self.assertGreater(audit["requirement_revision"], 0)
        self.assertRegex(audit["requirement_semantic_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(audit["llm"]["input_tokens"], 12)
        self.assertEqual(audit["llm"]["output_tokens"], 9)
        self.assertEqual(audit["llm"]["reasoning_tokens"], 4)
        serialized_audit = json.dumps(audit, sort_keys=True)
        self.assertNotIn(QUERY, serialized_audit)
        self.assertNotIn(QUERY_CONTENT, serialized_audit)
        self.assertNotIn("contents", audit["llm"])
        self.assertNotIn(grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY, second)

    async def test_grounding_audit_attachment_failure_is_fail_open(self):
        state = {
            "task_id": "p6a-audit-failure",
            "system_agent_llm_calls": [],
        }
        request = {"contents": [{"text": "unchanged"}]}
        before = copy.deepcopy(request)

        async def baseline_delegate(callback_context, llm_request):
            callback_context.state["system_agent_llm_calls"] = [
                {"request": copy.deepcopy(llm_request), "actions": []}
            ]
            callback_context.state["_active_llm_call_index"] = 0
            return "baseline-result"

        with (
            patch.object(
                grounding_callbacks,
                "_attach_model_call_audit",
                side_effect=RuntimeError("synthetic audit failure"),
            ),
            patch.object(
                baseline_callbacks,
                "before_model_callback",
                AsyncMock(side_effect=baseline_delegate),
            ),
        ):
            result = await grounding_callbacks.before_model_callback(
                SimpleNamespace(state=state),
                request,
            )
        self.assertEqual(result, "baseline-result")
        self.assertEqual(request, before)


class OrchestratorAdditiveExportTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _task():
        return {
            "instance_id": "p6a-task",
            "selected_database": "p6a_db",
            "amb_user_query": "Find the rows.",
            "user_query_ambiguity": {"critical_ambiguity": []},
            "knowledge_ambiguity": [],
        }

    @staticmethod
    def _run_state():
        runtime = RequirementGroundingRuntime()
        return {
            grounding_callbacks.GROUNDING_RUNTIME_KEY: runtime.model_dump(mode="json"),
            "phase1_completed": True,
            "phase2_completed": False,
            "total_reward": 1.0,
            "budget_remaining": 4.0,
            "dialogue_history": [{"role": "user", "text": "bounded"}],
            "tool_trajectory": [
                {
                    "tool": "submit_sql",
                    "phase": 1,
                    "args": {"sql": "SELECT 1"},
                }
            ],
            "adk_events": [{"final": True}],
            "system_agent_llm_calls": [],
            "system_agent_token_usage": {
                "model_calls": 0,
                "input_tokens": 10,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "total_tokens": 15,
            },
        }

    async def _run(self, *, exporter=None):
        from orchestrator import ainteract

        task = self._task()
        run_state = self._run_state()
        events = []

        async def fake_post(url, payload, timeout=120.0):
            events.append(
                {
                    "method": "POST",
                    "url": url,
                    "payload": copy.deepcopy(payload),
                    "timeout": timeout,
                }
            )
            if url.endswith("/run_session"):
                return {"state": copy.deepcopy(run_state), "response": "done"}
            return {"status": "ok"}

        async def fake_get(url, timeout=120.0):
            events.append(
                {"method": "GET", "url": url, "timeout": timeout}
            )
            return {
                "llm_calls": [],
                "token_usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "reasoning_tokens": 1,
                    "total_tokens": 5,
                },
            }

        clock = iter((100.0, 105.0))
        patches = [
            patch.object(ainteract, "_post", side_effect=fake_post),
            patch.object(ainteract, "_get", side_effect=fake_get),
            patch.object(
                ainteract,
                "time",
                SimpleNamespace(time=lambda: next(clock)),
            ),
        ]
        if exporter is not None:
            patches.append(
                patch.object(
                    ainteract,
                    "export_valibra_result",
                    side_effect=exporter,
                )
            )
        for item in patches:
            item.start()
        try:
            result = await ainteract.run_single_task(copy.deepcopy(task))
        finally:
            for item in reversed(patches):
                item.stop()
        return task, result, events

    async def test_http_order_payloads_and_legacy_result_are_unchanged(self):
        from orchestrator import ainteract

        task, result, events = await self._run()
        initial_budget = ainteract.calculate_initial_budget(task)
        expected_message = (
            "Database: p6a_db\n"
            "Task ID: p6a-task\n\n"
            "User Query:\nFind the rows.\n\n"
            f"You have a budget of {initial_budget:.1f} bird-coins. "
            "Use your tools to explore the database, clarify ambiguities with the user, "
            "and submit your final SQL efficiently."
        )
        self.assertEqual(
            events,
            [
                {
                    "method": "POST",
                    "url": f"{ainteract.DB_ENV_URL}/init_task",
                    "payload": {
                        "task_id": "p6a-task",
                        "task_data": {**task, "_interact_mode": "a-interact"},
                    },
                    "timeout": 120.0,
                },
                {
                    "method": "POST",
                    "url": f"{ainteract.USER_SIM_URL}/init_task",
                    "payload": {
                        "task_id": "p6a-task",
                        "task_data": {**task, "_interact_mode": "a-interact"},
                    },
                    "timeout": 120.0,
                },
                {
                    "method": "POST",
                    "url": f"{ainteract.SYSTEM_AGENT_URL}/init_session",
                    "payload": {
                        "task_id": "p6a-task",
                        "mode": "a-interact",
                        "state": {
                            "task_id": "p6a-task",
                            "db_name": "p6a_db",
                            "user_query": "Find the rows.",
                            "current_phase": 1,
                            "budget_remaining": initial_budget,
                            "initial_budget": initial_budget,
                            "total_reward": 0.0,
                            "dialogue_history": [],
                            "tool_trajectory": [],
                            "adk_events": [],
                            "phase1_completed": False,
                            "phase2_completed": False,
                            "task_done": False,
                        },
                        "reset": True,
                    },
                    "timeout": 30.0,
                },
                {
                    "method": "POST",
                    "url": f"{ainteract.SYSTEM_AGENT_URL}/run_session",
                    "payload": {
                        "task_id": "p6a-task",
                        "mode": "a-interact",
                        "message": expected_message,
                    },
                    "timeout": 1800.0,
                },
                {
                    "method": "GET",
                    "url": f"{ainteract.USER_SIM_URL}/audit/p6a-task",
                    "timeout": 120.0,
                },
                {
                    "method": "POST",
                    "url": f"{ainteract.DB_ENV_URL}/cleanup_task",
                    "payload": {"task_id": "p6a-task"},
                    "timeout": 30.0,
                },
            ],
        )
        legacy = {key: value for key, value in result.items() if key != "valibra"}
        self.assertEqual(
            legacy,
            {
                "task_id": "p6a-task",
                "instance_id": "p6a-task",
                "database": "p6a_db",
                "phase1_passed": True,
                "phase2_passed": False,
                "has_follow_up": False,
                "total_reward": 1.0,
                "elapsed_seconds": 5.0,
                "initial_budget": initial_budget,
                "budget_used": initial_budget - 4.0,
                "budget_remaining": 4.0,
                "system_agent_model": ainteract.settings.system_agent_model,
                "user_simulator_model": ainteract.settings.user_sim_model,
                "user_simulator_prompt_version": ainteract.settings.prompt_version,
                "initial_message": expected_message,
                "subtask_1_predicted_sql": ["SELECT 1"],
                "subtask_2_predicted_sql": [],
                "all_submitted_sql": {
                    "phase1": ["SELECT 1"],
                    "phase2": [],
                },
                "prompt_flow": [],
                "token_usage": {
                    "system_agent": self._run_state()["system_agent_token_usage"],
                    "user_simulator": {
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "reasoning_tokens": 1,
                        "total_tokens": 5,
                    },
                    "combined": {
                        "input_tokens": 13,
                        "output_tokens": 7,
                        "total_tokens": 20,
                        "cached_tokens": 0,
                        "reasoning_tokens": 3,
                        "tool_prompt_tokens": 0,
                    },
                },
                "dialogue_history": [{"role": "user", "text": "bounded"}],
                "tool_trajectory": [
                    {
                        "tool": "submit_sql",
                        "phase": 1,
                        "args": {"sql": "SELECT 1"},
                    }
                ],
                "adk_events": [{"final": True}],
                "user_simulator_audit": {
                    "llm_calls": [],
                    "token_usage": {
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "reasoning_tokens": 1,
                        "total_tokens": 5,
                    },
                },
                "final_response": "done",
            },
        )
        self.assertEqual(result["valibra"]["export_status"], "succeeded")
        self.assertEqual(result["token_usage"]["combined"]["total_tokens"], 20)

    async def test_export_exception_keeps_reward_sql_and_trajectories(self):
        _, result, events = await self._run(
            exporter=RuntimeError("synthetic export failure with private details")
        )
        self.assertEqual(result["total_reward"], 1.0)
        self.assertEqual(result["subtask_1_predicted_sql"], ["SELECT 1"])
        self.assertEqual(result["tool_trajectory"], self._run_state()["tool_trajectory"])
        self.assertEqual(result["dialogue_history"], self._run_state()["dialogue_history"])
        self.assertEqual(result["adk_events"], self._run_state()["adk_events"])
        self.assertEqual(
            result["valibra"],
            {"export_status": "failed", "error_type": "RuntimeError"},
        )
        self.assertEqual(len(events), 6)


if __name__ == "__main__":
    unittest.main()
