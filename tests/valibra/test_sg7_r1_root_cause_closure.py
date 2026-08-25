from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import unittest
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.control import evaluate_first_submit_gate
from valibra_agent.sql_grounding.models import (
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    ValidationContext,
    _validate_expression_identifiers,
    canonicalize_field_expression,
    canonicalize_relation_expression,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.service import process_sql_grounding_observation
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    DEFAULT_GROUNDING_TIMEOUT_SECONDS,
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT,
    SQL_GROUNDING_PROMPT_SHA256,
    SQL_GROUNDING_STAGE_PROMPTS,
    GroundingUpdaterError,
    GroundingUpdaterResult,
    LiteLLMSQLGroundingClient,
    SQLGroundingUpdater,
    load_sql_grounding_llm_config,
    load_sql_grounding_provider_config,
)
from valibra_agent.sql_grounding import updater as updater_module


OLD_PROMPT_SHA = "312a5c019c68d09aaf3c54e3991ef381d4dc2ded7564fdb344bd7113305ac594"
PROMPT_SHA = "db5a44e94a0e7fba92e78e6ca99e6b324affdee92aaed4dcd45010ff03876454"
FORM_SHA = "728fc43c6ed85e72e60c9ebf85b00487059a2871020a28764b9641c69b84ed81"
PRE_R1_CONFIG_SHA = "489a7185cb711429b4c5346481ae639851f02ad41893554cbedb6ca703d2ba9e"
PRE_PROMPT_FIX_R1_CONFIG_SHA = "af6c8d9378da50a2d167e6bdf6f247dc0978691a21dd82c2fcc6dadbd9ee0bf7"
CANONICAL_PROMPT_90S_CONFIG_SHA = "f6f674db77cf2a811802155738e20297fc2ecbd0b86a5d369c20e3bf64c032f7"
CONFIG_SHA = "bc3a636fbd12796184e74d850ae69d7a89fbf5c39165d2c4e9db856b36f1a43a"
QUERY = "Show the maintenance cost."
SCHEMA = """CREATE TABLE operational_metrics (
  maintcost NUMERIC
);"""


def _provider_environment() -> dict[str, str]:
    return {
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "600",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "8",
        "GROUNDING_PROMPT_SHA256": PROMPT_SHA,
    }


def _state(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "current_phase": 1,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "budget_remaining": 10.0,
        "initial_budget": 10.0,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
    }


def _observation(
    task_id: str,
    sequence: int,
    *,
    kind: str = "schema",
    tool_name: str = "get_schema",
    content: Any = SCHEMA,
):
    return build_sql_grounding_observation(
        task_id=task_id,
        phase=1,
        sequence=sequence,
        observation_type=kind,
        content=content,
        summary=f"offline {tool_name} observation",
        tool_name=tool_name,
        function_call_id=f"{task_id}-call-{sequence}",
    )


def _telemetry(*, attempted: bool = True, status: str = "succeeded"):
    return GroundingLLMTelemetry(
        attempted=attempted,
        status=status,
        timed_out=status == "timed_out",
        error_type="timeout" if status == "timed_out" else None,
        request_sha256="1" * 64 if attempted else "",
        response_sha256="2" * 64 if status == "succeeded" else "",
        prompt_sha256=PROMPT_SHA,
        form_schema_sha256=FORM_SHA,
        configuration_sha256=CONFIG_SHA,
    )


class DelayedSchemaUpdater:
    def __init__(self, *, preserve_focus: bool = False, delay: float = 0.02):
        self.preserve_focus = preserve_focus
        self.delay = delay
        self.calls: list[str] = []
        self.inflight = 0
        self.max_inflight = 0

    async def propose(self, runtime, observation, **kwargs):
        del kwargs
        self.calls.append(observation.tool_name or observation.observation_type)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay)
            if self.preserve_focus:
                state = runtime.grounding_state
                focus = runtime.focus_dimension
            else:
                state = SQLGroundingState(
                    tables=("operational_metrics",),
                    join_keys=(),
                    column_mapping=(),
                    domain_knowledge=(),
                )
                focus = "none"
            return GroundingUpdaterResult(
                response=GroundingLLMResponse(
                    sql_grounding_state=state,
                    user_clarification_requests=(),
                    next_focus_dimension=focus,
                ),
                telemetry=_telemetry(),
                transport_normalization="none",
            )
        finally:
            self.inflight -= 1


class TimeoutUpdater:
    def __init__(self):
        self.calls = 0

    async def propose(self, runtime, observation, **kwargs):
        del runtime, observation, kwargs
        self.calls += 1
        raise GroundingUpdaterError("timeout", _telemetry(status="timed_out"))


async def _bound_handle(state, observation):
    token = grounding_callbacks._bind_turn_message(state["task_id"], "a-interact", QUERY)
    try:
        return await grounding_callbacks._handle_observation(state, observation)
    finally:
        grounding_callbacks._reset_turn_message(token)


class TwoSiblingCallsLlm(BaseLlm):
    calls: int = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        self.calls += 1
        if self.calls == 1:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="r1-schema-call",
                                name="get_schema",
                                args={},
                            )
                        ),
                        types.Part(
                            function_call=types.FunctionCall(
                                id="r1-knowledge-names-call",
                                name="get_all_external_knowledge_names",
                                args={},
                            )
                        ),
                    ],
                )
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="R1_LOCAL_DONE")],
            )
        )


class SG7R1SchedulingAndConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_query_is_audited_without_provider_or_state_change(self):
        state = _state("r1-user-query")
        updater = DelayedSchemaUpdater()

        async def baseline_before(context, request):
            del request
            context.state["system_agent_llm_calls"].append({"actions": []})
            context.state["_active_llm_call_index"] = 0
            return None

        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, _provider_environment(), clear=False),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
                patch(
                    "system_agent.callbacks.before_model_callback",
                    baseline_before,
                ),
            ):
                await grounding_callbacks.before_model_callback(
                    SimpleNamespace(state=state), {"contents": []}
                )
        finally:
            grounding_callbacks._reset_turn_message(token)
        runtime = GroundingRuntime.model_validate(
            state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        audit = state["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_UPDATE_AUDIT_KEY
        ]
        self.assertEqual(updater.calls, [])
        self.assertEqual(
            audit["service_status"], "skipped_provider_no_state_evidence"
        )
        self.assertEqual(runtime, GroundingRuntime())
        self.assertEqual(
            state.get(grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY, 0), 0
        )

    async def test_same_task_intermediate_observations_never_reserve_provider(self):
        state = _state("r1-reservation")
        updater = DelayedSchemaUpdater(preserve_focus=True)
        with (
            patch.dict(os.environ, _provider_environment(), clear=False),
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
        ):
            first, second = await asyncio.gather(
                _bound_handle(state, _observation(state["task_id"], 1)),
                _bound_handle(state, _observation(state["task_id"], 2)),
            )
            third = await _bound_handle(
                state, _observation(state["task_id"], 3)
            )
        self.assertEqual(
            [first.service_status, second.service_status, third.service_status],
            [
                "stored_official_evidence_only",
                "stored_official_evidence_only",
                "stored_official_evidence_only",
            ],
        )
        self.assertEqual(updater.calls, [])
        self.assertEqual(updater.max_inflight, 0)
        self.assertEqual(
            state.get(grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY, 0), 0
        )

    async def test_intermediate_observations_cannot_consume_timeout_slots(self):
        state = _state("r1-timeout-slots")
        updater = TimeoutUpdater()
        results = []
        with (
            patch.dict(os.environ, _provider_environment(), clear=False),
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
        ):
            for sequence in (1, 2, 3):
                results.append(
                    await _bound_handle(
                        state,
                        _observation(state["task_id"], sequence),
                    )
                )
        self.assertEqual(updater.calls, 0)
        self.assertEqual(
            [item.service_status for item in results],
            [
                "stored_official_evidence_only",
                "stored_official_evidence_only",
                "stored_official_evidence_only",
            ],
        )
        self.assertEqual(
            state.get(grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY, 0), 0
        )

    async def test_different_tasks_skip_intermediate_provider_independently(self):
        updater = DelayedSchemaUpdater(preserve_focus=True, delay=0.04)
        first = _state("r1-parallel-a")
        second = _state("r1-parallel-b")
        with (
            patch.dict(os.environ, _provider_environment(), clear=False),
            patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
        ):
            await asyncio.gather(
                _bound_handle(first, _observation(first["task_id"], 1)),
                _bound_handle(second, _observation(second["task_id"], 1)),
            )
        self.assertEqual(updater.max_inflight, 0)
        self.assertEqual(
            first.get(grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY, 0), 0
        )
        self.assertEqual(
            second.get(grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY, 0), 0
        )

    async def test_actual_adk_sibling_tools_have_exact_audits_and_one_call(self):
        counts = {"get_schema": 0, "get_all_external_knowledge_names": 0}

        def get_schema() -> str:
            counts["get_schema"] += 1
            return SCHEMA

        def get_all_external_knowledge_names() -> str:
            counts["get_all_external_knowledge_names"] += 1
            return json.dumps(["RATIO_RULE"])

        updater = DelayedSchemaUpdater(delay=0.04)
        model = TwoSiblingCallsLlm(model="r1-local-sibling")
        agent = LlmAgent(
            name="r1_local_agent",
            model=model,
            instruction="Offline SG7-R1 sibling lifecycle fixture.",
            tools=[get_schema, get_all_external_knowledge_names],
            before_model_callback=grounding_callbacks.before_model_callback,
            after_model_callback=grounding_callbacks.after_model_callback,
            before_tool_callback=grounding_callbacks.before_tool_callback,
            after_tool_callback=grounding_callbacks.after_tool_callback,
            on_tool_error_callback=grounding_callbacks.on_tool_error_callback,
        )
        runner = InMemoryRunner(agent=agent, app_name="r1_sibling")
        state = _state("r1-adk-sibling")
        session = await runner.session_service.create_session(
            app_name="r1_sibling",
            user_id="r1-user",
            state=state,
        )
        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, _provider_environment(), clear=False),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
            ):
                async for _ in runner.run_async(
                    user_id="r1-user",
                    session_id=session.id,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=QUERY)],
                    ),
                ):
                    pass
        finally:
            grounding_callbacks._reset_turn_message(token)
        final = await runner.session_service.get_session(
            app_name="r1_sibling",
            user_id="r1-user",
            session_id=session.id,
        )
        final_state = dict(final.state)
        self.assertEqual(counts, {"get_schema": 1, "get_all_external_knowledge_names": 1})
        self.assertEqual(final_state["budget_remaining"], 8.5)
        self.assertEqual(
            sorted(event["tool"] for event in final_state["tool_trajectory"]),
            ["get_all_external_knowledge_names", "get_schema"],
        )
        self.assertEqual(updater.calls, [])
        self.assertEqual(updater.max_inflight, 0)
        self.assertEqual(
            final_state.get(grounding_callbacks.GROUNDING_PROVIDER_CALL_COUNT_KEY, 0), 0
        )
        exact = final_state[grounding_callbacks.GROUNDING_TOOL_AUDITS_KEY]
        self.assertEqual(set(exact), {"r1-schema-call", "r1-knowledge-names-call"})
        self.assertEqual(
            exact["r1-schema-call"][grounding_callbacks.SHADOW_AUDIT_KEY][
                "service_status"
            ],
            "stored_official_evidence_only",
        )
        self.assertEqual(
            exact["r1-knowledge-names-call"][grounding_callbacks.SHADOW_AUDIT_KEY][
                "service_status"
            ],
            "stored_official_evidence_only",
        )
        runtime = GroundingRuntime.model_validate(
            final_state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertIsNone(runtime.grounding_state.tables)


class SG7R1EvidenceAndAuditTests(unittest.IsolatedAsyncioTestCase):
    def _context(self, state, latest):
        token = grounding_callbacks._bind_turn_message(
            state["task_id"], "a-interact", QUERY
        )
        try:
            return grounding_callbacks._build_validation_context(state, latest)
        finally:
            grounding_callbacks._reset_turn_message(token)

    def test_official_schema_projects_without_shadow_audit(self):
        state = _state("r1-official-context")
        state["tool_trajectory"] = [
            {"tool": "get_schema", "phase": 1, "args": {}, "result": SCHEMA}
        ]
        context = self._context(state, _observation(state["task_id"], 2))
        self.assertEqual(context.known_tables, {"operational_metrics"})
        self.assertEqual(context.known_columns, {"operational_metrics.maintcost"})
        self.assertEqual(len(context.official_trajectory_observation_ids), 1)

    def test_budget_error_and_blocked_submit_do_not_project(self):
        state = _state("r1-fail-closed-context")
        state["tool_trajectory"] = [
            {
                "tool": "get_schema",
                "phase": 1,
                "args": {},
                "result": {"error": "Budget exhausted"},
            }
        ]
        state[grounding_callbacks.GROUNDING_BLOCKED_SUBMITS_KEY] = {
            "blocked-call": {"synthetic": True}
        }
        latest = _observation(
            state["task_id"],
            2,
            kind="metadata",
            tool_name="get_column_meaning",
            content={"meaning": "maintenance cost"},
        )
        context = self._context(state, latest)
        self.assertEqual(context.known_tables, frozenset())
        self.assertEqual(context.known_columns, frozenset())

    async def test_timeout_private_audit_preexists_and_finishes_timed_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key_file = root / "grounding.key"
            key_file.write_text("offline-r1-credential\n", encoding="utf-8")
            environment = {
                **_provider_environment(),
                "GROUNDING_API_BASE": "https://provider.invalid/v1",
                "GROUNDING_API_KEY": "",
                "GROUNDING_API_KEY_FILE": str(key_file),
                "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE": "false",
            }
            llm = load_sql_grounding_llm_config(Path.cwd(), environment)
            provider = load_sql_grounding_provider_config(
                Path.cwd(), llm, environment
            ).model_copy(update={"raw_audit_dir": str(root / "audit")})
            started_paths: list[Path] = []

            async def completion(**kwargs):
                del kwargs
                started_paths.extend((root / "audit").glob("*.json"))
                self.assertEqual(len(started_paths), 1)
                self.assertEqual(
                    json.loads(started_paths[0].read_text())["status"], "started"
                )
                await asyncio.sleep(0.05)
                raise AssertionError("cancel should occur first")

            updater = SQLGroundingUpdater(
                LiteLLMSQLGroundingClient(
                    llm,
                    provider,
                    environment=environment,
                    completion=completion,
                )
            )
            observation = _observation("r1-private-timeout", 1)
            state = _state("r1-private-timeout")
            token = grounding_callbacks._bind_turn_message(
                state["task_id"], "a-interact", QUERY
            )
            try:
                context = grounding_callbacks._build_validation_context(
                    state, observation
                )
            finally:
                grounding_callbacks._reset_turn_message(token)
            with patch.object(
                updater_module, "DEFAULT_GROUNDING_TIMEOUT_SECONDS", 0.01
            ):
                result = await process_sql_grounding_observation(
                    GroundingRuntime(), observation, context, updater
                )
            self.assertEqual(result.llm_telemetry.status, "timed_out")
            self.assertTrue(result.llm_telemetry.attempted)
            audit_path = (
                Path(provider.raw_audit_dir).parents[1]
                / result.llm_telemetry.raw_private_audit_ref
            )
            self.assertTrue(audit_path.is_file())
            self.assertEqual(stat.S_IMODE(audit_path.stat().st_mode), 0o600)
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit["status"], "timed_out")
            self.assertEqual(audit["observation_id"], observation.observation_id)
            self.assertEqual(audit["observation_type"], "schema")
            self.assertEqual(
                audit["request_sha256"],
                updater_module._stable_json_sha256(audit["request"]),
            )
            self.assertNotIn("offline-r1-credential", audit_path.read_text())

    def test_gate_semantics_and_frozen_contract_are_unchanged(self):
        partial = GroundingRuntime(
            grounding_state=SQLGroundingState(
                tables=("operational_metrics",),
                join_keys=None,
                column_mapping=None,
                domain_knowledge=None,
            ),
            focus_dimension="join_keys",
        )
        complete = GroundingRuntime(
            grounding_state=SQLGroundingState(
                tables=("operational_metrics",),
                join_keys=(),
                column_mapping=(),
                domain_knowledge=(),
            ),
            focus_dimension="none",
        )
        self.assertFalse(evaluate_first_submit_gate(partial, first_submit=True).open)
        self.assertTrue(evaluate_first_submit_gate(complete, first_submit=True).open)
        self.assertEqual(DEFAULT_GROUNDING_TIMEOUT_SECONDS, 600.0)
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)
        self.assertNotEqual(SQL_GROUNDING_CONFIGURATION_SHA256, PRE_R1_CONFIG_SHA)
        self.assertNotEqual(
            SQL_GROUNDING_CONFIGURATION_SHA256,
            CANONICAL_PROMPT_90S_CONFIG_SHA,
        )

    def test_canonical_expression_prompt_closure_and_regression(self):
        offending = (
            "electrical_performance.elec_perf_snapshot->'power'->>'power_now_w'"
        )
        canonical = (
            "electrical_performance.elec_perf_snapshot -> 'power' ->> "
            "'power_now_w'"
        )
        with self.assertRaisesRegex(
            ValueError,
            "must use canonical PostgreSQL expression form",
        ):
            canonicalize_field_expression(offending)
        self.assertEqual(canonicalize_field_expression(canonical), canonical)

        context = ValidationContext(
            current_query="Show the actual juice for each plant.",
            latest_observation_id="offline-r1-canonical-audit",
            known_tables=frozenset(
                {"electrical_performance", "plant_record", "plants"}
            ),
            known_columns=frozenset(
                {
                    "electrical_performance.elec_perf_snapshot",
                    "electrical_performance.snaplink",
                    "plant_record.snapkey",
                    "plant_record.sitetie",
                    "plants.sitekey",
                    "plants.sitelabel",
                }
            ),
            supported_json_paths=frozenset(
                {
                    (
                        "electrical_performance.elec_perf_snapshot",
                        ("power", "power_now_w"),
                    )
                }
            ),
        )
        self.assertEqual(
            _validate_expression_identifiers(
                canonical,
                context=context,
                relation=False,
            ),
            frozenset({"electrical_performance"}),
        )
        self.assertEqual(canonicalize_field_expression("plants.sitelabel"), "plants.sitelabel")
        relations = (
            "electrical_performance.snaplink = plant_record.snapkey",
            "plant_record.sitetie = plants.sitekey",
        )
        for relation in relations:
            with self.subTest(relation=relation):
                self.assertEqual(canonicalize_relation_expression(relation), relation)
                self.assertEqual(
                    len(
                        _validate_expression_identifiers(
                            relation,
                            context=context,
                            relation=True,
                        )
                    ),
                    2,
                )

        executable_prompts = "\n".join(SQL_GROUNDING_STAGE_PROMPTS.values())
        for fragment in (
            "精确词法",
            "sqlglot 26.16.4",
            'expression.sql(dialect="postgres")',
            "两侧都必须各有一个 ASCII 空格",
            "-> 和 ->>",
            "仅用于展示语法的示例",
            "t.c -> 'key' ->> 'leaf'",
            "只展示格式",
        ):
            with self.subTest(prompt_fragment=fragment):
                self.assertIn(fragment, executable_prompts)
        for forbidden in ("solar_panel", "actual juice", "electrical_performance"):
            with self.subTest(forbidden_prompt_content=forbidden):
                self.assertNotIn(forbidden, SQL_GROUNDING_PROMPT)
        self.assertNotEqual(SQL_GROUNDING_PROMPT_SHA256, OLD_PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertNotEqual(
            SQL_GROUNDING_CONFIGURATION_SHA256,
            PRE_PROMPT_FIX_R1_CONFIG_SHA,
        )


if __name__ == "__main__":
    unittest.main()
