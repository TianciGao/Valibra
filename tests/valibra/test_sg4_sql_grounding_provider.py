from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import litellm
from openai import AsyncOpenAI

from shared.config import PROJECT_ROOT
from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent.server import _configuration_summary
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    ValidationContext,
    canonical_json,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.service import process_sql_grounding_observation
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingClientResponse,
    GroundingLLMRequest,
    SQLGroundingProviderError,
    GroundingUpdaterError,
    GroundingUpdaterResult,
    LiteLLMSQLGroundingClient,
    SQLGroundingProviderConfig,
    SQLGroundingUpdater,
    build_real_sql_grounding_updater,
    load_sql_grounding_llm_config,
    load_sql_grounding_provider_config,
    sql_grounding_provider_health_report,
    _build_provider_request,
    _parse_grounding_credential_file,
)


OLD_PROMPT_SHA = "17455ea076632901a9c2aa3bada96fe4c06baa500e0ce271be854d681ab74962"
OLD_CONFIG_SHA = "405704b6798c4662df4dbe425ca0d15f284776551827e636bd0297f4f53a13f0"
PROMPT_SHA = "312a5c019c68d09aaf3c54e3991ef381d4dc2ded7564fdb344bd7113305ac594"
FORM_SHA = "2d60e788b2a3c1efc581f95945331a124805678fedc857bb2bc39f7462500406"
CONFIG_SHA = "489a7185cb711429b4c5346481ae639851f02ad41893554cbedb6ca703d2ba9e"
QUERY = "What is the maintenance cost?"


def environment(key_file: Path | None = None) -> dict[str, str]:
    result = {
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "30",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "2",
        "GROUNDING_PROMPT_SHA256": PROMPT_SHA,
        "GROUNDING_API_BASE": "https://provider.invalid/v1",
        "GROUNDING_API_KEY": "",
        "GROUNDING_API_KEY_FILE": str(key_file) if key_file else "",
        "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE": "false",
    }
    return result


def response_content(state: SQLGroundingState, focus: str) -> str:
    return json.dumps(
        {
            "sql_grounding_state": state.model_dump(mode="json"),
            "next_focus_dimension": focus,
        },
        sort_keys=True,
    )


def observation(kind="user_query", content=QUERY, tool_name=None, sequence=1, phase=1):
    return build_sql_grounding_observation(
        task_id="sg4-offline",
        phase=phase,
        sequence=sequence,
        observation_type=kind,
        content=content,
        summary="synthetic SG4 offline observation",
        tool_name=tool_name,
        function_call_id=f"call-{sequence}" if tool_name else None,
    )


class ProviderAdapterOfflineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key_file = self.root / "grounding.key"
        self.key_file.write_text("unit-test-credential\n", encoding="utf-8")
        self.env = environment(self.key_file)

    def tearDown(self):
        self.temp.cleanup()

    def configs(self):
        llm = load_sql_grounding_llm_config(PROJECT_ROOT, self.env)
        provider = load_sql_grounding_provider_config(PROJECT_ROOT, llm, self.env)
        provider = provider.model_copy(
            update={"raw_audit_dir": str(self.root / "research-runtime" / "audit")}
        )
        return llm, SQLGroundingProviderConfig.model_validate(provider)

    def test_credential_file_parser_accepts_only_two_frozen_formats(self):
        raw = "synthetic-raw-token-1234567890"
        bearer = "synthetic-bearer-token-1234567890"
        accepted = (
            (raw, raw),
            (f"Authorization: Bearer {bearer}", bearer),
            (
                "saved request\n"
                f"--header 'Authorization: Bearer {bearer}'\n"
                "non credential notes\n",
                bearer,
            ),
            (f"aUtHoRiZaTiOn : bEaReR {bearer}", bearer),
        )
        for content, expected in accepted:
            with self.subTest(content_class="accepted"):
                self.assertEqual(_parse_grounding_credential_file(content), expected)

        rejected = (
            "",
            "raw-token-one-1234567890\nraw-token-two-1234567890\n",
            (
                "Authorization: Bearer synthetic-bearer-one-1234567890\n"
                "Authorization: Bearer synthetic-bearer-two-1234567890\n"
            ),
            (
                "raw-token-extra-1234567890\n"
                "Authorization: Bearer synthetic-bearer-token-1234567890\n"
            ),
            '"quoted-token-1234567890"',
            "token with whitespace",
            '{"api_key":"synthetic-token-1234567890"}',
            "GROUNDING_API_KEY=synthetic-token-1234567890",
            "Authorization: Bearer 'quoted-token-1234567890'",
            "Authorization: Bearer token\\with-boundary",
        )
        for content in rejected:
            with self.subTest(content_class="rejected"):
                with self.assertRaises(SQLGroundingProviderError):
                    _parse_grounding_credential_file(content)

    def test_missing_credential_file_remains_fail_closed(self):
        llm = load_sql_grounding_llm_config(PROJECT_ROOT, self.env)
        missing = self.root / "missing.key"
        with self.assertRaisesRegex(ValueError, "non-empty file"):
            load_sql_grounding_provider_config(
                PROJECT_ROOT,
                llm,
                self.env | {"GROUNDING_API_KEY_FILE": str(missing)},
            )

    def test_mode_and_credentials_are_strict_and_never_fall_back_to_system(self):
        self.assertEqual(
            sql_grounding_provider_health_report(PROJECT_ROOT, {}),
            {
                "requested_updater_mode": "passthrough",
                "effective_updater_mode": "passthrough",
                "provider_configuration_valid": True,
                "provider_enabled": False,
                "configuration_error_type": None,
            },
        )
        invalid = sql_grounding_provider_health_report(
            PROJECT_ROOT, {"GROUNDING_UPDATER_MODE": "rule"}
        )
        self.assertEqual(invalid["effective_updater_mode"], "invalid")
        self.assertFalse(invalid["provider_enabled"])

        llm = load_sql_grounding_llm_config(PROJECT_ROOT, self.env)
        missing = self.env | {
            "GROUNDING_API_KEY_FILE": "",
            "SYSTEM_AGENT_API_KEY": "must-not-be-read",
            "SYSTEM_AGENT_API_BASE": "https://system.invalid",
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_sql_grounding_provider_config(PROJECT_ROOT, llm, missing)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_sql_grounding_provider_config(
                PROJECT_ROOT,
                llm,
                self.env | {"GROUNDING_API_KEY": "ambiguous"},
            )

    async def test_native_async_single_request_json_object_no_tools_usage_and_permissions(self):
        llm, provider = self.configs()
        calls = []

        async def completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [{"message": {"content": response_content(SQLGroundingState(), "tables")}}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 18,
                    "cost": 0.125,
                },
                "_hidden_params": {"custom_llm_provider": "offline"},
            }

        updater = SQLGroundingUpdater(
            LiteLLMSQLGroundingClient(
                llm,
                provider,
                environment=self.env,
                completion=completion,
            )
        )
        result = await updater.propose(
            GroundingRuntime(), observation(), original_query=QUERY
        )
        self.assertEqual(len(calls), 1)
        sent = calls[0]
        self.assertEqual(sent["num_retries"], 0)
        self.assertEqual(sent["max_retries"], 0)
        self.assertNotIn("tools", sent)
        self.assertNotIn("tool_choice", sent)
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(result.telemetry.usage.total_tokens, 18)
        self.assertEqual(result.telemetry.usage.reasoning_tokens, 3)
        self.assertEqual(result.telemetry.provider_reported_cost, 0.125)
        self.assertEqual(result.telemetry.credential_source, "file")
        audit_path = self.root / result.telemetry.raw_private_audit_ref
        self.assertTrue(audit_path.is_file())
        self.assertEqual(stat.S_IMODE(audit_path.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(audit_path.stat().st_mode), 0o600)
        audit_text = audit_path.read_text(encoding="utf-8")
        self.assertNotIn("unit-test-credential", audit_text)
        request = json.loads(audit_text)["request"]
        self.assertEqual(request["tools"], [])
        self.assertIsNone(request["tool_choice"])

    async def test_litellm_transformation_preserves_json_object_in_http_body(self):
        llm, provider = self.configs()
        request = GroundingLLMRequest(
            prompt=SQL_GROUNDING_PROMPT,
            input_json="{}",
            response_schema=SQL_GROUNDING_FORM_SCHEMA,
        )
        _, provider_kwargs = _build_provider_request(
            request,
            llm,
            provider,
            api_key="synthetic-offline-credential",
        )
        self.assertEqual(provider_kwargs["num_retries"], 0)
        self.assertEqual(provider_kwargs["max_retries"], 0)

        captured = {}

        async def handle_http(http_request):
            captured["path"] = http_request.url.path
            captured["body"] = json.loads(http_request.content)
            return httpx.Response(
                200,
                json={
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "glm-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "{}"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_http))
        openai_client = AsyncOpenAI(
            api_key="synthetic-offline-credential",
            base_url=provider_kwargs["api_base"],
            http_client=http_client,
        )
        transformed_kwargs = dict(provider_kwargs)
        transformed_kwargs["client"] = openai_client
        try:
            await litellm.acompletion(**transformed_kwargs)
        finally:
            await openai_client.close()

        expected_path = (
            httpx.URL(provider_kwargs["api_base"]).path.rstrip("/")
            + "/chat/completions"
        )
        self.assertEqual(captured["path"], expected_path)
        body = captured["body"]
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)

    async def _process_provider_content(self, content):
        llm, provider = self.configs()
        calls = []

        async def completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "_hidden_params": {"custom_llm_provider": "offline"},
            }

        runtime = GroundingRuntime(
            grounding_state=SQLGroundingState(
                tables=("operational_metrics",),
                join_keys=(),
                column_mapping=None,
                domain_knowledge=(),
            ),
            focus_dimension="column_mapping",
        )
        current_observation = observation(
            "metadata",
            {"meaning": "maintenance cost"},
            "get_column_meaning",
        )
        result = await process_sql_grounding_observation(
            runtime,
            current_observation,
            ValidationContext(
                current_query=QUERY,
                latest_observation_id=current_observation.observation_id,
                known_tables=frozenset({"operational_metrics"}),
                known_columns=frozenset({"operational_metrics.maintcost"}),
            ),
            SQLGroundingUpdater(
                LiteLLMSQLGroundingClient(
                    llm,
                    provider,
                    environment=self.env,
                    completion=completion,
                )
            ),
        )
        return runtime, result, calls

    async def test_local_strict_form_rejects_invalid_provider_objects_atomically(self):
        valid = {
            "sql_grounding_state": {
                "tables": ["operational_metrics"],
                "join_keys": [],
                "column_mapping": [
                    {
                        "phrase": "maintenance cost",
                        "targets": ["operational_metrics.maintcost"],
                    }
                ],
                "domain_knowledge": [],
            },
            "next_focus_dimension": "none",
        }
        singular_target = json.loads(json.dumps(valid))
        singular_target["sql_grounding_state"]["column_mapping"][0] = {
            "phrase": "maintenance cost",
            "target": "operational_metrics.maintcost",
        }
        extra_field = json.loads(json.dumps(valid))
        extra_field["sql_grounding_state"]["column_mapping"][0]["confidence"] = 1
        missing_required = json.loads(json.dumps(valid))
        del missing_required["next_focus_dimension"]
        wrong_type = json.loads(json.dumps(valid))
        wrong_type["sql_grounding_state"]["tables"] = "operational_metrics"
        unknown_focus = json.loads(json.dumps(valid))
        unknown_focus["next_focus_dimension"] = "schema"
        invalid_expression = json.loads(json.dumps(valid))
        invalid_expression["sql_grounding_state"]["column_mapping"][0]["targets"] = [
            "operational_metrics.not_real"
        ]
        encoded = json.dumps(valid, separators=(",", ":"), sort_keys=True)
        duplicate = encoded.replace(
            '"next_focus_dimension":"none"',
            '"next_focus_dimension":"none","next_focus_dimension":"none"',
        )
        cases = {
            "singular_target": (json.dumps(singular_target), "form_validation_failed"),
            "extra_field": (json.dumps(extra_field), "form_validation_failed"),
            "missing_required": (json.dumps(missing_required), "form_validation_failed"),
            "wrong_type": (json.dumps(wrong_type), "form_validation_failed"),
            "duplicate_key": (duplicate, "duplicate_json_key"),
            "prose_json": ("Here is JSON:\n" + encoded, "json_invalid"),
            "malformed_fence": ("```json\n" + encoded, "transport_format_invalid"),
            "unknown_focus": (json.dumps(unknown_focus), "form_validation_failed"),
            "invalid_expression": (
                json.dumps(invalid_expression),
                "state_validation_failed",
            ),
        }
        for name, (content, expected) in cases.items():
            with self.subTest(name=name):
                runtime, result, calls = await self._process_provider_content(content)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
                self.assertEqual(result.runtime, runtime)
                self.assertEqual(result.runtime.grounding_revision, 0)
                observed = (
                    result.state_update.error_type
                    if expected == "state_validation_failed"
                    else result.llm_telemetry.error_type
                )
                self.assertEqual(observed, expected)

    async def test_valid_local_form_and_semantics_replace_state_atomically(self):
        content = json.dumps(
            {
                "sql_grounding_state": {
                    "tables": ["operational_metrics"],
                    "join_keys": [],
                    "column_mapping": [
                        {
                            "phrase": "maintenance cost",
                            "targets": ["operational_metrics.maintcost"],
                        }
                    ],
                    "domain_knowledge": [],
                },
                "next_focus_dimension": "none",
            }
        )
        runtime, result, calls = await self._process_provider_content(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(result.runtime.grounding_revision, 1)
        self.assertEqual(result.runtime.focus_dimension, "none")
        self.assertEqual(
            result.runtime.grounding_state.column_mapping,
            (
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("operational_metrics.maintcost",),
                ),
            ),
        )
        self.assertNotEqual(result.runtime, runtime)

    async def test_provider_exception_is_sanitized_and_timeout_is_fail_open(self):
        llm, provider = self.configs()

        async def fail(**kwargs):
            del kwargs
            raise RuntimeError("unit-test-credential must never escape")

        updater = SQLGroundingUpdater(
            LiteLLMSQLGroundingClient(
                llm, provider, environment=self.env, completion=fail
            )
        )
        with self.assertRaises(GroundingUpdaterError) as raised:
            await updater.propose(GroundingRuntime(), observation(), original_query=QUERY)
        self.assertEqual(raised.exception.reason, "provider_error")
        self.assertTrue(raised.exception.telemetry.raw_private_audit_ref)
        self.assertNotIn("credential", str(raised.exception))

        class TimedOut:
            provider_may_continue_after_cancel = True
            provider_may_bill_after_cancel = True
            config = SimpleNamespace(model_id="model", credential_source="file")

            async def complete(self, request):
                del request
                raise asyncio.TimeoutError

        runtime = GroundingRuntime()
        service = await process_sql_grounding_observation(
            runtime,
            observation(),
            ValidationContext(
                current_query=QUERY,
                latest_observation_id=observation().observation_id,
            ),
            SQLGroundingUpdater(TimedOut()),
        )
        self.assertEqual(service.runtime, runtime)
        self.assertEqual(service.llm_telemetry.status, "timed_out")
        self.assertTrue(service.llm_telemetry.provider_may_bill_after_cancel)

    def test_frozen_contract_and_local_health_are_secret_free(self):
        self.assertEqual(SQL_GROUNDING_PROMPT_SHA256, PROMPT_SHA)
        self.assertEqual(SQL_GROUNDING_FORM_SCHEMA_SHA256, FORM_SHA)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION_SHA256, CONFIG_SHA)
        self.assertNotEqual(SQL_GROUNDING_PROMPT_SHA256, OLD_PROMPT_SHA)
        self.assertNotEqual(SQL_GROUNDING_CONFIGURATION_SHA256, OLD_CONFIG_SHA)
        self.assertEqual(
            hashlib.sha256(
                canonical_json(GroundingLLMResponse.model_json_schema()).encode("utf-8")
            ).hexdigest(),
            FORM_SHA,
        )
        self.assertIn('"targets"', SQL_GROUNDING_PROMPT)
        self.assertIn('There is no field named "target".', SQL_GROUNDING_PROMPT)
        self.assertIn('"targets" is always a JSON array', SQL_GROUNDING_PROMPT)
        self.assertNotIn("maintenance cost", SQL_GROUNDING_PROMPT.lower())
        self.assertNotIn("operational_metrics", SQL_GROUNDING_PROMPT.lower())
        report = sql_grounding_provider_health_report(PROJECT_ROOT, self.env)
        self.assertTrue(report["provider_enabled"])
        self.assertNotIn("key", json.dumps(report).lower())


class ValidationContextProjectionTests(unittest.TestCase):
    def bind(self, current_state):
        return grounding_callbacks._bind_turn_message(
            current_state["task_id"], "a-interact", QUERY
        )

    def test_schema_parser_uses_only_ddl_and_ignores_sample_rows(self):
        schema = '''CREATE TABLE operational_metrics (
  id integer NOT NULL,
  maintcost numeric,
  PRIMARY KEY (id)
);

First 3 rows:
value
CREATE TABLE invented_from_sample (fake integer)
...
CREATE TABLE work_orders (
  id integer NOT NULL,
  metric_id integer,
  FOREIGN KEY (metric_id) REFERENCES operational_metrics(id)
);'''
        tables, columns = grounding_callbacks._parse_schema_projection(schema)
        self.assertEqual(tables, {"operational_metrics", "work_orders"})
        self.assertIn("operational_metrics.maintcost", columns)
        self.assertNotIn("invented_from_sample", tables)
        with self.assertRaises(ValueError):
            grounding_callbacks._parse_schema_projection("not DDL")

    def test_schema_and_exact_knowledge_survive_later_trajectory_observations(self):
        current = {
            "task_id": "sg4-context",
            "tool_trajectory": [
                {
                    "tool": "get_schema",
                    "result": "CREATE TABLE operational_metrics (\n  maintcost numeric\n);",
                    grounding_callbacks.SHADOW_AUDIT_KEY: {
                        "observation_id": "obs-schema",
                        "observation_type": "schema",
                    },
                },
                {
                    "tool": "get_knowledge_definition",
                    "result": json.dumps({"definition": "RATIO_RULE"}),
                    grounding_callbacks.SHADOW_AUDIT_KEY: {
                        "observation_id": "obs-knowledge",
                        "observation_type": "knowledge",
                    },
                },
            ],
        }
        latest = observation(
            "metadata", {"meaning": "maintenance cost"}, "get_column_meaning", 3
        )
        token = self.bind(current)
        try:
            context = grounding_callbacks._build_validation_context(current, latest)
        finally:
            grounding_callbacks._reset_turn_message(token)
        self.assertEqual(context.known_tables, {"operational_metrics"})
        self.assertEqual(context.known_columns, {"operational_metrics.maintcost"})
        self.assertEqual(
            context.supported_domain_knowledge,
            {("business_rule", "RATIO_RULE")},
        )
        self.assertNotIn("tool_trajectory", repr(context))
        self.assertEqual(
            grounding_callbacks._exact_knowledge_definition(
                json.dumps({"definition": "RATIO_RULE"})
            ),
            "RATIO_RULE",
        )
        self.assertIsNone(
            grounding_callbacks._exact_knowledge_definition(json.dumps(["name"]))
        )


class CallbackFakeProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_focus_and_schema_state_are_shadow_only(self):
        current = {
            "task_id": "sg4-callback",
            "current_phase": 1,
            "budget_remaining": 10.0,
            "initial_budget": 10.0,
            "tool_trajectory": [],
            "system_agent_llm_calls": [],
        }
        query_updater = _ScriptedUpdater(
            GroundingLLMResponse(
                sql_grounding_state=SQLGroundingState(),
                next_focus_dimension="column_mapping",
            )
        )

        async def baseline_before(callback_context, llm_request):
            del llm_request
            callback_context.state["system_agent_llm_calls"].append({"actions": []})
            callback_context.state["_active_llm_call_index"] = 0
            return "baseline"

        request = {"contents": [{"text": "unchanged"}]}
        original = json.loads(json.dumps(request))
        token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", query_updater),
                patch.object(baseline_callbacks, "before_model_callback", baseline_before),
            ):
                result = await grounding_callbacks.before_model_callback(
                    SimpleNamespace(state=current), request
                )
        finally:
            grounding_callbacks._reset_turn_message(token)
        self.assertEqual(result, "baseline")
        self.assertEqual(request, original)
        self.assertEqual(query_updater.calls, 1)
        runtime = GroundingRuntime.model_validate(
            current[grounding_callbacks.GROUNDING_RUNTIME_KEY]
        )
        self.assertEqual(runtime.grounding_revision, 0)
        self.assertEqual(runtime.focus_dimension, "column_mapping")
        view = current["system_agent_llm_calls"][0][
            grounding_callbacks.GROUNDING_VIEW_AUDIT_KEY
        ]
        self.assertFalse(view["injected"])
        self.assertTrue(view["request_unchanged"])
        self.assertEqual(current["budget_remaining"], 10.0)

    def test_health_truthfully_reports_shadow_and_no_control(self):
        with patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}, clear=False):
            summary = _configuration_summary()
        self.assertEqual(summary["grounding_core"], "sql_grounding_v1")
        self.assertEqual(summary["grounding_updater"], "passthrough")
        self.assertFalse(summary["prompt_view_injection_enabled"])
        self.assertFalse(summary["control_enabled"])
        self.assertFalse(summary["attempt_gate_enabled"])

    async def test_schema_metadata_and_exact_knowledge_update_with_transient_context(self):
        current = {
            "task_id": "sg4-fake-tools",
            "current_phase": 1,
            "budget_remaining": 10.0,
            "initial_budget": 10.0,
            "tool_trajectory": [],
            "system_agent_llm_calls": [],
        }
        after_schema = SQLGroundingState(
            tables=("operational_metrics",),
            join_keys=(),
            column_mapping=None,
            domain_knowledge=None,
        )
        after_metadata = after_schema.model_copy(
            update={
                "column_mapping": (
                    ColumnMapping(
                        phrase="maintenance cost",
                        targets=("operational_metrics.maintcost",),
                    ),
                )
            },
        )
        after_knowledge = SQLGroundingState(
            tables=after_metadata.tables,
            join_keys=after_metadata.join_keys,
            column_mapping=after_metadata.column_mapping,
            domain_knowledge=(
                DomainKnowledge(kind="business_rule", content="RATIO_RULE"),
            ),
        )
        updater = _QueueUpdater(
            [
                GroundingLLMResponse(
                    sql_grounding_state=after_schema,
                    next_focus_dimension="column_mapping",
                ),
                GroundingLLMResponse(
                    sql_grounding_state=after_metadata,
                    next_focus_dimension="domain_knowledge",
                ),
                GroundingLLMResponse(
                    sql_grounding_state=after_knowledge, next_focus_dimension="none"
                ),
            ]
        )
        schema_text = (
            "CREATE TABLE operational_metrics (\n"
            "  id integer,\n"
            "  maintcost numeric\n"
            ");"
        )
        schema_observation = observation("schema", schema_text, "get_schema", 1)
        metadata_observation = observation(
            "metadata",
            {"meaning": "maintenance cost"},
            "get_column_meaning",
            2,
        )
        knowledge_observation = observation(
            "knowledge",
            json.dumps({"definition": "RATIO_RULE"}),
            "get_knowledge_definition",
            3,
        )
        token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", updater),
            ):
                first = await grounding_callbacks._handle_observation(
                    current, schema_observation, GroundingRuntime()
                )
                grounding_callbacks._store_runtime(current, first.runtime)
                current["tool_trajectory"].append(
                    {
                        "tool": "get_schema",
                        "result": schema_text,
                        grounding_callbacks.SHADOW_AUDIT_KEY: {
                            "observation_id": schema_observation.observation_id,
                            "observation_type": "schema",
                        },
                    }
                )
                second = await grounding_callbacks._handle_observation(
                    current, metadata_observation, first.runtime
                )
                grounding_callbacks._store_runtime(current, second.runtime)
                current["tool_trajectory"].append(
                    {
                        "tool": "get_column_meaning",
                        "result": {"meaning": "maintenance cost"},
                        grounding_callbacks.SHADOW_AUDIT_KEY: {
                            "observation_id": metadata_observation.observation_id,
                            "observation_type": "metadata",
                        },
                    }
                )
                third = await grounding_callbacks._handle_observation(
                    current, knowledge_observation, second.runtime
                )
        finally:
            grounding_callbacks._reset_turn_message(token)
        self.assertEqual(updater.calls, 3)
        self.assertEqual(first.runtime.grounding_revision, 1)
        self.assertEqual(second.runtime.grounding_revision, 2)
        self.assertEqual(third.runtime.grounding_revision, 3)
        self.assertEqual(third.runtime.grounding_state, after_knowledge)

    async def test_invalid_response_and_ineligible_observations_preserve_runtime(self):
        current = {
            "task_id": "sg4-fail-closed",
            "current_phase": 1,
            "tool_trajectory": [],
        }
        invalid = _ScriptedUpdater(
            GroundingLLMResponse(
                sql_grounding_state=SQLGroundingState(
                    tables=("invented",),
                    join_keys=(),
                    column_mapping=(),
                    domain_knowledge=(),
                ),
                next_focus_dimension="none",
            )
        )
        initial = GroundingRuntime()
        token = grounding_callbacks._bind_turn_message(
            current["task_id"], "a-interact", QUERY
        )
        try:
            with (
                patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                patch.object(grounding_callbacks, "_SQL_GROUNDING_UPDATER", invalid),
            ):
                rejected = await grounding_callbacks._handle_observation(
                    current,
                    observation(
                        "schema",
                        "CREATE TABLE operational_metrics (\n  maintcost numeric\n);",
                        "get_schema",
                    ),
                    initial,
                )
                ineligible = (
                    observation("submission", "failed", "submit_sql", 2),
                    observation("p2_follow_up", "follow up", None, 3, phase=2),
                    observation("tool_error", {"error_type": "TimeoutError"}, "execute_sql", 4),
                    observation("user_answer", "clarification", "ask_user", 5),
                )
                for item in ineligible:
                    skipped = await grounding_callbacks._handle_observation(
                        current,
                        item,
                        initial,
                    )
                    self.assertEqual(skipped.runtime, initial)
        finally:
            grounding_callbacks._reset_turn_message(token)
        self.assertEqual(rejected.runtime, initial)
        self.assertEqual(rejected.service_status, "rejected")
        self.assertEqual(invalid.calls, 1)


class _ScriptedUpdater:
    def __init__(self, response: GroundingLLMResponse):
        self.response = response
        self.calls = 0

    async def propose(self, runtime, observation, **kwargs):
        del runtime, observation, kwargs
        self.calls += 1
        return GroundingUpdaterResult(
            response=self.response,
            telemetry=grounding_callbacks.GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="a" * 64,
                response_sha256="b" * 64,
                prompt_sha256=PROMPT_SHA,
                form_schema_sha256=FORM_SHA,
                configuration_sha256=CONFIG_SHA,
                model="offline/fake",
                provider="fake",
                credential_source="file",
            ),
            transport_normalization="none",
        )


class _QueueUpdater(_ScriptedUpdater):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def propose(self, runtime, observation, **kwargs):
        del runtime, observation, kwargs
        response = self.responses[self.calls]
        self.calls += 1
        return GroundingUpdaterResult(
            response=response,
            telemetry=grounding_callbacks.GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="a" * 64,
                response_sha256="b" * 64,
                prompt_sha256=PROMPT_SHA,
                form_schema_sha256=FORM_SHA,
                configuration_sha256=CONFIG_SHA,
                model="offline/fake",
                provider="fake",
                credential_source="file",
            ),
            transport_normalization="none",
        )


if __name__ == "__main__":
    unittest.main()
