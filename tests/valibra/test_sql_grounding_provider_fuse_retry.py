from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from shared.config import PROJECT_ROOT
from valibra_agent.sql_grounding.models import (
    GroundingRuntime,
    SQLGroundingState,
    ValidationContext,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.service import process_sql_grounding_observation
from valibra_agent.sql_grounding.updater import (
    MAPPING_VALIDATION_CORRECTION_PROMPT,
    SQL_GROUNDING_CONFIGURATION,
    SQL_GROUNDING_EXACT_EMPTY_RETRY_REASON,
    SQL_GROUNDING_STAGE_FORM_SCHEMAS,
    SQL_GROUNDING_STAGE_MAX_TOKENS,
    SQL_GROUNDING_STAGE_PROMPTS,
    GroundingLLMRequest,
    GroundingUpdaterError,
    LiteLLMSQLGroundingClient,
    SQLGroundingProviderConfig,
    SQLGroundingUpdater,
    load_sql_grounding_llm_config,
    load_sql_grounding_provider_config,
)


PROMPT_SHA = "abcd64292037ba6fa5f6672c04383d47f9742da0ae63763afd66cc4ee8affccd"


def _environment(key_file: Path) -> dict[str, str]:
    return {
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "600",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "8",
        "GROUNDING_PROMPT_SHA256": PROMPT_SHA,
        "GROUNDING_API_BASE": "https://provider.invalid/v1",
        "GROUNDING_API_KEY": "",
        "GROUNDING_API_KEY_FILE": str(key_file),
        "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE": "false",
    }


def _request(call_kind: str) -> GroundingLLMRequest:
    return GroundingLLMRequest(
        prompt=SQL_GROUNDING_STAGE_PROMPTS[call_kind],
        input_json="{}",
        response_schema=SQL_GROUNDING_STAGE_FORM_SCHEMAS[call_kind],
        call_kind=call_kind,
        observation_id=f"fuse-{call_kind}",
        observation_type="schema",
    )


def _valid_content(call_kind: str) -> str:
    payloads = {
        "structure": {"tables": [], "join_keys": []},
        "mapping": {"tables": [], "join_keys": [], "column_mapping": [], "unresolved_mappings": []},
        "knowledge": {"column_mapping": [], "selected_knowledge_ids": []},
        "check": {
            "status": "complete",
            "missing_information": None,
            "next_tool": None,
            "column_mapping": [],
            "domain_knowledge": [],
        },
    }
    return json.dumps(payloads[call_kind], sort_keys=True)


def _mapping_runtime() -> GroundingRuntime:
    return GroundingRuntime(
        grounding_revision=1,
        stage="INITIAL_GROUNDING",
        focus_dimension="column_mapping",
        grounding_state=SQLGroundingState(
            tables=("metrics",),
            join_keys=(),
        ),
    )


def _mapping_observation(task_id: str):
    return build_sql_grounding_observation(
        task_id=task_id,
        phase=1,
        sequence=2,
        observation_type="metadata",
        content={"metrics.value": "numeric metric value"},
        summary="Official column meanings",
        tool_name="get_all_column_meanings",
        function_call_id=f"{task_id}-meanings",
    )


def _mapping_input(runtime: GroundingRuntime) -> dict[str, object]:
    return {
        "query": "show value",
        "current_state": runtime.grounding_state.model_dump(mode="json"),
        "column_meanings": {"metrics.value": "numeric metric value"},
        "unresolved_mappings": [],
    }


def _knowledge_runtime() -> GroundingRuntime:
    return GroundingRuntime(
        grounding_revision=2,
        stage="INITIAL_GROUNDING",
        focus_dimension="domain_knowledge",
        grounding_state=SQLGroundingState(
            tables=("metrics",),
            join_keys=(),
        ),
    )


def _knowledge_observation(task_id: str):
    return build_sql_grounding_observation(
        task_id=task_id,
        phase=1,
        sequence=3,
        observation_type="knowledge",
        content=[{"id": 32, "definition": "Use the official metric rule."}],
        summary="Official knowledge definitions",
        tool_name="get_all_knowledge_definitions",
        function_call_id=f"{task_id}-knowledge",
    )


def _knowledge_input(runtime: GroundingRuntime) -> dict[str, object]:
    return {
        "query": "show value",
        "current_state": runtime.grounding_state.model_dump(mode="json"),
        "knowledge_definitions": [
            {"id": 32, "definition": "Use the official metric rule."}
        ],
        "relevant_column_meanings": {},
        "unresolved_mappings": [],
    }


def _response(
    content: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 7,
    reasoning_tokens: int = 3,
) -> dict[str, object]:
    return {
        "choices": [
            {"finish_reason": finish_reason, "message": {"content": content}}
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
            "total_tokens": 11 + completion_tokens,
        },
        "_hidden_params": {"custom_llm_provider": "offline"},
    }


class StageSpecificFuseRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key_file = self.root / "grounding.key"
        self.key_file.write_text("unit-test-credential\n", encoding="utf-8")
        self.environment = _environment(self.key_file)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _client(self, completion):
        llm = load_sql_grounding_llm_config(PROJECT_ROOT, self.environment)
        provider = load_sql_grounding_provider_config(
            PROJECT_ROOT, llm, self.environment
        ).model_copy(
            update={"raw_audit_dir": str(self.root / "research-runtime" / "audit")}
        )
        return LiteLLMSQLGroundingClient(
            llm,
            SQLGroundingProviderConfig.model_validate(provider),
            environment=self.environment,
            completion=completion,
        )

    async def test_stage_specific_fuse_and_sdk_retry_zero(self):
        expected = {
            "structure": 12_288,
            "mapping": 32_768,
            "knowledge": 24_576,
            "check": 12_288,
        }
        self.assertEqual(SQL_GROUNDING_STAGE_MAX_TOKENS, expected)
        self.assertEqual(SQL_GROUNDING_CONFIGURATION["stage_max_tokens"], expected)
        self.assertEqual(
            SQL_GROUNDING_CONFIGURATION["provider_execution_policy"],
            {
                "exact_empty_retry_stages": [
                    "check",
                    "knowledge",
                    "mapping",
                    "structure",
                ],
                "max_identical_retries": 1,
                "retry_reason": SQL_GROUNDING_EXACT_EMPTY_RETRY_REASON,
                "sdk_retry_count": 0,
            },
        )
        calls = []

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return _response(_valid_content(current_kind))

        client = self._client(completion)
        for current_kind in expected:
            calls.clear()
            await client.complete(_request(current_kind))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["max_tokens"], expected[current_kind])
            self.assertEqual(calls[0]["num_retries"], 0)
            self.assertEqual(calls[0]["max_retries"], 0)

    async def test_all_stages_exact_empty_fuse_retry_once_identically(self):
        for call_kind in ("structure", "mapping", "knowledge", "check"):
            with self.subTest(call_kind=call_kind):
                calls = []
                fuse = SQL_GROUNDING_STAGE_MAX_TOKENS[call_kind]

                async def completion(**kwargs):
                    calls.append(copy.deepcopy(kwargs))
                    if len(calls) == 1:
                        return _response(
                            "",
                            finish_reason="MAX_TOKENS",
                            completion_tokens=fuse + 1,
                            reasoning_tokens=fuse,
                        )
                    return _response(_valid_content(call_kind))

                response = await self._client(completion).complete(
                    _request(call_kind)
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0], calls[1])
                self.assertEqual(calls[0]["max_tokens"], fuse)
                self.assertEqual(response.attempt_count, 2)
                self.assertTrue(response.retry_triggered)
                self.assertEqual(
                    response.retry_trigger_reason,
                    SQL_GROUNDING_EXACT_EMPTY_RETRY_REASON,
                )
                self.assertEqual(
                    response.provider_attempts[0].request_sha256,
                    response.provider_attempts[1].request_sha256,
                )
                self.assertEqual(response.final_selected_attempt, 2)
                self.assertEqual(response.content, _valid_content(call_kind))
                self.assertEqual(
                    response.usage.output_tokens,
                    fuse + 1 + 7,
                )

    async def test_normal_pass_nonempty_length_and_insufficient_usage_do_not_retry(self):
        cases = (
            _response(_valid_content("mapping")),
            _response(
                '{"tables":',
                finish_reason="length",
                completion_tokens=32_769,
                reasoning_tokens=32_768,
            ),
            _response(
                "",
                finish_reason="length",
                completion_tokens=0,
                reasoning_tokens=0,
            ),
        )
        for provider_response in cases:
            with self.subTest(provider_response=provider_response):
                calls = []

                async def completion(**kwargs):
                    calls.append(copy.deepcopy(kwargs))
                    return provider_response

                await self._client(completion).complete(_request("mapping"))
                self.assertEqual(len(calls), 1)

    async def test_nonempty_stop_invalid_json_is_not_retried(self):
        cases = (
            ("not-json", "json_invalid"),
            (
                json.dumps(
                    {
                        "tables": ["metrics"],
                        "join_keys": [],
                        "unresolved_mappings": [],
                        "column_mapping": [],
                        "unexpected": True,
                    }
                ),
                "form_validation_failed",
            ),
        )
        for content, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                calls = []

                async def completion(**kwargs):
                    calls.append(copy.deepcopy(kwargs))
                    return _response(content, finish_reason="stop")

                updater = SQLGroundingUpdater(self._client(completion))
                runtime = _mapping_runtime()
                observation = _mapping_observation(f"fuse-{expected_reason}")
                with self.assertRaises(GroundingUpdaterError) as raised:
                    await updater.propose(
                        runtime,
                        observation,
                        original_query="show value",
                        grounding_input=_mapping_input(runtime),
                    )
                self.assertEqual(raised.exception.reason, expected_reason)
                self.assertEqual(len(calls), 1)
                self.assertFalse(raised.exception.telemetry.retry_triggered)

    async def test_check_incomplete_and_terminal_incomplete_do_not_retry(self):
        contents = (
            {
                "status": "incomplete",
                "missing_information": "one specific gap",
                "next_tool": None,
                "column_mapping": [],
                "domain_knowledge": [],
            },
            {
                "status": "incomplete",
                "missing_information": "one specific gap",
                "next_tool": {
                    "tool_name": "get_all_external_knowledge_names",
                    "arguments": {},
                    "user_clarification_request": None,
                },
                "column_mapping": [],
                "domain_knowledge": [],
            },
        )
        for payload in contents:
            with self.subTest(next_tool=payload["next_tool"]):
                calls = []

                async def completion(**kwargs):
                    calls.append(copy.deepcopy(kwargs))
                    return _response(json.dumps(payload), finish_reason="stop")

                response = await self._client(completion).complete(
                    _request("check")
                )
                self.assertEqual(len(calls), 1)
                self.assertFalse(response.retry_triggered)

    async def test_second_nonempty_invalid_response_stops_without_third_attempt(self):
        calls = []
        fuse = SQL_GROUNDING_STAGE_MAX_TOKENS["mapping"]

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if len(calls) == 1:
                return _response(
                    "",
                    finish_reason="length",
                    completion_tokens=fuse,
                    reasoning_tokens=fuse,
                )
            return _response("not-json", finish_reason="stop")

        updater = SQLGroundingUpdater(self._client(completion))
        runtime = _mapping_runtime()
        observation = _mapping_observation("fuse-invalid-second")
        with self.assertRaises(GroundingUpdaterError) as raised:
            await updater.propose(
                runtime,
                observation,
                original_query="show value",
                grounding_input=_mapping_input(runtime),
            )
        self.assertEqual(raised.exception.reason, "json_invalid")
        self.assertEqual(len(calls), 2)
        self.assertEqual(raised.exception.telemetry.attempt_count, 2)

    async def test_knowledge_exact_empty_fuse_retries_once_at_24576(self):
        calls = []
        fuse = SQL_GROUNDING_STAGE_MAX_TOKENS["knowledge"]

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if len(calls) == 1:
                return _response(
                    "",
                    finish_reason="length",
                    completion_tokens=fuse + 1,
                    reasoning_tokens=fuse,
                )
            return _response(_valid_content("knowledge"))

        response = await self._client(completion).complete(
            _request("knowledge")
        )
        self.assertEqual(fuse, 24_576)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0]["max_tokens"], 24_576)
        self.assertEqual(calls[0]["num_retries"], 0)
        self.assertEqual(calls[0]["max_retries"], 0)
        self.assertTrue(response.retry_triggered)
        self.assertEqual(response.attempt_count, 2)
        self.assertEqual(response.final_selected_attempt, 2)
        self.assertEqual(
            response.provider_attempts[0].request_sha256,
            response.provider_attempts[1].request_sha256,
        )

    async def test_second_empty_fuse_hit_stops_at_two_and_fails_closed(self):
        cases = (
            (
                "mapping",
                _mapping_runtime,
                _mapping_observation,
                _mapping_input,
            ),
            (
                "knowledge",
                _knowledge_runtime,
                _knowledge_observation,
                _knowledge_input,
            ),
        )
        for call_kind, runtime_factory, observation_factory, input_factory in cases:
            with self.subTest(call_kind=call_kind):
                calls = []
                fuse = SQL_GROUNDING_STAGE_MAX_TOKENS[call_kind]

                async def completion(**kwargs):
                    calls.append(copy.deepcopy(kwargs))
                    return _response(
                        "",
                        finish_reason="length",
                        completion_tokens=fuse + 1,
                        reasoning_tokens=fuse,
                    )

                updater = SQLGroundingUpdater(self._client(completion))
                observation = observation_factory(f"fuse-twice-{call_kind}")
                runtime = runtime_factory()
                with self.assertRaises(GroundingUpdaterError) as raised:
                    await updater.propose(
                        runtime,
                        observation,
                        original_query="show value",
                        grounding_input=input_factory(runtime),
                    )
                self.assertEqual(raised.exception.reason, "json_invalid")
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    calls[0],
                    calls[1],
                )
                telemetry = raised.exception.telemetry
                self.assertEqual(telemetry.attempt_count, 2)
                self.assertTrue(telemetry.retry_triggered)
                self.assertEqual(telemetry.final_selected_attempt, 2)

    async def test_knowledge_bad_json_and_form_output_are_not_retried(self):
        contents = (
            "not-json",
            json.dumps(
                {
                    "column_mapping": [],
                    "selected_knowledge_ids": [],
                    "unexpected": True,
                }
            ),
        )
        for content in contents:
            with self.subTest(content=content):
                calls = []

                async def completion(**kwargs):
                    calls.append(copy.deepcopy(kwargs))
                    return _response(content, finish_reason="stop")

                runtime = _knowledge_runtime()
                observation = _knowledge_observation("knowledge-invalid")
                with self.assertRaises(GroundingUpdaterError):
                    await SQLGroundingUpdater(self._client(completion)).propose(
                        runtime,
                        observation,
                        original_query="show value",
                        grounding_input=_knowledge_input(runtime),
                    )
                self.assertEqual(len(calls), 1)

    async def test_knowledge_state_validation_failure_is_not_retried(self):
        calls = []

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return _response(
                json.dumps(
                    {
                        "column_mapping": [],
                        "selected_knowledge_ids": [99],
                    }
                )
            )

        observation = _knowledge_observation("knowledge-semantic-reject")
        runtime = _knowledge_runtime()
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            ValidationContext(
                current_query="show value",
                latest_observation_id=observation.observation_id,
                known_tables=frozenset({"metrics"}),
                known_columns=frozenset(),
            ),
            SQLGroundingUpdater(self._client(completion)),
            grounding_input=_knowledge_input(runtime),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.state_update.error_type, "state_validation_failed")
        self.assertEqual(result.runtime, runtime)
        self.assertFalse(result.llm_telemetry.retry_triggered)

    async def test_second_attempt_form_pass_is_selected_without_extra_revision(self):
        calls = []
        fuse = SQL_GROUNDING_STAGE_MAX_TOKENS["mapping"]

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if len(calls) == 1:
                return _response(
                    "",
                    finish_reason="length",
                    completion_tokens=fuse + 1,
                    reasoning_tokens=fuse,
                )
            return _response(
                json.dumps(
                    {
                        "tables": ["metrics"],
                        "join_keys": [],
                        "unresolved_mappings": [],
                        "column_mapping": [
                            {"phrase": "value", "targets": ["metrics.value"]}
                        ],
                    }
                )
            )

        updater = SQLGroundingUpdater(self._client(completion))
        observation = _mapping_observation("fuse-salvaged")
        runtime = _mapping_runtime()
        result = await updater.propose(
            runtime,
            observation,
            original_query="show value",
            grounding_input=_mapping_input(runtime),
        )
        self.assertEqual(result.response.tables, ("metrics",))
        self.assertEqual(result.telemetry.attempt_count, 2)
        self.assertEqual(result.telemetry.final_selected_attempt, 2)
        self.assertEqual(runtime.grounding_revision, 1)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("tools", calls[0])
        self.assertNotIn("tool_choice", calls[0])

    async def test_salvaged_response_reaches_service_with_one_state_revision(self):
        calls = []
        fuse = SQL_GROUNDING_STAGE_MAX_TOKENS["mapping"]

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if len(calls) == 1:
                return _response(
                    "",
                    finish_reason="length",
                    completion_tokens=fuse + 1,
                    reasoning_tokens=fuse,
                )
            return _response(
                json.dumps(
                    {
                        "tables": ["metrics"],
                        "join_keys": [],
                        "unresolved_mappings": [],
                        "column_mapping": [
                            {"phrase": "value", "targets": ["metrics.value"]}
                        ],
                    }
                )
            )

        observation = _mapping_observation("fuse-service")
        runtime = _mapping_runtime()
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            ValidationContext(
                current_query="show value",
                latest_observation_id=observation.observation_id,
                known_tables=frozenset({"metrics"}),
                known_columns=frozenset({"metrics.value"}),
            ),
            SQLGroundingUpdater(self._client(completion)),
            grounding_input=_mapping_input(runtime),
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(runtime.grounding_revision, 1)
        self.assertEqual(result.runtime.grounding_revision, 2)
        self.assertEqual(result.runtime.grounding_state.tables, ("metrics",))
        self.assertEqual(
            result.runtime.grounding_state.column_mapping[0].targets,
            ("metrics.value",),
        )
        self.assertEqual(result.llm_telemetry.attempt_count, 2)

    async def test_state_validation_failure_gets_one_logical_correction_not_retry(self):
        calls = []

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return _response(
                json.dumps(
                    {
                        "tables": ["metrics"],
                        "join_keys": [],
                        "unresolved_mappings": [],
                        "column_mapping": [
                            {
                                "phrase": "value",
                                "targets": ["metrics.invented"],
                            }
                        ],
                    }
                )
            )

        observation = _mapping_observation("fuse-semantic-reject")
        runtime = _mapping_runtime()
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            ValidationContext(
                current_query="show value",
                latest_observation_id=observation.observation_id,
                known_tables=frozenset({"metrics"}),
                known_columns=frozenset({"metrics.value"}),
            ),
            SQLGroundingUpdater(self._client(completion)),
            grounding_input=_mapping_input(runtime),
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["messages"][0]["content"], MAPPING_VALIDATION_CORRECTION_PROMPT)
        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.state_update.error_type, "state_validation_failed")
        self.assertEqual(result.runtime, runtime)
        self.assertFalse(result.llm_telemetry.retry_triggered)
        self.assertEqual(result.llm_telemetry.attempt_count, 1)
        self.assertIsNotNone(result.mapping_validation_repair)
        self.assertEqual(
            result.mapping_validation_repair.outcome,
            "repair_validation_failed",
        )

    async def test_private_audit_records_both_real_attempts(self):
        calls = []
        fuse = SQL_GROUNDING_STAGE_MAX_TOKENS["knowledge"]

        async def completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if len(calls) == 1:
                return _response(
                    "",
                    finish_reason="length",
                    completion_tokens=fuse + 1,
                    reasoning_tokens=fuse,
                )
            return _response(_valid_content("knowledge"))

        response = await self._client(completion).complete(_request("knowledge"))
        audit_path = self.root / response.raw_private_audit_ref
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["call_kind"], "knowledge")
        self.assertEqual(audit["configured_max_tokens"], 24_576)
        self.assertEqual(audit["attempt_count"], 2)
        self.assertTrue(audit["retry_triggered"])
        self.assertEqual(
            audit["retry_trigger_reason"],
            SQL_GROUNDING_EXACT_EMPTY_RETRY_REASON,
        )
        self.assertEqual(len(audit["attempts"]), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(
            audit["attempts"][0]["request_sha256"],
            audit["attempts"][1]["request_sha256"],
        )
        self.assertEqual(audit["attempts"][0]["finish_reason"], "length")
        self.assertTrue(audit["attempts"][0]["content_empty"])
        self.assertEqual(
            audit["attempts"][0]["completion_tokens"],
            24_577,
        )
        self.assertEqual(audit["attempts"][1]["finish_reason"], "stop")
        self.assertFalse(audit["attempts"][1]["content_empty"])
        self.assertGreaterEqual(audit["attempts"][0]["latency_ms"], 0)
        self.assertGreaterEqual(audit["attempts"][1]["latency_ms"], 0)
        self.assertEqual(audit["final_selected_attempt"], 2)


if __name__ == "__main__":
    unittest.main()
