import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from valibra_agent import grounding_callbacks
from valibra_agent import server as valibra_server
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.service import (
    process_observation_with_llm,
)
from valibra_agent.requirement_grounding.telemetry import (
    increment_metrics,
    summarize_model_usage_totals,
)
from valibra_agent.requirement_grounding.updater import (
    GroundingProviderError,
    LLM_FRAME_FORM_SCHEMA,
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLMUpdater,
    LLM_FRAME_PROMPT_SHA256,
    LiteLLMGroundingClient,
    load_grounding_llm_config,
    load_grounding_provider_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_QUESTION = (
    "Show the top 5 customer names from 2024 sorted by total descending."
)
VALID_CONTENT = json.dumps(
    {
        "proposal_outcome": "populated",
        "value_slots": [
            {
                "slot_role": "time_constraint",
                "mention": "2024",
                "interpretation": "calendar year 2024",
                "value_type": "time",
            }
        ],
        "schema_slots": [
            {
                "slot_role": "schema_candidate",
                "mention": "customer names",
                "interpretation": "candidate concept customer names",
            }
        ],
        "operation_slots": [
            {
                "slot_role": "top_k",
                "mention": "top 5",
                "interpretation": "descending ranked limit 5",
                "operation_type": "limit",
                "parameters": {"direction": "desc", "limit": 5},
            }
        ],
        "ambiguities": [],
    },
    sort_keys=True,
    separators=(",", ":"),
)


def _llm_environment(**updates):
    values = {
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "30",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "2",
        "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
    }
    values.update(updates)
    return values


def _llm_config(**updates):
    return load_grounding_llm_config(
        PROJECT_ROOT,
        _llm_environment(**updates),
    )


def _query_observation(observation_type="user_query", sequence=1):
    tool_kwargs = {}
    if observation_type == "user_answer":
        tool_kwargs = {
            "function_call_id": f"call-{sequence}",
            "tool_name": "ask_user",
        }
    return build_observation(
        task_id="task-p4c-offline",
        observation_type=observation_type,
        phase=1,
        sequence=sequence,
        source="synthetic_offline_test",
        raw=SYNTHETIC_QUESTION,
        summary=SYNTHETIC_QUESTION,
        **tool_kwargs,
    )


def _ineligible_observation(observation_type, sequence):
    tool_types = {
        "schema",
        "metadata",
        "knowledge",
        "sql_execution",
        "submission",
        "tool_error",
    }
    tool_kwargs = {}
    if observation_type in tool_types:
        tool_kwargs = {
            "function_call_id": f"call-{sequence}",
            "tool_name": "synthetic_tool",
        }
    return build_observation(
        task_id="task-p4c-offline",
        observation_type=observation_type,
        phase=1,
        sequence=sequence,
        source="synthetic_offline_test",
        raw={"kind": observation_type},
        **tool_kwargs,
    )


def _provider_environment(key_file, **updates):
    values = {
        "GROUNDING_API_BASE": "https://provider.invalid/v1",
        "GROUNDING_API_KEY": "",
        "GROUNDING_API_KEY_FILE": str(key_file),
        "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE": "false",
    }
    values.update(updates)
    return values


def _provider_config(temp_root, environment):
    config = load_grounding_provider_config(
        PROJECT_ROOT,
        _llm_config(),
        environment,
    )
    data = config.model_dump(mode="python")
    data["raw_audit_dir"] = str(temp_root / "research-runtime" / "grounding-llm")
    return type(config).model_validate(data)


class FakeCompletion:
    def __init__(self, *, error=None, cost_marker=False):
        self.error = error
        self.cost_marker = cost_marker
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        usage = {
            "prompt_tokens": 21,
            "completion_tokens": 13,
            "total_tokens": 34,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
        if self.cost_marker:
            usage["cost"] = 0.004
        return {
            "choices": [{"message": {"content": VALID_CONTENT}}],
            "usage": usage,
            "_hidden_params": {
                "custom_llm_provider": "openai",
                "response_cost": 999.0,
            },
        }


class GroundingProviderConfigurationTests(unittest.TestCase):
    def test_provider_loader_uses_only_explicit_grounding_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "grounding.key"
            key_file.write_text("offline-placeholder\n", encoding="utf-8")
            environment = _provider_environment(key_file)
            environment.update(
                {
                    "SYSTEM_AGENT_API_BASE": "https://forbidden.invalid/v1",
                    "SYSTEM_AGENT_API_KEY": "must-not-be-read",
                    "SYSTEM_AGENT_MODEL": "openai/forbidden-model",
                }
            )
            config = load_grounding_provider_config(
                PROJECT_ROOT,
                _llm_config(),
                environment,
            )
        self.assertEqual(config.api_base, "https://provider.invalid/v1")
        self.assertEqual(config.model_id, "openai/glm-5.2")
        self.assertEqual(config.credential_source, "file")
        self.assertEqual(config.retry_count, 0)

    def test_missing_grounding_values_never_fall_back_to_system_role(self):
        environment = {
            "SYSTEM_AGENT_API_BASE": "https://system.invalid/v1",
            "SYSTEM_AGENT_API_KEY": "system-only",
        }
        with self.assertRaisesRegex(ValueError, "GROUNDING_API_BASE"):
            load_grounding_provider_config(
                PROJECT_ROOT,
                _llm_config(),
                environment,
            )

    def test_exactly_one_credential_source_and_nonempty_file_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_file = Path(directory) / "empty.key"
            empty_file.touch()
            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_grounding_provider_config(
                    PROJECT_ROOT,
                    _llm_config(),
                    _provider_environment(empty_file),
                )

            key_file = Path(directory) / "grounding.key"
            key_file.write_text("offline-placeholder\n", encoding="utf-8")
            both = _provider_environment(
                key_file,
                GROUNDING_API_KEY="also-present",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_grounding_provider_config(
                    PROJECT_ROOT,
                    _llm_config(),
                    both,
                )

    def test_example_exposes_only_empty_grounding_configuration(self):
        values = {}
        for line in (PROJECT_ROOT / ".env.example").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("GROUNDING_") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        self.assertEqual(
            set(values),
            {
                "GROUNDING_UPDATER_MODE",
                "GROUNDING_MODEL_PRESET",
                "GROUNDING_TIMEOUT_SECONDS",
                "GROUNDING_MAX_TOKENS",
                "GROUNDING_MAX_CALLS_PER_TASK",
                "GROUNDING_PROMPT_SHA256",
                "GROUNDING_API_BASE",
                "GROUNDING_API_KEY",
                "GROUNDING_API_KEY_FILE",
                "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE",
            },
        )
        self.assertTrue(all(value == "" for value in values.values()))


class NativeAsyncProviderAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_native_async_request_has_no_tools_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            key_file = temp_root / "grounding.key"
            key_value = "offline-grounding-credential"
            key_file.write_text(key_value + "\n", encoding="utf-8")
            environment = _provider_environment(key_file)
            config = _provider_config(temp_root, environment)
            completion = FakeCompletion()
            client = LiteLLMGroundingClient(
                config,
                environment=environment,
                completion=completion,
            )
            result = await process_observation_with_llm(
                RequirementGroundingRuntime(),
                _query_observation(),
                updater=LLMUpdater(client, _llm_config()),
            )

            self.assertEqual(result.status, "processed")
            self.assertEqual(len(completion.calls), 1)
            request = completion.calls[0]
            self.assertEqual(request["model"], "openai/glm-5.2")
            self.assertEqual(request["api_base"], "https://provider.invalid/v1")
            self.assertEqual(request["api_key"], key_value)
            self.assertEqual(request["max_tokens"], 32768)
            self.assertEqual(request["num_retries"], 0)
            self.assertEqual(request["max_retries"], 0)
            self.assertEqual(
                request["response_format"],
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "valibra_requirement_frame",
                        "strict": True,
                        "schema": LLM_FRAME_FORM_SCHEMA,
                    },
                },
            )
            self.assertNotIn("tools", request)
            self.assertNotIn("tool_choice", request)
            self.assertNotIn("SYSTEM_AGENT_API_KEY", request)

            audit_files = list(
                (temp_root / "research-runtime" / "grounding-llm").glob("*.json")
            )
            self.assertEqual(len(audit_files), 1)
            audit_text = audit_files[0].read_text(encoding="utf-8")
            audit = json.loads(audit_text)
            self.assertNotIn(key_value, audit_text)
            self.assertEqual(audit["request"]["tools"], [])
            self.assertIsNone(audit["request"]["tool_choice"])
            self.assertEqual(audit["request"]["provider_retry_count"], 0)
            self.assertEqual(
                audit["request"]["form_schema_sha256"],
                LLM_FRAME_FORM_SCHEMA_SHA256,
            )
            self.assertEqual(
                audit["request"]["response_format"]["json_schema"]["schema"],
                LLM_FRAME_FORM_SCHEMA,
            )
            self.assertIn(SYNTHETIC_QUESTION, audit_text)
            self.assertEqual(
                audit["response"]["choices"][0]["message"]["content"],
                VALID_CONTENT,
            )
            self.assertEqual(
                stat.S_IMODE(audit_files[0].stat().st_mode),
                0o600,
            )
            self.assertEqual(
                stat.S_IMODE(audit_files[0].parent.stat().st_mode),
                0o700,
            )

            self.assertEqual(result.llm_audit.credential_source, "file")
            self.assertEqual(result.llm_audit.model, "openai/glm-5.2")
            self.assertEqual(result.llm_audit.input_tokens, 21)
            self.assertEqual(result.llm_audit.output_tokens, 13)
            self.assertEqual(result.llm_audit.reasoning_tokens, 5)
            self.assertEqual(result.llm_audit.total_tokens, 34)
            self.assertIsNone(result.llm_audit.cost)
            self.assertTrue(result.llm_audit.raw_audit_ref.startswith(
                "research-runtime/grounding-llm/"
            ))

    async def test_only_provider_reported_usage_cost_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            key_file = temp_root / "grounding.key"
            key_file.write_text("offline-placeholder\n", encoding="utf-8")
            environment = _provider_environment(key_file)
            completion = FakeCompletion(cost_marker=True)
            client = LiteLLMGroundingClient(
                _provider_config(temp_root, environment),
                environment=environment,
                completion=completion,
            )
            result = await process_observation_with_llm(
                RequirementGroundingRuntime(),
                _query_observation(),
                updater=LLMUpdater(client, _llm_config()),
            )
        self.assertAlmostEqual(result.llm_audit.cost, 0.004)
        self.assertAlmostEqual(
            result.runtime.metrics.root["llm_updater_cost"],
            0.004,
        )

    async def test_provider_failure_is_sanitized_and_fail_open(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            key_file = temp_root / "grounding.key"
            key_value = "offline-sensitive-placeholder"
            key_file.write_text(key_value + "\n", encoding="utf-8")
            environment = _provider_environment(key_file)
            completion = FakeCompletion(
                error=RuntimeError(f"provider exposed {key_value}"),
            )
            client = LiteLLMGroundingClient(
                _provider_config(temp_root, environment),
                environment=environment,
                completion=completion,
            )
            result = await process_observation_with_llm(
                RequirementGroundingRuntime(),
                _query_observation(),
                updater=LLMUpdater(client, _llm_config()),
            )
            audit_files = list(
                (temp_root / "research-runtime" / "grounding-llm").glob("*.json")
            )
            self.assertEqual(len(audit_files), 1)
            audit_text = audit_files[0].read_text(encoding="utf-8")

        self.assertEqual(len(completion.calls), 1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.processed_observation_ids, ())
        self.assertEqual(result.llm_audit.error_type, "GroundingProviderError")
        self.assertNotIn(key_value, result.runtime.model_dump_json())
        self.assertNotIn(key_value, audit_text)
        self.assertNotIn("provider exposed", audit_text)

    async def test_model_or_configuration_mismatch_fails_before_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            key_file = temp_root / "grounding.key"
            key_file.write_text("offline-placeholder\n", encoding="utf-8")
            environment = _provider_environment(key_file)
            config = _provider_config(temp_root, environment)
            completion = FakeCompletion()
            client = LiteLLMGroundingClient(
                config,
                environment=environment,
                completion=completion,
            )
            mismatched_llm_config = _llm_config(
                GROUNDING_MODEL_PRESET="glm47_matched_32768",
            )
            result = await process_observation_with_llm(
                RequirementGroundingRuntime(),
                _query_observation(),
                updater=LLMUpdater(client, mismatched_llm_config),
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(completion.calls, [])


class ObservationEligibilityAndLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_user_query_and_user_answer_consume_call_budget(self):
        class CountingClient:
            provider_may_continue_after_cancel = False
            provider_may_bill_after_cancel = False

            def __init__(self):
                self.calls = 0

            async def complete(self, request):
                self.calls += 1
                return {
                    "content": json.dumps(
                        {
                            "proposal_outcome": "no_extractable_requirement",
                            "value_slots": [],
                            "schema_slots": [],
                            "operation_slots": [],
                            "ambiguities": [],
                        }
                    ),
                    "usage": {},
                }

        client = CountingClient()
        updater = LLMUpdater(client, _llm_config())
        runtime = RequirementGroundingRuntime()
        ineligible = (
            "schema",
            "metadata",
            "knowledge",
            "sql_execution",
            "submission",
            "phase_transition",
            "tool_error",
        )
        for sequence, observation_type in enumerate(ineligible, start=1):
            result = await process_observation_with_llm(
                runtime,
                _ineligible_observation(observation_type, sequence),
                updater=updater,
            )
            self.assertEqual(result.status, "skipped")
            self.assertEqual(result.llm_audit.status, "ineligible")
            self.assertIs(result.runtime, runtime)
        self.assertEqual(client.calls, 0)
        self.assertEqual(runtime.metrics.root.get("llm_updater_calls", 0), 0)

        query = await process_observation_with_llm(
            runtime,
            _query_observation("user_query", 20),
            updater=updater,
        )
        answer = await process_observation_with_llm(
            query.runtime,
            _query_observation("user_answer", 21),
            updater=updater,
        )
        self.assertEqual(query.status, "processed")
        self.assertEqual(answer.status, "processed")
        self.assertEqual(client.calls, 2)
        self.assertEqual(answer.runtime.metrics.root["llm_updater_calls"], 2)

    async def test_main_and_grounding_ledgers_are_separate_with_read_only_total(self):
        runtime = increment_metrics(
            RequirementGroundingRuntime(),
            llm_updater_calls=1,
            llm_updater_input_tokens=20,
            llm_updater_output_tokens=10,
            llm_updater_reasoning_tokens=4,
            llm_updater_total_tokens=30,
        )
        before = runtime.model_dump_json()
        combined = summarize_model_usage_totals(
            {
                "model_calls": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "reasoning_tokens": 25,
                "total_tokens": 150,
            },
            runtime,
            main_agent_cost=None,
        )
        self.assertEqual(combined.main_agent.total_tokens, 150)
        self.assertEqual(combined.grounding.total_tokens, 30)
        self.assertEqual(combined.total_model_tokens, 180)
        self.assertIsNone(combined.main_agent.cost)
        self.assertIsNone(combined.grounding.cost)
        self.assertIsNone(combined.total_model_cost)
        self.assertNotIn("bird", combined.model_dump_json().lower())
        self.assertEqual(runtime.model_dump_json(), before)

    def test_empty_mode_keeps_rule_shadow(self):
        callback_source = Path(grounding_callbacks.__file__).read_text(
            encoding="utf-8"
        )
        self.assertIn("LLMUpdater", callback_source)
        self.assertIn("LiteLLMGroundingClient", callback_source)
        with patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}):
            summary = valibra_server._configuration_summary()
        self.assertEqual(summary["grounding_updater"], "rule")
        self.assertFalse(summary["prompt_view_injected"])


if __name__ == "__main__":
    unittest.main()
