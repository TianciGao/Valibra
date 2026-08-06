import asyncio
import hashlib
import inspect
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from valibra_agent import grounding_callbacks
from valibra_agent import server as valibra_server
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.reducer import apply_patch
from valibra_agent.requirement_grounding.service import (
    process_observation_with_llm,
)
from valibra_agent.requirement_grounding.telemetry import (
    LLMCallTelemetryRecorder,
    increment_metrics,
)
from valibra_agent.requirement_grounding import updater as updater_module
from valibra_agent.requirement_grounding.updater import (
    GroundingLLMConfig,
    LLMUpdater,
    LLM_FRAME_PROMPT,
    LLM_FRAME_PROMPT_SHA256,
    load_grounding_llm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _environment(**updates):
    values = {
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_MODEL_PRESET": "glm47_matched_32768",
        "GROUNDING_TIMEOUT_SECONDS": "0.05",
        "GROUNDING_MAX_TOKENS": "1024",
        "GROUNDING_MAX_CALLS_PER_TASK": "2",
        "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
    }
    values.update(updates)
    return values


def _config(**updates):
    return load_grounding_llm_config(
        PROJECT_ROOT,
        _environment(**updates),
    )


def _observation(text="Show the top 5 customer names from 2024.", sequence=1):
    return build_observation(
        task_id="task-llm-contract",
        observation_type="user_query",
        phase=1,
        sequence=sequence,
        source="synthetic_offline_test",
        raw=text,
        summary=text,
    )


def _proposal_content(
    *,
    value_slots=None,
    schema_slots=None,
    operation_slots=None,
    ambiguities=None,
    extra=None,
):
    value = {
        "value_slots": value_slots or [],
        "schema_slots": schema_slots or [],
        "operation_slots": operation_slots or [],
        "ambiguities": ambiguities or [],
    }
    if extra:
        value.update(extra)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


VALID_CONTENT = _proposal_content(
    value_slots=[
        {
            "slot_role": "time_constraint",
            "mention": "2024",
            "interpretation": "calendar year 2024",
            "value_type": "time",
        }
    ],
    schema_slots=[
        {
            "slot_role": "schema_candidate",
            "mention": "customer names",
            "interpretation": "candidate concept customer names",
        }
    ],
    operation_slots=[
        {
            "slot_role": "top_k",
            "mention": "top 5",
            "interpretation": "descending ranked limit 5",
            "operation_type": "limit",
            "parameters": {"direction": "desc", "limit": 5},
        }
    ],
)


class FakeClient:
    provider_may_continue_after_cancel = False
    provider_may_bill_after_cancel = False

    def __init__(
        self,
        *,
        content=VALID_CONTENT,
        response_extra=None,
        error=None,
        block=False,
        may_continue=False,
        may_bill=False,
    ):
        self.content = content
        self.response_extra = response_extra or {}
        self.error = error
        self.block = block
        self.provider_may_continue_after_cancel = may_continue
        self.provider_may_bill_after_cancel = may_bill
        self.calls = 0
        self.requests = []

    async def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return {
            "content": self.content,
            "usage": {
                "input_tokens": 120,
                "output_tokens": 40,
                "reasoning_tokens": 15,
                "cost": 0.0125,
            },
            **self.response_extra,
        }


class SequenceClock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FrozenConfigurationTests(unittest.TestCase):
    def test_prompt_and_configuration_sha_are_frozen_and_stable(self):
        expected_prompt_sha = hashlib.sha256(
            LLM_FRAME_PROMPT.encode("utf-8")
        ).hexdigest()
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, expected_prompt_sha)
        first = _config()
        second = _config()
        self.assertEqual(first, second)
        self.assertEqual(first.configuration_sha256, second.configuration_sha256)
        self.assertEqual(first.prompt_sha256, expected_prompt_sha)
        self.assertEqual(first.model_preset, "glm47_matched_32768")
        self.assertEqual(first.preset_config["model"], "openai/glm-4.7")

    def test_loader_is_read_only_and_never_activates_system_agent_preset(self):
        controlled = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SYSTEM_AGENT_")
        }
        with patch(
            "shared.model_presets.activate_model_preset",
            side_effect=AssertionError("activation is forbidden"),
        ):
            config = _config()
        self.assertEqual(config.mode, "llm")
        self.assertEqual(
            controlled,
            {
                key: value
                for key, value in os.environ.items()
                if key.startswith("SYSTEM_AGENT_")
            },
        )
        source = inspect.getsource(updater_module)
        self.assertNotIn("activate_model_preset", source)

    def test_missing_or_mismatched_frozen_configuration_fails_fast(self):
        missing = _environment()
        del missing["GROUNDING_MAX_CALLS_PER_TASK"]
        with self.assertRaisesRegex(ValueError, "missing frozen"):
            load_grounding_llm_config(PROJECT_ROOT, missing)
        with self.assertRaisesRegex(ValueError, "frozen prompt"):
            _config(GROUNDING_PROMPT_SHA256="0" * 64)
        with self.assertRaisesRegex(ValidationError, "must not exceed"):
            _config(GROUNDING_MAX_TOKENS="65536")

        config = _config()
        damaged = config.model_dump(mode="python")
        damaged["configuration_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValidationError, "configuration SHA256"):
            GroundingLLMConfig.model_validate(damaged)

    def test_all_six_environment_controls_are_required_and_validated(self):
        self.assertEqual(
            set(updater_module.GROUNDING_LLM_ENV_NAMES),
            {
                "GROUNDING_UPDATER_MODE",
                "GROUNDING_MODEL_PRESET",
                "GROUNDING_TIMEOUT_SECONDS",
                "GROUNDING_MAX_TOKENS",
                "GROUNDING_MAX_CALLS_PER_TASK",
                "GROUNDING_PROMPT_SHA256",
            },
        )
        with self.assertRaisesRegex(ValueError, "exactly 'llm'"):
            _config(GROUNDING_UPDATER_MODE="rule")
        with self.assertRaisesRegex(ValueError, "finite positive"):
            _config(GROUNDING_TIMEOUT_SECONDS="nan")


class LLMUpdaterContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_fake_response_produces_stable_patch_and_ids(self):
        observation = _observation()
        runtime = RequirementGroundingRuntime()
        first = LLMUpdater(FakeClient(), _config())
        second = LLMUpdater(FakeClient(), _config())
        first_patch = await first.propose(
            observation,
            runtime.grounding_state,
            base_revision=0,
            telemetry=LLMCallTelemetryRecorder(),
        )
        second_patch = await second.propose(
            observation,
            runtime.grounding_state,
            base_revision=0,
            telemetry=LLMCallTelemetryRecorder(),
        )
        self.assertEqual(first_patch, second_patch)
        self.assertEqual(
            [slot.slot_id for slot in first_patch.slot_additions],
            [slot.slot_id for slot in second_patch.slot_additions],
        )

    async def test_success_has_unbound_schema_empty_ambiguity_and_full_telemetry(self):
        client = FakeClient()
        runtime = RequirementGroundingRuntime()
        result = await process_observation_with_llm(
            runtime,
            _observation(),
            updater=LLMUpdater(client, _config()),
            monotonic=SequenceClock(10.0, 10.025),
        )
        self.assertEqual(result.status, "processed")
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.runtime.grounding_revision, 1)
        self.assertEqual(result.runtime.grounding_state.ambiguity_index, ())
        schema = result.runtime.grounding_state.requirement_frame.schema_slots[0]
        self.assertEqual(schema.binding_type, "unknown")
        self.assertIsNone(schema.bound_identifier)
        for slot in (
            *result.runtime.grounding_state.requirement_frame.value_slots,
            *result.runtime.grounding_state.requirement_frame.schema_slots,
            *result.runtime.grounding_state.requirement_frame.operation_slots,
        ):
            self.assertEqual(slot.grounding_status, "hypothesized")
            self.assertEqual(slot.origin, "llm_provisional")
            self.assertEqual(slot.ambiguity_refs, ())

        metrics = result.runtime.metrics.root
        self.assertEqual(metrics["llm_updater_calls"], 1)
        self.assertEqual(metrics["llm_updater_input_tokens"], 120)
        self.assertEqual(metrics["llm_updater_output_tokens"], 40)
        self.assertEqual(metrics["llm_updater_reasoning_tokens"], 15)
        self.assertAlmostEqual(metrics["llm_updater_latency_ms"], 25.0)
        self.assertAlmostEqual(metrics["llm_updater_cost"], 0.0125)
        self.assertEqual(result.llm_audit.status, "succeeded")
        self.assertEqual(result.llm_audit.cost, 0.0125)
        self.assertIsNone(result.llm_audit.provider_may_continue_after_cancel)

    async def test_request_contains_only_bounded_supplied_contract_inputs(self):
        client = FakeClient(content=_proposal_content())
        observation = _observation("Return average revenue from 2023.")
        result = await process_observation_with_llm(
            RequirementGroundingRuntime(),
            observation,
            updater=LLMUpdater(client, _config()),
        )
        self.assertEqual(result.status, "processed")
        request = client.requests[0]
        self.assertLessEqual(
            len(request.prompt),
            updater_module.MAX_LLM_FRAME_INPUT_CHARS,
        )
        self.assertIn(observation.observation_id, request.prompt)
        self.assertIn(observation.summary, request.prompt)
        self.assertIn('"base_revision":0', request.prompt)
        self.assertEqual(request.prompt_sha256, LLM_FRAME_PROMPT_SHA256)
        self.assertEqual(
            request.configuration_sha256,
            _config().configuration_sha256,
        )

    async def test_empty_frame_is_processed_without_business_revision(self):
        runtime = RequirementGroundingRuntime()
        result = await process_observation_with_llm(
            runtime,
            _observation("Uncertain request."),
            updater=LLMUpdater(FakeClient(content=_proposal_content()), _config()),
        )
        self.assertEqual(result.status, "processed")
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(len(result.runtime.processed_observation_ids), 1)
        self.assertEqual(result.runtime.metrics.root["llm_updater_calls"], 1)

    async def test_strict_json_and_pydantic_fail_open_for_every_invalid_shape(self):
        invalid_contents = {
            "invalid_json": "not-json",
            "extra_root": _proposal_content(extra={"extra": True}),
            "ambiguity": _proposal_content(ambiguities=[{"candidate": "x"}]),
            "schema_binding": _proposal_content(
                schema_slots=[
                    {
                        "slot_role": "schema_candidate",
                        "mention": "orders",
                        "interpretation": "orders",
                        "bound_identifier": "secret.orders",
                    }
                ]
            ),
            "scalar_coercion": _proposal_content(
                value_slots=[
                    {
                        "slot_role": 123,
                        "mention": "2024",
                        "interpretation": "year",
                        "value_type": "time",
                    }
                ]
            ),
            "duplicate_key": (
                '{"value_slots":[],"value_slots":[],"schema_slots":[],'
                '"operation_slots":[],"ambiguities":[]}'
            ),
            "non_finite_json": (
                '{"value_slots":[],"schema_slots":[],"operation_slots":['
                '{"slot_role":"filter","mention":"x","interpretation":"x",'
                '"operation_type":"filter","parameters":{"x":NaN}}],'
                '"ambiguities":[]}'
            ),
            "slot_count_limit": _proposal_content(
                value_slots=[
                    {
                        "slot_role": f"value_{index}",
                        "mention": str(index),
                        "interpretation": f"value {index}",
                        "value_type": "number",
                    }
                    for index in range(65)
                ]
            ),
        }
        for name, content in invalid_contents.items():
            with self.subTest(name=name):
                runtime = RequirementGroundingRuntime()
                result = await process_observation_with_llm(
                    runtime,
                    _observation(),
                    updater=LLMUpdater(FakeClient(content=content), _config()),
                )
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
                self.assertEqual(result.runtime.grounding_revision, 0)
                self.assertEqual(result.runtime.processed_observation_ids, ())
                self.assertEqual(
                    result.runtime.metrics.root["llm_updater_errors"],
                    1,
                )

    async def test_extra_client_envelope_field_is_rejected(self):
        runtime = RequirementGroundingRuntime()
        result = await process_observation_with_llm(
            runtime,
            _observation(),
            updater=LLMUpdater(
                FakeClient(response_extra={"provider_debug": "forbidden"}),
                _config(),
            ),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(result.runtime.grounding_revision, 0)

    async def test_oversized_response_is_rejected_without_state_change(self):
        runtime = RequirementGroundingRuntime()
        result = await process_observation_with_llm(
            runtime,
            _observation(),
            updater=LLMUpdater(
                FakeClient(content="x" * (updater_module.MAX_LLM_FRAME_RESPONSE_CHARS + 1)),
                _config(),
            ),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(result.runtime.grounding_revision, 0)

    async def test_oversized_input_fails_before_fake_provider_is_called(self):
        client = FakeClient()
        runtime = RequirementGroundingRuntime()
        with patch.object(updater_module, "MAX_LLM_FRAME_INPUT_CHARS", 32):
            result = await process_observation_with_llm(
                runtime,
                _observation(),
                updater=LLMUpdater(client, _config()),
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(client.calls, 0)
        self.assertFalse(result.llm_audit.attempted)
        self.assertEqual(
            result.runtime.metrics.root.get("llm_updater_calls", 0),
            0,
        )
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)

    async def test_timeout_records_cancellation_and_billing_uncertainty(self):
        client = FakeClient(block=True, may_continue=True, may_bill=True)
        runtime = RequirementGroundingRuntime()
        result = await process_observation_with_llm(
            runtime,
            _observation(),
            updater=LLMUpdater(
                client,
                _config(GROUNDING_TIMEOUT_SECONDS="0.001"),
            ),
            monotonic=SequenceClock(1.0, 1.002),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.llm_audit.status, "timed_out")
        self.assertTrue(result.llm_audit.timed_out)
        self.assertTrue(result.llm_audit.provider_may_continue_after_cancel)
        self.assertTrue(result.llm_audit.provider_may_bill_after_cancel)
        self.assertEqual(result.runtime.metrics.root["llm_updater_calls"], 1)
        self.assertEqual(result.runtime.metrics.root["llm_updater_timeouts"], 1)
        self.assertEqual(result.runtime.metrics.root["llm_updater_errors"], 1)
        self.assertEqual(result.runtime.grounding_revision, 0)

    async def test_client_exception_is_fail_open_and_independently_counted(self):
        runtime = RequirementGroundingRuntime()
        result = await process_observation_with_llm(
            runtime,
            _observation(),
            updater=LLMUpdater(
                FakeClient(error=RuntimeError("synthetic offline failure")),
                _config(),
            ),
            monotonic=SequenceClock(2.0, 2.01),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.llm_audit.status, "failed")
        self.assertEqual(result.llm_audit.error_type, "RuntimeError")
        self.assertEqual(result.runtime.metrics.root["llm_updater_calls"], 1)
        self.assertEqual(result.runtime.metrics.root["llm_updater_errors"], 1)
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)

    async def test_per_task_call_limit_rejects_without_calling_client(self):
        config = _config(GROUNDING_MAX_CALLS_PER_TASK="2")
        runtime = increment_metrics(
            RequirementGroundingRuntime(),
            llm_updater_calls=2,
        )
        client = FakeClient()
        result = await process_observation_with_llm(
            runtime,
            _observation(),
            updater=LLMUpdater(client, config),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.llm_audit.status, "limit_rejected")
        self.assertFalse(result.llm_audit.attempted)
        self.assertEqual(client.calls, 0)
        self.assertEqual(result.runtime.metrics.root["llm_updater_calls"], 2)
        self.assertEqual(result.runtime.grounding_revision, 0)

    async def test_reducer_failure_rolls_back_business_state_but_keeps_usage_audit(self):
        def broken_reducer(candidate_runtime, candidate_patch):
            stale_runtime = RequirementGroundingRuntime.model_validate(
                {
                    **candidate_runtime.model_dump(mode="python"),
                    "grounding_revision": candidate_runtime.grounding_revision + 1,
                }
            )
            return apply_patch(stale_runtime, candidate_patch)

        runtime = RequirementGroundingRuntime()
        result = await process_observation_with_llm(
            runtime,
            _observation(),
            updater=LLMUpdater(FakeClient(), _config()),
            reducer=broken_reducer,
            monotonic=SequenceClock(4.0, 4.02),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
        self.assertEqual(result.runtime.grounding_revision, 0)
        self.assertEqual(result.runtime.processed_observation_ids, ())
        self.assertEqual(result.runtime.metrics.root["patches_rejected"], 1)
        self.assertEqual(result.runtime.metrics.root["llm_updater_input_tokens"], 120)
        self.assertEqual(result.runtime.metrics.root["llm_updater_cost"], 0.0125)
        self.assertEqual(result.runtime.metrics.root["llm_updater_errors"], 1)

    async def test_duplicate_observation_does_not_repeat_llm_call(self):
        client = FakeClient(content=_proposal_content())
        updater = LLMUpdater(client, _config())
        observation = _observation()
        first = await process_observation_with_llm(
            RequirementGroundingRuntime(), observation, updater=updater
        )
        replay = await process_observation_with_llm(
            first.runtime, observation, updater=updater
        )
        self.assertEqual(first.status, "processed")
        self.assertEqual(replay.status, "duplicate")
        self.assertIs(replay.runtime, first.runtime)
        self.assertEqual(client.calls, 1)


class WiringBoundaryTests(unittest.TestCase):
    def test_current_runtime_remains_rule_shadow_and_llm_is_not_wired(self):
        callback_source = inspect.getsource(grounding_callbacks)
        self.assertNotIn("LLMUpdater", callback_source)
        self.assertNotIn("process_observation_with_llm", callback_source)
        summary = valibra_server._configuration_summary()
        self.assertEqual(summary["grounding_mode"], "shadow")
        self.assertEqual(summary["grounding_updater"], "rule")
        self.assertFalse(summary["prompt_view_injected"])

    def test_llm_contract_has_no_provider_db_tool_or_prompt_view_implementation(self):
        source = "\n".join(
            (
                inspect.getsource(updater_module),
                inspect.getsource(process_observation_with_llm),
            )
        )
        for forbidden in (
            "activate_model_preset",
            "render_prompt_view",
            "get_schema(",
            "task_data",
            "test_cases",
            "sol_sql",
            "litellm",
            "httpx",
            "requests",
            "spacy",
            "stanza",
            "torch",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__":
    unittest.main()
