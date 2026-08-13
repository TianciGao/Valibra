"""Offline acceptance tests for the P7.1c explicit empty-outcome contract.

The suite deliberately uses only synthetic clients and Session state.  It
freezes the V1.1 Form/Prompt/Runtime boundary without exercising the later
P7.1d provider-quality or evaluation scope.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent.requirement_grounding.models import (
    SCHEMA_VERSION,
    RequirementGroundingRuntime,
    migrate_requirement_grounding_runtime,
)
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.service import process_observation_with_llm
from valibra_agent.requirement_grounding.updater import (
    LLM_FRAME_FORM_SCHEMA,
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    LLMFrameUpdateError,
    LLMUpdater,
    load_grounding_llm_config,
)
from valibra_agent.requirement_grounding import updater as updater_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROMPT_SHA256 = (
    "ddaaa23fd3c824a704ea1e17a769b4c8b1af5c998949fccfe8d564b18f78f7c4"
)
EXPECTED_FORM_SHA256 = (
    "f7409a7267b3fddcb40d69574e6187320884951ad4e96980bb590ee0058c38d1"
)
EXPECTED_CONFIG_SHA256 = (
    "2ec2accb786a1f1e4d35027affe52c0402957861f59a93832583ac4094066dce"
)
QUESTION = "Show orders from 2024."
ANSWER = "I cannot provide more information."


def _value_slot(mention: str = "2024") -> dict[str, object]:
    return {
        "slot_role": "time_constraint",
        "mention": mention,
        "interpretation": f"calendar year {mention}",
        "value_type": "time",
    }


def _proposal(
    proposal_outcome: str,
    *,
    value_slots: list[dict[str, object]] | None = None,
    schema_slots: list[dict[str, object]] | None = None,
    operation_slots: list[dict[str, object]] | None = None,
    ambiguities: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "proposal_outcome": proposal_outcome,
        "value_slots": [] if value_slots is None else value_slots,
        "schema_slots": [] if schema_slots is None else schema_slots,
        "operation_slots": [] if operation_slots is None else operation_slots,
        "ambiguities": [] if ambiguities is None else ambiguities,
    }


def _content(proposal_outcome: str, *, populated: bool = False) -> str:
    return json.dumps(
        _proposal(
            proposal_outcome,
            value_slots=[_value_slot()] if populated else [],
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse(payload: dict[str, object], *, text: str = QUESTION):
    return updater_module._parse_llm_frame_response(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        observation_text=text,
    )


def _config():
    return load_grounding_llm_config(
        PROJECT_ROOT,
        {
            "GROUNDING_UPDATER_MODE": "llm",
            "GROUNDING_MODEL_PRESET": "glm52_high_32768",
            "GROUNDING_TIMEOUT_SECONDS": "300",
            "GROUNDING_MAX_TOKENS": "32768",
            "GROUNDING_MAX_CALLS_PER_TASK": "2",
            "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
        },
    )


def _observation(
    text: str = QUESTION,
    *,
    observation_type: str = "user_query",
    sequence: int = 1,
    phase: int = 1,
    task_id: str = "task-p71c",
):
    tool_identity = {}
    if observation_type == "user_answer":
        tool_identity = {
            "function_call_id": f"call-{sequence}",
            "tool_name": "ask_user",
        }
    return build_observation(
        task_id=task_id,
        observation_type=observation_type,
        phase=phase,
        sequence=sequence,
        source="p71c_synthetic",
        raw=text,
        summary=text,
        **tool_identity,
    )


class FakeClient:
    provider_may_continue_after_cancel = False
    provider_may_bill_after_cancel = False

    def __init__(self, *contents: str):
        if not contents:
            raise ValueError("at least one synthetic response is required")
        self._contents = list(contents)
        self.calls = 0
        self.requests = []

    async def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        await asyncio.sleep(0)
        index = min(self.calls - 1, len(self._contents) - 1)
        return {
            "content": self._contents[index],
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "reasoning_tokens": 0,
                "total_tokens": 3,
                "cost": None,
            },
            "model": "openai/glm-5.2",
            "provider": "offline-fake",
            "credential_source": "file",
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "raw_audit_ref": "private://p71c/fake.json",
        }


def _state(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "current_phase": 1,
        "budget_remaining": 20.0,
        "initial_budget": 20.0,
        "tool_trajectory": [],
        "system_agent_llm_calls": [{"actions": []}],
        "_active_llm_call_index": 0,
    }


def _runtime(state: dict[str, object]) -> RequirementGroundingRuntime:
    return RequirementGroundingRuntime.model_validate(
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY]
    )


async def _consume_query(state: dict[str, object], message: str = QUESTION):
    token = grounding_callbacks._bind_turn_message(
        str(state["task_id"]),
        "a-interact",
        message,
    )
    try:
        return await grounding_callbacks._consume_bound_user_message(state)
    finally:
        grounding_callbacks._reset_turn_message(token)


class ProposalOutcomeFormTests(unittest.TestCase):
    def test_populated_and_empty_outcomes_have_exact_cross_constraints(self):
        accepted = (
            _proposal("populated", value_slots=[_value_slot()]),
            _proposal("insufficient_information"),
            _proposal("no_extractable_requirement"),
        )
        for payload in accepted:
            with self.subTest(accepted=payload["proposal_outcome"]):
                parsed = _parse(payload)
                self.assertEqual(
                    parsed.proposal_outcome,
                    payload["proposal_outcome"],
                )

        rejected = (
            _proposal("populated"),
            _proposal(
                "insufficient_information",
                value_slots=[_value_slot()],
            ),
            _proposal(
                "no_extractable_requirement",
                value_slots=[_value_slot()],
            ),
        )
        for payload in rejected:
            with self.subTest(rejected=payload["proposal_outcome"]):
                with self.assertRaises(ValidationError):
                    _parse(payload)

    def test_old_form_nonempty_ambiguity_and_duplicate_key_stay_rejected(self):
        old_v1 = _proposal("populated", value_slots=[_value_slot()])
        del old_v1["proposal_outcome"]
        with self.assertRaises(ValidationError):
            _parse(old_v1)

        ambiguity = _proposal(
            "populated",
            value_slots=[_value_slot()],
            ambiguities=[{"candidate": "forbidden"}],
        )
        with self.assertRaises(ValidationError):
            _parse(ambiguity)

        duplicate = (
            '{"proposal_outcome":"populated",'
            '"proposal_outcome":"populated",'
            '"value_slots":[],"schema_slots":[],'
            '"operation_slots":[],"ambiguities":[]}'
        )
        with self.assertRaises(LLMFrameUpdateError) as duplicated:
            updater_module._parse_llm_frame_response(
                duplicate,
                observation_text=QUESTION,
            )
        self.assertEqual(duplicated.exception.reason, "duplicate_json_key")


class ServiceOutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_propagates_outcome_and_empty_keeps_revisions_zero(self):
        scenarios = (
            ("populated", True, (1, 1), 1),
            ("insufficient_information", False, (0, 0), 0),
            ("no_extractable_requirement", False, (0, 0), 0),
        )
        for outcome, populated, revisions, expected_slots in scenarios:
            with self.subTest(outcome=outcome):
                runtime = RequirementGroundingRuntime()
                client = FakeClient(_content(outcome, populated=populated))
                result = await process_observation_with_llm(
                    runtime,
                    _observation(task_id=f"service-{outcome}"),
                    updater=LLMUpdater(client, _config()),
                )

                self.assertEqual(result.status, "processed")
                self.assertEqual(result.proposal_outcome, outcome)
                self.assertEqual(client.calls, 1)
                self.assertEqual(
                    (
                        result.runtime.grounding_revision,
                        result.runtime.requirement_revision,
                    ),
                    revisions,
                )
                frame = result.runtime.grounding_state.requirement_frame
                self.assertEqual(
                    len(frame.value_slots)
                    + len(frame.schema_slots)
                    + len(frame.operation_slots),
                    expected_slots,
                )
                self.assertEqual(len(result.runtime.processed_observation_ids), 1)

    async def test_empty_user_answer_is_semantic_noop_without_evidence_pollution(self):
        for outcome in (
            "insufficient_information",
            "no_extractable_requirement",
        ):
            with self.subTest(outcome=outcome):
                task_id = f"service-answer-noop-{outcome}"
                updater = LLMUpdater(
                    FakeClient(
                        _content("populated", populated=True),
                        _content(outcome),
                    ),
                    _config(),
                )
                initial = await process_observation_with_llm(
                    RequirementGroundingRuntime(),
                    _observation(task_id=task_id),
                    updater=updater,
                )
                before = initial.runtime

                answer = await process_observation_with_llm(
                    before,
                    _observation(
                        ANSWER,
                        observation_type="user_answer",
                        sequence=2,
                        task_id=task_id,
                    ),
                    updater=updater,
                )

                self.assertEqual(answer.status, "processed")
                self.assertEqual(answer.proposal_outcome, outcome)
                self.assertEqual(
                    answer.runtime.grounding_state,
                    before.grounding_state,
                )
                self.assertEqual(
                    (
                        answer.runtime.grounding_revision,
                        answer.runtime.requirement_revision,
                    ),
                    (before.grounding_revision, before.requirement_revision),
                )
                self.assertEqual(
                    answer.runtime.grounding_state.evidence,
                    before.grounding_state.evidence,
                )
                self.assertEqual(
                    len(answer.runtime.processed_observation_ids),
                    len(before.processed_observation_ids) + 1,
                )


class CallbackInitializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_one_callback_records_ready_or_explicit_empty_once(self):
        scenarios = (
            ("populated", True, "ready", None, (1, 1)),
            (
                "insufficient_information",
                False,
                "empty",
                "insufficient_information",
                (0, 0),
            ),
            (
                "no_extractable_requirement",
                False,
                "empty",
                "no_extractable_requirement",
                (0, 0),
            ),
        )
        for outcome, populated, status, reason, revisions in scenarios:
            with self.subTest(outcome=outcome):
                state = _state(f"callback-{outcome}")
                client = FakeClient(_content(outcome, populated=populated))
                updater = LLMUpdater(client, _config())
                with (
                    patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
                    patch.object(
                        grounding_callbacks,
                        "_build_llm_updater",
                        return_value=updater,
                    ),
                ):
                    audit = await _consume_query(state)

                runtime = _runtime(state)
                self.assertEqual(client.calls, 1)
                self.assertEqual(audit["status"], "processed")
                self.assertEqual(audit["proposal_outcome"], outcome)
                self.assertEqual(runtime.frame_initialization_status, status)
                self.assertEqual(runtime.frame_initialization_reason, reason)
                self.assertEqual(
                    runtime.frame_initialization_observation_id,
                    audit["observation_id"],
                )
                self.assertEqual(
                    (runtime.grounding_revision, runtime.requirement_revision),
                    revisions,
                )

    async def test_callback_user_answer_empty_does_not_rewrite_ready_frame(self):
        state = _state("callback-answer-noop")
        client = FakeClient(
            _content("populated", populated=True),
            _content("insufficient_information"),
        )
        updater = LLMUpdater(client, _config())
        tool = SimpleNamespace(name="ask_user")
        context = SimpleNamespace(
            state=state,
            function_call_id="call-answer",
            invocation_id="inv-answer",
        )
        baseline_override = object()
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
            patch.object(
                baseline_callbacks,
                "before_tool_callback",
                AsyncMock(return_value=None),
            ),
            patch.object(
                baseline_callbacks,
                "after_tool_callback",
                AsyncMock(return_value=baseline_override),
            ),
        ):
            await _consume_query(state)
            before = _runtime(state)
            rejection = await grounding_callbacks.before_tool_callback(
                tool,
                {"question": "Can you clarify?"},
                context,
            )
            returned = await grounding_callbacks.after_tool_callback(
                tool,
                {"question": "Can you clarify?"},
                context,
                ANSWER,
            )

        after = _runtime(state)
        self.assertIsNone(rejection)
        self.assertIs(returned, baseline_override)
        self.assertEqual(client.calls, 2)
        self.assertEqual(after.grounding_state, before.grounding_state)
        self.assertEqual(
            (after.grounding_revision, after.requirement_revision),
            (before.grounding_revision, before.requirement_revision),
        )
        self.assertEqual(
            after.grounding_state.evidence,
            before.grounding_state.evidence,
        )
        self.assertEqual(after.frame_initialization_status, "ready")
        self.assertEqual(
            after.frame_initialization_observation_id,
            before.frame_initialization_observation_id,
        )
        self.assertEqual(after.pending_tool_calls, {})
        self.assertEqual(
            len(after.processed_observation_ids),
            len(before.processed_observation_ids) + 1,
        )

    async def test_phase_two_empty_is_noop_and_keeps_initialization_history(self):
        state = _state("callback-phase2-empty")
        client = FakeClient(
            _content("populated", populated=True),
            _content("no_extractable_requirement"),
        )
        updater = LLMUpdater(client, _config())
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=updater,
            ),
        ):
            first = await _consume_query(state)
            before = _runtime(state)
            state["current_phase"] = 2
            second = await _consume_query(
                state,
                "No additional requirement applies in phase two.",
            )

        after = _runtime(state)
        self.assertEqual(first["proposal_outcome"], "populated")
        self.assertEqual(
            second["proposal_outcome"],
            "no_extractable_requirement",
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(after.grounding_state, before.grounding_state)
        self.assertEqual(
            (after.grounding_revision, after.requirement_revision),
            (before.grounding_revision, before.requirement_revision),
        )
        self.assertEqual(after.frame_initialization_status, "ready")
        self.assertEqual(
            after.frame_initialization_observation_id,
            before.frame_initialization_observation_id,
        )
        self.assertEqual(
            len(after.processed_observation_ids),
            len(before.processed_observation_ids) + 1,
        )

    async def test_legacy_unknown_history_is_migrated_but_never_backfilled(self):
        state = _state("callback-legacy-unknown")
        legacy = RequirementGroundingRuntime().model_dump(mode="json")
        legacy["schema_version"] = "1.0"
        legacy["grounding_revision"] = 3
        legacy["metrics"] = {"observations_seen": 7}
        for field in (
            "requirement_revision",
            "frame_initialization_status",
            "frame_initialization_reason",
            "frame_initialization_observation_id",
        ):
            legacy.pop(field)
        state[grounding_callbacks.GROUNDING_RUNTIME_KEY] = legacy
        client = FakeClient(_content("populated", populated=True))
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": "llm"}),
            patch.object(
                grounding_callbacks,
                "_build_llm_updater",
                return_value=LLMUpdater(client, _config()),
            ),
        ):
            await _consume_query(state)

        runtime = _runtime(state)
        self.assertEqual(runtime.schema_version, "1.1")
        self.assertTrue(
            state[grounding_callbacks.GROUNDING_LEGACY_INITIALIZATION_UNKNOWN_KEY]
        )
        self.assertEqual(runtime.frame_initialization_status, "not_attempted")
        self.assertIsNone(runtime.frame_initialization_reason)
        self.assertIsNone(runtime.frame_initialization_observation_id)
        self.assertEqual(
            (runtime.grounding_revision, runtime.requirement_revision),
            (4, 1),
        )
        self.assertEqual(runtime.metrics.root["observations_seen"], 8)


class FrozenV11AndMigrationTests(unittest.TestCase):
    def test_runtime_form_prompt_and_configuration_are_frozen_as_v11(self):
        self.assertEqual(SCHEMA_VERSION, "1.1")
        runtime = RequirementGroundingRuntime()
        self.assertEqual(runtime.schema_version, "1.1")
        self.assertEqual(runtime.model_dump(mode="json")["schema_version"], "1.1")
        self.assertEqual(
            LLM_FRAME_FORM_SCHEMA["required"],
            [
                "proposal_outcome",
                "value_slots",
                "schema_slots",
                "operation_slots",
                "ambiguities",
            ],
        )
        self.assertEqual(
            LLM_FRAME_FORM_SCHEMA["properties"]["proposal_outcome"]["enum"],
            [
                "populated",
                "insufficient_information",
                "no_extractable_requirement",
            ],
        )
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, EXPECTED_PROMPT_SHA256)
        self.assertEqual(LLM_FRAME_FORM_SCHEMA_SHA256, EXPECTED_FORM_SHA256)
        self.assertEqual(_config().configuration_sha256, EXPECTED_CONFIG_SHA256)

    def test_explicit_v1_migration_preserves_unknown_history_and_input(self):
        legacy = RequirementGroundingRuntime().model_dump(mode="json")
        legacy["schema_version"] = "1.0"
        for field in (
            "requirement_revision",
            "frame_initialization_status",
            "frame_initialization_reason",
            "frame_initialization_observation_id",
        ):
            legacy.pop(field)
        before = copy.deepcopy(legacy)

        migrated = migrate_requirement_grounding_runtime(legacy)

        self.assertEqual(legacy, before)
        self.assertTrue(migrated.legacy_unknown_history)
        self.assertEqual(migrated.runtime.schema_version, "1.1")
        self.assertEqual(migrated.runtime.requirement_revision, 0)
        self.assertEqual(
            migrated.runtime.frame_initialization_status,
            "not_attempted",
        )
        self.assertEqual(
            migrated.runtime.model_dump(mode="json")["schema_version"],
            "1.1",
        )
        with self.assertRaises(ValidationError):
            RequirementGroundingRuntime.model_validate(legacy)

    def test_current_payload_cannot_hide_history_and_unknown_versions_reject(self):
        current = RequirementGroundingRuntime().model_dump(mode="json")
        missing_history = dict(current)
        missing_history.pop("frame_initialization_status")
        with self.assertRaises((ValidationError, ValueError)):
            migrate_requirement_grounding_runtime(missing_history)

        for version in (None, "0.9", "1.2"):
            payload = dict(current)
            if version is None:
                payload.pop("schema_version")
            else:
                payload["schema_version"] = version
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    migrate_requirement_grounding_runtime(payload)


if __name__ == "__main__":
    unittest.main()
