import copy
import json
import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from system_agent import callbacks as baseline_callbacks
from valibra_agent import grounding_callbacks
from valibra_agent.requirement_grounding.evidence_bridge import (
    EVIDENCE_OBSERVATION_TYPES,
    EvidenceBridgeUpdater,
)
from valibra_agent.requirement_grounding.models import (
    RequirementFrame,
    RequirementGroundingRuntime,
    RequirementGroundingState,
    ValueSlot,
)
from valibra_agent.requirement_grounding.observations import (
    ObservationNormalizationError,
    build_observation,
    classify_tool_observation_type,
    extract_submit_follow_up,
)
from valibra_agent.requirement_grounding.service import (
    process_observation,
    process_observation_with_llm,
)
from valibra_agent.requirement_grounding.updater import (
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    LLMUpdater,
)
from tests.valibra.test_p4d_llm_callback_shadow import (
    EXPECTED_FORM_SHA256,
    EXPECTED_PROMPT_SHA256,
    FakeClient,
    QUERY,
    _content,
    _context,
    _llm_config,
    _run_bound_query,
    _runtime,
    _state,
)


EXPECTED_CONFIG_SHA256 = (
    "2ec2accb786a1f1e4d35027affe52c0402957861f59a93832583ac4094066dce"
)
PRE_P71C_CONFIG_SHA256 = (
    "83ba93c060b110a0e48485f8d5083052d96a67c8a79892ab77024be3c4b5ccd9"
)
ANSWER = "Use 2023 instead."
FOLLOW_UP = "For phase 2, sort ascending and include limit 10."


def _frame(*, values=(), schemas=(), operations=()):
    values = list(values)
    schemas = list(schemas)
    operations = list(operations)
    return json.dumps(
        {
            "proposal_outcome": (
                "populated"
                if values or schemas or operations
                else "no_extractable_requirement"
            ),
            "value_slots": values,
            "schema_slots": schemas,
            "operation_slots": operations,
            "ambiguities": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _observation(
    observation_type,
    text,
    *,
    phase=1,
    sequence=1,
    task_id="task-p4g",
):
    tool_kwargs = {}
    if observation_type not in {"user_query", "phase_transition"}:
        tool_kwargs = {
            "function_call_id": f"call-{sequence}",
            "tool_name": "ask_user" if observation_type == "user_answer" else "get_schema",
        }
    return build_observation(
        task_id=task_id,
        observation_type=observation_type,
        phase=phase,
        sequence=sequence,
        source="p4g_synthetic",
        raw=text,
        summary=text,
        **tool_kwargs,
    )


def _duplicate_role_runtime(*, phase=1):
    slots = tuple(
        ValueSlot(
            slot_id=f"slot.value.{index}",
            slot_role="time_constraint",
            mention=str(year),
            current_interpretation=f"calendar year {year}",
            grounding_status="hypothesized",
            origin="llm_provisional",
            lifecycle="active",
            introduced_in_phase=1,
            last_updated_phase=1,
            sequence=index,
            value_type="time",
        )
        for index, year in ((1, 2022), (2, 2024))
    )
    return RequirementGroundingRuntime(
        phase=phase,
        grounding_state=RequirementGroundingState(
            requirement_frame=RequirementFrame(value_slots=slots)
        ),
    )


class SlotReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def _apply(self, runtime, observation, content):
        client = FakeClient(content=content)
        result = await process_observation_with_llm(
            runtime,
            observation,
            updater=LLMUpdater(client, _llm_config()),
        )
        self.assertEqual(client.calls, 1)
        return result

    async def _initial_value_runtime(self):
        result = await self._apply(
            RequirementGroundingRuntime(),
            _observation("user_query", "Show orders from 2024.", sequence=1),
            _content(value_mention="2024"),
        )
        self.assertEqual(result.status, "processed")
        return result.runtime

    async def test_user_answer_unique_match_reuses_slot_identity(self):
        initial = await self._initial_value_runtime()
        old = initial.grounding_state.requirement_frame.value_slots[0]
        result = await self._apply(
            initial,
            _observation("user_answer", ANSWER, sequence=2),
            _content(value_mention="2023"),
        )
        updated = result.runtime.grounding_state.requirement_frame.value_slots[0]
        self.assertEqual(updated.slot_id, old.slot_id)
        self.assertEqual(updated.introduced_in_phase, old.introduced_in_phase)
        self.assertEqual(updated.last_updated_phase, 1)
        self.assertEqual(updated.mention, "2023")
        self.assertEqual(updated.grounding_status, "hypothesized")
        self.assertEqual(updated.origin, "llm_provisional")
        self.assertEqual(len(updated.evidence_refs), len(old.evidence_refs) + 1)

    async def test_user_answer_zero_or_multiple_match_is_atomic_noop(self):
        cases = (
            (RequirementGroundingRuntime(), "zero"),
            (_duplicate_role_runtime(), "multiple"),
        )
        for runtime, label in cases:
            with self.subTest(label=label):
                before = runtime.grounding_state
                revision = runtime.grounding_revision
                result = await self._apply(
                    runtime,
                    _observation("user_answer", ANSWER, sequence=3),
                    _content(value_mention="2023"),
                )
                self.assertEqual(result.status, "processed")
                self.assertEqual(result.runtime.grounding_state, before)
                self.assertEqual(result.runtime.grounding_revision, revision)
                self.assertEqual(len(result.runtime.processed_observation_ids), 1)

    async def test_duplicate_proposal_key_rejects_entire_semantic_update(self):
        initial = await self._initial_value_runtime()
        answer = "Use 2023, not 2022."
        duplicate = _frame(
            values=(
                {
                    "slot_role": "time_constraint",
                    "mention": "2023",
                    "interpretation": "calendar year 2023",
                    "value_type": "time",
                },
                {
                    "slot_role": "time_constraint",
                    "mention": "2022",
                    "interpretation": "calendar year 2022",
                    "value_type": "time",
                },
            )
        )
        before = initial.grounding_state
        result = await self._apply(
            initial,
            _observation("user_answer", answer, sequence=2),
            duplicate,
        )
        self.assertEqual(result.runtime.grounding_state, before)
        self.assertEqual(result.runtime.grounding_revision, initial.grounding_revision)

    async def test_phase_two_unique_updates_zero_adds_and_multiple_refuses(self):
        initial = await self._initial_value_runtime()
        phase_two = RequirementGroundingRuntime.model_validate(
            {**initial.model_dump(mode="python"), "phase": 2}
        )
        update = await self._apply(
            phase_two,
            _observation(
                "user_query",
                "For phase 2 use 2025.",
                phase=2,
                sequence=2,
            ),
            _content(value_mention="2025"),
        )
        old_id = initial.grounding_state.requirement_frame.value_slots[0].slot_id
        updated = update.runtime.grounding_state.requirement_frame.value_slots[0]
        self.assertEqual(updated.slot_id, old_id)
        self.assertEqual(updated.introduced_in_phase, 1)
        self.assertEqual(updated.last_updated_phase, 2)

        add_text = "For phase 2 add customer names."
        addition = await self._apply(
            update.runtime,
            _observation("user_query", add_text, phase=2, sequence=3),
            _content(schema_mention="customer names"),
        )
        schemas = addition.runtime.grounding_state.requirement_frame.schema_slots
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0].introduced_in_phase, 2)
        self.assertEqual(schemas[0].binding_type, "unknown")
        self.assertIsNone(schemas[0].bound_identifier)

        ambiguous = _duplicate_role_runtime(phase=2)
        before = ambiguous.grounding_state
        refused = await self._apply(
            ambiguous,
            _observation(
                "user_query",
                "For phase 2 use 2023.",
                phase=2,
                sequence=4,
            ),
            _content(value_mention="2023"),
        )
        self.assertEqual(refused.runtime.grounding_state, before)
        self.assertEqual(refused.runtime.grounding_revision, 0)


class EvidenceAndToolErrorTests(unittest.TestCase):
    def test_all_five_success_types_are_evidence_only_and_duplicate_is_idempotent(self):
        runtime = RequirementGroundingRuntime()
        updater = EvidenceBridgeUpdater()
        for sequence, observation_type in enumerate(
            sorted(EVIDENCE_OBSERVATION_TYPES), start=1
        ):
            observation = _observation(
                observation_type,
                f"bounded {observation_type} result",
                sequence=sequence,
            )
            result = process_observation(runtime, observation, updater=updater)
            self.assertEqual(result.status, "processed")
            self.assertEqual(
                result.runtime.grounding_state.requirement_frame,
                runtime.grounding_state.requirement_frame,
            )
            self.assertEqual(result.runtime.grounding_state.ambiguity_index, ())
            runtime = result.runtime
            duplicate = process_observation(runtime, observation, updater=updater)
            self.assertEqual(duplicate.status, "duplicate")
            self.assertEqual(duplicate.runtime, runtime)
        self.assertEqual(runtime.grounding_revision, 5)
        self.assertEqual(len(runtime.grounding_state.evidence), 5)

    def test_official_error_prefixes_are_finite_and_exact(self):
        cases = (
            ("execute_sql", "SQL Error: bad SQL", "sql_execution"),
            (
                "execute_sql",
                "Error calling DB environment: ConnectError",
                "sql_execution",
            ),
            ("get_schema", "Error: unavailable", "schema"),
            ("submit_sql", "Error: rejected", "submission"),
        )
        for tool_name, response, success_type in cases:
            with self.subTest(tool=tool_name, response=response):
                self.assertEqual(
                    classify_tool_observation_type(
                        tool_name=tool_name,
                        tool_response=response,
                        success_type=success_type,
                    ),
                    "tool_error",
                )
                self.assertEqual(
                    classify_tool_observation_type(
                        tool_name=tool_name,
                        tool_response=f" {response}",
                        success_type=success_type,
                    ),
                    success_type,
                )

    def test_follow_up_parser_accepts_only_the_official_current_response(self):
        legal = (
            "Phase 1 passed\nReward: 1\n"
            f"Follow-up question: {FOLLOW_UP}\n"
            "Budget remaining: 12 bird-coins"
        )
        self.assertEqual(extract_submit_follow_up(legal), FOLLOW_UP)
        invalid = (
            "no marker\nBudget remaining: 12 bird-coins",
            "Follow-up question: q\nFollow-up question: q2\n"
            "Budget remaining: 12 bird-coins",
            "Follow-up question: q",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ObservationNormalizationError):
                    extract_submit_follow_up(value)

class FrozenContractTests(unittest.TestCase):
    def test_p71c_refreezes_hashes_without_changing_lifecycle_contract(self):
        config = _llm_config(timeout="300", max_calls="2")
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, EXPECTED_PROMPT_SHA256)
        self.assertEqual(LLM_FRAME_FORM_SCHEMA_SHA256, EXPECTED_FORM_SHA256)
        self.assertEqual(config.configuration_sha256, EXPECTED_CONFIG_SHA256)
        self.assertNotEqual(config.configuration_sha256, PRE_P71C_CONFIG_SHA256)
