import inspect
import json
import unittest

from pydantic import ValidationError

from shared.config import PROJECT_ROOT
from valibra_agent.requirement_grounding import updater as updater_module
from valibra_agent.requirement_grounding.models import RequirementGroundingRuntime
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.service import process_observation_with_llm
from valibra_agent.requirement_grounding.updater import (
    LLMUpdater,
    LLM_FRAME_FORM_SCHEMA,
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT,
    LLM_FRAME_PROMPT_SHA256,
    load_grounding_llm_config,
)


OLD_PROMPT_SHA256 = (
    "b5c55caa8dd06d8d8e9c54670e118bae1f2e5753ed48a9a1329f1d89ff812e00"
)
PARAMETER_OBJECT_PROMPT_SHA256 = (
    "5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d"
)
PARAMETER_OBJECT_FORM_SCHEMA_SHA256 = (
    "441a59c410a99ef0db53b8e974aeeeaea1bcd1735aabdc3cb51f59e0b6e069a2"
)
OLD_CONFIGURATION_SHA256 = (
    "d6ec85e73d6061bdfc8b4eb23196595ed9086a05b622fb75ccfbaaedc4d154b4"
)
PARAMETER_OBJECT_CONFIGURATION_SHA256 = (
    "83ba93c060b110a0e48485f8d5083052d96a67c8a79892ab77024be3c4b5ccd9"
)
P71C_PROMPT_SHA256 = (
    "ddaaa23fd3c824a704ea1e17a769b4c8b1af5c998949fccfe8d564b18f78f7c4"
)
P71C_FORM_SCHEMA_SHA256 = (
    "f7409a7267b3fddcb40d69574e6187320884951ad4e96980bb590ee0058c38d1"
)
P71C_CONFIGURATION_SHA256 = (
    "2ec2accb786a1f1e4d35027affe52c0402957861f59a93832583ac4094066dce"
)
QUESTION = "Show the top 5 customer names from 2024 sorted by total descending."


def _environment():
    return {
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "300",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "2",
        "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
    }


def _config():
    return load_grounding_llm_config(PROJECT_ROOT, _environment())


def _frame(parameters):
    return {
        "proposal_outcome": "populated",
        "value_slots": [],
        "schema_slots": [
            {
                "slot_role": "schema_candidate",
                "mention": "customer names",
                "interpretation": "candidate customer-name concept",
            }
        ],
        "operation_slots": [
            {
                "slot_role": "ordering",
                "mention": "sorted by total descending",
                "interpretation": "order by total descending",
                "operation_type": "order",
                "parameters": parameters,
            }
        ],
        "ambiguities": [],
    }


def _parse(frame):
    return updater_module._parse_llm_frame_response(
        json.dumps(frame, sort_keys=True, separators=(",", ":")),
        observation_text=QUESTION,
    )


def _observation():
    return build_observation(
        task_id="p4e-parameter-object-recovery",
        observation_type="user_query",
        phase=1,
        sequence=1,
        source="synthetic_offline_test",
        raw=QUESTION,
        summary=QUESTION,
    )


class _StaticClient:
    provider_may_continue_after_cancel = False
    provider_may_bill_after_cancel = False

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return {
            "content": json.dumps(
                self.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "usage": {},
        }


class ParameterObjectPromptTests(unittest.TestCase):
    def test_prompt_uses_explicit_object_examples_and_keeps_all_boundaries(self):
        for text in (
            "parameters MUST always be",
            'Correct: "parameters": {"direction": "desc"}',
            'Correct: "parameters": {"limit": 5}',
            'Correct when there are no parameters: "parameters": {}',
            'Wrong: "parameters": "desc"',
            'Wrong: "parameters": 5',
            'Wrong: "parameters": true',
            "projection,",
            "filter, aggregation, group, order, limit, distinct, or other",
            "copied verbatim as one continuous",
            "case-sensitive substring",
            "never emit",
            "ambiguities must always be an empty array",
            "no Markdown, prose, comments, or additional fields",
            "never nested",
        ):
            with self.subTest(text=text):
                self.assertIn(text, LLM_FRAME_PROMPT)

    def test_p71c_refreezes_prompt_form_and_configuration_hashes(self):
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, P71C_PROMPT_SHA256)
        self.assertNotEqual(LLM_FRAME_PROMPT_SHA256, OLD_PROMPT_SHA256)
        self.assertNotEqual(
            LLM_FRAME_PROMPT_SHA256,
            PARAMETER_OBJECT_PROMPT_SHA256,
        )
        self.assertEqual(LLM_FRAME_FORM_SCHEMA_SHA256, P71C_FORM_SCHEMA_SHA256)
        self.assertNotEqual(
            LLM_FRAME_FORM_SCHEMA_SHA256,
            PARAMETER_OBJECT_FORM_SCHEMA_SHA256,
        )
        config = _config()
        self.assertEqual(config.prompt_sha256, P71C_PROMPT_SHA256)
        self.assertEqual(config.form_schema_sha256, P71C_FORM_SCHEMA_SHA256)
        self.assertEqual(config.configuration_sha256, P71C_CONFIGURATION_SHA256)
        self.assertNotEqual(
            config.configuration_sha256,
            OLD_CONFIGURATION_SHA256,
        )
        self.assertNotEqual(
            config.configuration_sha256,
            PARAMETER_OBJECT_CONFIGURATION_SHA256,
        )

    def test_operation_enum_is_unchanged(self):
        enum = LLM_FRAME_FORM_SCHEMA["$defs"]["LLMOperationSlotProposal"][
            "properties"
        ]["operation_type"]["enum"]
        self.assertEqual(
            enum,
            [
                "projection",
                "filter",
                "aggregation",
                "group",
                "order",
                "limit",
                "distinct",
                "other",
            ],
        )

    def test_source_has_no_alias_or_parameter_auto_repair(self):
        source = inspect.getsource(updater_module)
        for forbidden in (
            "order_by",
            "sorting",
            "normalize_parameters",
            "repair_parameters",
            "wrap_parameters",
            "scalar_to_object",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class ParameterObjectValidationTests(unittest.TestCase):
    def test_legal_parameter_objects_accept_every_scalar_value_type(self):
        legal = (
            {"direction": "desc"},
            {"limit": 5},
            {},
            {"enabled": True},
            {"threshold": 1.5},
            {"label": "x"},
            {"value": None},
        )
        for parameters in legal:
            with self.subTest(parameters=parameters):
                proposal = _parse(_frame(parameters))
                self.assertEqual(proposal.operation_slots[0].parameters, parameters)

    def test_scalar_null_and_array_outer_parameters_are_rejected(self):
        illegal = ("desc", 5, True, None, [])
        for parameters in illegal:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    _parse(_frame(parameters))

    def test_nested_object_and_array_parameter_values_are_rejected(self):
        illegal = ({"nested": {"x": 1}}, {"items": [1, 2]})
        for parameters in illegal:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    _parse(_frame(parameters))

    def test_mention_anchor_remains_exact(self):
        proposal = _parse(_frame({"direction": "desc"}))
        self.assertEqual(
            proposal.operation_slots[0].mention,
            "sorted by total descending",
        )
        invalid = _frame({"direction": "desc"})
        invalid["operation_slots"][0]["mention"] = "Sort customers"
        with self.assertRaises(ValueError):
            _parse(invalid)


class ParameterObjectServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_legal_object_keeps_schema_unbound_and_ambiguity_empty(self):
        client = _StaticClient(_frame({"direction": "desc"}))
        result = await process_observation_with_llm(
            RequirementGroundingRuntime(),
            _observation(),
            updater=LLMUpdater(client, _config()),
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.status, "processed")
        self.assertEqual(result.runtime.grounding_revision, 1)
        frame = result.runtime.grounding_state.requirement_frame
        self.assertTrue(
            all(
                slot.binding_type == "unknown" and slot.bound_identifier is None
                for slot in frame.schema_slots
            )
        )
        self.assertEqual(result.runtime.grounding_state.ambiguity_index, ())

    async def test_scalar_parameters_fail_open_without_frame_or_revision(self):
        for parameters in (True, 5, "desc"):
            with self.subTest(parameters=parameters):
                runtime = RequirementGroundingRuntime()
                client = _StaticClient(_frame(parameters))
                result = await process_observation_with_llm(
                    runtime,
                    _observation(),
                    updater=LLMUpdater(client, _config()),
                )
                self.assertEqual(client.calls, 1)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.runtime.grounding_revision, 0)
                self.assertEqual(
                    result.runtime.grounding_state,
                    runtime.grounding_state,
                )
                self.assertEqual(result.runtime.processed_observation_ids, ())


if __name__ == "__main__":
    unittest.main()
