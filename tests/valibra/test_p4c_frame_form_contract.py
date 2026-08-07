import hashlib
import inspect
import json
import math
import unittest
from pathlib import Path

from pydantic import ValidationError

from valibra_agent.requirement_grounding import updater as updater_module
from valibra_agent.requirement_grounding.models import RequirementGroundingRuntime
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.service import process_observation_with_llm
from valibra_agent.requirement_grounding.updater import (
    LLMUpdater,
    LLM_FRAME_FORM_SCHEMA,
    LLM_FRAME_FORM_SCHEMA_JSON,
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT,
    LLM_FRAME_PROMPT_SHA256,
    load_grounding_llm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Show the top 5 customer names from 2024 sorted by total descending."


def _environment():
    return {
        "GROUNDING_UPDATER_MODE": "llm",
        "GROUNDING_MODEL_PRESET": "glm52_high_32768",
        "GROUNDING_TIMEOUT_SECONDS": "30",
        "GROUNDING_MAX_TOKENS": "32768",
        "GROUNDING_MAX_CALLS_PER_TASK": "2",
        "GROUNDING_PROMPT_SHA256": LLM_FRAME_PROMPT_SHA256,
    }


def _config():
    return load_grounding_llm_config(PROJECT_ROOT, _environment())


def _observation():
    return build_observation(
        task_id="p4c-fixed-form-offline",
        observation_type="user_query",
        phase=1,
        sequence=1,
        source="synthetic_offline_test",
        raw=QUESTION,
        summary=QUESTION,
    )


def _frame(*, operation_type="order", mention="sorted by total descending"):
    return {
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
                "interpretation": "candidate customer-name concept",
            }
        ],
        "operation_slots": [
            {
                "slot_role": "ordering",
                "mention": mention,
                "interpretation": "order by total descending",
                "operation_type": operation_type,
                "parameters": {"direction": "desc"},
            }
        ],
        "ambiguities": [],
    }


def _parse(value):
    return updater_module._parse_llm_frame_response(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        observation_text=QUESTION,
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


class FixedFormSchemaTests(unittest.TestCase):
    def test_installed_litellm_async_interface_exposes_response_format(self):
        import litellm

        parameters = inspect.signature(litellm.acompletion).parameters
        self.assertIn("response_format", parameters)

    def test_schema_and_hash_are_generated_stably_from_strict_models(self):
        self.assertEqual(
            LLM_FRAME_FORM_SCHEMA_JSON,
            updater_module.preset_canonical_json(LLM_FRAME_FORM_SCHEMA),
        )
        self.assertEqual(
            LLM_FRAME_FORM_SCHEMA_SHA256,
            hashlib.sha256(LLM_FRAME_FORM_SCHEMA_JSON.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(LLM_FRAME_FORM_SCHEMA, updater_module.LLMFrameProposal.model_json_schema())
        self.assertFalse(LLM_FRAME_FORM_SCHEMA["additionalProperties"])
        self.assertEqual(
            LLM_FRAME_FORM_SCHEMA["required"],
            ["value_slots", "schema_slots", "operation_slots", "ambiguities"],
        )

    def test_schema_fixes_all_fields_enum_scalars_and_empty_ambiguity(self):
        definitions = LLM_FRAME_FORM_SCHEMA["$defs"]
        value_form = definitions["LLMValueSlotProposal"]
        schema_form = definitions["LLMSchemaSlotProposal"]
        operation_form = definitions["LLMOperationSlotProposal"]
        self.assertEqual(
            value_form["required"],
            ["slot_role", "mention", "interpretation", "value_type"],
        )
        self.assertEqual(
            schema_form["required"],
            ["slot_role", "mention", "interpretation"],
        )
        self.assertEqual(
            operation_form["required"],
            [
                "slot_role",
                "mention",
                "interpretation",
                "operation_type",
                "parameters",
            ],
        )
        self.assertFalse(value_form["additionalProperties"])
        self.assertFalse(schema_form["additionalProperties"])
        self.assertFalse(operation_form["additionalProperties"])
        self.assertEqual(
            operation_form["properties"]["operation_type"]["enum"],
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
        scalar_schema = operation_form["properties"]["parameters"][
            "additionalProperties"
        ]["anyOf"]
        self.assertEqual(
            {item["type"] for item in scalar_schema},
            {"string", "integer", "number", "boolean", "null"},
        )
        self.assertEqual(
            LLM_FRAME_FORM_SCHEMA["properties"]["ambiguities"]["maxItems"],
            0,
        )

    def test_prompt_and_configuration_freeze_new_form_contract(self):
        self.assertEqual(
            LLM_FRAME_PROMPT_SHA256,
            hashlib.sha256(LLM_FRAME_PROMPT.encode("utf-8")).hexdigest(),
        )
        for required_text in (
            "filling a form",
            "projection",
            "other",
            "copied verbatim",
            "case-sensitive substring",
            "ambiguities must always be an empty array",
            "a real table, column, database identifier",
        ):
            self.assertIn(required_text, LLM_FRAME_PROMPT)
        config = _config()
        self.assertEqual(config.form_schema_sha256, LLM_FRAME_FORM_SCHEMA_SHA256)
        self.assertEqual(config.prompt_sha256, LLM_FRAME_PROMPT_SHA256)

    def test_source_has_no_llm_enum_alias_normalizer_or_synonym_table(self):
        source = inspect.getsource(updater_module)
        self.assertNotIn("_normalize_llm_frame_enum_aliases", source)
        self.assertNotIn("order_by", source)
        self.assertNotIn("sorting", source)


class FixedFormValidationTests(unittest.TestCase):
    def test_legal_fixed_form_and_exact_order_pass(self):
        proposal = _parse(_frame())
        self.assertEqual(proposal.operation_slots[0].operation_type, "order")
        self.assertEqual(
            proposal.operation_slots[0].mention,
            "sorted by total descending",
        )

    def test_operation_aliases_and_unknown_values_are_rejected(self):
        for operation_type in ("order_by", "sort", "sorting", "ORDER", "rank"):
            with self.subTest(operation_type=operation_type):
                with self.assertRaises(ValidationError):
                    _parse(_frame(operation_type=operation_type))

    def test_extra_and_missing_fields_are_rejected(self):
        extra_root = _frame()
        extra_root["extra"] = True
        extra_slot = _frame()
        extra_slot["schema_slots"][0]["bound_identifier"] = "forbidden.table"
        missing_top = _frame()
        del missing_top["ambiguities"]
        missing_value_field = _frame()
        del missing_value_field["value_slots"][0]["value_type"]
        missing_operation_field = _frame()
        del missing_operation_field["operation_slots"][0]["parameters"]
        for value in (
            extra_root,
            extra_slot,
            missing_top,
            missing_value_field,
            missing_operation_field,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    _parse(value)

    def test_nested_parameter_objects_and_arrays_are_rejected(self):
        for nested in ({"direction": "desc"}, ["desc"]):
            value = _frame()
            value["operation_slots"][0]["parameters"] = {"nested": nested}
            with self.subTest(nested=nested):
                with self.assertRaises(ValidationError):
                    _parse(value)

    def test_non_finite_parameter_is_rejected(self):
        value = _frame()
        value["operation_slots"][0]["parameters"] = {"score": math.inf}
        with self.assertRaises((ValidationError, ValueError)):
            _parse(value)

    def test_empty_or_rewritten_mentions_reject_the_entire_form(self):
        for mention in ("", "   ", "rank customers", "Sorted by total descending"):
            with self.subTest(mention=mention):
                with self.assertRaises((ValidationError, ValueError)):
                    _parse(_frame(mention=mention))

    def test_verbatim_anchor_is_case_and_space_sensitive(self):
        proposal = _parse(_frame(mention="top 5 customer names"))
        self.assertEqual(proposal.operation_slots[0].mention, "top 5 customer names")
        for mention in ("Top 5 customer names", "top  5 customer names"):
            with self.subTest(mention=mention):
                with self.assertRaises(ValueError):
                    _parse(_frame(mention=mention))

    def test_nonempty_ambiguity_is_rejected(self):
        value = _frame()
        value["ambiguities"] = [{"candidate": "x"}]
        with self.assertRaises(ValidationError):
            _parse(value)


class FixedFormServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_order_form_creates_atomic_unbound_patch(self):
        client = _StaticClient(_frame())
        result = await process_observation_with_llm(
            RequirementGroundingRuntime(),
            _observation(),
            updater=LLMUpdater(client, _config()),
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.status, "processed")
        self.assertEqual(result.runtime.grounding_revision, 1)
        frame = result.runtime.grounding_state.requirement_frame
        self.assertEqual(frame.operation_slots[0].operation_type, "order")
        self.assertTrue(
            all(
                slot.grounding_status == "hypothesized"
                for slot in (
                    *frame.value_slots,
                    *frame.schema_slots,
                    *frame.operation_slots,
                )
            )
        )
        self.assertTrue(
            all(
                slot.binding_type == "unknown" and slot.bound_identifier is None
                for slot in frame.schema_slots
            )
        )
        self.assertEqual(result.runtime.grounding_state.ambiguity_index, ())

    async def test_invalid_form_fails_open_without_business_state_change(self):
        for operation_type in ("order_by", "sort"):
            with self.subTest(operation_type=operation_type):
                runtime = RequirementGroundingRuntime()
                result = await process_observation_with_llm(
                    runtime,
                    _observation(),
                    updater=LLMUpdater(
                        _StaticClient(_frame(operation_type=operation_type)),
                        _config(),
                    ),
                )
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.runtime.grounding_revision, 0)
                self.assertEqual(result.runtime.grounding_state, runtime.grounding_state)
                self.assertEqual(result.runtime.processed_observation_ids, ())


class OfflineSmokeCheckerTests(unittest.TestCase):
    def test_checker_uses_correct_ambiguity_index_and_has_no_provider_path(self):
        source = (PROJECT_ROOT / "scripts/check_grounding_frame_smoke_offline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("grounding_state.ambiguity_index", source)
        self.assertNotIn("grounding_state.ambiguities", source)
        self.assertNotIn("litellm", source.lower())
        self.assertNotIn("acompletion", source.lower())
        self.assertIn('"provider_requests": 0', source)


if __name__ == "__main__":
    unittest.main()
