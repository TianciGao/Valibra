"""Offline acceptance tests for the narrow P7.1d JSON transport boundary.

Only a single outer Markdown fence labelled ``json`` (case-insensitive), or
an otherwise unlabelled fence, may be removed.  Everything inside that
transport wrapper remains subject to the existing strict JSON, duplicate-key,
Pydantic, mention, and atomic reducer boundaries.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from valibra_agent.requirement_grounding import updater as updater_module
from valibra_agent.requirement_grounding.models import (
    SCHEMA_VERSION,
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.service import process_observation_with_llm
from valibra_agent.requirement_grounding.updater import (
    LLM_FRAME_FORM_SCHEMA_SHA256,
    LLM_FRAME_PROMPT_SHA256,
    LLMFrameUpdateError,
    LLMUpdater,
    load_grounding_llm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Show orders from 2024."
EXPECTED_PROMPT_SHA256 = (
    "ddaaa23fd3c824a704ea1e17a769b4c8b1af5c998949fccfe8d564b18f78f7c4"
)
EXPECTED_FORM_SHA256 = (
    "f7409a7267b3fddcb40d69574e6187320884951ad4e96980bb590ee0058c38d1"
)
EXPECTED_CONFIG_SHA256 = (
    "2ec2accb786a1f1e4d35027affe52c0402957861f59a93832583ac4094066dce"
)


def _proposal(*, mention: str = "2024") -> dict[str, object]:
    return {
        "proposal_outcome": "populated",
        "value_slots": [
            {
                "slot_role": "time_constraint",
                "mention": mention,
                "interpretation": "calendar year 2024",
                "value_type": "time",
            }
        ],
        "schema_slots": [],
        "operation_slots": [],
        "ambiguities": [],
    }


def _canonical_json(value: dict[str, object] | None = None) -> str:
    return json.dumps(
        _proposal() if value is None else value,
        sort_keys=True,
        separators=(",", ":"),
    )


VALID_JSON = _canonical_json()


def _fence(content: str, language: str = "json") -> str:
    return f"```{language}\n{content}\n```"


def _parse(content: str):
    return updater_module._parse_llm_frame_response(
        content,
        observation_text=QUESTION,
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


def _observation():
    return build_observation(
        task_id="task-p71d-transport",
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

    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return {"content": self.content, "usage": {}}


class TransportParsingTests(unittest.TestCase):
    def test_naked_json_and_each_allowed_single_fence_pass(self):
        quoted_backticks = _proposal()
        quoted_backticks["value_slots"][0]["interpretation"] = "literal ``` marker"
        cases = {
            "naked": VALID_JSON,
            "naked_quoted_backticks": _canonical_json(quoted_backticks),
            "unlabelled": _fence(VALID_JSON, ""),
            "json_lower": _fence(VALID_JSON, "json"),
            "json_upper": _fence(VALID_JSON, "JSON"),
            "json_title": _fence(VALID_JSON, "Json"),
            "json_mixed": _fence(VALID_JSON, "jSoN"),
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                parsed = _parse(content)
                self.assertEqual(parsed.proposal_outcome, "populated")
                if name != "naked_quoted_backticks":
                    self.assertEqual(
                        parsed.model_dump(mode="json"),
                        _parse(VALID_JSON).model_dump(mode="json"),
                    )

    def test_invalid_fence_wrappers_are_transport_format_invalid(self):
        cases = {
            "leading_prose": f"Here is the form:\n{_fence(VALID_JSON)}",
            "leading_prose_unclosed_quote": (
                f'Explanation "unterminated\n{_fence(VALID_JSON)}'
            ),
            "trailing_prose": f"{_fence(VALID_JSON)}\nThat is the form.",
            "two_blocks": f"{_fence(VALID_JSON)}\n{_fence(VALID_JSON)}",
            "unclosed": f"```json\n{VALID_JSON}",
            "python": _fence(VALID_JSON, "python"),
            "javascript": _fence(VALID_JSON, "javascript"),
            "text": _fence(VALID_JSON, "text"),
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(LLMFrameUpdateError) as raised:
                    _parse(content)
                self.assertEqual(raised.exception.reason, "transport_format_invalid")

    def test_valid_fence_with_bad_json_is_json_invalid(self):
        for name, content in {
            "malformed": _fence("{"),
            "non_finite": _fence(
                '{"proposal_outcome":"populated","value_slots":[],'
                '"schema_slots":[],"operation_slots":['
                '{"slot_role":"filter","mention":"2024",'
                '"interpretation":"year","operation_type":"filter",'
                '"parameters":{"value":NaN}}],"ambiguities":[]}'
            ),
        }.items():
            with self.subTest(name=name):
                with self.assertRaises(LLMFrameUpdateError) as raised:
                    _parse(content)
                self.assertEqual(raised.exception.reason, "json_invalid")

    def test_valid_fence_with_duplicate_key_is_duplicate_json_key(self):
        duplicate = (
            '{"proposal_outcome":"populated",'
            '"proposal_outcome":"populated",'
            '"value_slots":[],"schema_slots":[],"operation_slots":[],'
            '"ambiguities":[]}'
        )
        with self.assertRaises(LLMFrameUpdateError) as raised:
            _parse(_fence(duplicate))
        self.assertEqual(raised.exception.reason, "duplicate_json_key")

    def test_malformed_naked_json_with_quoted_backticks_stays_json_invalid(self):
        malformed = '{"proposal_outcome":"populated","note":"```",}'
        with self.assertRaises(LLMFrameUpdateError) as raised:
            _parse(malformed)
        self.assertEqual(raised.exception.reason, "json_invalid")


class TransportServiceTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, content: str):
        original = RequirementGroundingRuntime()
        client = _StaticClient(content)
        result = await process_observation_with_llm(
            original,
            _observation(),
            updater=LLMUpdater(client, _config()),
        )
        self.assertEqual(client.calls, 1)
        return original, result

    async def test_normalization_does_not_change_business_revisions_or_patch(self):
        _, naked = await self._run(VALID_JSON)
        for language in ("", "json", "JSON"):
            with self.subTest(language=language or "unlabelled"):
                _, fenced = await self._run(_fence(VALID_JSON, language))
                self.assertEqual(fenced.status, "processed")
                self.assertEqual(fenced.runtime.grounding_revision, 1)
                self.assertEqual(fenced.runtime.requirement_revision, 1)
                self.assertEqual(
                    fenced.runtime.grounding_state,
                    naked.runtime.grounding_state,
                )
                self.assertEqual(
                    fenced.runtime.processed_observation_ids,
                    naked.runtime.processed_observation_ids,
                )
                self.assertNotIn(
                    "transport_normalization",
                    fenced.runtime.model_dump(mode="json"),
                )
        self.assertEqual(naked.status, "processed")
        self.assertEqual(naked.runtime.grounding_revision, 1)
        self.assertEqual(naked.runtime.requirement_revision, 1)

    async def test_each_failure_class_fails_open_without_revision_or_frame(self):
        duplicate = (
            '{"proposal_outcome":"populated",'
            '"value_slots":[],"value_slots":[],"schema_slots":[],'
            '"operation_slots":[],"ambiguities":[]}'
        )
        form_invalid = _proposal()
        form_invalid["unexpected"] = True
        scenarios = {
            "transport_format_invalid": f"prose\n{_fence(VALID_JSON)}",
            "json_invalid": _fence("{"),
            "duplicate_json_key": _fence(duplicate),
            "form_validation_failed": _fence(_canonical_json(form_invalid)),
            "mention_validation_failed": _fence(
                _canonical_json(_proposal(mention="2099"))
            ),
        }
        for expected_reason, content in scenarios.items():
            with self.subTest(expected_reason=expected_reason):
                original, result = await self._run(content)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.failure_reason, expected_reason)
                self.assertEqual(result.runtime.grounding_revision, 0)
                self.assertEqual(result.runtime.requirement_revision, 0)
                self.assertEqual(
                    result.runtime.grounding_state,
                    original.grounding_state,
                )
                self.assertEqual(result.runtime.processed_observation_ids, ())


class FrozenContractTests(unittest.TestCase):
    def test_p71c_schema_prompt_form_and_configuration_stay_frozen(self):
        self.assertEqual(SCHEMA_VERSION, "1.1")
        self.assertEqual(LLM_FRAME_PROMPT_SHA256, EXPECTED_PROMPT_SHA256)
        self.assertEqual(LLM_FRAME_FORM_SCHEMA_SHA256, EXPECTED_FORM_SHA256)
        self.assertEqual(_config().configuration_sha256, EXPECTED_CONFIG_SHA256)


if __name__ == "__main__":
    unittest.main()
