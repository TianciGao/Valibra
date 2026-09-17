from __future__ import annotations

import copy
import json
import unittest

from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
    CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SCHEMA,
    CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT,
    GroundingCheckResolvedLiteralProposalShadowResponse,
    validate_resolved_literal_proposal_shadow,
)
from valibra_agent.sql_grounding.updater import (
    CHECK_GROUNDING_PROMPT,
    SQL_GROUNDING_STAGE_FORM_SCHEMAS,
)


def _input(
    *,
    query: str,
    phrase: str,
    target: str,
    result: str,
    table: str,
    column: str,
) -> dict:
    return {
        "query": query,
        "current_state": {
            "tables": [table],
            "join_keys": [],
            "column_mapping": [{"phrase": phrase, "targets": [target]}],
            "domain_knowledge": [],
        },
        "unresolved_mappings": [],
        "previous_official_calls": [],
        "answered_clarifications": [],
        "latest_tool": {
            "name": "get_column_meaning",
            "arguments": {"table_name": table, "column_name": column},
            "result": result,
        },
    }


def _response(payload: dict, proposal: dict | None):
    return GroundingCheckResolvedLiteralProposalShadowResponse.model_validate(
        {
            "status": "complete",
            "clarification_route": "none",
            "missing_information": None,
            "next_tool": None,
            "column_mapping": payload["current_state"]["column_mapping"],
            "domain_knowledge": [],
            "resolved_literal_proposals": [] if proposal is None else [proposal],
        }
    )


def _validate(payload: dict, proposal: dict | None):
    before = copy.deepcopy(payload)
    result = validate_resolved_literal_proposal_shadow(
        _response(payload, proposal),
        grounding_input=payload,
        task_id="shadow-task",
        phase=1,
        grounding_revision=7,
    )
    if payload != before:
        raise AssertionError("Shadow validator mutated its input")
    return result


class ResolvedLiteralProposalShadowR1Tests(unittest.TestCase):
    def test_shadow_prompt_and_form_do_not_modify_authoritative_contract(self) -> None:
        self.assertNotIn("resolved_literal_proposals", CHECK_GROUNDING_PROMPT)
        self.assertNotIn(
            "resolved_literal_proposals",
            json.dumps(SQL_GROUNDING_STAGE_FORM_SCHEMAS["check"]),
        )
        self.assertIn(
            "resolved_literal_proposals",
            CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT,
        )
        self.assertIn(
            "resolved_literal_proposals",
            json.dumps(CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SCHEMA),
        )

    def test_solar9_fails_to_failed_is_valid(self) -> None:
        phrase = "electrical grounding fails"
        target = (
            "mechanical_condition.mech_health_snapshot -> "
            "'electrical_integrity' ->> 'grounding_status'"
        )
        payload = _input(
            query="When our electrical grounding fails, how much money do we lose?",
            phrase=phrase,
            target=target,
            table="mechanical_condition",
            column="mech_health_snapshot",
            result=json.dumps(
                {
                    "fields_meaning": {
                        "electrical_integrity": {
                            "grounding_status": (
                                "TEXT. Grounding status. Possible values: "
                                "Check Required, Failed, Normal."
                            )
                        }
                    }
                }
            ),
        )
        verdict = _validate(
            payload,
            {"phrase": phrase, "target": target, "literal": "Failed"},
        )
        self.assertEqual(verdict.verdict, "VALID")
        self.assertEqual(verdict.matching_enum_literals, ("Failed",))
        self.assertEqual(verdict.carrier.literal, "Failed")
        self.assertEqual(verdict.carrier.operator, "EXACT_EQUALITY")
        self.assertEqual(
            verdict.carrier.authority,
            "query_lexical_literal+official_column_meaning",
        )

    def test_cyber12_highest_to_high_is_valid(self) -> None:
        phrase = "highest priority level alert"
        target = "alerts.alert_case_management -> 'invest_priority_stat'"
        payload = _input(
            query="On average, how long does it take to close the highest priority level alert?",
            phrase=phrase,
            target=target,
            table="alerts",
            column="alert_case_management",
            result=json.dumps(
                {
                    "fields_meaning": {
                        "invest_priority_stat": (
                            "TEXT. Investigation priority. Possible values: "
                            "High, Low, Medium."
                        )
                    }
                }
            ),
        )
        verdict = _validate(
            payload,
            {"phrase": phrase, "target": target, "literal": "High"},
        )
        self.assertEqual(verdict.verdict, "VALID")
        self.assertEqual(verdict.matching_enum_literals, ("High",))

    def test_two_lexical_enum_matches_are_rejected(self) -> None:
        phrase = "active claimed warranty status"
        target = "plants.warrstate"
        payload = _input(
            query=f"Show plants with {phrase}",
            phrase=phrase,
            target=target,
            table="plants",
            column="warrstate",
            result="TEXT. Possible values: Active, Claimed, Expired.",
        )
        verdict = _validate(
            payload,
            {"phrase": phrase, "target": target, "literal": "Claimed"},
        )
        self.assertEqual(verdict.verdict, "REJECTED")
        self.assertEqual(verdict.reason, "query_enum_lexical_match_not_unique")
        self.assertEqual(verdict.matching_enum_literals, ("Active", "Claimed"))

    def test_semantic_synonym_without_lexical_match_is_rejected(self) -> None:
        phrase = "panels that break"
        target = "plant.condition ->> 'severity'"
        payload = _input(
            query=f"Find {phrase}",
            phrase=phrase,
            target=target,
            table="plant",
            column="condition",
            result=json.dumps(
                {
                    "fields_meaning": {
                        "severity": "TEXT. Possible values: Critical, Normal."
                    }
                }
            ),
        )
        verdict = _validate(
            payload,
            {"phrase": phrase, "target": target, "literal": "Critical"},
        )
        self.assertEqual(verdict.verdict, "REJECTED")
        self.assertEqual(verdict.reason, "query_enum_lexical_match_not_unique")
        self.assertEqual(verdict.matching_enum_literals, ())

    def test_wrong_target_is_rejected(self) -> None:
        payload = _input(
            query="Show failed jobs",
            phrase="failed jobs",
            target="jobs.status",
            table="jobs",
            column="status",
            result="TEXT. Possible values: Failed, Running.",
        )
        verdict = _validate(
            payload,
            {
                "phrase": "failed jobs",
                "target": "jobs.other_status",
                "literal": "Failed",
            },
        )
        self.assertEqual(verdict.verdict, "REJECTED")
        self.assertEqual(verdict.reason, "target_not_unique_exact_current_mapping")

    def test_stale_official_evidence_is_rejected(self) -> None:
        payload = _input(
            query="Show failed jobs",
            phrase="failed jobs",
            target="jobs.status",
            table="jobs",
            column="status",
            result="TEXT. Possible values: Failed, Running.",
        )
        payload["latest_tool"]["arguments"]["column_name"] = "old_status"
        verdict = _validate(
            payload,
            {
                "phrase": "failed jobs",
                "target": "jobs.status",
                "literal": "Failed",
            },
        )
        self.assertEqual(verdict.verdict, "REJECTED")
        self.assertEqual(verdict.reason, "official_evidence_target_mismatch")

    def test_literal_outside_official_enum_is_rejected(self) -> None:
        payload = _input(
            query="Show failed jobs",
            phrase="failed jobs",
            target="jobs.status",
            table="jobs",
            column="status",
            result="TEXT. Possible values: Failed, Running.",
        )
        verdict = _validate(
            payload,
            {
                "phrase": "failed jobs",
                "target": "jobs.status",
                "literal": "Failure",
            },
        )
        self.assertEqual(verdict.verdict, "REJECTED")
        self.assertEqual(verdict.reason, "literal_not_exact_official_enum_member")

    def test_negative_phrase_is_not_positive_equality_authority(self) -> None:
        payload = _input(
            query="Show jobs not failed",
            phrase="not failed",
            target="jobs.status",
            table="jobs",
            column="status",
            result="TEXT. Possible values: Failed, Running.",
        )
        verdict = _validate(
            payload,
            {
                "phrase": "not failed",
                "target": "jobs.status",
                "literal": "Failed",
            },
        )
        self.assertEqual(verdict.verdict, "REJECTED")
        self.assertEqual(verdict.reason, "negative_or_exclusion_phrase_not_supported")

    def test_healthy_check_may_explicitly_emit_no_proposal(self) -> None:
        payload = _input(
            query="Show job names",
            phrase="job names",
            target="jobs.name",
            table="jobs",
            column="name",
            result="TEXT. Human-readable job name.",
        )
        verdict = _validate(payload, None)
        self.assertEqual(verdict.verdict, "OMITTED")
        self.assertIsNone(verdict.carrier)

    def test_provider_cannot_self_assert_authority(self) -> None:
        payload = _input(
            query="Show failed jobs",
            phrase="failed jobs",
            target="jobs.status",
            table="jobs",
            column="status",
            result="TEXT. Possible values: Failed, Running.",
        )
        with self.assertRaisesRegex(Exception, "extra"):
            _response(
                payload,
                {
                    "phrase": "failed jobs",
                    "target": "jobs.status",
                    "literal": "Failed",
                    "authority": "official_metadata",
                },
            )


if __name__ == "__main__":
    unittest.main()
