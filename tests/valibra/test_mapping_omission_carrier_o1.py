from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingRuntime,
    MappingGroundingResponse,
    SQLGroundingState,
    SQLGroundingValidationError,
    UnresolvedMapping,
    ValidationContext,
)
from valibra_agent.sql_grounding.service import (
    _validate_mapping_omission_phrases,
)


class MappingOmissionFormO1Tests(unittest.TestCase):
    def test_form_requires_explicit_mapping_partition(self) -> None:
        with self.assertRaises(ValidationError):
            MappingGroundingResponse.model_validate(
                {
                    "tables": ["metrics"],
                    "join_keys": [],
                    "column_mapping": [],
                }
            )

    def test_form_rejects_resolved_unresolved_overlap(self) -> None:
        with self.assertRaises(ValidationError):
            MappingGroundingResponse(
                tables=("metrics",),
                join_keys=(),
                column_mapping=(
                    ColumnMapping(
                        phrase="risk score",
                        targets=("metrics.risk_score",),
                    ),
                ),
                unresolved_mappings=(
                    UnresolvedMapping(
                        phrase="risk score",
                        reason="no_direct_metadata",
                    ),
                ),
            )

    def test_reason_is_closed_and_exact(self) -> None:
        with self.assertRaises(ValidationError):
            UnresolvedMapping.model_validate(
                {"phrase": "risk score", "reason": "other"}
            )

    def test_omission_phrase_must_be_verbatim_query_text(self) -> None:
        response = MappingGroundingResponse(
            tables=("metrics",),
            join_keys=(),
            column_mapping=(),
            unresolved_mappings=(
                UnresolvedMapping(
                    phrase="rewritten risk score",
                    reason="no_direct_metadata",
                ),
            ),
        )
        with self.assertRaisesRegex(
            SQLGroundingValidationError,
            "verbatim query substring",
        ):
            _validate_mapping_omission_phrases(
                response,
                ValidationContext(
                    current_query="Show the risk score.",
                    latest_observation_id="o1-verbatim",
                ),
            )

    def test_sidecar_is_projected_read_only_through_mapping_knowledge_and_check(
        self,
    ) -> None:
        query = "Show the risk score."
        runtime = GroundingRuntime(
            grounding_revision=2,
            focus_dimension="none",
            grounding_state=SQLGroundingState(
                tables=("metrics",),
                join_keys=(),
                column_mapping=(),
                domain_knowledge=(),
            ),
        )
        state = {
            "tool_trajectory": [
                {
                    "type": "tool",
                    "tool": "get_schema",
                    "phase": 1,
                    "args": {},
                    "result": (
                        "CREATE TABLE metrics (\n"
                        "risk_score NUMERIC\n"
                        ");"
                    ),
                    "cost": 0.0,
                    "budget_before": None,
                    "budget_after": None,
                    "action_input_tokens": 0,
                    "action_output_tokens": 0,
                    "timestamp": "2026-09-02T00:00:00Z",
                },
                {
                    "type": "tool",
                    "tool": "get_all_column_meanings",
                    "phase": 1,
                    "args": {},
                    "result": json.dumps(
                        {"o1|metrics|risk_score": "Stored risk score."}
                    ),
                    "cost": 0.0,
                    "budget_before": None,
                    "budget_after": None,
                    "action_input_tokens": 0,
                    "action_output_tokens": 0,
                    "timestamp": "2026-09-02T00:00:01Z",
                },
                {
                    "type": "tool",
                    "tool": "get_all_knowledge_definitions",
                    "phase": 1,
                    "args": {},
                    "result": "[]",
                    "cost": 0.0,
                    "budget_before": None,
                    "budget_after": None,
                    "action_input_tokens": 0,
                    "action_output_tokens": 0,
                    "timestamp": "2026-09-02T00:00:02Z",
                },
            ]
        }
        carrier = grounding_callbacks._MappingOmissionCarrier(
            phase=1,
            grounding_revision=2,
            query_sha256=grounding_callbacks._mapping_omission_query_sha256(
                query, None
            ),
            mapping_evidence_sha256="a" * 64,
            unresolved_mappings=(
                UnresolvedMapping(
                    phrase="risk score",
                    reason="derived_rule_required",
                ),
            ),
        )
        grounding_callbacks._store_mapping_omission_carrier(state, carrier)
        mapping_input = grounding_callbacks._build_staged_grounding_request(
            state,
            call_kind="mapping",
            query=query,
            runtime=runtime,
            phase=1,
        )
        knowledge_input = grounding_callbacks._build_staged_grounding_request(
            state,
            call_kind="knowledge",
            query=query,
            runtime=runtime,
            phase=1,
        )
        check_input = grounding_callbacks._build_check_grounding_request(
            state,
            query=query,
            runtime=runtime,
            phase=1,
            initial=True,
        )
        expected = [
            {
                "phrase": "risk score",
                "reason": "derived_rule_required",
            }
        ]
        self.assertEqual(
            mapping_input["unresolved_mappings"],
            expected,
        )
        self.assertEqual(
            knowledge_input["unresolved_mappings"],
            expected,
        )
        self.assertEqual(check_input["unresolved_mappings"], expected)


class MappingOmissionTransitionO1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = grounding_callbacks._MappingOmissionCarrier(
            phase=1,
            grounding_revision=4,
            query_sha256="a" * 64,
            mapping_evidence_sha256="b" * 64,
            unresolved_mappings=(
                UnresolvedMapping(
                    phrase="risk score",
                    reason="derived_rule_required",
                ),
            ),
        )

    def _resolved_response(self) -> MappingGroundingResponse:
        return MappingGroundingResponse(
            tables=("metrics",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="risk score",
                    targets=("metrics.risk_score",),
                ),
            ),
            unresolved_mappings=(),
        )

    def test_same_evidence_cannot_clear_omission(self) -> None:
        with self.assertRaisesRegex(ValueError, "new actionable evidence"):
            grounding_callbacks._validate_mapping_omission_transition(
                previous=self.previous,
                response=self._resolved_response(),
                mapping_evidence_sha256="b" * 64,
                legal_regrounding=True,
            )

    def test_new_evidence_legal_reground_can_clear_exact_phrase(self) -> None:
        grounding_callbacks._validate_mapping_omission_transition(
            previous=self.previous,
            response=self._resolved_response(),
            mapping_evidence_sha256="c" * 64,
            legal_regrounding=True,
        )

    def test_non_mapping_cannot_fill_omission(self) -> None:
        runtime = type(
            "Runtime",
            (),
            {
                "grounding_state": type(
                    "State",
                    (),
                    {
                        "column_mapping": self._resolved_response().column_mapping
                    },
                )()
            },
        )()
        with self.assertRaisesRegex(ValueError, "only Mapping"):
            grounding_callbacks._validate_non_mapping_does_not_fill_omissions(
                self.previous,
                runtime,
            )


if __name__ == "__main__":
    unittest.main()
