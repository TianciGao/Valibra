from __future__ import annotations

import unittest

from valibra_agent.sql_grounding.models import (
    GroundingCheckResponse,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    SQLGroundingValidationError,
)
from valibra_agent.sql_grounding.service import (
    _validate_check_route_and_repair_scope,
)
from valibra_agent.sql_grounding.updater import classify_grounding_input


def _runtime() -> GroundingRuntime:
    return GroundingRuntime(
        grounding_revision=3,
        focus_dimension="none",
        grounding_state=SQLGroundingState(
            tables=("metrics",),
            join_keys=(),
            column_mapping=(),
            domain_knowledge=(),
        ),
    )


def _materialized(runtime: GroundingRuntime) -> GroundingLLMResponse:
    return GroundingLLMResponse(
        sql_grounding_state=runtime.grounding_state,
        user_clarification_requests=(),
        next_focus_dimension="none",
    )


def _check(*, status: str) -> GroundingCheckResponse:
    return GroundingCheckResponse(
        status=status,
        clarification_route="none",
        missing_information=(
            None if status == "complete" else "risk score remains unresolved"
        ),
        next_tool=None,
        column_mapping=(),
        domain_knowledge=(),
    )


def _input(*, unresolved: bool) -> dict:
    return {
        "query": "Show the risk score.",
        "current_state": {
            "tables": ["metrics"],
            "join_keys": [],
            "column_mapping": [],
            "domain_knowledge": [],
        },
        "unresolved_mappings": (
            [
                {
                    "phrase": "risk score",
                    "reason": "derived_rule_required",
                }
            ]
            if unresolved
            else []
        ),
        "answered_clarifications": [],
        "previous_official_calls": [],
        "check_context": {"kind": "initial"},
    }


class MappingOmissionCheckO3Tests(unittest.TestCase):
    def test_proven_crypto_formula_and_equivalence_guard_is_inherited(self) -> None:
        from valibra_agent.sql_grounding.updater import CHECK_GROUNDING_PROMPT

        required = (
            "1.1 Exact Official formula operand completeness",
            "exact formula 明确要求的每个必要 operand",
            "没有上述 explicit Official proof 时不得自行推断 equivalence",
            "不得继续为旧 stored target 寻找语义合理化",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)

    def test_check_input_requires_read_only_omission_projection(self) -> None:
        payload = _input(unresolved=True)
        self.assertEqual(classify_grounding_input(payload, phase=1), "check")

        payload.pop("unresolved_mappings")
        with self.assertRaisesRegex(ValueError, "invalid stage-local field set"):
            classify_grounding_input(payload, phase=1)

    def test_complete_is_rejected_when_omission_is_nonempty(self) -> None:
        runtime = _runtime()
        with self.assertRaisesRegex(
            SQLGroundingValidationError,
            "cannot complete while unresolved_mappings is non-empty",
        ):
            _validate_check_route_and_repair_scope(
                runtime,
                _materialized(runtime),
                changed=(),
                grounding_input=_input(unresolved=True),
                check=_check(status="complete"),
            )

    def test_incomplete_is_not_auto_routed_when_omission_is_nonempty(self) -> None:
        runtime = _runtime()
        _validate_check_route_and_repair_scope(
            runtime,
            _materialized(runtime),
            changed=(),
            grounding_input=_input(unresolved=True),
            check=_check(status="incomplete"),
        )

    def test_empty_omission_does_not_block_healthy_complete(self) -> None:
        runtime = _runtime()
        _validate_check_route_and_repair_scope(
            runtime,
            _materialized(runtime),
            changed=(),
            grounding_input=_input(unresolved=False),
            check=_check(status="complete"),
        )


if __name__ == "__main__":
    unittest.main()
