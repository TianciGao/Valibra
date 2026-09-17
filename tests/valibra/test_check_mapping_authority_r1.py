from __future__ import annotations

import unittest

from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingCheckResponse,
    GroundingRuntime,
    SQLGroundingState,
    ValidationContext,
)
from valibra_agent.sql_grounding.observations import build_sql_grounding_observation
from valibra_agent.sql_grounding.service import process_sql_grounding_observation
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    CHECK_GROUNDING_PROMPT,
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_STAGE_PROMPT_SHA256,
    GroundingUpdaterResult,
)


QUERY = "What's the potential financial hit from our risky plants?"


class _FakeUpdater:
    def __init__(self, response: GroundingCheckResponse) -> None:
        self.response = response

    async def propose(self, *args: object, **kwargs: object) -> GroundingUpdaterResult:
        del args, kwargs
        return GroundingUpdaterResult(
            response=self.response,
            call_kind="check",
            telemetry=GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="1" * 64,
                response_sha256="2" * 64,
                prompt_sha256=SQL_GROUNDING_STAGE_PROMPT_SHA256["check"],
                form_schema_sha256=SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256["check"],
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


def _state(*mappings: ColumnMapping) -> SQLGroundingState:
    return SQLGroundingState(
        tables=("operational_metrics",),
        join_keys=(),
        column_mapping=mappings,
        domain_knowledge=(),
    )


async def _process(
    old: SQLGroundingState,
    response: GroundingCheckResponse,
):
    runtime = GroundingRuntime(
        grounding_revision=3,
        stage="INITIAL_GROUNDING",
        focus_dimension="none",
        grounding_state=old,
    )
    observation = build_sql_grounding_observation(
        task_id="check-mapping-authority-r1",
        phase=1,
        sequence=7,
        observation_type="metadata",
        content="TEXT. Revenue loss amount.",
        summary="Official column meaning observed",
        tool_name="get_column_meaning",
        function_call_id="check-mapping-authority-r1-call",
    )
    result = await process_sql_grounding_observation(
        runtime,
        observation,
        ValidationContext(
            current_query=QUERY,
            latest_observation_id=observation.observation_id,
            known_tables=frozenset({"operational_metrics"}),
            known_columns=frozenset(
                {
                    "operational_metrics.maintcost",
                    "operational_metrics.revloss",
                }
            ),
        ),
        _FakeUpdater(response),
        grounding_input={
            "query": QUERY,
            "current_state": old.model_dump(mode="json"),
            "previous_official_calls": [],
            "unresolved_mappings": [],
            "answered_clarifications": [],
            "latest_tool": {
                "name": "get_column_meaning",
                "arguments": {
                    "table_name": "operational_metrics",
                    "column_name": "revloss",
                },
                "result": "TEXT. Revenue loss amount.",
            },
        },
    )
    return runtime, result


class CheckMappingAuthorityR1Tests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_freezes_omission_and_forbids_new_phrase(self) -> None:
        required = (
            "Mapping omission contract",
            "唯一 owner 是 Mapping",
            "Check 不得新增、删除、修改 omission",
            "unresolved_mappings 非空时，本轮 Check 不得返回 complete",
            "本轮已经获得新的 actionable clarification / Official evidence",
            "交给 Final Regrounding Gate",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, CHECK_GROUNDING_PROMPT)

    async def test_column_meaning_cannot_materialize_omitted_phrase(self) -> None:
        old = _state()
        response = GroundingCheckResponse(
            status="complete",
            clarification_route="none",
            missing_information=None,
            next_tool=None,
            column_mapping=(
                ColumnMapping(
                    phrase="potential financial hit",
                    targets=("operational_metrics.revloss",),
                ),
            ),
            domain_knowledge=(),
        )
        runtime, result = await _process(old, response)

        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.state_update.error_type, "state_validation_failed")
        self.assertEqual(result.runtime, runtime)

    async def test_column_meaning_can_correct_one_existing_phrase(self) -> None:
        old = _state(
            ColumnMapping(
                phrase="potential financial hit",
                targets=("operational_metrics.maintcost",),
            )
        )
        response = GroundingCheckResponse(
            status="complete",
            clarification_route="none",
            missing_information=None,
            next_tool=None,
            column_mapping=(
                ColumnMapping(
                    phrase="potential financial hit",
                    targets=("operational_metrics.revloss",),
                ),
            ),
            domain_knowledge=(),
        )
        _, result = await _process(old, response)

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(
            result.runtime.grounding_state.column_mapping,
            response.column_mapping,
        )

    async def test_column_meaning_can_remove_one_invalid_existing_phrase(self) -> None:
        old = _state(
            ColumnMapping(
                phrase="potential financial hit",
                targets=("operational_metrics.revloss",),
            )
        )
        response = GroundingCheckResponse(
            status="incomplete",
            clarification_route="none",
            missing_information=(
                "The Official meaning does not directly define potential financial hit."
            ),
            next_tool=None,
            column_mapping=(),
            domain_knowledge=(),
        )
        _, result = await _process(old, response)

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(result.runtime.grounding_state.column_mapping, ())


if __name__ == "__main__":
    unittest.main()
