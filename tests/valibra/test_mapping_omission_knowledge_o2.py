from __future__ import annotations

import unittest

from pydantic import ValidationError

from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    SQLGroundingState,
    ValidationContext,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.service import (
    process_sql_grounding_observation,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    GroundingUpdaterResult,
    _strip_knowledge_omission_owned_mappings,
    classify_grounding_input,
)


class _FakeKnowledgeUpdater:
    def __init__(self, response: KnowledgeGroundingResponse) -> None:
        self._response = response

    async def propose(self, *args: object, **kwargs: object) -> GroundingUpdaterResult:
        del args, kwargs
        digest = "0" * 64
        return GroundingUpdaterResult(
            response=self._response,
            call_kind="knowledge",
            telemetry=GroundingLLMTelemetry(
                attempted=True,
                status="succeeded",
                request_sha256="",
                response_sha256="",
                prompt_sha256=digest,
                form_schema_sha256=digest,
                configuration_sha256=digest,
            ),
            transport_normalization="none",
        )


class MappingOmissionKnowledgeO2Tests(unittest.TestCase):
    def test_knowledge_input_requires_read_only_omission_projection(self) -> None:
        payload = {
            "query": "Show the risk score.",
            "current_state": {
                "tables": ["metrics"],
                "join_keys": [],
                "column_mapping": [],
                "domain_knowledge": [],
            },
            "knowledge_definitions": [],
            "relevant_column_meanings": {},
            "unresolved_mappings": [
                {
                    "phrase": "risk score",
                    "reason": "derived_rule_required",
                }
            ],
        }
        self.assertEqual(classify_grounding_input(payload, phase=1), "knowledge")

        payload.pop("unresolved_mappings")
        with self.assertRaisesRegex(ValueError, "invalid stage-local field set"):
            classify_grounding_input(payload, phase=1)

    def test_knowledge_response_has_no_omission_mutation_channel(self) -> None:
        with self.assertRaises(ValidationError):
            KnowledgeGroundingResponse.model_validate(
                {
                    "column_mapping": [],
                    "selected_knowledge_ids": [],
                    "unresolved_mappings": [],
                }
            )

    def test_selected_empty_remains_a_valid_knowledge_noop(self) -> None:
        response = KnowledgeGroundingResponse.model_validate(
            {
                "column_mapping": [],
                "selected_knowledge_ids": [],
            }
        )
        self.assertEqual(response.column_mapping, ())
        self.assertEqual(response.selected_knowledge_ids, ())

    def test_transport_drops_exact_omission_even_when_targets_are_empty(self) -> None:
        payload, ignored = _strip_knowledge_omission_owned_mappings(
            """{"column_mapping":[
                {"phrase":"asset","targets":["metrics.asset_id"]},
                {"phrase":"risk score","targets":[]}
            ],"selected_knowledge_ids":[7]}""",
            grounding_input={
                "unresolved_mappings": [
                    {
                        "phrase": "risk score",
                        "reason": "derived_rule_required",
                    }
                ]
            },
        )
        response = KnowledgeGroundingResponse.model_validate_json(payload)
        self.assertEqual(ignored, ("risk score",))
        self.assertEqual(
            response.column_mapping,
            (ColumnMapping(phrase="asset", targets=("metrics.asset_id",)),),
        )
        self.assertEqual(response.selected_knowledge_ids, (7,))


class KnowledgeOmissionOwnershipRuntimeTests(unittest.IsolatedAsyncioTestCase):
    query = "Show each asset's risk score."
    rule = "Risk score is calculated from risk_a and risk_b."

    def _runtime(self) -> GroundingRuntime:
        return GroundingRuntime(
            grounding_revision=2,
            stage="INITIAL_GROUNDING",
            focus_dimension="domain_knowledge",
            grounding_state=SQLGroundingState(
                tables=("metrics",),
                join_keys=(),
                column_mapping=(
                    ColumnMapping(
                        phrase="asset",
                        targets=("metrics.asset_id",),
                    ),
                ),
                domain_knowledge=None,
            ),
        )

    def _bundle(self, runtime: GroundingRuntime) -> dict[str, object]:
        return {
            "query": self.query,
            "current_state": runtime.grounding_state.model_dump(mode="json"),
            "knowledge_definitions": [
                {"id": 7, "definition": self.rule},
            ],
            "relevant_column_meanings": {
                "metrics": {
                    "asset_id": "Canonical asset identifier.",
                    "risk_a": "First risk operand.",
                    "risk_b": "Second risk operand.",
                    "risk_score": "Stored risk metric.",
                }
            },
            "unresolved_mappings": [
                {
                    "phrase": "risk score",
                    "reason": "derived_rule_required",
                }
            ],
        }

    def _context(self, observation_id: str) -> ValidationContext:
        return ValidationContext(
            current_query=self.query,
            latest_observation_id=observation_id,
            known_tables=frozenset({"metrics"}),
            known_columns=frozenset(
                {
                    "metrics.asset_id",
                    "metrics.canonical_asset_id",
                    "metrics.risk_a",
                    "metrics.risk_b",
                    "metrics.risk_score",
                }
            ),
            supported_domain_knowledge=frozenset(
                {("business_rule", self.rule)}
            ),
        )

    async def _run(
        self,
        response: KnowledgeGroundingResponse,
    ):
        runtime = self._runtime()
        observation = build_sql_grounding_observation(
            task_id="knowledge-omission-owner",
            phase=1,
            sequence=3,
            observation_type="knowledge",
            content=[{"id": 7, "definition": self.rule}],
            summary="knowledge",
            tool_name="get_all_knowledge_definitions",
            function_call_id="knowledge-omission-owner-call",
        )
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            self._context(observation.observation_id),
            _FakeKnowledgeUpdater(response),
            grounding_input=self._bundle(runtime),
        )
        return runtime, result

    async def test_knowledge_mapping_for_omission_is_ignored_but_rule_is_kept(
        self,
    ) -> None:
        old, result = await self._run(
            KnowledgeGroundingResponse(
                column_mapping=(
                    ColumnMapping(
                        phrase="asset",
                        targets=("metrics.asset_id",),
                    ),
                    ColumnMapping(
                        phrase="risk score",
                        targets=("metrics.risk_a", "metrics.risk_b"),
                    ),
                ),
                selected_knowledge_ids=(7,),
            )
        )
        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(
            result.knowledge_omission_mapping_ignored,
            ("risk score",),
        )
        self.assertEqual(
            result.runtime.grounding_state.column_mapping,
            old.grounding_state.column_mapping,
        )
        self.assertEqual(
            result.runtime.grounding_state.domain_knowledge,
            (DomainKnowledge(kind="business_rule", content=self.rule),),
        )
        self.assertEqual(result.response.selected_knowledge_ids, (7,))

    async def test_knowledge_retains_existing_non_omission_correction_authority(
        self,
    ) -> None:
        _, result = await self._run(
            KnowledgeGroundingResponse(
                column_mapping=(
                    ColumnMapping(
                        phrase="asset",
                        targets=("metrics.canonical_asset_id",),
                    ),
                ),
                selected_knowledge_ids=(7,),
            )
        )
        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(result.knowledge_omission_mapping_ignored, ())
        self.assertEqual(
            result.runtime.grounding_state.column_mapping,
            (
                ColumnMapping(
                    phrase="asset",
                    targets=("metrics.canonical_asset_id",),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
