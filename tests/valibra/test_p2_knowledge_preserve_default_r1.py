from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    DomainKnowledge,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    SQLGroundingState,
    ValidationContext,
    canonical_json,
    domain_knowledge_semantic_ref,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.service import (
    SQLGroundingDraftRuntime,
    _materialize_knowledge_update,
    process_sql_grounding_draft_stage,
    process_sql_grounding_observation,
)
from valibra_agent.sql_grounding.telemetry import (
    GroundingLLMTelemetry,
    GroundingTokenUsage,
)
from valibra_agent.sql_grounding.updater import (
    GroundingClientResponse,
    GroundingUpdaterError,
    GroundingUpdaterResult,
    KnowledgeRetirementAuditSidecar,
    KNOWLEDGE_GROUNDING_PROMPT,
    SQL_GROUNDING_STAGE_FORM_SCHEMAS,
    SQLGroundingUpdater,
    _validated_bundled_grounding_input,
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


class _RawKnowledgeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self._content = canonical_json(response)

    async def complete(self, request: object) -> GroundingClientResponse:
        del request
        return GroundingClientResponse(
            content=self._content,
            usage=GroundingTokenUsage(),
        )


class P2KnowledgePreserveDefaultRuntimeTests(unittest.IsolatedAsyncioTestCase):
    query = "Show the old metric and status."
    follow_up = "Now use the new metric while keeping status."
    old_rule = "Old metric uses field old_value."
    status_rule = "Status uses field state_code."
    new_rule = "New metric uses field new_value."

    def _runtime(self) -> GroundingRuntime:
        return GroundingRuntime(
            grounding_revision=8,
            stage="P2_INCREMENTAL",
            focus_dimension="domain_knowledge",
            grounding_state=SQLGroundingState(
                tables=("metrics",),
                join_keys=(),
                column_mapping=(
                    ColumnMapping(
                        phrase="old metric",
                        targets=("metrics.old_value",),
                    ),
                    ColumnMapping(
                        phrase="status",
                        targets=("metrics.state_code",),
                    ),
                ),
                domain_knowledge=(
                    DomainKnowledge(kind="business_rule", content=self.old_rule),
                    DomainKnowledge(kind="business_rule", content=self.status_rule),
                ),
            ),
        )

    def _observation(self):
        return build_sql_grounding_observation(
            task_id="p2-knowledge-preserve-default-r1",
            phase=2,
            sequence=9,
            observation_type="p2_follow_up",
            content=self.follow_up,
            summary="P2 follow-up",
        )

    def _bundle(self, runtime: GroundingRuntime) -> dict[str, object]:
        return {
            "query": self.query,
            "follow_up": self.follow_up,
            "user_clarifications": [],
            "current_state": runtime.grounding_state.model_dump(mode="json"),
            "knowledge_definitions": [
                {"id": 15, "definition": self.new_rule},
                {"id": 46, "definition": self.status_rule},
            ],
            "relevant_column_meanings": {
                "metrics": {
                    "old_value": "Old value.",
                    "state_code": "Status code.",
                    "new_value": "New value.",
                }
            },
            "unresolved_mappings": [],
        }

    def _context(self, observation_id: str) -> ValidationContext:
        return ValidationContext(
            current_query=self.query,
            follow_up_query=self.follow_up,
            latest_observation_id=observation_id,
            known_tables=frozenset({"metrics"}),
            known_columns=frozenset(
                {
                    "metrics.old_value",
                    "metrics.state_code",
                    "metrics.new_value",
                }
            ),
            supported_domain_knowledge=frozenset(
                {
                    ("business_rule", self.old_rule),
                    ("business_rule", self.status_rule),
                    ("business_rule", self.new_rule),
                }
            ),
        )

    async def _run(self, response: KnowledgeGroundingResponse):
        runtime = self._runtime()
        observation = self._observation()
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            self._context(observation.observation_id),
            _FakeKnowledgeUpdater(response),
            grounding_input=self._bundle(runtime),
        )
        return runtime, result

    def _response(self, *, selected_ids: tuple[int, ...]) -> KnowledgeGroundingResponse:
        return KnowledgeGroundingResponse(
            column_mapping=self._runtime().grounding_state.column_mapping or (),
            selected_knowledge_ids=selected_ids,
        )

    async def test_provider_omission_preserves_prior_and_adds_selected(self) -> None:
        old, result = await self._run(self._response(selected_ids=(15, 46)))

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(
            result.runtime.grounding_state.domain_knowledge,
            (
                DomainKnowledge(kind="business_rule", content=self.new_rule),
                DomainKnowledge(kind="business_rule", content=self.old_rule),
                DomainKnowledge(kind="business_rule", content=self.status_rule),
            ),
        )
        self.assertEqual(
            result.knowledge_preserved_by_default_refs,
            (
                domain_knowledge_semantic_ref(
                    DomainKnowledge(kind="business_rule", content=self.old_rule)
                ),
            ),
        )
        self.assertEqual(
            old.grounding_state.domain_knowledge,
            (
                DomainKnowledge(kind="business_rule", content=self.old_rule),
                DomainKnowledge(kind="business_rule", content=self.status_rule),
            ),
        )

    async def test_selected_empty_cannot_clear_prior_knowledge(self) -> None:
        _, result = await self._run(self._response(selected_ids=()))

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(
            result.runtime.grounding_state.domain_knowledge,
            self._runtime().grounding_state.domain_knowledge,
        )
        self.assertEqual(len(result.knowledge_preserved_by_default_refs), 2)

    def test_p1_materialization_behavior_is_unchanged(self) -> None:
        old = self._runtime().grounding_state
        response = self._response(selected_ids=(15,))
        materialized, preserved = _materialize_knowledge_update(
            old,
            response,
            grounding_input={
                "query": self.query,
                "knowledge_definitions": [
                    {"id": 15, "definition": self.new_rule}
                ],
            },
        )

        self.assertEqual(
            materialized,
            (DomainKnowledge(kind="business_rule", content=self.new_rule),),
        )
        self.assertEqual(preserved, ())

    async def test_atomic_draft_preserves_prior_without_mutating_formal_state(
        self,
    ) -> None:
        formal = self._runtime()
        draft = SQLGroundingDraftRuntime(
            grounding_revision=formal.grounding_revision,
            stage=formal.stage,
            focus_dimension=formal.focus_dimension,
            grounding_state=formal.grounding_state,
        )
        observation = self._observation()
        result = await process_sql_grounding_draft_stage(
            draft,
            observation,
            self._context(observation.observation_id),
            _FakeKnowledgeUpdater(self._response(selected_ids=(15, 46))),
            grounding_input=self._bundle(formal),
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(len(result.runtime.grounding_state.domain_knowledge or ()), 3)
        self.assertEqual(
            formal.grounding_state.domain_knowledge,
            self._runtime().grounding_state.domain_knowledge,
        )
        self.assertEqual(len(result.knowledge_preserved_by_default_refs), 1)

    async def test_unexpected_retirement_field_is_non_authoritative_sidecar(
        self,
    ) -> None:
        runtime = self._runtime()
        observation = self._observation()
        retired_value = [
            {
                "prior_knowledge_ref": "not-a-valid-ref",
                "evidence_span": "not verbatim",
            }
        ]
        response = {
            "column_mapping": [
                item.model_dump(mode="json")
                for item in (runtime.grounding_state.column_mapping or ())
            ],
            "selected_knowledge_ids": [15, 46],
            "knowledge_retirement_proposals": retired_value,
        }
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            self._context(observation.observation_id),
            SQLGroundingUpdater(_RawKnowledgeClient(response)),
            grounding_input=self._bundle(runtime),
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(len(result.runtime.grounding_state.domain_knowledge or ()), 3)
        self.assertEqual(
            result.knowledge_retirement_audit_sidecar,
            KnowledgeRetirementAuditSidecar(
                value_shape="array",
                item_count=1,
                value_sha256=hashlib.sha256(
                    canonical_json(retired_value).encode("utf-8")
                ).hexdigest(),
            ),
        )

    async def test_non_array_retirement_field_cannot_gate_core_parse(self) -> None:
        runtime = self._runtime()
        observation = self._observation()
        response = {
            "column_mapping": [
                item.model_dump(mode="json")
                for item in (runtime.grounding_state.column_mapping or ())
            ],
            "selected_knowledge_ids": [],
            "knowledge_retirement_proposals": "malformed legacy value",
        }
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            self._context(observation.observation_id),
            SQLGroundingUpdater(_RawKnowledgeClient(response)),
            grounding_input=self._bundle(runtime),
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(
            result.runtime.grounding_state.domain_knowledge,
            runtime.grounding_state.domain_knowledge,
        )
        assert result.knowledge_retirement_audit_sidecar is not None
        self.assertEqual(
            result.knowledge_retirement_audit_sidecar.value_shape,
            "non_array",
        )
        self.assertEqual(result.knowledge_retirement_audit_sidecar.item_count, 0)

    async def test_atomic_legacy_retirement_field_is_stripped_and_audited(
        self,
    ) -> None:
        formal = self._runtime()
        draft = SQLGroundingDraftRuntime(
            grounding_revision=formal.grounding_revision,
            stage=formal.stage,
            focus_dimension=formal.focus_dimension,
            grounding_state=formal.grounding_state,
        )
        observation = self._observation()
        retired_value = [{"authority": "provider-asserted-but-non-authoritative"}]
        response = {
            "column_mapping": [
                item.model_dump(mode="json")
                for item in (formal.grounding_state.column_mapping or ())
            ],
            "selected_knowledge_ids": [15, 46],
            "knowledge_retirement_proposals": retired_value,
        }
        result = await process_sql_grounding_draft_stage(
            draft,
            observation,
            self._context(observation.observation_id),
            SQLGroundingUpdater(_RawKnowledgeClient(response)),
            grounding_input=self._bundle(formal),
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(len(result.runtime.grounding_state.domain_knowledge or ()), 3)
        self.assertEqual(
            formal.grounding_state.domain_knowledge,
            self._runtime().grounding_state.domain_knowledge,
        )
        assert result.knowledge_retirement_audit_sidecar is not None
        self.assertEqual(result.knowledge_retirement_audit_sidecar.item_count, 1)
        audit = grounding_callbacks._atomic_draft_stage_audit("knowledge", result)
        self.assertEqual(
            audit["knowledge_retirement_audit_sidecar"],
            result.knowledge_retirement_audit_sidecar.model_dump(mode="json"),
        )
        self.assertNotIn("knowledge_retirement_proposal_refs", audit)

    def test_p2_bundle_has_no_provider_retirement_catalog(self) -> None:
        runtime = self._runtime()
        observation = self._observation()
        bundle = self._bundle(runtime)

        self.assertEqual(classify_grounding_input(bundle, phase=2), "knowledge")
        validated = _validated_bundled_grounding_input(
            bundle,
            runtime=runtime,
            original_query=self.query,
            follow_up_query=self.follow_up,
            observation=observation,
        )
        self.assertEqual(validated, bundle)

        legacy = dict(bundle)
        legacy["prior_knowledge_catalog"] = []
        with self.assertRaises(GroundingUpdaterError):
            _validated_bundled_grounding_input(
                legacy,
                runtime=runtime,
                original_query=self.query,
                follow_up_query=self.follow_up,
                observation=observation,
            )

    def test_real_p2_builder_excludes_retirement_catalog(self) -> None:
        runtime = self._runtime()
        bootstrap = [
            {"tool_name": "get_schema", "content": "schema"},
            {"tool_name": "get_all_column_meanings", "content": "meanings"},
            {"tool_name": "get_all_knowledge_definitions", "content": "definitions"},
        ]
        with (
            patch.object(
                grounding_callbacks,
                "_bootstrap_evidence_prefix",
                return_value=bootstrap,
            ),
            patch.object(
                grounding_callbacks,
                "_parse_schema_projection",
                return_value=(frozenset({"metrics"}), frozenset()),
            ),
            patch.object(
                grounding_callbacks,
                "_normalized_knowledge_definitions",
                return_value=[],
            ),
            patch.object(
                grounding_callbacks,
                "_relevant_mapping_column_meanings",
                return_value={},
            ),
            patch.object(
                grounding_callbacks,
                "_active_mapping_omission_carrier",
                return_value=None,
            ),
        ):
            payload = grounding_callbacks._build_staged_grounding_request(
                {},
                call_kind="knowledge",
                query=self.query,
                runtime=runtime,
                phase=2,
                follow_up=self.follow_up,
            )

        self.assertNotIn("prior_knowledge_catalog", payload)
        self.assertEqual(classify_grounding_input(payload, phase=2), "knowledge")


class P2KnowledgeCoreContractTests(unittest.TestCase):
    def test_retirement_is_absent_from_prompt_and_core_form(self) -> None:
        lowered = KNOWLEDGE_GROUNDING_PROMPT.lower()
        self.assertNotIn("retirement", lowered)
        self.assertNotIn("prior_knowledge_catalog", KNOWLEDGE_GROUNDING_PROMPT)
        self.assertNotIn("authority", lowered)
        properties = SQL_GROUNDING_STAGE_FORM_SCHEMAS["knowledge"]["properties"]
        self.assertEqual(
            set(properties),
            {"column_mapping", "selected_knowledge_ids"},
        )

    def test_stable_kref_remains_an_internal_semantic_identity(self) -> None:
        first = DomainKnowledge(kind="business_rule", content="A + B")
        same = DomainKnowledge(kind="business_rule", content="A + B")
        other_kind = DomainKnowledge(kind="runtime_state", content="A + B")
        other_content = DomainKnowledge(kind="business_rule", content="A - B")

        self.assertEqual(
            domain_knowledge_semantic_ref(first),
            domain_knowledge_semantic_ref(same),
        )
        self.assertNotEqual(
            domain_knowledge_semantic_ref(first),
            domain_knowledge_semantic_ref(other_kind),
        )
        self.assertNotEqual(
            domain_knowledge_semantic_ref(first),
            domain_knowledge_semantic_ref(other_content),
        )

    def test_sidecar_contains_no_provider_authority_or_content(self) -> None:
        sidecar = KnowledgeRetirementAuditSidecar(
            value_shape="array",
            item_count=1,
            value_sha256="a" * 64,
        )
        serialized = json.dumps(sidecar.model_dump(mode="json"))
        self.assertNotIn("authority", serialized)
        self.assertNotIn("evidence_span", serialized)
        self.assertNotIn("prior_knowledge_ref", serialized)


if __name__ == "__main__":
    unittest.main()
