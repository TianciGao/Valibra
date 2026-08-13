import json
import unittest

from pydantic import ValidationError

from tests.valibra.k0_fixtures import (
    graph_patch,
    make_ambiguity,
    make_candidate,
    make_evidence,
    make_observation,
    make_slot,
)
from valibra_agent.requirement_grounding.models import (
    GroundedAmbiguityHypothesis,
    GroundingEvidence,
    InterpretationCandidate,
    OperationSlot,
    PendingToolCall,
    PhaseTransition,
    RequirementFrame,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    RequirementGroundingState,
    RuntimeMetrics,
    SQLImpact,
    SchemaSlot,
    ValibraError,
    ValueSlot,
)
from valibra_agent.requirement_grounding.observations import build_observation
from valibra_agent.requirement_grounding.reducer import apply_patch


class K0ModelTests(unittest.TestCase):
    def test_default_runtime_exact_json_structure(self):
        runtime = RequirementGroundingRuntime()
        self.assertEqual(
            runtime.model_dump(mode="json"),
            {
                "schema_version": "1.0",
                "grounding_revision": 0,
                "requirement_revision": 0,
                "frame_initialization_status": "not_attempted",
                "frame_initialization_reason": None,
                "frame_initialization_observation_id": None,
                "phase": 1,
                "grounding_state": {
                    "requirement_frame": {
                        "value_slots": [],
                        "schema_slots": [],
                        "operation_slots": [],
                    },
                    "ambiguity_index": [],
                },
                "processed_observation_ids": [],
                "pending_tool_calls": {},
                "metrics": {},
                "last_error": None,
            },
        )

    def test_default_runtime_json_round_trip(self):
        runtime = RequirementGroundingRuntime()
        restored = RequirementGroundingRuntime.model_validate_json(
            runtime.model_dump_json()
        )
        self.assertEqual(restored, runtime)

    def test_v1_runtime_without_requirement_revision_loads_compatibly(self):
        payload = RequirementGroundingRuntime().model_dump(mode="json")
        del payload["requirement_revision"]
        restored = RequirementGroundingRuntime.model_validate(payload)
        self.assertEqual(restored.requirement_revision, 0)

    def test_requirement_revision_cannot_exceed_technical_revision(self):
        with self.assertRaises(ValidationError):
            RequirementGroundingRuntime(
                grounding_revision=0,
                requirement_revision=1,
            )

    def test_populated_runtime_json_round_trip(self):
        runtime = RequirementGroundingRuntime()
        observation = make_observation()
        candidates = (
            make_candidate("cand-a", "A", "effect-a"),
            make_candidate("cand-b", "B", "effect-b"),
        )
        ambiguity = make_ambiguity(
            candidates=candidates,
            status="unresolved",
        )
        populated = apply_patch(runtime, graph_patch(runtime, observation, ambiguity=ambiguity))
        payload = populated.model_dump(mode="json")
        self.assertIn("evidence", payload["grounding_state"])
        restored = RequirementGroundingRuntime.model_validate_json(
            json.dumps(payload)
        )
        self.assertEqual(restored, populated)

    def test_models_forbid_unknown_fields(self):
        with self.assertRaises(ValidationError):
            RequirementGroundingRuntime.model_validate({"unknown": True})

    def test_pending_key_must_match_function_call_id(self):
        digest = "a" * 64
        pending = PendingToolCall(
            function_call_id="call-1",
            tool_name="get_schema",
            args_summary={},
            args_digest=digest,
            phase_before=1,
            sequence=1,
        )
        with self.assertRaises(ValidationError):
            RequirementGroundingRuntime(
                pending_tool_calls={"call-2": pending}
            )

    def test_metrics_are_json_object_and_nonnegative(self):
        metrics = RuntimeMetrics.model_validate(
            {"observations_seen": 2, "updater_latency_ms": 3.5}
        )
        self.assertEqual(
            metrics.model_dump(mode="json"),
            {"observations_seen": 2, "updater_latency_ms": 3.5},
        )
        with self.assertRaises(ValidationError):
            RuntimeMetrics.model_validate({"observations_seen": -1})

    def test_operation_parameters_cannot_store_nested_or_large_raw_content(self):
        common = dict(
            slot_id="operation-1",
            slot_role="filter",
            origin="user_query",
            sequence=1,
            operation_type="filter",
        )
        with self.assertRaises(ValidationError):
            OperationSlot(parameters={"schema": {"raw": "not allowed"}}, **common)
        with self.assertRaises(ValidationError):
            OperationSlot(parameters={"prompt": "x" * 257}, **common)

    def test_every_public_data_object_supports_json_round_trip(self):
        observation = make_observation()
        evidence = make_evidence(observation)
        candidate = make_candidate("cand-a", "A", "effect-a")
        value_slot = make_slot()
        schema_slot = SchemaSlot(
            slot_id="schema-1",
            slot_role="dimension",
            origin="schema",
            sequence=1,
            binding_type="column",
            bound_identifier="table.column",
        )
        operation_slot = OperationSlot(
            slot_id="operation-1",
            slot_role="limit",
            origin="user_query",
            sequence=1,
            operation_type="limit",
            parameters={"count": 10},
        )
        ambiguity = make_ambiguity(status="deferred")
        frame = RequirementFrame(
            value_slots=(value_slot,),
            schema_slots=(schema_slot,),
            operation_slots=(operation_slot,),
        )
        state = RequirementGroundingState(
            requirement_frame=frame,
            evidence=(evidence,),
        )
        pending = PendingToolCall(
            function_call_id="call-1",
            tool_name="get_schema",
            args_summary={"table": "bounded"},
            args_digest="b" * 64,
            phase_before=1,
            sequence=1,
            started_at="caller-provided",
        )
        metrics = RuntimeMetrics.model_validate({"observations_seen": 1})
        error = ValibraError(
            stage="service",
            error_type="SyntheticError",
            message_preview="bounded",
            observation_id=observation.observation_id,
            sequence=1,
            timestamp="caller-provided",
        )
        transition = PhaseTransition(sequence=2)
        candidate_patch = RequirementGroundingPatch(
            patch_id="roundtrip-patch",
            base_revision=0,
            source_observation_ids=(observation.observation_id,),
            slot_additions=(value_slot, schema_slot, operation_slot),
            ambiguity_additions=(ambiguity,),
            evidence_additions=(evidence,),
            phase_transition=transition,
        )
        runtime = RequirementGroundingRuntime(
            grounding_state=state,
            pending_tool_calls={pending.function_call_id: pending},
            metrics=metrics,
            last_error=error,
        )
        impact = SQLImpact(effect_key="effect-a", summary="effect")

        objects = (
            observation,
            evidence,
            impact,
            candidate,
            value_slot,
            schema_slot,
            operation_slot,
            frame,
            ambiguity,
            state,
            pending,
            metrics,
            error,
            transition,
            candidate_patch,
            runtime,
        )
        for item in objects:
            with self.subTest(model=type(item).__name__):
                dumped = item.model_dump(mode="json")
                restored = type(item).model_validate_json(item.model_dump_json())
                self.assertEqual(restored.model_dump(mode="json"), dumped)


if __name__ == "__main__":
    unittest.main()
