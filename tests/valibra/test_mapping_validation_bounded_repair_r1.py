from __future__ import annotations

import json
import unittest
from collections import deque
from typing import Any

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    GroundingRuntime,
    SQLGroundingState,
    ValidationContext,
)
from valibra_agent.sql_grounding.observations import (
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.service import (
    process_sql_grounding_observation,
)
from valibra_agent.sql_grounding.telemetry import GroundingTokenUsage
from valibra_agent.sql_grounding.updater import (
    MAPPING_GROUNDING_PROMPT,
    MAPPING_VALIDATION_CORRECTION_PROMPT,
    GroundingClientResponse,
    SQLGroundingUpdater,
)


class _SequentialClient:
    def __init__(self, *responses: str) -> None:
        self.responses = deque(responses)
        self.requests: list[Any] = []

    async def complete(self, request: Any) -> GroundingClientResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected Mapping Provider call")
        return GroundingClientResponse(
            content=self.responses.popleft(),
            usage=GroundingTokenUsage(
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                total_tokens=2,
            ),
        )


def _observation(task_id: str = "mapping-repair") -> Any:
    return build_sql_grounding_observation(
        task_id=task_id,
        phase=1,
        sequence=2,
        observation_type="metadata",
        content={"source": "get_all_column_meanings"},
        summary="Official column meanings observed",
        tool_name="get_all_column_meanings",
        function_call_id=f"{task_id}-meanings",
    )


class MappingValidationBoundedRepairR1Tests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_targets_are_omitted_once_and_valid_mapping_is_preserved(
        self,
    ) -> None:
        query = "Show maintenance cost for active assets."
        state = SQLGroundingState(
            tables=("operational_metrics",),
            join_keys=(),
            column_mapping=None,
            domain_knowledge=None,
        )
        runtime = GroundingRuntime(
            focus_dimension="column_mapping",
            grounding_state=state,
        )
        initial = {
            "tables": ["operational_metrics"],
            "join_keys": [],
            "column_mapping": [
                {"phrase": "maintenance cost", "targets": []},
                {
                    "phrase": "active assets",
                    "targets": ["operational_metrics.status"],
                },
            ],
            "unresolved_mappings": [],
        }
        corrected = {
            "tables": ["operational_metrics"],
            "join_keys": [],
            "column_mapping": [
                {
                    "phrase": "active assets",
                    "targets": ["operational_metrics.status"],
                }
            ],
            "unresolved_mappings": [
                {
                    "phrase": "maintenance cost",
                    "reason": "no_direct_metadata",
                }
            ],
        }
        client = _SequentialClient(json.dumps(initial), json.dumps(corrected))
        observation = _observation("empty-target")
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            ValidationContext(
                current_query=query,
                latest_observation_id=observation.observation_id,
                official_trajectory_observation_ids=(observation.observation_id,),
                known_tables=frozenset({"operational_metrics"}),
                known_columns=frozenset(
                    {
                        "operational_metrics.maintcost",
                        "operational_metrics.status",
                    }
                ),
            ),
            SQLGroundingUpdater(client),
            grounding_input={
                "query": query,
                "current_state": state.model_dump(mode="json"),
                "column_meanings": {
                    "operational_metrics.maintcost": "Maintenance cost.",
                    "operational_metrics.status": "Asset status.",
                },
                "unresolved_mappings": [],
            },
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[0].prompt, MAPPING_GROUNDING_PROMPT)
        self.assertEqual(
            client.requests[1].prompt,
            MAPPING_VALIDATION_CORRECTION_PROMPT,
        )
        repair_input = json.loads(client.requests[1].input_json)[
            "mapping_validation_correction"
        ]
        self.assertEqual(repair_input["invalid_phrases"], ["maintenance cost"])
        self.assertEqual(
            result.runtime.grounding_state.column_mapping[0].phrase,
            "active assets",
        )
        self.assertEqual(result.mapping_validation_repair.outcome, "accepted")
        self.assertEqual(
            result.mapping_validation_repair.trigger,
            "form_validation_failed",
        )

    async def test_unsupported_json_path_gets_one_target_only_correction(self) -> None:
        query = "Compare short clips to longer ones."
        state = SQLGroundingState(
            tables=("content_info",),
            join_keys=(),
            column_mapping=None,
            domain_knowledge=None,
        )
        runtime = GroundingRuntime(
            focus_dimension="column_mapping",
            grounding_state=state,
        )
        initial = {
            "tables": ["content_info"],
            "join_keys": [],
            "column_mapping": [
                {
                    "phrase": "short clips",
                    "targets": ["content_info.mediacounts ->> 'Clips_Total'"],
                }
            ],
            "unresolved_mappings": [],
        }
        corrected = {
            "tables": ["content_info"],
            "join_keys": [],
            "column_mapping": [
                {
                    "phrase": "short clips",
                    "targets": [
                        "content_info.mediacounts -> 'content_volumes' ->> 'Clips_Total'"
                    ],
                }
            ],
            "unresolved_mappings": [],
        }
        client = _SequentialClient(json.dumps(initial), json.dumps(corrected))
        observation = _observation("json-path")
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            ValidationContext(
                current_query=query,
                latest_observation_id=observation.observation_id,
                official_trajectory_observation_ids=(observation.observation_id,),
                known_tables=frozenset({"content_info"}),
                known_columns=frozenset({"content_info.mediacounts"}),
                supported_json_paths=frozenset(
                    {
                        (
                            "content_info.mediacounts",
                            ("content_volumes", "Clips_Total"),
                        )
                    }
                ),
            ),
            SQLGroundingUpdater(client),
            grounding_input={
                "query": query,
                "current_state": state.model_dump(mode="json"),
                "column_meanings": {
                    "content_info.mediacounts": {
                        "fields_meaning": {
                            "content_volumes": {"Clips_Total": "Clip count."}
                        }
                    }
                },
                "unresolved_mappings": [],
            },
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(
            result.mapping_validation_repair.trigger,
            "state_validation_failed",
        )
        self.assertEqual(result.mapping_validation_repair.outcome, "accepted")
        self.assertEqual(
            result.runtime.grounding_state.column_mapping[0].targets,
            (
                "content_info.mediacounts -> 'content_volumes' ->> 'Clips_Total'",
            ),
        )

    async def test_form_valid_semantic_proxy_does_not_trigger_repair(self) -> None:
        query = "Show the downtime score."
        state = SQLGroundingState(
            tables=("metrics",),
            join_keys=(),
            column_mapping=None,
            domain_knowledge=None,
        )
        runtime = GroundingRuntime(
            focus_dimension="column_mapping",
            grounding_state=state,
        )
        response = {
            "tables": ["metrics"],
            "join_keys": [],
            "column_mapping": [
                {
                    "phrase": "downtime score",
                    "targets": ["metrics.availability_pct"],
                }
            ],
            "unresolved_mappings": [],
        }
        client = _SequentialClient(json.dumps(response))
        observation = _observation("semantic-proxy")
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            ValidationContext(
                current_query=query,
                latest_observation_id=observation.observation_id,
                official_trajectory_observation_ids=(observation.observation_id,),
                known_tables=frozenset({"metrics"}),
                known_columns=frozenset({"metrics.availability_pct"}),
            ),
            SQLGroundingUpdater(client),
            grounding_input={
                "query": query,
                "current_state": state.model_dump(mode="json"),
                "column_meanings": {
                    "metrics.availability_pct": "System availability percentage."
                },
                "unresolved_mappings": [],
            },
        )

        self.assertEqual(result.state_update.status, "accepted")
        self.assertEqual(len(client.requests), 1)
        self.assertIsNone(result.mapping_validation_repair)

    async def test_correction_cannot_change_a_previously_valid_mapping(self) -> None:
        query = "Show maintenance cost for active assets."
        state = SQLGroundingState(
            tables=("operational_metrics",),
            join_keys=(),
            column_mapping=None,
            domain_knowledge=None,
        )
        runtime = GroundingRuntime(
            focus_dimension="column_mapping",
            grounding_state=state,
        )
        initial = {
            "tables": ["operational_metrics"],
            "join_keys": [],
            "column_mapping": [
                {"phrase": "maintenance cost", "targets": []},
                {
                    "phrase": "active assets",
                    "targets": ["operational_metrics.status"],
                },
            ],
            "unresolved_mappings": [],
        }
        illegal_correction = {
            "tables": ["operational_metrics"],
            "join_keys": [],
            "column_mapping": [
                {
                    "phrase": "active assets",
                    "targets": ["operational_metrics.asset_id"],
                }
            ],
            "unresolved_mappings": [
                {
                    "phrase": "maintenance cost",
                    "reason": "no_direct_metadata",
                }
            ],
        }
        client = _SequentialClient(
            json.dumps(initial),
            json.dumps(illegal_correction),
        )
        observation = _observation("scope-violation")
        result = await process_sql_grounding_observation(
            runtime,
            observation,
            ValidationContext(
                current_query=query,
                latest_observation_id=observation.observation_id,
                official_trajectory_observation_ids=(observation.observation_id,),
                known_tables=frozenset({"operational_metrics"}),
                known_columns=frozenset(
                    {
                        "operational_metrics.asset_id",
                        "operational_metrics.maintcost",
                        "operational_metrics.status",
                    }
                ),
            ),
            SQLGroundingUpdater(client),
            grounding_input={
                "query": query,
                "current_state": state.model_dump(mode="json"),
                "column_meanings": {
                    "operational_metrics.asset_id": "Asset identity.",
                    "operational_metrics.maintcost": "Maintenance cost.",
                    "operational_metrics.status": "Asset status.",
                },
                "unresolved_mappings": [],
            },
        )

        self.assertEqual(result.state_update.status, "rejected")
        self.assertEqual(result.runtime, runtime)
        self.assertEqual(
            result.mapping_validation_repair.outcome,
            "repair_validation_failed",
        )


class MappingValidationRepairCounterTests(unittest.TestCase):
    def test_repair_is_a_second_logical_call_but_not_a_second_stage(self) -> None:
        state: dict[str, Any] = {}
        grounding_callbacks._record_provider_call(state, 1)
        grounding_callbacks._record_provider_call(
            state,
            1,
            mapping_validation_repair=True,
        )

        self.assertEqual(grounding_callbacks._provider_call_count(state), 2)
        self.assertEqual(grounding_callbacks._provider_phase_call_count(state, 1), 2)
        self.assertEqual(
            grounding_callbacks._provider_phase_stage_call_count(state, 1),
            1,
        )


if __name__ == "__main__":
    unittest.main()
