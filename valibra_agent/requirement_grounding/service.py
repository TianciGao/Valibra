"""Fail-open orchestration for the pure K0 NoOp kernel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import Field

from valibra_agent.requirement_grounding.models import (
    KernelModel,
    Observation,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    RequirementGroundingState,
)
from valibra_agent.requirement_grounding.reducer import apply_patch
from valibra_agent.requirement_grounding.telemetry import (
    increment_metrics,
    record_failure,
    set_last_error,
)
from valibra_agent.requirement_grounding.updater import NoOpUpdater


class Updater(Protocol):
    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch: ...


Reducer = Callable[
    [RequirementGroundingRuntime, RequirementGroundingPatch],
    RequirementGroundingRuntime,
]


class GroundingServiceResult(KernelModel):
    status: Literal["processed", "duplicate", "failed"]
    runtime: RequirementGroundingRuntime
    patch_id: str | None = Field(default=None, max_length=128)


def process_observation(
    runtime: RequirementGroundingRuntime,
    observation: Observation,
    *,
    updater: Updater | None = None,
    reducer: Reducer = apply_patch,
) -> GroundingServiceResult:
    """Process one Observation while preserving valid business state on error."""

    if observation.observation_id in runtime.processed_observation_ids:
        return GroundingServiceResult(
            status="duplicate",
            runtime=runtime,
            patch_id=None,
        )

    selected_updater = updater or NoOpUpdater()
    # All untrusted extension points receive a detached working copy.  Even a
    # faulty future Updater/Reducer that mutates before raising cannot corrupt
    # the caller's last valid Runtime.
    working = RequirementGroundingRuntime.model_validate_json(
        runtime.model_dump_json()
    )
    counted = increment_metrics(
        runtime,
        observations_seen=1,
        updater_calls=1,
    )
    try:
        patch = selected_updater.propose(
            observation,
            working.grounding_state,
            base_revision=working.grounding_revision,
        )
    except Exception as exc:
        failed = record_failure(
            counted,
            stage="updater",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
        )
        return GroundingServiceResult(status="failed", runtime=failed)

    try:
        reduced = reducer(working, patch)
        reduced = increment_metrics(
            reduced,
            observations_seen=1,
            updater_calls=1,
            patches_applied=1,
        )
        reduced = set_last_error(reduced, None)
        return GroundingServiceResult(
            status="processed",
            runtime=reduced,
            patch_id=patch.patch_id,
        )
    except Exception as exc:
        failed = record_failure(
            counted,
            stage="reducer",
            exception=exc,
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
        )
        return GroundingServiceResult(
            status="failed",
            runtime=failed,
            patch_id=patch.patch_id,
        )
