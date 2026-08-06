"""Fail-open orchestration for the pure K0 NoOp kernel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import Field

from valibra_agent.requirement_grounding.models import (
    KernelModel,
    Observation,
    PendingToolCall,
    PhaseTransition,
    RequirementGroundingPatch,
    RequirementGroundingRuntime,
    RequirementGroundingState,
)
from valibra_agent.requirement_grounding.reducer import apply_patch, validate_runtime
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


class GroundingControlError(ValueError):
    """A bounded runtime-control mutation is invalid or ambiguous."""


def add_pending_tool_call(
    runtime: RequirementGroundingRuntime,
    pending: PendingToolCall,
) -> RequirementGroundingRuntime:
    """Add one exactly identified Pending call without changing revision."""

    if pending.function_call_id in runtime.pending_tool_calls:
        raise GroundingControlError(
            f"pending tool call already exists: {pending.function_call_id}"
        )
    pending_calls = dict(runtime.pending_tool_calls)
    pending_calls[pending.function_call_id] = pending
    result = _validated_runtime_copy(
        runtime,
        pending_tool_calls=pending_calls,
    )
    _assert_revision_unchanged(runtime, result)
    validate_runtime(result)
    return result


def remove_pending_tool_call(
    runtime: RequirementGroundingRuntime,
    function_call_id: str,
) -> tuple[RequirementGroundingRuntime, PendingToolCall | None]:
    """Remove only the call matching ``function_call_id``."""

    pending = runtime.pending_tool_calls.get(function_call_id)
    if pending is None:
        return runtime, None
    pending_calls = dict(runtime.pending_tool_calls)
    del pending_calls[function_call_id]
    result = _validated_runtime_copy(
        runtime,
        pending_tool_calls=pending_calls,
    )
    _assert_revision_unchanged(runtime, result)
    validate_runtime(result)
    return result, pending


def process_phase_transition(
    runtime: RequirementGroundingRuntime,
    observation: Observation,
) -> GroundingServiceResult:
    """Apply an explicit PhaseTransition Observation without semantic inference."""

    if observation.observation_id in runtime.processed_observation_ids:
        return GroundingServiceResult(
            status="duplicate",
            runtime=runtime,
            patch_id=None,
        )
    if observation.observation_type != "phase_transition":
        failed = record_failure(
            runtime,
            stage="observation",
            exception=ValueError(
                "phase transition service requires phase_transition observation"
            ),
            sequence=observation.sequence,
            observation_id=observation.observation_id,
            function_call_id=observation.function_call_id,
        )
        return GroundingServiceResult(status="failed", runtime=failed)

    patch = RequirementGroundingPatch(
        patch_id=f"phase-transition:{observation.observation_id}",
        base_revision=runtime.grounding_revision,
        source_observation_ids=(observation.observation_id,),
        phase_transition=PhaseTransition(
            target_phase=2,
            reason="legal submit_sql transition from phase 1 to phase 2",
            sequence=observation.sequence,
        ),
        diagnostics=("explicit ADK lifecycle transition; no semantic update",),
    )
    try:
        reduced = apply_patch(runtime, patch)
        reduced = increment_metrics(
            reduced,
            observations_seen=1,
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
            runtime,
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


def _validated_runtime_copy(
    runtime: RequirementGroundingRuntime,
    **updates: object,
) -> RequirementGroundingRuntime:
    data = runtime.model_dump(mode="python")
    data.update(updates)
    return RequirementGroundingRuntime.model_validate(data)


def _assert_revision_unchanged(
    before: RequirementGroundingRuntime,
    after: RequirementGroundingRuntime,
) -> None:
    if before.grounding_revision != after.grounding_revision:
        raise AssertionError("runtime-control changes cannot increment revision")
