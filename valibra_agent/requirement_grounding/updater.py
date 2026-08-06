"""K0 updater contract with the only supported implementation: NoOp."""

from __future__ import annotations

from valibra_agent.requirement_grounding.models import (
    Observation,
    RequirementGroundingPatch,
    RequirementGroundingState,
)


class NoOpUpdater:
    """Produce a deterministic empty Patch without changing State."""

    mode = "noop"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        # Force contract validation of the supplied State without retaining or
        # mutating it.  K0 deliberately derives no semantic facts.
        RequirementGroundingState.model_validate(state.model_dump(mode="python"))
        return RequirementGroundingPatch(
            patch_id=f"noop:{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            diagnostics=("K0 NoOp updater: no business-state operations",),
        )


def build_noop_patch(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    return NoOpUpdater().propose(
        observation,
        state,
        base_revision=base_revision,
    )
