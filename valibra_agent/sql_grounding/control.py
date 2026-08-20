"""Pure SQL Grounding control decisions; never executes a BIRD tool."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias

from pydantic import Field

from valibra_agent.sql_grounding.models import (
    ContractModel,
    FocusDimension,
    GroundingRuntime,
    SQLGroundingValidationError,
    StateDiffAuthorization,
    validate_grounding_runtime_transition,
)

StageEvent: TypeAlias = Literal[
    "grounding_completed",
    "official_submit_failed",
    "official_p2_follow_up",
    "official_task_completed",
    "repair_completed_p1",
    "repair_completed_p2",
]

FOCUS_TOOL_DIRECTIONS: Mapping[FocusDimension, tuple[str, ...]] = MappingProxyType({
    "tables": ("get_schema",),
    "join_keys": ("get_schema", "execute_sql"),
    "column_mapping": (
        "get_schema",
        "get_column_meaning",
        "get_all_column_meanings",
        "execute_sql",
    ),
    "domain_knowledge": (
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
    ),
    "none": (),
})


class AttemptGateDecision(ContractModel):
    applicable: bool
    open: bool
    reason: Literal[
        "ready",
        "dimension_not_evaluated",
        "initial_focus_pending",
        "invalid_first_submit_stage",
        "subsequent_submit_not_gated",
    ]


class RenderedControlHint(ContractModel):
    text: str = Field(max_length=1_024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def tool_directions_for_focus(focus: FocusDimension) -> tuple[str, ...]:
    """Return stable tool directions, never a selected tool or invocation."""

    return FOCUS_TOOL_DIRECTIONS[focus]


def render_control_hint(focus: FocusDimension) -> RenderedControlHint:
    """Render a short, State-independent suggestion for a future model turn."""

    directions = tool_directions_for_focus(focus)
    lines = [
        "[VALIBRA CONTROL]",
        f"Current grounding focus: {focus}.",
    ]
    if directions:
        lines.append(
            "Prefer the relevant BIRD tool direction before submit_sql: "
            + " / ".join(directions)
            + "."
        )
    else:
        lines.append("No additional Grounding tool direction is suggested.")
    text = "\n".join(lines)
    return RenderedControlHint(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def transition_grounding_stage(
    runtime: GroundingRuntime,
    event: StageEvent,
    *,
    allow_initial_forced_exit: bool = False,
) -> GroundingRuntime:
    """Apply only an explicit future-callback lifecycle event."""

    if event == "grounding_completed":
        if (
            runtime.stage != "INITIAL_GROUNDING"
            or not runtime.grounding_state.all_dimensions_evaluated
            or runtime.focus_dimension != "none"
        ):
            raise SQLGroundingValidationError(
                "grounding_completed requires complete INITIAL_GROUNDING with focus none"
            )
        stage = "SQL_ATTEMPT"
    elif event == "official_submit_failed":
        allowed_stages = {"SQL_ATTEMPT", "P2_INCREMENTAL"}
        if allow_initial_forced_exit:
            allowed_stages.add("INITIAL_GROUNDING")
        if runtime.stage not in allowed_stages:
            raise SQLGroundingValidationError(
                "official_submit_failed requires an SQL attempt stage"
            )
        stage = "REPAIR"
    elif event == "official_p2_follow_up":
        allowed_stages = {"SQL_ATTEMPT"}
        if allow_initial_forced_exit:
            allowed_stages.add("INITIAL_GROUNDING")
        if runtime.stage not in allowed_stages:
            raise SQLGroundingValidationError(
                "official_p2_follow_up requires SQL_ATTEMPT"
            )
        stage = "P2_INCREMENTAL"
    elif event in {"repair_completed_p1", "repair_completed_p2"}:
        if (
            runtime.stage != "REPAIR"
            or not runtime.grounding_state.all_dimensions_evaluated
            or runtime.focus_dimension != "none"
        ):
            raise SQLGroundingValidationError(
                f"{event} requires complete REPAIR with focus none"
            )
        stage = "SQL_ATTEMPT" if event == "repair_completed_p1" else "P2_INCREMENTAL"
    else:
        if runtime.stage == "DONE":
            return runtime
        stage = "DONE"

    candidate = runtime.model_copy(update={"stage": stage})
    validate_grounding_runtime_transition(
        runtime,
        candidate,
        StateDiffAuthorization(stage=runtime.stage),
    )
    return candidate


def evaluate_first_submit_gate(
    runtime: GroundingRuntime,
    *,
    first_submit: bool,
) -> AttemptGateDecision:
    """Evaluate the SG2 pure gate; only the first submit is hard-gated."""

    if not first_submit:
        return AttemptGateDecision(
            applicable=False,
            open=True,
            reason="subsequent_submit_not_gated",
        )
    if not runtime.grounding_state.all_dimensions_evaluated:
        return AttemptGateDecision(
            applicable=True,
            open=False,
            reason="dimension_not_evaluated",
        )
    if runtime.stage == "INITIAL_GROUNDING" and runtime.focus_dimension != "none":
        return AttemptGateDecision(
            applicable=True,
            open=False,
            reason="initial_focus_pending",
        )
    if runtime.stage in {"INITIAL_GROUNDING", "SQL_ATTEMPT"} and (
        runtime.focus_dimension == "none"
    ):
        return AttemptGateDecision(applicable=True, open=True, reason="ready")
    return AttemptGateDecision(
        applicable=True,
        open=False,
        reason="invalid_first_submit_stage",
    )
