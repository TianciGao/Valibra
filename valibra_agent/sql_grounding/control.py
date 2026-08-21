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
        "pending_user_clarification",
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
        lines.append(
            "Grounding is final for this phase. Use the current State for targeted "
            "verification only; do not repeat get_schema, "
            "get_all_column_meanings, or get_all_knowledge_definitions."
        )
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
        # Submission feedback belongs to the frozen Official trajectory.  The
        # Main Agent may debug and resubmit, but Grounding does not re-open.
        stage = runtime.stage
    elif event == "official_p2_follow_up":
        allowed_stages = {"SQL_ATTEMPT"}
        if allow_initial_forced_exit:
            allowed_stages.add("INITIAL_GROUNDING")
        if runtime.stage not in allowed_stages:
            raise SQLGroundingValidationError(
                "official_p2_follow_up requires SQL_ATTEMPT"
            )
        stage = "P2_INCREMENTAL"
    elif event == "official_task_completed":
        if runtime.stage == "DONE":
            return runtime
        stage = "DONE"
    else:
        raise SQLGroundingValidationError("unsupported Grounding control event")

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
    pending_clarifications: int = 0,
) -> AttemptGateDecision:
    """Gate unresolved user-owned questions and premature first submits."""

    if (
        isinstance(pending_clarifications, bool)
        or not isinstance(pending_clarifications, int)
        or pending_clarifications < 0
    ):
        raise ValueError("pending_clarifications must be a non-negative integer")
    if pending_clarifications:
        return AttemptGateDecision(
            applicable=True,
            open=False,
            reason="pending_user_clarification",
        )

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
