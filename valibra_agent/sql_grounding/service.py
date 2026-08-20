"""Atomic, offline State replacement service for SQL Grounding V1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from valibra_agent.sql_grounding.models import (
    ContractModel,
    GROUNDING_DIMENSIONS,
    GroundingDimension,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingValidationError,
    StateDiffAuthorization,
    ValidationContext,
    sql_grounding_state_sha256,
    validate_grounding_llm_response,
    validate_grounding_runtime_transition,
    validate_grounding_state_transition,
    validate_sql_grounding_state,
)
from valibra_agent.sql_grounding.observations import SQLGroundingObservation
from valibra_agent.sql_grounding.telemetry import (
    GroundingLLMTelemetry,
    StateUpdateTelemetry,
)
from valibra_agent.sql_grounding.updater import (
    GroundingUpdaterError,
    SQLGroundingUpdater,
)


class SQLGroundingServiceResult(ContractModel):
    runtime: GroundingRuntime
    response: GroundingLLMResponse | None = None
    llm_telemetry: GroundingLLMTelemetry
    state_update: StateUpdateTelemetry
    transport_normalization: Literal["none", "single_json_fence"] | None = None


async def process_sql_grounding_observation(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    context: ValidationContext,
    updater: SQLGroundingUpdater,
    *,
    affected_dimensions: tuple[GroundingDimension, ...] = (),
    grounding_input: Mapping[str, Any] | None = None,
) -> SQLGroundingServiceResult:
    """Propose, validate, and atomically accept a complete State plus focus.

    ``affected_dimensions`` is a transient caller authorization used only for
    user-provided P2/follow-up semantics.  It is not an Observation field or a
    persisted Evidence object.  Omitting it is fail-closed.
    """

    old_sha = sql_grounding_state_sha256(runtime.grounding_state)
    try:
        _validate_service_inputs(
            runtime,
            observation,
            context,
            affected_dimensions,
            grounding_input,
        )
        updater_result = await updater.propose(
            runtime,
            observation,
            original_query=context.current_query,
            follow_up_query=context.follow_up_query,
            grounding_input=grounding_input,
        )
    except GroundingUpdaterError as exc:
        return SQLGroundingServiceResult(
            runtime=runtime,
            llm_telemetry=exc.telemetry,
            state_update=_rejected_update(
                runtime,
                observation,
                old_sha=old_sha,
                error_type=exc.reason,
            ),
        )
    except Exception as exc:
        # Input validation occurs before a fake-client call.  Keep this path
        # bounded and do not copy exception text into telemetry.
        return SQLGroundingServiceResult(
            runtime=runtime,
            llm_telemetry=_not_attempted_telemetry("service_input_invalid"),
            state_update=_rejected_update(
                runtime,
                observation,
                old_sha=old_sha,
                error_type=_service_error_type(exc),
            ),
        )

    response = updater_result.response
    try:
        validate_sql_grounding_state(response.sql_grounding_state, context)
        validate_grounding_llm_response(
            response,
            stage=runtime.stage,
            context=context,
        )
        authorization = _authorization_for_observation(
            runtime,
            observation,
            affected_dimensions=affected_dimensions,
            bundled_grounding=grounding_input is not None,
        )
        changed = validate_grounding_state_transition(
            runtime.grounding_state,
            response.sql_grounding_state,
            authorization,
        )
        _validate_observation_specific_diff(
            runtime,
            observation,
            response,
            changed=changed,
            bundled_grounding=grounding_input is not None,
        )
        candidate = GroundingRuntime(
            grounding_revision=runtime.grounding_revision + int(bool(changed)),
            stage=runtime.stage,
            focus_dimension=response.next_focus_dimension,
            grounding_state=response.sql_grounding_state,
        )
        validate_grounding_runtime_transition(runtime, candidate, authorization)
    except Exception as exc:
        return SQLGroundingServiceResult(
            runtime=runtime,
            response=response,
            llm_telemetry=updater_result.telemetry,
            state_update=_rejected_update(
                runtime,
                observation,
                old_sha=old_sha,
                error_type=_service_error_type(exc),
            ),
            transport_normalization=updater_result.transport_normalization,
        )

    new_sha = sql_grounding_state_sha256(candidate.grounding_state)
    changed_control = candidate.focus_dimension != runtime.focus_dimension
    status: Literal["accepted", "noop"] = (
        "accepted" if changed or changed_control else "noop"
    )
    accepted_runtime = runtime if status == "noop" else candidate
    return SQLGroundingServiceResult(
        runtime=accepted_runtime,
        response=response,
        llm_telemetry=updater_result.telemetry,
        state_update=StateUpdateTelemetry(
            observation_id=observation.observation_id,
            stage=runtime.stage,
            status=status,
            old_state_sha256=old_sha,
            new_state_sha256=new_sha,
            changed_dimensions=changed,
            revision_before=runtime.grounding_revision,
            revision_after=candidate.grounding_revision,
            focus_before=runtime.focus_dimension,
            focus_after=candidate.focus_dimension,
        ),
        transport_normalization=updater_result.transport_normalization,
    )


def _validate_service_inputs(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    context: ValidationContext,
    affected_dimensions: tuple[GroundingDimension, ...],
    grounding_input: Mapping[str, Any] | None,
) -> None:
    if context.latest_observation_id != observation.observation_id:
        raise SQLGroundingValidationError(
            "ValidationContext must identify the latest legal Observation"
        )
    if observation.observation_type == "user_query":
        if runtime.stage != "INITIAL_GROUNDING" or observation.phase != 1:
            raise SQLGroundingValidationError(
                "user_query initialization requires Phase 1 INITIAL_GROUNDING"
            )
        if observation.content != context.current_query:
            raise SQLGroundingValidationError(
                "user_query Observation must equal ValidationContext.current_query"
            )
    if observation.observation_type == "p2_follow_up":
        if runtime.stage != "P2_INCREMENTAL" or observation.phase != 2:
            raise SQLGroundingValidationError(
                "p2_follow_up requires Phase 2 P2_INCREMENTAL"
            )
        if context.follow_up_query is None or observation.content != context.follow_up_query:
            raise SQLGroundingValidationError(
                "p2_follow_up must equal ValidationContext.follow_up_query"
            )
    if affected_dimensions and observation.observation_type not in {
        "p2_follow_up",
        "user_answer",
    }:
        raise SQLGroundingValidationError(
            "explicit affected_dimensions are only valid for new user semantics"
        )
    if grounding_input is not None:
        if affected_dimensions:
            raise SQLGroundingValidationError(
                "Primary Grounding cannot combine affected_dimensions"
            )
        primary_fields = {
            "query",
            "schema",
            "column_meanings",
            "knowledge_definitions",
            "current_state",
        }
        repair_fields = primary_fields | {
            "execute_sql_evidence",
            "submit_failure",
        }
        primary = (
            runtime.stage == "INITIAL_GROUNDING"
            and observation.phase == 1
            and observation.observation_type == "knowledge"
            and observation.tool_name == "get_all_knowledge_definitions"
        )
        repair = (
            runtime.stage == "REPAIR"
            and observation.observation_type == "submission"
            and observation.tool_name == "submit_sql"
        )
        if not primary and not repair:
            raise SQLGroundingValidationError(
                "bundled Grounding requires Primary bootstrap or REPAIR submission"
            )
        expected_fields = primary_fields if primary else repair_fields
        if set(grounding_input) != expected_fields:
            raise SQLGroundingValidationError(
                "bundled Grounding input has an invalid field set"
            )
        if grounding_input.get("query") != context.current_query:
            raise SQLGroundingValidationError(
                "Primary Grounding query differs from ValidationContext"
            )
        if grounding_input.get("current_state") != runtime.grounding_state.model_dump(
            mode="json"
        ):
            raise SQLGroundingValidationError(
                "bundled Grounding current_state differs from Runtime"
            )
        if repair:
            _validate_repair_bundle(grounding_input, observation)
    # The strict model performs de-duplication and canonical ordering checks.
    StateDiffAuthorization(
        stage=runtime.stage,
        authorized_dimensions=affected_dimensions,
    )


def _authorization_for_observation(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    *,
    affected_dimensions: tuple[GroundingDimension, ...],
    bundled_grounding: bool,
) -> StateDiffAuthorization:
    if bundled_grounding:
        return StateDiffAuthorization(
            stage=runtime.stage,
            authorized_dimensions=GROUNDING_DIMENSIONS,
        )
    observation_type = observation.observation_type
    if observation_type == "schema":
        dimensions: tuple[GroundingDimension, ...] = (
            "tables",
            "join_keys",
            "column_mapping",
            "domain_knowledge",
        )
    elif observation_type == "metadata":
        dimensions = ("column_mapping", "domain_knowledge")
    elif observation_type == "knowledge":
        dimensions = ("domain_knowledge",)
    elif observation_type in {"p2_follow_up", "user_answer"}:
        dimensions = affected_dimensions
    elif observation_type == "sql_execution" and runtime.focus_dimension != "none":
        dimensions = (runtime.focus_dimension,)
    else:
        dimensions = ()
    return StateDiffAuthorization(
        stage=runtime.stage,
        authorized_dimensions=dimensions,
    )


def _validate_observation_specific_diff(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    response: GroundingLLMResponse,
    *,
    changed: tuple[GroundingDimension, ...],
    bundled_grounding: bool,
) -> None:
    if (
        observation.observation_type in {"schema", "metadata"}
        and "domain_knowledge" in changed
    ):
        old = runtime.grounding_state.domain_knowledge
        new = response.sql_grounding_state.domain_knowledge
        if old is not None or new != ():
            raise SQLGroundingValidationError(
                f"{observation.observation_type} can only resolve null "
                "domain_knowledge to []"
            )
    if observation.observation_type in {
        "user_query",
        "tool_error",
    } and changed:
        raise SQLGroundingValidationError(
            f"{observation.observation_type} cannot directly change Grounding State"
        )
    if (
        observation.observation_type == "submission"
        and changed
        and not bundled_grounding
    ):
        raise SQLGroundingValidationError(
            "submission cannot change Grounding State without a Repair bundle"
        )
    if (
        observation.observation_type == "submission"
        and runtime.stage != "REPAIR"
        and response.next_focus_dimension != runtime.focus_dimension
    ):
        raise SQLGroundingValidationError(
            "submission can change focus only after an official REPAIR transition"
        )


def _validate_repair_bundle(
    grounding_input: Mapping[str, Any],
    observation: SQLGroundingObservation,
) -> None:
    """Validate the bounded shape and exact current submit evidence."""

    expected_event_fields = {"trajectory_index", "phase", "args", "result"}
    execute_evidence = grounding_input.get("execute_sql_evidence")
    submit_failure = grounding_input.get("submit_failure")
    if not isinstance(execute_evidence, list) or not isinstance(
        submit_failure, Mapping
    ):
        raise SQLGroundingValidationError("Repair evidence has an invalid shape")
    indexes: list[int] = []
    for event in execute_evidence:
        if not isinstance(event, Mapping) or set(event) != expected_event_fields:
            raise SQLGroundingValidationError(
                "execute_sql Repair evidence has an invalid shape"
            )
        index = event.get("trajectory_index")
        phase = event.get("phase")
        args = event.get("args")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise SQLGroundingValidationError(
                "execute_sql Repair evidence has an invalid trajectory index"
            )
        if (
            isinstance(phase, bool)
            or phase not in {1, 2}
            or not isinstance(args, dict)
        ):
            raise SQLGroundingValidationError(
                "execute_sql Repair evidence has invalid Official fields"
            )
        indexes.append(index)
    if indexes != sorted(set(indexes)):
        raise SQLGroundingValidationError(
            "execute_sql Repair evidence must be unique and ordered"
        )
    if set(submit_failure) != expected_event_fields:
        raise SQLGroundingValidationError(
            "submit failure evidence has an invalid shape"
        )
    submit_phase = submit_failure.get("phase")
    if isinstance(submit_phase, bool) or submit_phase != observation.phase:
        raise SQLGroundingValidationError(
            "submit failure phase differs from Observation"
        )
    if submit_failure.get("result") != observation.content:
        raise SQLGroundingValidationError(
            "submit failure result differs from Observation"
        )
    submit_index = submit_failure.get("trajectory_index")
    if (
        isinstance(submit_index, bool)
        or not isinstance(submit_index, int)
        or submit_index < 0
        or (indexes and submit_index <= indexes[-1])
        or not isinstance(submit_failure.get("args"), dict)
    ):
        raise SQLGroundingValidationError(
            "submit failure evidence is not the ordered current Official result"
        )


def _service_error_type(exc: Exception) -> str:
    if isinstance(exc, SQLGroundingValidationError):
        text = str(exc)
        if "unauthorized" in text or "can only" in text or "cannot directly" in text:
            return "authorization_rejected"
        if "revision" in text or "Runtime" in text:
            return "runtime_transition_rejected"
        return "state_validation_failed"
    return "unexpected_error"


def _rejected_update(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    *,
    old_sha: str,
    error_type: str,
) -> StateUpdateTelemetry:
    return StateUpdateTelemetry(
        observation_id=observation.observation_id,
        stage=runtime.stage,
        status="rejected",
        old_state_sha256=old_sha,
        new_state_sha256=old_sha,
        changed_dimensions=(),
        revision_before=runtime.grounding_revision,
        revision_after=runtime.grounding_revision,
        focus_before=runtime.focus_dimension,
        focus_after=runtime.focus_dimension,
        error_type=error_type,
    )


def _not_attempted_telemetry(error_type: str) -> GroundingLLMTelemetry:
    # Import constants here avoids copying hashes into a second source of truth.
    from valibra_agent.sql_grounding.updater import (
        SQL_GROUNDING_CONFIGURATION_SHA256,
        SQL_GROUNDING_FORM_SCHEMA_SHA256,
        SQL_GROUNDING_PROMPT_SHA256,
    )

    return GroundingLLMTelemetry(
        attempted=False,
        status="rejected",
        error_type=error_type,
        request_sha256="",
        response_sha256="",
        prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
        form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )
