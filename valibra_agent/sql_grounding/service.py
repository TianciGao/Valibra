"""Atomic, offline State replacement service for SQL Grounding V1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from valibra_agent.sql_grounding.models import (
    ContractModel,
    DomainKnowledge,
    GroundingCheckResponse,
    GroundingDimension,
    GroundingLLMResponse,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    MappingGroundingResponse,
    SQLGroundingState,
    SQLGroundingValidationError,
    StageGroundingResponse,
    StateDiffAuthorization,
    StructureGroundingResponse,
    UserClarificationRecord,
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
    GroundingCallKind,
    GroundingUpdaterError,
    SQLGroundingUpdater,
    classify_grounding_input,
)


class SQLGroundingServiceResult(ContractModel):
    runtime: GroundingRuntime
    response: GroundingLLMResponse | StageGroundingResponse | None = None
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
        call_kind = _validate_service_inputs(
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
        materialized = _materialize_stage_response(
            runtime,
            response,
            call_kind=call_kind,
            grounding_input=grounding_input,
        )
        validate_sql_grounding_state(materialized.sql_grounding_state, context)
        if isinstance(response, GroundingCheckResponse):
            tool = response.next_tool
            clarification = (
                tool.materialize_user_clarification_request()
                if tool is not None
                else None
            )
            if clarification is not None and not any(
                clarification.phrase in source for source in context.query_texts
            ):
                raise SQLGroundingValidationError(
                    "clarification phrase is not a verbatim query substring"
                )
        validate_grounding_llm_response(
            materialized,
            stage=runtime.stage,
            context=context,
        )
        authorization = _authorization_for_observation(
            runtime,
            observation,
            affected_dimensions=affected_dimensions,
            call_kind=call_kind,
        )
        changed = validate_grounding_state_transition(
            runtime.grounding_state,
            materialized.sql_grounding_state,
            authorization,
        )
        _validate_observation_specific_diff(
            runtime,
            observation,
            materialized,
            changed=changed,
            call_kind=call_kind,
            grounding_input=grounding_input,
        )
        candidate = GroundingRuntime(
            grounding_revision=runtime.grounding_revision + int(bool(changed)),
            stage=runtime.stage,
            focus_dimension=materialized.next_focus_dimension,
            grounding_state=materialized.sql_grounding_state,
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


def _materialize_stage_response(
    runtime: GroundingRuntime,
    response: GroundingLLMResponse | StageGroundingResponse,
    *,
    call_kind: GroundingCallKind | None,
    grounding_input: Mapping[str, Any] | None,
) -> GroundingLLMResponse:
    """Merge one small 1.3 form into a complete candidate State.

    The merge is deterministic and dimension authorization remains enforced by
    the ordinary State transition validator below.  Legacy full responses are
    accepted only by offline compatibility callers.
    """

    if isinstance(response, GroundingLLMResponse):
        return response
    old = runtime.grounding_state
    if call_kind == "structure" and isinstance(response, StructureGroundingResponse):
        state = old.model_copy(
            update={"tables": response.tables, "join_keys": response.join_keys}
        )
        focus = "column_mapping"
    elif call_kind == "mapping" and isinstance(response, MappingGroundingResponse):
        state = old.model_copy(
            update={
                "tables": response.tables,
                "join_keys": response.join_keys,
                "column_mapping": response.column_mapping,
            }
        )
        focus = "domain_knowledge"
    elif call_kind == "knowledge" and isinstance(
        response, KnowledgeGroundingResponse
    ):
        domain_knowledge = _materialize_official_business_rules(
            response.selected_knowledge_ids,
            grounding_input=grounding_input,
        )
        state = SQLGroundingState.model_validate(
            {
                **old.model_dump(mode="json"),
                "column_mapping": response.column_mapping,
                "domain_knowledge": domain_knowledge,
            }
        )
        focus = "none"
    elif call_kind == "check" and isinstance(response, GroundingCheckResponse):
        state = old.model_copy(
            update={
                "column_mapping": response.column_mapping,
                "domain_knowledge": response.domain_knowledge,
            }
        )
        focus = "none"
    else:
        raise SQLGroundingValidationError("stage response does not match call kind")
    return GroundingLLMResponse(
        sql_grounding_state=state,
        user_clarification_requests=(),
        next_focus_dimension=focus,
    )


_OFFICIAL_BULK_KNOWLEDGE_FIELDS = frozenset(
    {"id", "knowledge", "description", "definition"}
)


def _materialize_official_business_rules(
    selected_ids: tuple[int, ...],
    *,
    grounding_input: Mapping[str, Any] | None,
) -> tuple[DomainKnowledge, ...]:
    """Resolve exact Official definitions without LLM copying or classification."""

    if grounding_input is None:
        raise SQLGroundingValidationError(
            "Knowledge selection requires current Official knowledge definitions"
        )
    entries = grounding_input.get("knowledge_definitions")
    if not isinstance(entries, list):
        raise SQLGroundingValidationError(
            "Official knowledge definitions must be a JSON array"
        )
    by_id: dict[int, DomainKnowledge] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise SQLGroundingValidationError(
                "Official knowledge entry must be an object"
            )
        if not set(entry).issubset(_OFFICIAL_BULK_KNOWLEDGE_FIELDS):
            raise SQLGroundingValidationError(
                "Official knowledge entry contains a hidden or unsupported field"
            )
        knowledge_id = entry.get("id")
        if isinstance(knowledge_id, bool) or not isinstance(knowledge_id, int):
            raise SQLGroundingValidationError(
                "Official knowledge id must be an integer"
            )
        if knowledge_id in by_id:
            raise SQLGroundingValidationError(
                "Official knowledge ids must be unique"
            )
        definition = entry.get("definition")
        try:
            knowledge = DomainKnowledge(
                kind="business_rule",
                content=definition,
            )
        except Exception as exc:
            raise SQLGroundingValidationError(
                "Official knowledge definition is invalid"
            ) from exc
        by_id[knowledge_id] = knowledge
    try:
        return tuple(by_id[knowledge_id] for knowledge_id in selected_ids)
    except KeyError as exc:
        raise SQLGroundingValidationError(
            "selected knowledge id is absent from Official evidence"
        ) from exc


def _validate_service_inputs(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    context: ValidationContext,
    affected_dimensions: tuple[GroundingDimension, ...],
    grounding_input: Mapping[str, Any] | None,
) -> GroundingCallKind | None:
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
    if affected_dimensions and observation.observation_type != "p2_follow_up":
        raise SQLGroundingValidationError(
            "explicit affected_dimensions are only valid for a P2 follow-up"
        )
    call_kind: GroundingCallKind | None = None
    if grounding_input is not None:
        if affected_dimensions:
            raise SQLGroundingValidationError(
                "staged Grounding cannot combine affected_dimensions"
            )
        call_kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
        expected_observation = {
            "structure": ("schema", "get_schema"),
            "mapping": ("metadata", "get_all_column_meanings"),
            "knowledge": ("knowledge", "get_all_knowledge_definitions"),
        }.get(call_kind)
        check_input_kind = next(
            (
                name
                for name in ("check_context", "latest_tool", "latest_user_answer")
                if name in grounding_input
            ),
            None,
        )
        p1_staged = runtime.stage == "INITIAL_GROUNDING" and observation.phase == 1
        if call_kind != "check":
            p1_staged = bool(
                p1_staged
                and expected_observation is not None
                and observation.observation_type == expected_observation[0]
                and observation.tool_name == expected_observation[1]
            )
        elif check_input_kind == "check_context":
            p1_staged = bool(
                p1_staged
                and observation.observation_type == "knowledge"
                and observation.tool_name == "get_all_knowledge_definitions"
            )
        elif check_input_kind == "latest_user_answer":
            p1_staged = bool(
                p1_staged
                and observation.observation_type == "user_answer"
                and observation.tool_name == "ask_user"
            )
        p2_staged = (
            runtime.stage == "P2_INCREMENTAL"
            and observation.phase == 2
            and (
                observation.observation_type == "p2_follow_up"
                if call_kind != "check" or check_input_kind == "check_context"
                else (
                    observation.observation_type == "user_answer"
                    and observation.tool_name == "ask_user"
                    if check_input_kind == "latest_user_answer"
                    else observation.tool_name is not None
                )
            )
        )
        if call_kind == "check" and check_input_kind == "latest_tool":
            p1_staged = bool(
                runtime.stage == "INITIAL_GROUNDING"
                and observation.phase == 1
                and observation.tool_name is not None
            )
        if not p1_staged and not p2_staged:
            raise SQLGroundingValidationError(
                "staged Grounding requires the matching P1 evidence or P2 follow-up"
            )
        if call_kind == "check" and check_input_kind == "latest_tool":
            latest_tool = grounding_input.get("latest_tool")
            if (
                not isinstance(latest_tool, Mapping)
                or latest_tool.get("name") != observation.tool_name
            ):
                raise SQLGroundingValidationError(
                    "Check latest tool differs from the legal Observation"
                )
        if call_kind == "check" and check_input_kind == "latest_user_answer":
            if observation.observation_type != "user_answer":
                raise SQLGroundingValidationError(
                    "Check latest user answer requires a user_answer Observation"
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
        if observation.phase == 2:
            if grounding_input.get("follow_up") != context.follow_up_query:
                raise SQLGroundingValidationError(
                    "P2 Grounding follow_up differs from ValidationContext"
                )
            clarifications = grounding_input.get("user_clarifications")
            if not isinstance(clarifications, list):
                raise SQLGroundingValidationError(
                    "P2 Grounding user_clarifications must be a list"
                )
            records = tuple(
                UserClarificationRecord.model_validate(item)
                for item in clarifications
            )
            if any(item.phase != 1 or item.answer is None for item in records):
                raise SQLGroundingValidationError(
                    "P2 Grounding may include only answered Phase-1 clarifications"
                )
            questions = [item.question for item in records]
            if len(questions) != len(set(questions)):
                raise SQLGroundingValidationError(
                    "P2 Grounding clarification questions must be unique"
                )
    # The strict model performs de-duplication and canonical ordering checks.
    StateDiffAuthorization(
        stage=runtime.stage,
        authorized_dimensions=affected_dimensions,
    )
    return call_kind


def _authorization_for_observation(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    *,
    affected_dimensions: tuple[GroundingDimension, ...],
    call_kind: GroundingCallKind | None,
) -> StateDiffAuthorization:
    if call_kind is not None:
        dimensions_by_kind: dict[
            GroundingCallKind,
            tuple[GroundingDimension, ...],
        ] = {
            "structure": ("tables", "join_keys"),
            "mapping": ("tables", "join_keys", "column_mapping"),
            "knowledge": ("column_mapping", "domain_knowledge"),
            "check": ("column_mapping", "domain_knowledge"),
        }
        return StateDiffAuthorization(
            stage=runtime.stage,
            authorized_dimensions=dimensions_by_kind[call_kind],
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
    elif observation_type == "p2_follow_up":
        dimensions = affected_dimensions
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
    call_kind: GroundingCallKind | None,
    grounding_input: Mapping[str, Any] | None,
) -> None:
    if call_kind is not None:
        _validate_staged_response(
            runtime,
            response,
            call_kind=call_kind,
            grounding_input=grounding_input,
        )
    if (
        call_kind == "check"
        and grounding_input is not None
        and "latest_user_answer" in grounding_input
        and changed
    ):
        raise SQLGroundingValidationError(
            "Check latest_user_answer cannot directly change Grounding State"
        )
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
    if call_kind is None and observation.observation_type in {
        "user_query",
        "tool_error",
        "user_answer",
        "sql_execution",
    } and changed:
        raise SQLGroundingValidationError(
            f"{observation.observation_type} cannot directly change Grounding State"
        )
    if call_kind is None and observation.observation_type in {
        "tool_error",
        "user_answer",
        "sql_execution",
        "submission",
    } and response.next_focus_dimension != runtime.focus_dimension:
        raise SQLGroundingValidationError(
            f"{observation.observation_type} cannot change Grounding focus"
        )
    if call_kind is None and observation.observation_type == "submission" and changed:
        raise SQLGroundingValidationError(
            "submission cannot change Grounding State"
        )


def _validate_staged_response(
    runtime: GroundingRuntime,
    response: GroundingLLMResponse,
    *,
    call_kind: GroundingCallKind,
    grounding_input: Mapping[str, Any] | None,
) -> None:
    old = runtime.grounding_state
    new = response.sql_grounding_state
    if call_kind == "structure":
        if new.tables is None or new.join_keys is None:
            raise SQLGroundingValidationError(
                "Structure Grounding must evaluate tables and join_keys"
            )
        if (
            new.column_mapping != old.column_mapping
            or new.domain_knowledge != old.domain_knowledge
        ):
            raise SQLGroundingValidationError(
                "Structure Grounding cannot change mapping or knowledge"
            )
        expected_focus = "column_mapping"
    elif call_kind == "mapping":
        if (
            new.tables is None
            or new.join_keys is None
            or new.column_mapping is None
        ):
            raise SQLGroundingValidationError(
                "Mapping Grounding must evaluate structure and column_mapping"
            )
        if new.domain_knowledge != old.domain_knowledge:
            raise SQLGroundingValidationError(
                "Mapping Grounding cannot change domain_knowledge"
            )
        expected_focus = "domain_knowledge"
    elif call_kind == "knowledge":
        if not new.all_dimensions_evaluated:
            raise SQLGroundingValidationError(
                "Knowledge Grounding must complete all four dimensions"
            )
        if new.tables != old.tables or new.join_keys != old.join_keys:
            raise SQLGroundingValidationError(
                "Knowledge Grounding cannot change tables or join_keys"
            )
        expected_focus = "none"
    else:
        if not new.all_dimensions_evaluated:
            raise SQLGroundingValidationError("Check requires a complete State")
        if new.tables != old.tables or new.join_keys != old.join_keys:
            raise SQLGroundingValidationError(
                "Check cannot change tables or join_keys"
            )
        expected_focus = "none"
    if response.next_focus_dimension != expected_focus:
        raise SQLGroundingValidationError(
            f"{call_kind} Grounding must set focus to {expected_focus}"
        )
    if response.user_clarification_requests:
        raise SQLGroundingValidationError(
            "stage-specific Grounding forms cannot embed clarification requests"
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
