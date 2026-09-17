"""Atomic, offline State replacement service for SQL Grounding V1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, ValidationError

from valibra_agent.sql_grounding.models import (
    ContractModel,
    ColumnMapping,
    DomainKnowledge,
    GroundingCheckResponse,
    GroundingDimension,
    GroundingLLMResponse,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    MappingGroundingResponse,
    UnresolvedMapping,
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
    domain_knowledge_semantic_ref,
)
from valibra_agent.sql_grounding.observations import SQLGroundingObservation
from valibra_agent.sql_grounding.telemetry import (
    GroundingLLMTelemetry,
    StateUpdateTelemetry,
)
from valibra_agent.sql_grounding.updater import (
    GroundingCallKind,
    KnowledgeRetirementAuditSidecar,
    GroundingUpdaterError,
    SQLGroundingUpdater,
    classify_grounding_input,
    normalize_grounding_transport,
)


class MappingValidationRepairAudit(ContractModel):
    trigger: Literal["form_validation_failed", "state_validation_failed"]
    outcome: Literal[
        "accepted",
        "repair_provider_failed",
        "repair_validation_failed",
    ]
    invalid_phrases: tuple[str, ...]
    initial_llm_telemetry: GroundingLLMTelemetry
    repair_llm_telemetry: GroundingLLMTelemetry | None = None


class SQLGroundingServiceResult(ContractModel):
    runtime: GroundingRuntime
    response: GroundingLLMResponse | StageGroundingResponse | None = None
    llm_telemetry: GroundingLLMTelemetry
    state_update: StateUpdateTelemetry
    transport_normalization: Literal["none", "single_json_fence"] | None = None
    mapping_validation_repair: MappingValidationRepairAudit | None = None
    knowledge_omission_mapping_ignored: tuple[str, ...] = ()
    knowledge_preserved_by_default_refs: tuple[str, ...] = ()
    knowledge_retirement_audit_sidecar: (
        KnowledgeRetirementAuditSidecar | None
    ) = None


@dataclass(frozen=True, slots=True)
class _MappingValidationRepairScope:
    rejected_mapping: dict[str, Any]
    tables: tuple[str, ...]
    join_keys: tuple[str, ...]
    valid_mappings: tuple[ColumnMapping, ...]
    unresolved_mappings: tuple[UnresolvedMapping, ...]
    invalid_phrases: tuple[str, ...]
    validation_error: str


@dataclass(frozen=True, slots=True)
class SQLGroundingDraftRuntime:
    """Private, possibly cross-dimensionally inconsistent Draft carrier."""

    grounding_revision: int
    stage: Any
    focus_dimension: Any
    grounding_state: SQLGroundingState


@dataclass(frozen=True, slots=True)
class _DraftMaterializedResponse:
    sql_grounding_state: SQLGroundingState
    next_focus_dimension: Any
    user_clarification_requests: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class SQLGroundingDraftStageResult:
    """One private Draft step; never a persisted formal Runtime."""

    runtime: SQLGroundingDraftRuntime
    llm_telemetry: GroundingLLMTelemetry
    state_update: StateUpdateTelemetry
    response: StageGroundingResponse | None = None
    transport_normalization: Literal["none", "single_json_fence"] | None = None
    knowledge_omission_mapping_ignored: tuple[str, ...] = ()
    knowledge_preserved_by_default_refs: tuple[str, ...] = ()
    knowledge_retirement_audit_sidecar: (
        KnowledgeRetirementAuditSidecar | None
    ) = None


async def process_sql_grounding_draft_stage(
    runtime: SQLGroundingDraftRuntime,
    observation: SQLGroundingObservation,
    context: ValidationContext,
    updater: SQLGroundingUpdater,
    *,
    grounding_input: Mapping[str, Any],
) -> SQLGroundingDraftStageResult:
    """Materialize one Draft stage without validating a mixed complete State.

    Stage scope, cumulative preservation, typed forms, and transition authority
    remain enforced. Cross-dimensional State validation is deliberately
    deferred until ``commit_sql_grounding_draft`` so a new Structure is never
    rejected merely because the Draft still contains the old Mapping.
    """

    old_sha = sql_grounding_state_sha256(runtime.grounding_state)
    try:
        call_kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
        if call_kind not in {"structure", "mapping", "knowledge", "check"}:
            raise SQLGroundingValidationError("unsupported Draft stage")
        if grounding_input.get("query") != context.current_query:
            raise SQLGroundingValidationError(
                "Draft query differs from ValidationContext"
            )
        if grounding_input.get("current_state") != runtime.grounding_state.model_dump(
            mode="json"
        ):
            raise SQLGroundingValidationError(
                "Draft input State differs from private Runtime"
            )
        updater_result = await updater.propose(
            runtime,
            observation,
            original_query=context.current_query,
            follow_up_query=context.follow_up_query,
            grounding_input=grounding_input,
        )
    except GroundingUpdaterError as exc:
        return SQLGroundingDraftStageResult(
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
        return SQLGroundingDraftStageResult(
            runtime=runtime,
            llm_telemetry=_not_attempted_telemetry("draft_input_invalid"),
            state_update=_rejected_update(
                runtime,
                observation,
                old_sha=old_sha,
                error_type=_service_error_type(exc),
            ),
        )

    response = updater_result.response
    knowledge_omission_mapping_ignored = (
        updater_result.knowledge_omission_mapping_ignored
    )
    knowledge_retirement_audit_sidecar = (
        updater_result.knowledge_retirement_audit_sidecar
    )
    knowledge_preserved_by_default_refs: tuple[str, ...] = ()
    try:
        if not isinstance(
            response,
            (
                StructureGroundingResponse,
                MappingGroundingResponse,
                KnowledgeGroundingResponse,
                GroundingCheckResponse,
            ),
        ):
            raise SQLGroundingValidationError("Draft received a control response")
        if call_kind == "knowledge" and isinstance(
            response, KnowledgeGroundingResponse
        ):
            response, service_ignored = (
                _ignore_knowledge_mapping_for_unresolved_phrases(
                    runtime.grounding_state,
                    response,
                    grounding_input=grounding_input,
                )
            )
            knowledge_omission_mapping_ignored = tuple(
                sorted(
                    set(knowledge_omission_mapping_ignored) | set(service_ignored)
                )
            )
            knowledge_preserved_by_default_refs = (
                _knowledge_preserved_by_default_refs(
                    runtime.grounding_state,
                    response,
                    grounding_input=grounding_input,
                )
            )
        _validate_mapping_omission_phrases(response, context)
        materialized = _materialize_stage_response(
            runtime,
            response,
            call_kind=call_kind,
            grounding_input=grounding_input,
            draft_cycle=True,
        )
        authorization = _authorization_for_observation(
            runtime,
            observation,
            affected_dimensions=(),
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
            stage_response=(
                response if isinstance(response, GroundingCheckResponse) else None
            ),
            draft_cycle=True,
        )
        candidate = SQLGroundingDraftRuntime(
            grounding_revision=runtime.grounding_revision + int(bool(changed)),
            stage=runtime.stage,
            focus_dimension=materialized.next_focus_dimension,
            grounding_state=materialized.sql_grounding_state,
        )
    except Exception as exc:
        return SQLGroundingDraftStageResult(
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
            knowledge_omission_mapping_ignored=(
                knowledge_omission_mapping_ignored
            ),
            knowledge_preserved_by_default_refs=(
                knowledge_preserved_by_default_refs
            ),
            knowledge_retirement_audit_sidecar=(
                knowledge_retirement_audit_sidecar
            ),
        )

    new_sha = sql_grounding_state_sha256(candidate.grounding_state)
    status: Literal["accepted", "noop"] = (
        "accepted"
        if changed or candidate.focus_dimension != runtime.focus_dimension
        else "noop"
    )
    accepted_runtime = runtime if status == "noop" else candidate
    return SQLGroundingDraftStageResult(
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
        knowledge_omission_mapping_ignored=(
            knowledge_omission_mapping_ignored
        ),
        knowledge_preserved_by_default_refs=(
            knowledge_preserved_by_default_refs
        ),
        knowledge_retirement_audit_sidecar=(
            knowledge_retirement_audit_sidecar
        ),
    )


def commit_sql_grounding_draft(
    formal_runtime: GroundingRuntime,
    draft_runtime: SQLGroundingDraftRuntime,
    *,
    context: ValidationContext,
    final_check: GroundingCheckResponse,
) -> GroundingRuntime:
    """Validate a complete Draft and create exactly one formal revision."""

    if (
        final_check.status != "complete"
        or final_check.clarification_route != "none"
        or final_check.next_tool is not None
        or final_check.missing_information is not None
    ):
        raise SQLGroundingValidationError(
            "atomic Draft commits only after a complete final Check"
        )
    materialized = GroundingLLMResponse(
        sql_grounding_state=draft_runtime.grounding_state,
        user_clarification_requests=(),
        next_focus_dimension="none",
    )
    validate_sql_grounding_state(materialized.sql_grounding_state, context)
    validate_grounding_llm_response(
        materialized,
        stage=formal_runtime.stage,
        context=context,
    )
    authorization = StateDiffAuthorization(
        stage=formal_runtime.stage,
        authorized_dimensions=(
            "tables",
            "join_keys",
            "column_mapping",
            "domain_knowledge",
        ),
    )
    changed = validate_grounding_state_transition(
        formal_runtime.grounding_state,
        materialized.sql_grounding_state,
        authorization,
    )
    candidate = GroundingRuntime(
        grounding_revision=(
            formal_runtime.grounding_revision + int(bool(changed))
        ),
        stage=formal_runtime.stage,
        focus_dimension="none",
        grounding_state=materialized.sql_grounding_state,
    )
    return validate_grounding_runtime_transition(
        formal_runtime,
        candidate,
        authorization,
    )


def _mapping_form_repair_scope(
    error: GroundingUpdaterError,
) -> _MappingValidationRepairScope | None:
    """Return a target-only repair scope for one parseable Mapping Form error."""

    raw_payload = error.rejected_mapping_payload
    if error.reason != "form_validation_failed" or raw_payload is None:
        return None
    try:
        normalized, _ = normalize_grounding_transport(raw_payload)
        raw = json.loads(normalized)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or set(raw) != {
        "tables",
        "join_keys",
        "column_mapping",
        "unresolved_mappings",
    }:
        return None
    if not isinstance(raw.get("column_mapping"), list):
        return None
    try:
        common = MappingGroundingResponse.model_validate(
            {
                "tables": raw["tables"],
                "join_keys": raw["join_keys"],
                "column_mapping": [],
                "unresolved_mappings": raw["unresolved_mappings"],
            }
        )
    except ValidationError:
        return None

    valid: list[ColumnMapping] = []
    invalid: list[str] = []
    seen_phrases: set[str] = set()
    for item in raw["column_mapping"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"phrase", "targets"}
            or not isinstance(item.get("phrase"), str)
            or not item["phrase"].strip()
            or item["phrase"] in seen_phrases
        ):
            return None
        phrase = item["phrase"]
        seen_phrases.add(phrase)
        try:
            valid.append(ColumnMapping.model_validate(item))
        except ValidationError as exc:
            # R1 repairs only target shape/expression errors.  Phrase, table,
            # join, duplicate, and whole-response failures remain terminal.
            if not exc.errors() or any(
                not detail.get("loc") or detail["loc"][0] != "targets"
                for detail in exc.errors()
            ):
                return None
            invalid.append(phrase)
    if seen_phrases & {item.phrase for item in common.unresolved_mappings}:
        return None
    if not invalid:
        return None
    return _MappingValidationRepairScope(
        rejected_mapping=raw,
        tables=common.tables,
        join_keys=common.join_keys,
        valid_mappings=tuple(valid),
        unresolved_mappings=common.unresolved_mappings,
        invalid_phrases=tuple(sorted(invalid)),
        validation_error=error.validation_detail or error.reason,
    )


def _mapping_state_repair_scope(
    runtime: GroundingRuntime,
    response: GroundingLLMResponse | StageGroundingResponse,
    context: ValidationContext,
    error: SQLGroundingValidationError,
) -> _MappingValidationRepairScope | None:
    """Identify only Mapping entries whose targets violate Official evidence."""

    if not isinstance(response, MappingGroundingResponse):
        return None
    old = runtime.grounding_state
    common_state = old.model_copy(
        update={
            "tables": response.tables,
            "join_keys": response.join_keys,
            "column_mapping": (),
        }
    )
    try:
        validate_sql_grounding_state(common_state, context)
    except SQLGroundingValidationError:
        # A table/join/common-State error is outside target-only R1.
        return None

    valid: list[ColumnMapping] = []
    invalid: list[str] = []
    for mapping in response.column_mapping:
        if not any(mapping.phrase in source for source in context.query_texts):
            return None
        candidate = common_state.model_copy(update={"column_mapping": (mapping,)})
        try:
            validate_sql_grounding_state(candidate, context)
            valid.append(mapping)
        except SQLGroundingValidationError:
            invalid.append(mapping.phrase)
    if not invalid:
        return None
    return _MappingValidationRepairScope(
        rejected_mapping=response.model_dump(mode="json"),
        tables=response.tables,
        join_keys=response.join_keys,
        valid_mappings=tuple(valid),
        unresolved_mappings=response.unresolved_mappings,
        invalid_phrases=tuple(sorted(invalid)),
        validation_error=str(error),
    )


def _mapping_repair_input(
    scope: _MappingValidationRepairScope,
) -> dict[str, Any]:
    return {
        "invalid_phrases": list(scope.invalid_phrases),
        "rejected_mapping": scope.rejected_mapping,
        "validation_error": scope.validation_error,
    }


def _validate_mapping_repair_scope(
    response: GroundingLLMResponse | StageGroundingResponse,
    scope: _MappingValidationRepairScope,
) -> None:
    if not isinstance(response, MappingGroundingResponse):
        raise SQLGroundingValidationError(
            "Mapping validation repair returned the wrong response type"
        )
    if response.tables != scope.tables or response.join_keys != scope.join_keys:
        raise SQLGroundingValidationError(
            "Mapping validation repair changed tables or join_keys"
        )
    original_phrases = {
        item.get("phrase")
        for item in scope.rejected_mapping.get("column_mapping", [])
        if isinstance(item, dict) and isinstance(item.get("phrase"), str)
    }
    repaired = {item.phrase: item for item in response.column_mapping}
    if not set(repaired).issubset(original_phrases):
        raise SQLGroundingValidationError(
            "Mapping validation repair added a new phrase"
        )
    for original in scope.valid_mappings:
        if repaired.get(original.phrase) != original:
            raise SQLGroundingValidationError(
                "Mapping validation repair changed a previously valid mapping"
            )
    previous_unresolved = {
        item.phrase: item for item in scope.unresolved_mappings
    }
    repaired_unresolved = {
        item.phrase: item for item in response.unresolved_mappings
    }
    for phrase, original in previous_unresolved.items():
        if repaired_unresolved.get(phrase) != original:
            raise SQLGroundingValidationError(
                "Mapping validation repair changed an existing omission"
            )
    allowed_new_unresolved = set(scope.invalid_phrases)
    new_unresolved = set(repaired_unresolved) - set(previous_unresolved)
    if not new_unresolved.issubset(allowed_new_unresolved):
        raise SQLGroundingValidationError(
            "Mapping validation repair added an unrelated omission"
        )
    if any(
        repaired_unresolved[phrase].reason != "no_direct_metadata"
        for phrase in new_unresolved
    ):
        raise SQLGroundingValidationError(
            "deleted invalid mappings must use no_direct_metadata"
        )
    for phrase in scope.invalid_phrases:
        if phrase not in repaired and phrase not in repaired_unresolved:
            raise SQLGroundingValidationError(
                "Mapping validation repair silently dropped an invalid phrase"
            )


def _validate_mapping_omission_phrases(
    response: GroundingLLMResponse | StageGroundingResponse,
    context: ValidationContext,
) -> None:
    if not isinstance(response, MappingGroundingResponse):
        return
    for omission in response.unresolved_mappings:
        if not any(omission.phrase in source for source in context.query_texts):
            raise SQLGroundingValidationError(
                "unresolved mapping phrase is not a verbatim query substring"
            )


async def process_sql_grounding_observation(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    context: ValidationContext,
    updater: SQLGroundingUpdater,
    *,
    affected_dimensions: tuple[GroundingDimension, ...] = (),
    grounding_input: Mapping[str, Any] | None = None,
    allow_mapping_validation_repair: bool = True,
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

    repair_scope: _MappingValidationRepairScope | None = None
    repair_trigger: Literal[
        "form_validation_failed", "state_validation_failed"
    ] | None = None
    initial_telemetry: GroundingLLMTelemetry | None = None
    try:
        updater_result = await updater.propose(
            runtime,
            observation,
            original_query=context.current_query,
            follow_up_query=context.follow_up_query,
            grounding_input=grounding_input,
        )
    except GroundingUpdaterError as exc:
        if allow_mapping_validation_repair and call_kind == "mapping":
            repair_scope = _mapping_form_repair_scope(exc)
        if repair_scope is None:
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
        repair_trigger = "form_validation_failed"
        initial_telemetry = exc.telemetry
        try:
            updater_result = await updater.propose(
                runtime,
                observation,
                original_query=context.current_query,
                follow_up_query=context.follow_up_query,
                grounding_input=grounding_input,
                mapping_validation_correction=_mapping_repair_input(repair_scope),
            )
        except GroundingUpdaterError as repair_exc:
            return SQLGroundingServiceResult(
                runtime=runtime,
                llm_telemetry=repair_exc.telemetry,
                state_update=_rejected_update(
                    runtime,
                    observation,
                    old_sha=old_sha,
                    error_type=repair_exc.reason,
                ),
                mapping_validation_repair=MappingValidationRepairAudit(
                    trigger=repair_trigger,
                    outcome="repair_provider_failed",
                    invalid_phrases=repair_scope.invalid_phrases,
                    initial_llm_telemetry=initial_telemetry,
                    repair_llm_telemetry=repair_exc.telemetry,
                ),
            )
    except Exception as exc:
        return SQLGroundingServiceResult(
            runtime=runtime,
            llm_telemetry=_not_attempted_telemetry("updater_failed"),
            state_update=_rejected_update(
                runtime,
                observation,
                old_sha=old_sha,
                error_type=_service_error_type(exc),
            ),
        )

    response = updater_result.response
    knowledge_omission_mapping_ignored = (
        updater_result.knowledge_omission_mapping_ignored
    )
    knowledge_retirement_audit_sidecar = (
        updater_result.knowledge_retirement_audit_sidecar
    )
    knowledge_preserved_by_default_refs: tuple[str, ...] = ()
    try:
        try:
            if repair_scope is not None:
                _validate_mapping_repair_scope(response, repair_scope)
            if call_kind == "knowledge" and isinstance(
                response, KnowledgeGroundingResponse
            ):
                response, service_ignored = (
                    _ignore_knowledge_mapping_for_unresolved_phrases(
                        runtime.grounding_state,
                        response,
                        grounding_input=grounding_input,
                    )
                )
                knowledge_omission_mapping_ignored = tuple(
                    sorted(
                        set(knowledge_omission_mapping_ignored)
                        | set(service_ignored)
                    )
                )
                knowledge_preserved_by_default_refs = (
                    _knowledge_preserved_by_default_refs(
                        runtime.grounding_state,
                        response,
                        grounding_input=grounding_input,
                    )
                )
            _validate_mapping_omission_phrases(response, context)
            materialized = _materialize_stage_response(
                runtime,
                response,
                call_kind=call_kind,
                grounding_input=grounding_input,
            )
            validate_sql_grounding_state(materialized.sql_grounding_state, context)
        except SQLGroundingValidationError as exc:
            if repair_scope is not None:
                raise
            if allow_mapping_validation_repair and call_kind == "mapping":
                repair_scope = _mapping_state_repair_scope(
                    runtime,
                    response,
                    context,
                    exc,
                )
            if repair_scope is None:
                raise
            repair_trigger = "state_validation_failed"
            initial_telemetry = updater_result.telemetry
            updater_result = await updater.propose(
                runtime,
                observation,
                original_query=context.current_query,
                follow_up_query=context.follow_up_query,
                grounding_input=grounding_input,
                mapping_validation_correction=_mapping_repair_input(repair_scope),
            )
            response = updater_result.response
            _validate_mapping_repair_scope(response, repair_scope)
            _validate_mapping_omission_phrases(response, context)
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
            stage_response=(
                response if isinstance(response, GroundingCheckResponse) else None
            ),
        )
        candidate = GroundingRuntime(
            grounding_revision=runtime.grounding_revision + int(bool(changed)),
            stage=runtime.stage,
            focus_dimension=materialized.next_focus_dimension,
            grounding_state=materialized.sql_grounding_state,
        )
        validate_grounding_runtime_transition(runtime, candidate, authorization)
    except GroundingUpdaterError as exc:
        return SQLGroundingServiceResult(
            runtime=runtime,
            response=response,
            llm_telemetry=exc.telemetry,
            state_update=_rejected_update(
                runtime,
                observation,
                old_sha=old_sha,
                error_type=exc.reason,
            ),
            mapping_validation_repair=(
                MappingValidationRepairAudit(
                    trigger=repair_trigger or "state_validation_failed",
                    outcome="repair_provider_failed",
                    invalid_phrases=repair_scope.invalid_phrases,
                    initial_llm_telemetry=initial_telemetry
                    or updater_result.telemetry,
                    repair_llm_telemetry=exc.telemetry,
                )
                if repair_scope is not None
                else None
            ),
            knowledge_omission_mapping_ignored=(
                knowledge_omission_mapping_ignored
            ),
            knowledge_preserved_by_default_refs=(
                knowledge_preserved_by_default_refs
            ),
            knowledge_retirement_audit_sidecar=(
                knowledge_retirement_audit_sidecar
            ),
        )
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
            mapping_validation_repair=(
                MappingValidationRepairAudit(
                    trigger=repair_trigger or "state_validation_failed",
                    outcome="repair_validation_failed",
                    invalid_phrases=repair_scope.invalid_phrases,
                    initial_llm_telemetry=initial_telemetry
                    or updater_result.telemetry,
                    repair_llm_telemetry=updater_result.telemetry,
                )
                if repair_scope is not None
                else None
            ),
            knowledge_omission_mapping_ignored=(
                knowledge_omission_mapping_ignored
            ),
            knowledge_preserved_by_default_refs=(
                knowledge_preserved_by_default_refs
            ),
            knowledge_retirement_audit_sidecar=(
                knowledge_retirement_audit_sidecar
            ),
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
        mapping_validation_repair=(
            MappingValidationRepairAudit(
                trigger=repair_trigger or "state_validation_failed",
                outcome="accepted",
                invalid_phrases=repair_scope.invalid_phrases,
                initial_llm_telemetry=initial_telemetry
                or updater_result.telemetry,
                repair_llm_telemetry=updater_result.telemetry,
            )
            if repair_scope is not None
            else None
        ),
        knowledge_omission_mapping_ignored=(
            knowledge_omission_mapping_ignored
        ),
        knowledge_preserved_by_default_refs=(
            knowledge_preserved_by_default_refs
        ),
        knowledge_retirement_audit_sidecar=(
            knowledge_retirement_audit_sidecar
        ),
    )


def _materialize_stage_response(
    runtime: GroundingRuntime,
    response: GroundingLLMResponse | StageGroundingResponse,
    *,
    call_kind: GroundingCallKind | None,
    grounding_input: Mapping[str, Any] | None,
    draft_cycle: bool = False,
) -> GroundingLLMResponse | _DraftMaterializedResponse:
    """Merge one small 1.3 form into a complete candidate State.

    The merge is deterministic and dimension authorization remains enforced by
    the ordinary State transition validator below.  Legacy full responses are
    accepted only by offline compatibility callers.
    """

    if isinstance(response, GroundingLLMResponse):
        return response
    old = runtime.grounding_state
    initial_clarification_cycle = draft_cycle or bool(
        runtime.stage == "INITIAL_GROUNDING"
        and old.all_dimensions_evaluated
        and grounding_input is not None
        and grounding_input.get("user_clarifications")
    )
    if call_kind == "structure" and isinstance(response, StructureGroundingResponse):
        state = old.model_copy(
            update={"tables": response.tables, "join_keys": response.join_keys}
        )
        focus = "none" if initial_clarification_cycle else "column_mapping"
    elif call_kind == "mapping" and isinstance(response, MappingGroundingResponse):
        state = old.model_copy(
            update={
                "tables": response.tables,
                "join_keys": response.join_keys,
                "column_mapping": response.column_mapping,
            }
        )
        focus = "none" if initial_clarification_cycle else "domain_knowledge"
    elif call_kind == "knowledge" and isinstance(
        response, KnowledgeGroundingResponse
    ):
        _validate_knowledge_mapping_mutation(
            old,
            response,
        )
        domain_knowledge, _ = _materialize_knowledge_update(
            old,
            response,
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
    response_type = _DraftMaterializedResponse if draft_cycle else GroundingLLMResponse
    return response_type(
        sql_grounding_state=state,
        user_clarification_requests=(),
        next_focus_dimension=focus,
    )


def _validate_knowledge_mapping_mutation(
    old: SQLGroundingState,
    response: KnowledgeGroundingResponse,
) -> None:
    """Forbid mapping edits when no Official knowledge was selected."""

    if not response.selected_knowledge_ids:
        if response.column_mapping != (old.column_mapping or ()):
            raise SQLGroundingValidationError(
                "Knowledge cannot change column_mapping without selected Official knowledge"
            )


def _is_p2_grounding_input(
    grounding_input: Mapping[str, Any] | None,
) -> bool:
    return bool(
        grounding_input is not None
        and isinstance(grounding_input.get("follow_up"), str)
    )


def _materialize_knowledge_update(
    old: SQLGroundingState,
    response: KnowledgeGroundingResponse,
    *,
    grounding_input: Mapping[str, Any] | None,
) -> tuple[tuple[DomainKnowledge, ...], tuple[str, ...]]:
    """Materialize Knowledge with P2 preserve-by-default semantics.

    P2_KNOWLEDGE_RETIREMENT_DECOUPLING_R1 treats Provider omission as no
    deletion authority. Retirement is not part of the authoritative Knowledge
    response; the updater strips any legacy field into a digest-only sidecar.
    """

    selected = _materialize_official_business_rules(
        response.selected_knowledge_ids,
        grounding_input=grounding_input,
    )
    if not _is_p2_grounding_input(grounding_input):
        return selected, ()

    prior = old.domain_knowledge or ()
    selected_keys = {(item.kind, item.content) for item in selected}
    preserved_refs = tuple(
        domain_knowledge_semantic_ref(item)
        for item in prior
        if (item.kind, item.content) not in selected_keys
    )
    merged_by_semantics = {
        (item.kind, item.content): item
        for item in (*prior, *selected)
    }
    merged = tuple(
        sorted(
            merged_by_semantics.values(),
            key=lambda item: (item.kind, item.content),
        )
    )
    return merged, preserved_refs


def _knowledge_preserved_by_default_refs(
    old: SQLGroundingState,
    response: KnowledgeGroundingResponse,
    *,
    grounding_input: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    _, refs = _materialize_knowledge_update(
        old,
        response,
        grounding_input=grounding_input,
    )
    return refs


def _ignore_knowledge_mapping_for_unresolved_phrases(
    old: SQLGroundingState,
    response: KnowledgeGroundingResponse,
    *,
    grounding_input: Mapping[str, Any] | None,
) -> tuple[KnowledgeGroundingResponse, tuple[str, ...]]:
    """Remove only Knowledge mappings owned by the Mapping omission carrier.

    Knowledge may still select Official rules and may retain its existing
    bounded correction authority for every other phrase.  An unresolved phrase
    can move into ``column_mapping`` only through a later Mapping revision.
    """

    if grounding_input is None:
        return response, ()
    raw_unresolved = grounding_input.get("unresolved_mappings")
    if not isinstance(raw_unresolved, list):
        return response, ()
    unresolved = tuple(
        UnresolvedMapping.model_validate(item) for item in raw_unresolved
    )
    unresolved_phrases = {item.phrase for item in unresolved}
    if not unresolved_phrases:
        return response, ()

    old_phrases = {item.phrase for item in (old.column_mapping or ())}
    if old_phrases & unresolved_phrases:
        raise SQLGroundingValidationError(
            "Mapping omission carrier conflicts with current State"
        )

    ignored = tuple(
        sorted(
            item.phrase
            for item in response.column_mapping
            if item.phrase in unresolved_phrases
        )
    )
    if not ignored:
        return response, ()
    sanitized = response.model_copy(
        update={
            "column_mapping": tuple(
                item
                for item in response.column_mapping
                if item.phrase not in unresolved_phrases
            )
        }
    )
    return sanitized, ignored


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
        if call_kind == "final_gate":
            raise SQLGroundingValidationError(
                "Final re-Grounding Gate is control-only and cannot materialize State"
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
        clarification_records: tuple[UserClarificationRecord, ...] = ()
        if "user_clarifications" in grounding_input:
            clarifications = grounding_input.get("user_clarifications")
            if not isinstance(clarifications, list):
                raise SQLGroundingValidationError(
                    "Grounding user_clarifications must be a list"
                )
            clarification_records = tuple(
                UserClarificationRecord.model_validate(item)
                for item in clarifications
            )
            if any(
                item.phase > observation.phase or item.answer is None
                for item in clarification_records
            ):
                raise SQLGroundingValidationError(
                    "Grounding may include only prior answered clarifications"
                )
            questions = [item.question for item in clarification_records]
            if len(questions) != len(set(questions)):
                raise SQLGroundingValidationError(
                    "Grounding clarification questions must be unique"
                )
        clarification_regrounding = bool(
            observation.observation_type == "user_answer"
            and observation.tool_name == "ask_user"
            and isinstance(observation.content, str)
            and any(
                item.phase == observation.phase
                and item.answer == observation.content
                for item in clarification_records
            )
        )
        p1_staged = runtime.stage == "INITIAL_GROUNDING" and observation.phase == 1
        if clarification_regrounding:
            p1_staged = bool(p1_staged)
        elif call_kind != "check":
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
                clarification_regrounding
                or observation.observation_type == "p2_follow_up"
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
            if "user_clarifications" not in grounding_input:
                raise SQLGroundingValidationError(
                    "P2 Grounding user_clarifications must be present"
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
    runtime: GroundingRuntime | SQLGroundingDraftRuntime,
    observation: SQLGroundingObservation,
    response: GroundingLLMResponse | _DraftMaterializedResponse,
    *,
    changed: tuple[GroundingDimension, ...],
    call_kind: GroundingCallKind | None,
    grounding_input: Mapping[str, Any] | None,
    stage_response: GroundingCheckResponse | None = None,
    draft_cycle: bool = False,
) -> None:
    if call_kind is not None:
        _validate_staged_response(
            runtime,
            response,
            call_kind=call_kind,
            grounding_input=grounding_input,
            draft_cycle=draft_cycle,
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
    if call_kind == "check" and grounding_input is not None:
        _validate_check_route_and_repair_scope(
            runtime,
            response,
            changed=changed,
            grounding_input=grounding_input,
            check=stage_response,
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


def _validate_check_route_and_repair_scope(
    runtime: GroundingRuntime | SQLGroundingDraftRuntime,
    response: GroundingLLMResponse | _DraftMaterializedResponse,
    *,
    changed: tuple[GroundingDimension, ...],
    grounding_input: Mapping[str, Any],
    check: GroundingCheckResponse | None,
) -> None:
    """Bind Check control and State repair to its one latest evidence source."""

    if check is None:
        # Materialized staged responses retain the typed source response so
        # this branch is only for retired whole-State compatibility fixtures.
        return
    unresolved_mappings = grounding_input.get("unresolved_mappings")
    if not isinstance(unresolved_mappings, list):
        raise SQLGroundingValidationError(
            "Check requires the read-only unresolved_mappings carrier"
        )
    if check.status == "complete" and unresolved_mappings:
        raise SQLGroundingValidationError(
            "Check cannot complete while unresolved_mappings is non-empty"
        )
    latest_user_answer = "latest_user_answer" in grounding_input
    if latest_user_answer:
        if check.clarification_route == "none":
            raise SQLGroundingValidationError(
                "Check latest_user_answer requires one clarification route"
            )
    elif check.clarification_route != "none":
        raise SQLGroundingValidationError(
            "clarification route is legal only for latest_user_answer"
        )

    if latest_user_answer:
        return

    latest_tool = grounding_input.get("latest_tool")
    if latest_tool is None:
        if changed:
            raise SQLGroundingValidationError(
                "initial Check cannot change Grounding State without new Official evidence"
            )
        return
    if not isinstance(latest_tool, Mapping):
        raise SQLGroundingValidationError("Check latest_tool must be a mapping")
    tool_name = latest_tool.get("name")
    old = runtime.grounding_state
    new = response.sql_grounding_state
    if tool_name == "get_column_meaning":
        if new.domain_knowledge != old.domain_knowledge:
            raise SQLGroundingValidationError(
                "column meaning evidence cannot change domain_knowledge"
            )
        old_phrases = {item.phrase for item in old.column_mapping or ()}
        new_phrases = {item.phrase for item in new.column_mapping or ()}
        if new_phrases - old_phrases:
            raise SQLGroundingValidationError(
                "column meaning evidence cannot create a new mapping phrase"
            )
        if _column_mapping_change_count(old, new) > 1:
            raise SQLGroundingValidationError(
                "one column meaning may repair at most one mapping phrase"
            )
        return
    if tool_name == "get_knowledge_definition":
        if new.column_mapping != old.column_mapping:
            raise SQLGroundingValidationError(
                "knowledge definition evidence cannot change column_mapping"
            )
        old_knowledge = {
            (item.kind, item.content) for item in old.domain_knowledge or ()
        }
        new_knowledge = {
            (item.kind, item.content) for item in new.domain_knowledge or ()
        }
        added = new_knowledge - old_knowledge
        if not old_knowledge.issubset(new_knowledge) or len(
            added
        ) > 1:
            raise SQLGroundingValidationError(
                "one knowledge definition may only preserve or add its exact Official rule"
            )
        if added and added != {
            (
                "business_rule",
                _latest_official_knowledge_definition(latest_tool),
            )
        }:
            raise SQLGroundingValidationError(
                "Check domain_knowledge must materialize the exact latest Official definition"
            )
        return
    if changed:
        raise SQLGroundingValidationError(
            "this Check evidence source cannot change Grounding State"
        )


def _latest_official_knowledge_definition(latest_tool: Mapping[str, Any]) -> str:
    """Return the exact definition carried by the latest Official tool result."""

    raw = latest_tool.get("result")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            pass
    if isinstance(raw, Mapping):
        definition = raw.get("definition")
        if isinstance(definition, str) and definition.strip() == definition:
            return definition
        nested = raw.get("knowledge")
        if isinstance(nested, Mapping):
            definition = nested.get("definition")
            if isinstance(definition, str) and definition.strip() == definition:
                return definition
    raise SQLGroundingValidationError(
        "latest knowledge definition is missing an exact Official definition"
    )


def _column_mapping_change_count(
    old: SQLGroundingState,
    new: SQLGroundingState,
) -> int:
    old_by_phrase = {item.phrase: item for item in old.column_mapping or ()}
    new_by_phrase = {item.phrase: item for item in new.column_mapping or ()}
    return sum(
        old_by_phrase.get(phrase) != new_by_phrase.get(phrase)
        for phrase in old_by_phrase.keys() | new_by_phrase.keys()
    )


def _validate_staged_response(
    runtime: GroundingRuntime | SQLGroundingDraftRuntime,
    response: GroundingLLMResponse | _DraftMaterializedResponse,
    *,
    call_kind: GroundingCallKind,
    grounding_input: Mapping[str, Any] | None,
    draft_cycle: bool = False,
) -> None:
    old = runtime.grounding_state
    new = response.sql_grounding_state
    initial_clarification_cycle = draft_cycle or bool(
        runtime.stage == "INITIAL_GROUNDING"
        and old.all_dimensions_evaluated
        and grounding_input is not None
        and grounding_input.get("user_clarifications")
    )
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
        expected_focus = "none" if initial_clarification_cycle else "column_mapping"
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
        expected_focus = "none" if initial_clarification_cycle else "domain_knowledge"
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
