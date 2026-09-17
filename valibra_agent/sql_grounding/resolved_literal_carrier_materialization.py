"""Deterministic materialization for one verified resolved-literal proposal.

``CHECK_RESOLVED_LITERAL_CARRIER_MATERIALIZATION_R1`` remains isolated from
authoritative Check, Atomic re-Grounding and Main wiring.  It converts a valid
Shadow proposal into a typed execution constraint, and models commit/rollback
lifecycles without persisting anything in four-dimensional Grounding State.

The execution projection is deliberately narrow.  A direct column or existing
``->>`` target is retained verbatim.  A JSON target ending in ``->`` may change
only its outermost operator to ``->>`` after the exact Official terminal leaf
has supplied a finite ``Possible values`` enum.  The Provider never supplies
the comparison expression or its authority.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal, Mapping

import sqlglot
from pydantic import Field, field_validator, model_validator
from sqlglot import exp

from valibra_agent.sql_grounding.models import (
    ContractModel,
    MAX_EXPRESSION_CHARS,
    MAX_PHRASE_CHARS,
    SQLGroundingState,
    canonical_json,
    canonicalize_field_expression,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
    MAX_RESOLVED_LITERAL_CHARS,
    ResolvedLiteralShadowValidation,
    _NEGATIVE_TOKENS,
    _TOKEN_RE,
    _literal_matches_phrase,
    _possible_values,
    _target_description,
    _target_reference,
)


MAX_ATOMIC_DRAFT_ID_CHARS = 256
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ResolvedLiteralCarrierMaterializationError(ValueError):
    """A deterministic proposal, evidence, target, or lifecycle rejection."""


class ResolvedLiteralExecutableCarrier(ContractModel):
    """Runtime-authored constraint, inactive until its Draft commits."""

    task_id: Annotated[str, Field(min_length=1, max_length=256)]
    atomic_draft_id: Annotated[
        str,
        Field(min_length=1, max_length=MAX_ATOMIC_DRAFT_ID_CHARS),
    ]
    phase: Literal[1, 2]
    state_revision: Annotated[int, Field(ge=0)]
    state_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    phrase: Annotated[str, Field(min_length=1, max_length=MAX_PHRASE_CHARS)]
    source_target: Annotated[
        str,
        Field(min_length=1, max_length=MAX_EXPRESSION_CHARS),
    ]
    comparison_target: Annotated[
        str,
        Field(min_length=1, max_length=MAX_EXPRESSION_CHARS),
    ]
    literal: Annotated[
        str,
        Field(min_length=1, max_length=MAX_RESOLVED_LITERAL_CHARS),
    ]
    operator: Literal["EXACT_EQUALITY"] = "EXACT_EQUALITY"
    authority: Literal[
        "query_lexical_literal+official_column_meaning"
    ] = "query_lexical_literal+official_column_meaning"
    source_request_digest: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    source_result_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]

    @field_validator("phrase", "literal", "task_id", "atomic_draft_id")
    @classmethod
    def validate_bounded_text(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("carrier text must have no outer whitespace or controls")
        return value

    @field_validator("source_target", "comparison_target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        canonicalize_field_expression(value)
        return value

    @model_validator(mode="after")
    def validate_execution_projection(self) -> "ResolvedLiteralExecutableCarrier":
        source = sqlglot.parse_one(self.source_target, read="postgres")
        comparison = sqlglot.parse_one(self.comparison_target, read="postgres")
        if _target_reference(source) != _target_reference(comparison):
            raise ValueError("comparison target must retain the exact source leaf")
        if isinstance(source, (exp.Column, exp.JSONExtractScalar)):
            if self.comparison_target != self.source_target:
                raise ValueError("already-scalar target must remain byte-identical")
        elif isinstance(source, exp.JSONExtract):
            if not isinstance(comparison, exp.JSONExtractScalar):
                raise ValueError("JSON comparison target must extract scalar text")
            expected = _json_scalar_projection(source)
            if comparison.sql(dialect="postgres") != expected:
                raise ValueError("only the terminal JSON operator may become scalar")
        else:
            raise ValueError("carrier source target is not a direct scalar or JSON leaf")
        return self


class ResolvedLiteralCarrierLifecycle(ContractModel):
    """One immutable, Draft-local lifecycle; no Main renderer is provided."""

    status: Literal["DRAFT", "COMMITTED", "ROLLED_BACK"]
    atomic_draft_id: Annotated[
        str,
        Field(min_length=1, max_length=MAX_ATOMIC_DRAFT_ID_CHARS),
    ]
    task_id: Annotated[str, Field(min_length=1, max_length=256)]
    phase: Literal[1, 2]
    base_revision: Annotated[int, Field(ge=0)]
    base_state_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    draft_revision: Annotated[int, Field(ge=0)]
    draft_state_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    committed_revision: Annotated[int, Field(ge=0)] | None = None
    staged_carriers: Annotated[
        tuple[ResolvedLiteralExecutableCarrier, ...],
        Field(max_length=1),
    ] = ()
    active_carriers: Annotated[
        tuple[ResolvedLiteralExecutableCarrier, ...],
        Field(max_length=1),
    ] = ()

    @model_validator(mode="after")
    def validate_lifecycle_shape(self) -> "ResolvedLiteralCarrierLifecycle":
        if self.draft_revision < self.base_revision:
            raise ValueError("Draft revision predates the formal base revision")
        for item in self.staged_carriers + self.active_carriers:
            if (
                item.atomic_draft_id != self.atomic_draft_id
                or item.task_id != self.task_id
                or item.phase != self.phase
            ):
                raise ValueError("carrier identity differs from its lifecycle")
        if self.status == "DRAFT":
            if (
                len(self.staged_carriers) != 1
                or self.active_carriers
                or self.committed_revision is not None
            ):
                raise ValueError("Draft lifecycle must contain one inactive carrier")
            if any(
                item.state_revision != self.draft_revision
                or item.state_sha256 != self.draft_state_sha256
                for item in self.staged_carriers
            ):
                raise ValueError("staged carrier is not bound to Draft State")
        elif self.status == "COMMITTED":
            if (
                self.staged_carriers
                or len(self.active_carriers) != 1
                or self.committed_revision is None
            ):
                raise ValueError("committed lifecycle must contain one active carrier")
            if any(
                item.state_revision != self.committed_revision
                or item.state_sha256 != self.draft_state_sha256
                for item in self.active_carriers
            ):
                raise ValueError("active carrier is not bound to committed State")
        elif (
            self.staged_carriers
            or self.active_carriers
            or self.committed_revision is not None
        ):
            raise ValueError("rollback must destroy staged and active carriers")
        return self


def materialize_resolved_literal_carrier(
    validation: ResolvedLiteralShadowValidation,
    *,
    grounding_input: Mapping[str, Any],
    atomic_draft_id: str,
    task_id: str,
    phase: Literal[1, 2],
    state_revision: int,
) -> ResolvedLiteralExecutableCarrier | None:
    """Revalidate one Shadow result and materialize an inactive carrier."""

    if validation.verdict == "OMITTED":
        return None
    if validation.verdict != "VALID" or validation.carrier is None:
        raise ResolvedLiteralCarrierMaterializationError(
            "only a valid Shadow proposal may be materialized"
        )
    shadow = validation.carrier
    if (
        shadow.task_id != task_id
        or shadow.phase != phase
        or shadow.grounding_revision != state_revision
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "Shadow proposal task, phase, or revision is stale"
        )
    try:
        state = SQLGroundingState.model_validate(grounding_input.get("current_state"))
    except Exception as exc:
        raise ResolvedLiteralCarrierMaterializationError(
            "current Draft State is invalid"
        ) from exc
    state_sha = sql_grounding_state_sha256(state)
    if shadow.state_sha256 != state_sha:
        raise ResolvedLiteralCarrierMaterializationError(
            "Shadow proposal State digest is stale"
        )
    mappings = tuple(
        item for item in state.column_mapping or () if item.phrase == shadow.phrase
    )
    if (
        len(mappings) != 1
        or len(mappings[0].targets) != 1
        or mappings[0].targets[0] != shadow.target
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "Shadow source target is not the exact current Draft mapping"
        )
    query_parts = [grounding_input.get("query")]
    if phase == 2:
        query_parts.append(grounding_input.get("follow_up"))
    if not any(
        isinstance(item, str) and shadow.phrase in item for item in query_parts
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "Shadow phrase is no longer a verbatim Query span"
        )
    phrase_words = {
        token.casefold() for token in _TOKEN_RE.findall(shadow.phrase)
    }
    if phrase_words & _NEGATIVE_TOKENS:
        raise ResolvedLiteralCarrierMaterializationError(
            "negative or exclusion phrase cannot become exact equality"
        )

    latest_tool = grounding_input.get("latest_tool")
    if not isinstance(latest_tool, Mapping) or latest_tool.get("name") != (
        "get_column_meaning"
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "current Official evidence is not get_column_meaning"
        )
    arguments = latest_tool.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "table_name",
        "column_name",
    }:
        raise ResolvedLiteralCarrierMaterializationError(
            "current Official request arguments are invalid"
        )
    request_digest = _sha256(
        canonical_json(
            {
                "arguments": dict(arguments),
                "name": "get_column_meaning",
            }
        )
    )
    raw_result = latest_tool.get("result")
    result_text = raw_result if isinstance(raw_result, str) else canonical_json(raw_result)
    result_sha = _sha256(result_text)
    if (
        shadow.source_request_digest != request_digest
        or shadow.source_result_sha256 != result_sha
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "Official evidence provenance is stale or tampered"
        )

    try:
        expression = sqlglot.parse_one(shadow.target, read="postgres")
        table, column, path = _target_reference(expression)
    except Exception as exc:
        raise ResolvedLiteralCarrierMaterializationError(
            "source target is not one canonical mapped leaf"
        ) from exc
    if (
        arguments.get("table_name") != table
        or arguments.get("column_name") != column
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "Official evidence does not belong to source target"
        )
    try:
        description = _target_description(raw_result, path=path)
        enum_values = _possible_values(description)
    except ValueError as exc:
        raise ResolvedLiteralCarrierMaterializationError(
            "terminal target is not a bounded scalar enum leaf"
        ) from exc
    if shadow.literal not in enum_values:
        raise ResolvedLiteralCarrierMaterializationError(
            "literal is not an exact current Official enum member"
        )
    lexical_matches = tuple(
        value
        for value in enum_values
        if _literal_matches_phrase(shadow.phrase, value)
    )
    if lexical_matches != (shadow.literal,):
        raise ResolvedLiteralCarrierMaterializationError(
            "Query-to-enum lexical authority is no longer unique"
        )

    comparison_target = _comparison_target(expression, shadow.target)
    return ResolvedLiteralExecutableCarrier(
        task_id=task_id,
        atomic_draft_id=atomic_draft_id,
        phase=phase,
        state_revision=state_revision,
        state_sha256=state_sha,
        phrase=shadow.phrase,
        source_target=shadow.target,
        comparison_target=comparison_target,
        literal=shadow.literal,
        source_request_digest=request_digest,
        source_result_sha256=result_sha,
    )


def begin_resolved_literal_carrier_draft(
    carrier: ResolvedLiteralExecutableCarrier,
    *,
    base_revision: int,
    base_state_sha256: str,
) -> ResolvedLiteralCarrierLifecycle:
    """Stage one carrier without making it available to a consumer."""

    if base_revision < 0 or carrier.state_revision < base_revision:
        raise ResolvedLiteralCarrierMaterializationError(
            "carrier Draft revision is invalid"
        )
    return ResolvedLiteralCarrierLifecycle(
        status="DRAFT",
        atomic_draft_id=carrier.atomic_draft_id,
        task_id=carrier.task_id,
        phase=carrier.phase,
        base_revision=base_revision,
        base_state_sha256=base_state_sha256,
        draft_revision=carrier.state_revision,
        draft_state_sha256=carrier.state_sha256,
        staged_carriers=(carrier,),
    )


def commit_resolved_literal_carrier_draft(
    lifecycle: ResolvedLiteralCarrierLifecycle,
    *,
    task_id: str,
    phase: Literal[1, 2],
    formal_revision_before_commit: int,
    formal_state_before_commit: SQLGroundingState,
    committed_revision: int,
    committed_state: SQLGroundingState,
) -> ResolvedLiteralCarrierLifecycle:
    """Activate a carrier only alongside the exact atomic State commit."""

    if lifecycle.status != "DRAFT":
        raise ResolvedLiteralCarrierMaterializationError(
            "only a live Draft carrier may commit"
        )
    if lifecycle.task_id != task_id or lifecycle.phase != phase:
        raise ResolvedLiteralCarrierMaterializationError(
            "carrier Draft task or phase does not match commit"
        )
    if (
        formal_revision_before_commit != lifecycle.base_revision
        or sql_grounding_state_sha256(formal_state_before_commit)
        != lifecycle.base_state_sha256
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "formal State changed before carrier commit"
        )
    committed_sha = sql_grounding_state_sha256(committed_state)
    if committed_sha != lifecycle.draft_state_sha256:
        raise ResolvedLiteralCarrierMaterializationError(
            "carrier Draft State differs from committed State"
        )
    expected_revision = lifecycle.base_revision + int(
        committed_sha != lifecycle.base_state_sha256
    )
    if committed_revision != expected_revision:
        raise ResolvedLiteralCarrierMaterializationError(
            "carrier commit revision does not match atomic State semantics"
        )
    for item in lifecycle.staged_carriers:
        mappings = tuple(
            mapping
            for mapping in committed_state.column_mapping or ()
            if mapping.phrase == item.phrase
        )
        if (
            len(mappings) != 1
            or len(mappings[0].targets) != 1
            or mappings[0].targets[0] != item.source_target
        ):
            raise ResolvedLiteralCarrierMaterializationError(
                "committed State no longer contains the carrier source mapping"
            )
    active = tuple(
        ResolvedLiteralExecutableCarrier.model_validate(
            {
                **item.model_dump(mode="json"),
                "state_revision": committed_revision,
                "state_sha256": committed_sha,
            }
        )
        for item in lifecycle.staged_carriers
    )
    return ResolvedLiteralCarrierLifecycle(
        status="COMMITTED",
        atomic_draft_id=lifecycle.atomic_draft_id,
        task_id=lifecycle.task_id,
        phase=lifecycle.phase,
        base_revision=lifecycle.base_revision,
        base_state_sha256=lifecycle.base_state_sha256,
        draft_revision=lifecycle.draft_revision,
        draft_state_sha256=lifecycle.draft_state_sha256,
        committed_revision=committed_revision,
        active_carriers=active,
    )


def rollback_resolved_literal_carrier_draft(
    lifecycle: ResolvedLiteralCarrierLifecycle,
) -> ResolvedLiteralCarrierLifecycle:
    """Destroy the Draft-local proposal and expose no active carrier."""

    if lifecycle.status != "DRAFT":
        raise ResolvedLiteralCarrierMaterializationError(
            "only a live Draft carrier may roll back"
        )
    return ResolvedLiteralCarrierLifecycle(
        status="ROLLED_BACK",
        atomic_draft_id=lifecycle.atomic_draft_id,
        task_id=lifecycle.task_id,
        phase=lifecycle.phase,
        base_revision=lifecycle.base_revision,
        base_state_sha256=lifecycle.base_state_sha256,
        draft_revision=lifecycle.draft_revision,
        draft_state_sha256=lifecycle.draft_state_sha256,
    )


def active_resolved_literal_carriers(
    lifecycle: ResolvedLiteralCarrierLifecycle,
    *,
    task_id: str,
    phase: Literal[1, 2],
    state_revision: int,
    state: SQLGroundingState,
) -> tuple[ResolvedLiteralExecutableCarrier, ...]:
    """Return committed carriers only for their exact task, phase and State."""

    if lifecycle.status != "COMMITTED":
        return ()
    if lifecycle.task_id != task_id or lifecycle.phase != phase:
        return ()
    state_sha = sql_grounding_state_sha256(state)
    if (
        lifecycle.committed_revision != state_revision
        or lifecycle.draft_state_sha256 != state_sha
    ):
        raise ResolvedLiteralCarrierMaterializationError(
            "committed literal carrier is stale for current State"
        )
    return lifecycle.active_carriers


def _comparison_target(expression: exp.Expression, source_target: str) -> str:
    if isinstance(expression, (exp.Column, exp.JSONExtractScalar)):
        return source_target
    if isinstance(expression, exp.JSONExtract):
        comparison = _json_scalar_projection(expression)
        canonicalize_field_expression(comparison)
        return comparison
    raise ResolvedLiteralCarrierMaterializationError(
        "source target is not scalar-projectable"
    )


def _json_scalar_projection(expression: exp.JSONExtract) -> str:
    scalar = exp.JSONExtractScalar(
        this=expression.this.copy(),
        expression=expression.expression.copy(),
        only_json_types=expression.args.get("only_json_types"),
    )
    return scalar.sql(dialect="postgres")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
