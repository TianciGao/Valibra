"""Pure data and validation contracts for four-dimensional SQL Grounding V1.

This module performs no I/O and does not import ADK, Provider, database, HTTP,
or benchmark code.  SQL expressions remain strings at the LLM boundary, but
are accepted only after PostgreSQL parsing, a context-specific AST allowlist,
and identifier validation against a transient :class:`ValidationContext`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated, Any, Literal, TypeAlias

import sqlglot
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlglot import exp
from sqlglot.errors import ParseError

SQL_GROUNDING_RUNTIME_KEY = "valibra:sql_grounding_runtime"
LEGACY_GROUNDING_RUNTIME_KEY = "valibra:grounding_runtime"

EXPECTED_SQLGLOT_VERSION = "26.16.4"
SQLGLOT_VERSION = version("sqlglot")
if SQLGLOT_VERSION != EXPECTED_SQLGLOT_VERSION:
    raise RuntimeError(
        "SQL Grounding V1 requires sqlglot "
        f"{EXPECTED_SQLGLOT_VERSION}, found {SQLGLOT_VERSION}"
    )

MAX_TABLES = 64
MAX_JOIN_KEYS = 64
MAX_COLUMN_MAPPINGS = 128
MAX_TARGETS_PER_MAPPING = 16
MAX_DOMAIN_KNOWLEDGE = 64
MAX_IDENTIFIER_CHARS = 128
MAX_EXPRESSION_CHARS = 1024
MAX_PHRASE_CHARS = 256
MAX_KNOWLEDGE_CHARS = 2048
MAX_QUERY_CHARS = 32_768
MAX_TRAJECTORY_REFS = 512

GroundingStage: TypeAlias = Literal[
    "INITIAL_GROUNDING",
    "SQL_ATTEMPT",
    "REPAIR",
    "P2_INCREMENTAL",
    "DONE",
]
FocusDimension: TypeAlias = Literal[
    "tables",
    "join_keys",
    "column_mapping",
    "domain_knowledge",
    "none",
]
DomainKnowledgeKind: TypeAlias = Literal[
    "business_rule",
    "runtime_state",
    "database_capability",
]

_IDENTIFIER_PART = r"[A-Za-z_][A-Za-z0-9_$]*"
_TABLE_IDENTIFIER_RE = re.compile(
    rf"^{_IDENTIFIER_PART}(?:\.{_IDENTIFIER_PART})?$"
)
_CONTROL_TOKEN_RE = re.compile(r";|--|/\*|\*/")

# Frozen from already-completed Full-600 taxonomy findings.  The list is
# deliberately narrow: plain columns, JSON/JSONB paths, PostgreSQL arrays,
# casts, UNNEST/STRING_TO_ARRAY, and JSON expansion functions.
FIELD_EXPRESSION_AST_WHITELIST = frozenset(
    {
        "Anonymous",
        "Bracket",
        "Cast",
        "Column",
        "DataType",
        "Explode",
        "Identifier",
        "JSONExtract",
        "JSONExtractScalar",
        "JSONPath",
        "JSONPathKey",
        "JSONPathRoot",
        "Literal",
        "Paren",
        "Slice",
        "StringToArray",
    }
)
FIELD_EXPRESSION_FUNCTION_WHITELIST = frozenset(
    {
        "JSONB_ARRAY_ELEMENTS",
        "JSONB_ARRAY_ELEMENTS_TEXT",
        "JSONB_EACH",
        "JSONB_EACH_TEXT",
    }
)
ALLOWED_CAST_TYPES = frozenset(
    {
        "BIGINT",
        "BOOLEAN",
        "DATE",
        "DECIMAL",
        "DOUBLE",
        "INT",
        "JSON",
        "JSONB",
        "TEXT",
        "TIMESTAMP",
        "TIMESTAMPTZ",
    }
)

# Frozen relation families: ordinary equality, normalization via LOWER/TRIM,
# range predicates, temporal normalization, and ordinal +/- relationships.
RELATION_EXPRESSION_AST_WHITELIST = frozenset(
    {
        "Add",
        "And",
        "Between",
        "Cast",
        "Column",
        "DataType",
        "EQ",
        "Extract",
        "GT",
        "GTE",
        "Identifier",
        "LT",
        "LTE",
        "Literal",
        "Lower",
        "NEQ",
        "Or",
        "Paren",
        "Sub",
        "TimeToStr",
        "Trim",
        "Var",
    }
)
_RELATION_ROOTS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.And, exp.Or)


class SQLGroundingValidationError(ValueError):
    """A bounded SQL Grounding contract violation."""


class ContractModel(BaseModel):
    """Strict immutable base for all persisted or LLM-facing contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )


def _require_bounded_text(value: str, *, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty with no outer whitespace")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    return value


def _require_table_identifier(value: str) -> str:
    value = _require_bounded_text(
        value,
        label="table identifier",
        maximum=MAX_IDENTIFIER_CHARS,
    )
    if not _TABLE_IDENTIFIER_RE.fullmatch(value):
        raise ValueError("table identifier must be one or two SQL identifier parts")
    return value


def _sorted_unique(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(values))


class ColumnMapping(ContractModel):
    """A verbatim query phrase mapped to validated field expressions."""

    phrase: Annotated[str, Field(min_length=1, max_length=MAX_PHRASE_CHARS)]
    targets: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)], ...],
        Field(min_length=1, max_length=MAX_TARGETS_PER_MAPPING),
    ]

    @field_validator("phrase")
    @classmethod
    def validate_phrase_shape(cls, value: str) -> str:
        return _require_bounded_text(
            value,
            label="column_mapping.phrase",
            maximum=MAX_PHRASE_CHARS,
        )

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for target in value:
            canonicalize_field_expression(target)
        return _sorted_unique(value, label="column_mapping.targets")


class DomainKnowledge(ContractModel):
    """A bounded database-grounding fact from a closed three-kind set."""

    kind: DomainKnowledgeKind
    content: Annotated[str, Field(min_length=1, max_length=MAX_KNOWLEDGE_CHARS)]

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _require_bounded_text(
            value,
            label="domain_knowledge.content",
            maximum=MAX_KNOWLEDGE_CHARS,
        )


class SQLGroundingState(ContractModel):
    """The complete four-dimensional database Grounding state.

    ``None`` means not evaluated, an empty tuple means evaluated and not
    needed, and a non-empty tuple contains the current validated result.
    """

    tables: Annotated[tuple[str, ...], Field(max_length=MAX_TABLES)] | None = None
    join_keys: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)], ...],
        Field(max_length=MAX_JOIN_KEYS),
    ] | None = None
    column_mapping: Annotated[
        tuple[ColumnMapping, ...],
        Field(max_length=MAX_COLUMN_MAPPINGS),
    ] | None = None
    domain_knowledge: Annotated[
        tuple[DomainKnowledge, ...],
        Field(max_length=MAX_DOMAIN_KNOWLEDGE),
    ] | None = None

    @field_validator("tables")
    @classmethod
    def validate_tables(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        for table in value:
            _require_table_identifier(table)
        return _sorted_unique(value, label="tables")

    @field_validator("join_keys")
    @classmethod
    def validate_join_keys(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        for relation in value:
            canonicalize_relation_expression(relation)
        return _sorted_unique(value, label="join_keys")

    @field_validator("column_mapping")
    @classmethod
    def validate_column_mappings(
        cls,
        value: tuple[ColumnMapping, ...] | None,
    ) -> tuple[ColumnMapping, ...] | None:
        if value is None:
            return None
        phrases = [mapping.phrase for mapping in value]
        if len(phrases) != len(set(phrases)):
            raise ValueError("column_mapping must not contain duplicate phrases")
        return tuple(sorted(value, key=lambda mapping: mapping.phrase))

    @field_validator("domain_knowledge")
    @classmethod
    def validate_domain_knowledge(
        cls,
        value: tuple[DomainKnowledge, ...] | None,
    ) -> tuple[DomainKnowledge, ...] | None:
        if value is None:
            return None
        keys = [(item.kind, item.content) for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("domain_knowledge must not contain duplicates")
        return tuple(sorted(value, key=lambda item: (item.kind, item.content)))

    @property
    def all_dimensions_evaluated(self) -> bool:
        return all(
            value is not None
            for value in (
                self.tables,
                self.join_keys,
                self.column_mapping,
                self.domain_knowledge,
            )
        )


class GroundingLLMResponse(ContractModel):
    """The entire state proposed by Grounding LLM plus its next focus."""

    sql_grounding_state: SQLGroundingState
    next_focus_dimension: FocusDimension


class GroundingRuntime(ContractModel):
    """Persisted SQL Grounding runtime; focus is control state, not dimension five."""

    grounding_revision: Annotated[int, Field(ge=0)] = 0
    stage: GroundingStage = "INITIAL_GROUNDING"
    focus_dimension: FocusDimension = "tables"
    grounding_state: SQLGroundingState = Field(default_factory=SQLGroundingState)

    @model_validator(mode="after")
    def validate_initial_focus(self) -> "GroundingRuntime":
        if (
            self.stage == "INITIAL_GROUNDING"
            and not self.grounding_state.all_dimensions_evaluated
            and self.focus_dimension == "none"
        ):
            raise ValueError(
                "INITIAL_GROUNDING cannot focus none while a dimension is null"
            )
        return self


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Transient, non-persisted projection of allowed validation evidence.

    The caller may derive this projection only from the current query, latest
    legal Observation, and completed, paired Official BIRD trajectory entries
    in the current Session.  It deliberately cannot carry GT, test cases,
    hidden follow-up, User Simulator internals, database handles, or network
    clients, and it must never be stored in SQLGroundingState or Runtime.
    """

    current_query: str
    latest_observation_id: str
    official_trajectory_observation_ids: tuple[str, ...] = ()
    follow_up_query: str | None = None
    known_tables: frozenset[str] = frozenset()
    known_columns: frozenset[str] = frozenset()
    table_aliases: tuple[tuple[str, str], ...] = ()
    supported_domain_knowledge: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "official_trajectory_observation_ids",
            tuple(self.official_trajectory_observation_ids),
        )
        object.__setattr__(self, "known_tables", frozenset(self.known_tables))
        object.__setattr__(self, "known_columns", frozenset(self.known_columns))
        object.__setattr__(
            self,
            "table_aliases",
            tuple(tuple(pair) for pair in self.table_aliases),
        )
        object.__setattr__(
            self,
            "supported_domain_knowledge",
            frozenset(tuple(item) for item in self.supported_domain_knowledge),
        )
        _require_bounded_text(
            self.current_query,
            label="ValidationContext.current_query",
            maximum=MAX_QUERY_CHARS,
        )
        _require_bounded_text(
            self.latest_observation_id,
            label="ValidationContext.latest_observation_id",
            maximum=MAX_IDENTIFIER_CHARS,
        )
        if self.follow_up_query is not None:
            _require_bounded_text(
                self.follow_up_query,
                label="ValidationContext.follow_up_query",
                maximum=MAX_QUERY_CHARS,
            )
        if len(self.official_trajectory_observation_ids) > MAX_TRAJECTORY_REFS:
            raise ValueError("too many Official BIRD trajectory references")
        if len(self.official_trajectory_observation_ids) != len(
            set(self.official_trajectory_observation_ids)
        ):
            raise ValueError("Official BIRD trajectory references must be unique")
        for observation_id in self.official_trajectory_observation_ids:
            _require_bounded_text(
                observation_id,
                label="Official BIRD trajectory observation ID",
                maximum=MAX_IDENTIFIER_CHARS,
            )
        for table in self.known_tables:
            _require_table_identifier(table)
        for column in self.known_columns:
            _validate_context_column(column, self.known_tables)
        aliases = dict(self.table_aliases)
        if len(aliases) != len(self.table_aliases):
            raise ValueError("table aliases must be unique")
        for alias, table in self.table_aliases:
            _require_table_identifier(alias)
            if "." in alias:
                raise ValueError("table alias must be a single identifier")
            if table not in self.known_tables:
                raise ValueError("table alias target must be a known table")
        for kind, content in self.supported_domain_knowledge:
            if kind not in {"business_rule", "runtime_state", "database_capability"}:
                raise ValueError("unsupported domain knowledge kind in context")
            _require_bounded_text(
                content,
                label="supported domain knowledge",
                maximum=MAX_KNOWLEDGE_CHARS,
            )

    @property
    def query_texts(self) -> tuple[str, ...]:
        if self.follow_up_query is None:
            return (self.current_query,)
        return (self.current_query, self.follow_up_query)


def _validate_context_column(column: str, known_tables: frozenset[str]) -> None:
    column = _require_bounded_text(
        column,
        label="known column",
        maximum=MAX_IDENTIFIER_CHARS * 2 + 1,
    )
    if "." not in column:
        raise ValueError("known column must be table-qualified")
    table, name = column.rsplit(".", 1)
    _require_table_identifier(table)
    if not re.fullmatch(_IDENTIFIER_PART, name):
        raise ValueError("known column name is not a SQL identifier")
    if table not in known_tables:
        raise ValueError("known column must belong to a known table")


def _parse_expression(value: str, *, label: str) -> exp.Expression:
    value = _require_bounded_text(
        value,
        label=label,
        maximum=MAX_EXPRESSION_CHARS,
    )
    if _CONTROL_TOKEN_RE.search(value):
        raise SQLGroundingValidationError(
            f"{label} cannot contain semicolons or SQL comments"
        )
    try:
        statements = sqlglot.parse(value, read="postgres")
    except ParseError as exc:
        raise SQLGroundingValidationError(f"{label} is not valid PostgreSQL") from exc
    if len(statements) != 1 or statements[0] is None:
        raise SQLGroundingValidationError(f"{label} must contain exactly one expression")
    return statements[0]


def _require_canonical(value: str, expression: exp.Expression, *, label: str) -> str:
    canonical = expression.sql(dialect="postgres")
    if value != canonical:
        raise SQLGroundingValidationError(
            f"{label} must use canonical PostgreSQL expression form: {canonical}"
        )
    return canonical


def _validate_ast(
    expression: exp.Expression,
    *,
    allowed_nodes: frozenset[str],
    label: str,
) -> None:
    for node in expression.walk():
        node_name = type(node).__name__
        if node_name not in allowed_nodes:
            raise SQLGroundingValidationError(
                f"{label} contains unapproved AST node {node_name}"
            )


def _validate_cast_types(expression: exp.Expression, *, label: str) -> None:
    for data_type in expression.find_all(exp.DataType):
        cast_type = getattr(data_type.this, "value", str(data_type.this))
        if cast_type not in ALLOWED_CAST_TYPES:
            raise SQLGroundingValidationError(
                f"{label} contains unapproved cast type {cast_type}"
            )


def canonicalize_field_expression(value: str) -> str:
    """Validate and return an already-canonical field expression."""

    expression = _parse_expression(value, label="field expression")
    if type(expression).__name__ not in {
        "Anonymous",
        "Bracket",
        "Cast",
        "Column",
        "Explode",
        "JSONExtract",
        "JSONExtractScalar",
    }:
        raise SQLGroundingValidationError("field expression has an unapproved root")
    _validate_ast(
        expression,
        allowed_nodes=FIELD_EXPRESSION_AST_WHITELIST,
        label="field expression",
    )
    for function in expression.find_all(exp.Anonymous):
        if function.name.upper() not in FIELD_EXPRESSION_FUNCTION_WHITELIST:
            raise SQLGroundingValidationError(
                f"field expression contains unapproved function {function.name}"
            )
    _validate_cast_types(expression, label="field expression")
    return _require_canonical(value, expression, label="field expression")


def canonicalize_relation_expression(value: str) -> str:
    """Validate and return an already-canonical relation expression."""

    expression = _parse_expression(value, label="relation expression")
    if not isinstance(expression, _RELATION_ROOTS):
        raise SQLGroundingValidationError(
            "relation expression must be a comparison or bounded boolean relation"
        )
    _validate_ast(
        expression,
        allowed_nodes=RELATION_EXPRESSION_AST_WHITELIST,
        label="relation expression",
    )
    _validate_cast_types(expression, label="relation expression")
    qualifiers = {column.table for column in expression.find_all(exp.Column) if column.table}
    if len(qualifiers) < 2:
        raise SQLGroundingValidationError(
            "relation expression must relate at least two qualified records"
        )
    for comparison in expression.find_all(
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.Between,
    ):
        comparison_qualifiers = {
            column.table
            for column in comparison.find_all(exp.Column)
            if column.table
        }
        if len(comparison_qualifiers) < 2:
            raise SQLGroundingValidationError(
                "each relation comparison must connect two qualified records"
            )
    return _require_canonical(value, expression, label="relation expression")


def _resolve_column(column: exp.Column, context: ValidationContext) -> str:
    name = column.name
    qualifier = column.table
    aliases = dict(context.table_aliases)
    if qualifier:
        table = aliases.get(qualifier, qualifier)
        candidate = f"{table}.{name}"
        if candidate not in context.known_columns:
            raise SQLGroundingValidationError(
                f"expression references unknown column {candidate}"
            )
        return candidate
    matches = sorted(
        known for known in context.known_columns if known.rsplit(".", 1)[1] == name
    )
    if len(matches) != 1:
        raise SQLGroundingValidationError(
            f"unqualified column {name} is unknown or ambiguous"
        )
    return matches[0]


def _validate_expression_identifiers(
    value: str,
    *,
    context: ValidationContext,
    relation: bool,
) -> None:
    expression = _parse_expression(
        value,
        label="relation expression" if relation else "field expression",
    )
    resolved = [_resolve_column(column, context) for column in expression.find_all(exp.Column)]
    if not resolved:
        raise SQLGroundingValidationError("expression must reference a known column")


def validate_sql_grounding_state(
    state: SQLGroundingState,
    context: ValidationContext,
) -> SQLGroundingState:
    """Validate State only against the explicitly allowed transient evidence."""

    if state.tables is not None:
        for table in state.tables:
            if table not in context.known_tables:
                raise SQLGroundingValidationError(f"unknown table {table}")
    if state.join_keys is not None:
        for relation in state.join_keys:
            _validate_expression_identifiers(relation, context=context, relation=True)
    if state.column_mapping is not None:
        for mapping in state.column_mapping:
            if not any(mapping.phrase in source for source in context.query_texts):
                raise SQLGroundingValidationError(
                    f"phrase is not a verbatim query substring: {mapping.phrase}"
                )
            for target in mapping.targets:
                _validate_expression_identifiers(target, context=context, relation=False)
    if state.domain_knowledge is not None:
        supported = context.supported_domain_knowledge
        for knowledge in state.domain_knowledge:
            if (knowledge.kind, knowledge.content) not in supported:
                raise SQLGroundingValidationError(
                    "domain knowledge lacks an allowed Official BIRD evidence source"
                )
    return state


def validate_grounding_llm_response(
    response: GroundingLLMResponse,
    *,
    stage: GroundingStage,
    context: ValidationContext,
) -> GroundingLLMResponse:
    """Validate an offline LLM response contract for the supplied Stage."""

    validate_sql_grounding_state(response.sql_grounding_state, context)
    if (
        stage == "INITIAL_GROUNDING"
        and not response.sql_grounding_state.all_dimensions_evaluated
        and response.next_focus_dimension == "none"
    ):
        raise SQLGroundingValidationError(
            "INITIAL_GROUNDING cannot return none while a dimension is null"
        )
    return response


def canonical_json(value: BaseModel | Any) -> str:
    """Return deterministic UTF-8 JSON for a JSON-safe contract value."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sql_grounding_state_sha256(state: SQLGroundingState) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def validate_grounding_runtime_transition(
    previous: GroundingRuntime,
    current: GroundingRuntime,
) -> GroundingRuntime:
    """Freeze revision semantics without implementing a state-update service."""

    state_changed = sql_grounding_state_sha256(previous.grounding_state) != (
        sql_grounding_state_sha256(current.grounding_state)
    )
    expected_revision = previous.grounding_revision + int(state_changed)
    if current.grounding_revision != expected_revision:
        raise SQLGroundingValidationError(
            "grounding_revision must increase exactly once iff canonical State changes"
        )
    return current
