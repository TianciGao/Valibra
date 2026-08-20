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
GroundingDimension: TypeAlias = Literal[
    "tables",
    "join_keys",
    "column_mapping",
    "domain_knowledge",
]
DomainKnowledgeKind: TypeAlias = Literal[
    "business_rule",
    "runtime_state",
    "database_capability",
]

GROUNDING_DIMENSIONS: tuple[GroundingDimension, ...] = (
    "tables",
    "join_keys",
    "column_mapping",
    "domain_knowledge",
)

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
CORRELATED_RELATION_AST_WHITELIST = frozenset(
    {
        "Column",
        "EQ",
        "From",
        "Identifier",
        "Max",
        "Select",
        "Subquery",
        "Table",
        "TableAlias",
        "Where",
    }
)
SAME_TABLE_MULTI_RECORD_AST_WHITELIST = frozenset(
    {
        "Column",
        "Identifier",
        "Lag",
        "NEQ",
        "Order",
        "Ordered",
        "Window",
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

    @model_validator(mode="after")
    def validate_cross_dimension_references(self) -> "SQLGroundingState":
        referenced_tables: set[str] = set()
        for relation in self.join_keys or ():
            referenced_tables.update(_relation_real_table_references(relation))
        for mapping in self.column_mapping or ():
            for target in mapping.targets:
                referenced_tables.update(_field_real_table_references(target))

        declared_tables = set(self.tables or ())
        undeclared = sorted(referenced_tables - declared_tables)
        if undeclared:
            raise ValueError(
                "field and relation expressions reference tables absent from "
                f"SQLGroundingState.tables: {', '.join(undeclared)}"
            )
        return self


class GroundingLLMResponse(ContractModel):
    """The entire state proposed by Grounding LLM plus its next focus."""

    sql_grounding_state: SQLGroundingState
    next_focus_dimension: FocusDimension


class StateDiffAuthorization(ContractModel):
    """Stage-scoped permission to change complete State dimensions.

    This is not a Patch or Evidence store.  A later Service must derive the
    authorized dimensions from the current legal Observation before using this
    pure contract.
    """

    stage: GroundingStage
    authorized_dimensions: tuple[GroundingDimension, ...] = ()

    @field_validator("authorized_dimensions")
    @classmethod
    def validate_authorized_dimensions(
        cls,
        value: tuple[GroundingDimension, ...],
    ) -> tuple[GroundingDimension, ...]:
        if len(value) != len(set(value)):
            raise ValueError("authorized_dimensions must not contain duplicates")
        order = {dimension: index for index, dimension in enumerate(GROUNDING_DIMENSIONS)}
        return tuple(sorted(value, key=order.__getitem__))


class GroundingRuntime(ContractModel):
    """Persisted SQL Grounding runtime; focus is control state, not dimension five."""

    grounding_revision: Annotated[int, Field(ge=0)] = 0
    stage: GroundingStage = "INITIAL_GROUNDING"
    focus_dimension: FocusDimension = "tables"
    grounding_state: SQLGroundingState = Field(default_factory=SQLGroundingState)

    @model_validator(mode="after")
    def validate_initial_focus(self) -> "GroundingRuntime":
        if self.stage == "INITIAL_GROUNDING":
            if (
                not self.grounding_state.all_dimensions_evaluated
                and self.focus_dimension == "none"
            ):
                raise ValueError(
                    "INITIAL_GROUNDING cannot focus none while a dimension is null"
                )
            if (
                self.grounding_state.all_dimensions_evaluated
                and self.focus_dimension != "none"
            ):
                raise ValueError(
                    "INITIAL_GROUNDING must focus none after all dimensions are evaluated"
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
    supported_json_paths: frozenset[tuple[str, tuple[str, ...]]] = frozenset()
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
            "supported_json_paths",
            frozenset(
                (str(column), tuple(path))
                for column, path in self.supported_json_paths
            ),
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
        for column, path in self.supported_json_paths:
            if column not in self.known_columns:
                raise ValueError("supported JSON path must belong to a known column")
            if not path:
                raise ValueError("supported JSON path must contain at least one key")
            for key in path:
                _require_bounded_text(
                    key,
                    label="supported JSON path key",
                    maximum=MAX_IDENTIFIER_CHARS,
                )
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


def _qualified_column_table(column: exp.Column) -> str:
    parts = [part for part in (column.catalog, column.db, column.table) if part]
    if not parts:
        raise SQLGroundingValidationError(
            f"column {column.name} must use a real, fully qualified table identifier"
        )
    if len(parts) > 2:
        raise SQLGroundingValidationError("three-part table identifiers are not allowed")
    table = ".".join(parts)
    _require_table_identifier(table)
    return table


def _table_expression_identifier(table: exp.Table) -> str:
    parts = [part for part in (table.catalog, table.db, table.name) if part]
    if not parts or len(parts) > 2:
        raise SQLGroundingValidationError(
            "correlated FROM must name one real table identifier"
        )
    identifier = ".".join(parts)
    _require_table_identifier(identifier)
    return identifier


@dataclass(frozen=True, slots=True)
class _CorrelatedRelationShape:
    inner_table: str
    inner_alias: str
    root_outer_column: exp.Column
    projected_inner_column: exp.Column
    predicate_inner_column: exp.Column
    predicate_outer_column: exp.Column


def _require_correlated_relation_shape(
    expression: exp.Expression,
) -> _CorrelatedRelationShape:
    """Accept only the two scalar correlated shapes found in the Full-600 audit.

    Supported projections are either one locally-qualified column (the
    ``planets_data_M_6`` lookup shape) or ``MAX`` over one locally-qualified
    column (the ``mental_health_M_6`` latest-record shape).  The subquery must
    contain exactly one explicitly aliased table and one equality correlation.
    """

    if not isinstance(expression, exp.EQ):
        raise SQLGroundingValidationError(
            "correlated relation must be a top-level equality"
        )
    if not isinstance(expression.left, exp.Column) or not isinstance(
        expression.right, exp.Subquery
    ):
        raise SQLGroundingValidationError(
            "correlated relation must be outer_table.column = (SELECT ...)"
        )
    if len(list(expression.find_all(exp.Subquery))) != 1 or len(
        list(expression.find_all(exp.Select))
    ) != 1:
        raise SQLGroundingValidationError(
            "correlated relation must contain exactly one scalar SELECT"
        )

    select = expression.right.this
    if not isinstance(select, exp.Select):
        raise SQLGroundingValidationError("correlated subquery must contain SELECT")
    allowed_select_args = {
        "kind",
        "hint",
        "distinct",
        "expressions",
        "limit",
        "operation_modifiers",
        "from",
        "where",
    }
    if any(
        value not in (None, [], False)
        for key, value in select.args.items()
        if key not in allowed_select_args
    ):
        raise SQLGroundingValidationError(
            "correlated SELECT contains an unapproved clause"
        )
    for key in ("kind", "hint", "distinct", "limit", "operation_modifiers"):
        if select.args.get(key) not in (None, [], False):
            raise SQLGroundingValidationError(
                "correlated SELECT cannot use modifiers, DISTINCT, or LIMIT"
            )
    if len(select.expressions) != 1:
        raise SQLGroundingValidationError(
            "correlated SELECT must project exactly one field"
        )

    from_clause = select.args.get("from")
    if not isinstance(from_clause, exp.From) or not isinstance(
        from_clause.this, exp.Table
    ):
        raise SQLGroundingValidationError(
            "correlated SELECT must read exactly one real table"
        )
    if from_clause.expressions or len(list(select.find_all(exp.Table))) != 1:
        raise SQLGroundingValidationError(
            "correlated SELECT cannot contain joins or additional tables"
        )
    inner_table_expression = from_clause.this
    inner_table = _table_expression_identifier(inner_table_expression)
    inner_alias = inner_table_expression.alias
    if not inner_alias or not re.fullmatch(_IDENTIFIER_PART, inner_alias):
        raise SQLGroundingValidationError(
            "correlated FROM table must declare one explicit local alias"
        )
    if inner_alias == inner_table.rsplit(".", 1)[-1]:
        raise SQLGroundingValidationError(
            "correlated local alias must be distinct from the real table name"
        )
    alias_expression = inner_table_expression.args.get("alias")
    if not isinstance(alias_expression, exp.TableAlias) or alias_expression.columns:
        raise SQLGroundingValidationError(
            "correlated local alias cannot declare a column list"
        )

    projection = select.expressions[0]
    if isinstance(projection, exp.Column):
        projected_column = projection
    elif isinstance(projection, exp.Max):
        projected_column = projection.this
        if not isinstance(projected_column, exp.Column):
            raise SQLGroundingValidationError(
                "correlated MAX must wrap exactly one field"
            )
        if len(list(projection.find_all(exp.Column))) != 1:
            raise SQLGroundingValidationError(
                "correlated MAX must wrap exactly one field"
            )
    else:
        raise SQLGroundingValidationError(
            "correlated SELECT supports only a field or MAX(field) projection"
        )
    if _qualified_column_table(projected_column) != inner_alias:
        raise SQLGroundingValidationError(
            "correlated projection must use its locally declared alias"
        )

    where = select.args.get("where")
    if not isinstance(where, exp.Where) or not isinstance(where.this, exp.EQ):
        raise SQLGroundingValidationError(
            "correlated SELECT must contain exactly one equality WHERE predicate"
        )
    if not isinstance(where.this.left, exp.Column) or not isinstance(
        where.this.right, exp.Column
    ):
        raise SQLGroundingValidationError(
            "correlation predicate must connect two qualified columns"
        )
    predicate_columns = (where.this.left, where.this.right)
    inner_columns = [
        column
        for column in predicate_columns
        if _qualified_column_table(column) == inner_alias
    ]
    outer_columns = [
        column
        for column in predicate_columns
        if _qualified_column_table(column) != inner_alias
    ]
    if len(inner_columns) != 1 or len(outer_columns) != 1:
        raise SQLGroundingValidationError(
            "correlation predicate must connect the local alias to one outer table"
        )

    root_outer_table = _qualified_column_table(expression.left)
    predicate_outer_table = _qualified_column_table(outer_columns[0])
    if inner_alias in {root_outer_table, predicate_outer_table}:
        raise SQLGroundingValidationError(
            "local alias cannot stand in for an outer real table"
        )

    return _CorrelatedRelationShape(
        inner_table=inner_table,
        inner_alias=inner_alias,
        root_outer_column=expression.left,
        projected_inner_column=projected_column,
        predicate_inner_column=inner_columns[0],
        predicate_outer_column=outer_columns[0],
    )


def _require_same_table_multi_record_shape(
    expression: exp.Expression,
) -> None:
    """Accept only the audited adjacent-record window relation.

    Full-600 audit cases ``planets_data_11`` and ``planets_data_M_9`` pair
    adjacent rows within one host partition after ordering by orbital period.
    ``LAG`` makes the previous record instance explicit without a hidden alias,
    SELECT, JOIN fragment, or derived ``rn`` identifier in persisted State.
    """

    if not isinstance(expression, exp.NEQ):
        raise SQLGroundingValidationError(
            "same-table multi-record relation must compare LAG(record_id) "
            "with the current record_id"
        )
    if not isinstance(expression.left, exp.Window) or not isinstance(
        expression.right, exp.Column
    ):
        raise SQLGroundingValidationError(
            "same-table multi-record relation must be "
            "LAG(table.record_id) OVER (...) <> table.record_id"
        )

    window = expression.left
    if len(list(expression.find_all(exp.Window))) != 1 or len(
        list(expression.find_all(exp.Lag))
    ) != 1:
        raise SQLGroundingValidationError(
            "same-table multi-record relation must contain exactly one LAG window"
        )
    if any(
        value not in (None, [], False, "OVER")
        for key, value in window.args.items()
        if key not in {"this", "partition_by", "order", "over"}
    ):
        raise SQLGroundingValidationError(
            "same-table multi-record window contains an unapproved clause"
        )

    lag = window.this
    if not isinstance(lag, exp.Lag) or not isinstance(lag.this, exp.Column):
        raise SQLGroundingValidationError(
            "same-table multi-record window must apply LAG to one record identifier"
        )
    if any(
        value not in (None, [], False)
        for key, value in lag.args.items()
        if key != "this"
    ):
        raise SQLGroundingValidationError(
            "same-table multi-record LAG cannot use offset or default arguments"
        )
    previous_record_id = lag.this
    current_record_id = expression.right
    if previous_record_id.sql(dialect="postgres") != current_record_id.sql(
        dialect="postgres"
    ):
        raise SQLGroundingValidationError(
            "LAG and current record must use the same real record identifier"
        )

    partition_by = window.args.get("partition_by")
    if not isinstance(partition_by, list) or len(partition_by) != 1 or not isinstance(
        partition_by[0], exp.Column
    ):
        raise SQLGroundingValidationError(
            "same-table multi-record window requires exactly one partition column"
        )
    order = window.args.get("order")
    if not isinstance(order, exp.Order) or len(order.expressions) != 1:
        raise SQLGroundingValidationError(
            "same-table multi-record window requires exactly one order column"
        )
    ordered = order.expressions[0]
    if not isinstance(ordered, exp.Ordered) or not isinstance(
        ordered.this, exp.Column
    ):
        raise SQLGroundingValidationError(
            "same-table multi-record window order must be one real column"
        )
    if ordered.args.get("desc") not in (None, False) or ordered.args.get(
        "nulls_first"
    ) not in (None, False):
        raise SQLGroundingValidationError(
            "same-table multi-record window must use canonical ascending order"
        )

    record_table = _qualified_column_table(previous_record_id)
    if _qualified_column_table(current_record_id) != record_table:
        raise SQLGroundingValidationError(
            "LAG and current record must belong to the same real table"
        )
    if _qualified_column_table(ordered.this) != record_table:
        raise SQLGroundingValidationError(
            "adjacent-record ordering must use the same real record table"
        )
    _qualified_column_table(partition_by[0])


def _field_real_table_references(value: str) -> frozenset[str]:
    expression = _parse_expression(value, label="field expression")
    columns = list(expression.find_all(exp.Column))
    if not columns:
        raise SQLGroundingValidationError(
            "field expression must reference a fully qualified real column"
        )
    return frozenset(_qualified_column_table(column) for column in columns)


def _relation_real_table_references(value: str) -> frozenset[str]:
    expression = _parse_expression(value, label="relation expression")
    if any(expression.find_all(exp.Subquery)) or any(expression.find_all(exp.Select)):
        shape = _require_correlated_relation_shape(expression)
        return frozenset(
            {
                shape.inner_table,
                _qualified_column_table(shape.root_outer_column),
                _qualified_column_table(shape.predicate_outer_column),
            }
        )
    return frozenset(
        _qualified_column_table(column) for column in expression.find_all(exp.Column)
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
    _field_real_table_references(value)
    return _require_canonical(value, expression, label="field expression")


def canonicalize_relation_expression(value: str) -> str:
    """Validate and return an already-canonical relation expression."""

    expression = _parse_expression(value, label="relation expression")
    if any(expression.find_all(exp.Subquery)) or any(expression.find_all(exp.Select)):
        _validate_ast(
            expression,
            allowed_nodes=CORRELATED_RELATION_AST_WHITELIST,
            label="correlated relation expression",
        )
        _require_correlated_relation_shape(expression)
        return _require_canonical(value, expression, label="relation expression")
    if any(expression.find_all(exp.Window)) or any(expression.find_all(exp.Lag)):
        _validate_ast(
            expression,
            allowed_nodes=SAME_TABLE_MULTI_RECORD_AST_WHITELIST,
            label="same-table multi-record relation expression",
        )
        _require_same_table_multi_record_shape(expression)
        return _require_canonical(value, expression, label="relation expression")
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
    qualifiers = {
        _qualified_column_table(column)
        for column in expression.find_all(exp.Column)
    }
    if len(qualifiers) < 2:
        raise SQLGroundingValidationError(
            "relation expression must relate at least two real qualified tables"
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
            _qualified_column_table(column)
            for column in comparison.find_all(exp.Column)
        }
        if len(comparison_qualifiers) < 2:
            raise SQLGroundingValidationError(
                "each relation comparison must connect two real qualified tables"
            )
    return _require_canonical(value, expression, label="relation expression")


def _resolve_column(
    column: exp.Column,
    context: ValidationContext,
    *,
    local_aliases: dict[str, str] | None = None,
) -> str:
    name = column.name
    qualifier = _qualified_column_table(column)
    table = (local_aliases or {}).get(qualifier, qualifier)
    if table not in context.known_tables:
        raise SQLGroundingValidationError(
            f"expression references unknown real table {table}"
        )
    candidate = f"{table}.{name}"
    if candidate not in context.known_columns:
        raise SQLGroundingValidationError(
            f"expression references unknown column {candidate}"
        )
    return candidate


def _validate_expression_identifiers(
    value: str,
    *,
    context: ValidationContext,
    relation: bool,
) -> frozenset[str]:
    expression = _parse_expression(
        value,
        label="relation expression" if relation else "field expression",
    )
    local_aliases: dict[str, str] = {}
    if relation and (
        any(expression.find_all(exp.Subquery)) or any(expression.find_all(exp.Select))
    ):
        shape = _require_correlated_relation_shape(expression)
        local_aliases[shape.inner_alias] = shape.inner_table
    resolved = [
        _resolve_column(column, context, local_aliases=local_aliases)
        for column in expression.find_all(exp.Column)
    ]
    if not resolved:
        raise SQLGroundingValidationError("expression must reference a known column")
    _validate_json_path_evidence(
        expression,
        context=context,
        local_aliases=local_aliases,
    )
    return frozenset(column.rsplit(".", 1)[0] for column in resolved)


def _validate_json_path_evidence(
    expression: exp.Expression,
    *,
    context: ValidationContext,
    local_aliases: dict[str, str],
) -> None:
    """Require every outer JSON path to exist in Official fields_meaning."""

    json_nodes = tuple(
        node
        for node in expression.walk()
        if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar))
        and not isinstance(node.parent, (exp.JSONExtract, exp.JSONExtractScalar))
    )
    for node in json_nodes:
        column, path = _json_path_reference(node)
        resolved = _resolve_column(column, context, local_aliases=local_aliases)
        if (resolved, path) not in context.supported_json_paths:
            raise SQLGroundingValidationError(
                "JSON path lacks an allowed Official column-meaning source"
            )


def _json_path_reference(
    expression: exp.Expression,
) -> tuple[exp.Column, tuple[str, ...]]:
    if isinstance(expression, (exp.JSONExtract, exp.JSONExtractScalar)):
        column, prefix = _json_path_reference(expression.this)
        path = expression.args.get("expression")
        if not isinstance(path, exp.JSONPath):
            raise SQLGroundingValidationError("JSON path is not a fixed key path")
        keys = tuple(
            str(part.this)
            for part in path.expressions
            if isinstance(part, exp.JSONPathKey)
        )
        if not keys:
            raise SQLGroundingValidationError("JSON path is not a fixed key path")
        return column, prefix + keys
    if isinstance(expression, (exp.Cast, exp.Paren)):
        return _json_path_reference(expression.this)
    if isinstance(expression, exp.Column):
        return expression, ()
    raise SQLGroundingValidationError("JSON path must originate from a known column")


def validate_sql_grounding_state(
    state: SQLGroundingState,
    context: ValidationContext,
) -> SQLGroundingState:
    """Validate State only against the explicitly allowed transient evidence."""

    if state.tables is not None:
        for table in state.tables:
            if table not in context.known_tables:
                raise SQLGroundingValidationError(f"unknown table {table}")
    declared_tables = set(state.tables or ())
    if state.join_keys is not None:
        for relation in state.join_keys:
            referenced = _validate_expression_identifiers(
                relation,
                context=context,
                relation=True,
            )
            if not referenced.issubset(declared_tables):
                raise SQLGroundingValidationError(
                    "relation expression references a table absent from State.tables"
                )
    if state.column_mapping is not None:
        for mapping in state.column_mapping:
            if not any(mapping.phrase in source for source in context.query_texts):
                raise SQLGroundingValidationError(
                    f"phrase is not a verbatim query substring: {mapping.phrase}"
                )
            for target in mapping.targets:
                referenced = _validate_expression_identifiers(
                    target,
                    context=context,
                    relation=False,
                )
                if not referenced.issubset(declared_tables):
                    raise SQLGroundingValidationError(
                        "field expression references a table absent from State.tables"
                    )
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
    if (
        stage == "INITIAL_GROUNDING"
        and response.sql_grounding_state.all_dimensions_evaluated
        and response.next_focus_dimension != "none"
    ):
        raise SQLGroundingValidationError(
            "INITIAL_GROUNDING must return none after all dimensions are evaluated"
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


def validate_grounding_state_transition(
    previous: SQLGroundingState,
    candidate: SQLGroundingState,
    authorization: StateDiffAuthorization,
) -> tuple[GroundingDimension, ...]:
    """Validate a complete-State proposal against a minimal stage authorization.

    ``authorized_dimensions`` represents dimensions supported by the current
    legal Observation.  It is deliberately not an Evidence object or Patch.
    Returning the stable changed-dimension tuple gives a later Service an
    auditable, deterministic diff without implementing SG2 update machinery.
    """

    changed = tuple(
        dimension
        for dimension in GROUNDING_DIMENSIONS
        if getattr(previous, dimension) != getattr(candidate, dimension)
    )
    authorized = set(authorization.authorized_dimensions)
    unauthorized = [dimension for dimension in changed if dimension not in authorized]
    if unauthorized:
        raise SQLGroundingValidationError(
            "State changes an unauthorized dimension: " + ", ".join(unauthorized)
        )

    if authorization.stage in {"SQL_ATTEMPT", "DONE"} and changed:
        raise SQLGroundingValidationError(
            f"{authorization.stage} does not authorize Grounding State changes"
        )

    for dimension in changed:
        old_value = getattr(previous, dimension)
        new_value = getattr(candidate, dimension)
        if old_value is not None and new_value is None:
            raise SQLGroundingValidationError(
                f"{dimension} cannot regress from evaluated to null"
            )
    return changed


def validate_grounding_runtime_transition(
    previous: GroundingRuntime,
    current: GroundingRuntime,
    authorization: StateDiffAuthorization | None = None,
) -> GroundingRuntime:
    """Freeze authorization and revision semantics without an SG2 Service."""

    effective_authorization = authorization or StateDiffAuthorization(
        stage=previous.stage,
    )
    if effective_authorization.stage != previous.stage:
        raise SQLGroundingValidationError(
            "State diff authorization stage must match the previous Runtime stage"
        )
    validate_grounding_state_transition(
        previous.grounding_state,
        current.grounding_state,
        effective_authorization,
    )

    state_changed = sql_grounding_state_sha256(previous.grounding_state) != (
        sql_grounding_state_sha256(current.grounding_state)
    )
    expected_revision = previous.grounding_revision + int(state_changed)
    if current.grounding_revision != expected_revision:
        raise SQLGroundingValidationError(
            "grounding_revision must increase exactly once iff canonical State changes"
        )
    return current
