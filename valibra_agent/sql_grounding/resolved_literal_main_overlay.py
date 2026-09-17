"""Bounded Main overlay and SQL gate for resolved literal carriers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from valibra_agent.sql_grounding.models import canonical_json
from valibra_agent.sql_grounding.resolved_literal_carrier_materialization import (
    ResolvedLiteralExecutableCarrier,
)
from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
    _json_path_keys,
)


RESOLVED_LITERAL_MAIN_OVERLAY_VERSION = "resolved-literal-main-overlay/v1"
MAX_RESOLVED_LITERAL_MAIN_OVERLAY_CHARS = 4_096


@dataclass(frozen=True, slots=True)
class RenderedResolvedLiteralMainOverlay:
    text: str
    sha256: str
    item_count: int


@dataclass(frozen=True, slots=True)
class ResolvedLiteralSQLGateDecision:
    allowed: bool
    reason: str
    involved_carrier_count: int = 0


@dataclass(frozen=True, slots=True)
class _TargetReference:
    table: str
    column: str
    path: tuple[str, ...]
    scalar: bool


def render_resolved_literal_main_overlay(
    carriers: tuple[ResolvedLiteralExecutableCarrier, ...],
) -> RenderedResolvedLiteralMainOverlay | None:
    """Render only executable equality facts, never hidden evidence."""

    if not carriers:
        return None
    if len(carriers) > 1:
        raise ValueError("resolved literal R1 permits one active carrier")
    items = []
    for carrier in carriers:
        literal_sql = exp.Literal.string(carrier.literal).sql(dialect="postgres")
        items.append(
            {
                "condition_sql": (
                    f"{carrier.comparison_target} = {literal_sql}"
                ),
                "literal": carrier.literal,
                "operator": carrier.operator,
                "phrase": carrier.phrase,
                "target": carrier.comparison_target,
            }
        )
    payload = {
        "constraints": items,
        "version": RESOLVED_LITERAL_MAIN_OVERLAY_VERSION,
    }
    text = "\n".join(
        [
            "[RESOLVED LITERAL EXECUTION CONSTRAINTS]",
            canonical_json(payload),
            (
                "These are committed Grounding execution constraints. If the "
                "candidate references a listed target, it must use the exact "
                "target = literal equality shown in condition_sql. Do not "
                "replace it with MAX/MIN, ILIKE, another spelling or another "
                "literal. This overlay supplies no other predicate, formula, "
                "aggregation, grain, ordering or output rule."
            ),
        ]
    )
    if len(text) > MAX_RESOLVED_LITERAL_MAIN_OVERLAY_CHARS:
        raise ValueError("resolved literal Main overlay is too large")
    return RenderedResolvedLiteralMainOverlay(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        item_count=len(items),
    )


def evaluate_resolved_literal_sql_gate(
    sql: str,
    carriers: tuple[ResolvedLiteralExecutableCarrier, ...],
) -> ResolvedLiteralSQLGateDecision:
    """Require exact equality whenever candidate SQL uses a carrier target."""

    if not carriers:
        return ResolvedLiteralSQLGateDecision(True, "not_applicable")
    if len(carriers) > 1:
        return ResolvedLiteralSQLGateDecision(False, "carrier_count_invalid")
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except ParseError:
        return ResolvedLiteralSQLGateDecision(False, "sql_validation_failed")
    if len(statements) != 1 or statements[0] is None:
        return ResolvedLiteralSQLGateDecision(False, "sql_validation_failed")
    statement = statements[0]
    aliases = _table_aliases(statement)
    involved = 0
    for carrier in carriers:
        expected = _target_reference(carrier.comparison_target)
        target_nodes = tuple(
            node
            for node in _candidate_target_nodes(statement)
            if _same_target(node, expected, aliases=aliases)
        )
        if not target_nodes:
            continue
        involved += 1
        exact_equalities = tuple(
            predicate
            for predicate in statement.find_all(exp.EQ)
            if _is_exact_carrier_equality(
                predicate,
                carrier=carrier,
                expected=expected,
                aliases=aliases,
            )
        )
        if not exact_equalities:
            return ResolvedLiteralSQLGateDecision(
                False,
                "resolved_literal_exact_equality_missing",
                involved,
            )
        if any(
            _expression_contains_target(function, expected, aliases=aliases)
            for function in (
                *statement.find_all(exp.Max),
                *statement.find_all(exp.Min),
            )
        ):
            return ResolvedLiteralSQLGateDecision(
                False,
                "resolved_literal_aggregate_reinterpretation",
                involved,
            )
        for predicate in _comparison_predicates(statement):
            if not _expression_contains_target(predicate, expected, aliases=aliases):
                continue
            if isinstance(predicate, exp.EQ) and _is_exact_carrier_equality(
                predicate,
                carrier=carrier,
                expected=expected,
                aliases=aliases,
            ):
                continue
            return ResolvedLiteralSQLGateDecision(
                False,
                "resolved_literal_conflicting_comparison",
                involved,
            )
    return ResolvedLiteralSQLGateDecision(True, "allowed", involved)


def _is_exact_carrier_equality(
    predicate: exp.EQ,
    *,
    carrier: ResolvedLiteralExecutableCarrier,
    expected: _TargetReference,
    aliases: dict[str, frozenset[str]],
) -> bool:
    return (
        _is_exact_target_expression(predicate.this, expected, aliases=aliases)
        and _is_exact_literal(predicate.expression, carrier.literal)
    ) or (
        _is_exact_target_expression(predicate.expression, expected, aliases=aliases)
        and _is_exact_literal(predicate.this, carrier.literal)
    )


def _is_exact_target_expression(
    expression: exp.Expression,
    expected: _TargetReference,
    *,
    aliases: dict[str, frozenset[str]],
) -> bool:
    try:
        actual = _target_reference(expression)
    except ValueError:
        return False
    return actual.scalar == expected.scalar and _same_target(
        actual,
        expected,
        aliases=aliases,
    )


def _is_exact_literal(expression: exp.Expression, literal: str) -> bool:
    return (
        isinstance(expression, exp.Literal)
        and expression.is_string
        and expression.this == literal
    )


def _candidate_target_nodes(statement: exp.Expression) -> Iterable[_TargetReference]:
    for node in statement.walk():
        if not isinstance(node, (exp.Column, exp.JSONExtract, exp.JSONExtractScalar)):
            continue
        try:
            yield _target_reference(node)
        except ValueError:
            continue


def _target_reference(expression: exp.Expression | str) -> _TargetReference:
    node = (
        sqlglot.parse_one(expression, read="postgres")
        if isinstance(expression, str)
        else expression
    )
    if isinstance(node, exp.Paren):
        return _target_reference(node.this)
    if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar)):
        base = _target_reference(node.this)
        return _TargetReference(
            table=base.table,
            column=base.column,
            path=base.path + _json_path_keys(node.expression),
            scalar=isinstance(node, exp.JSONExtractScalar),
        )
    if isinstance(node, exp.Column) and node.name:
        return _TargetReference(
            table=node.table,
            column=node.name,
            path=(),
            scalar=True,
        )
    raise ValueError("expression is not a direct column or JSON leaf")


def _table_aliases(statement: exp.Expression) -> dict[str, frozenset[str]]:
    aliases: dict[str, set[str]] = {}
    for table in statement.find_all(exp.Table):
        actual = table.name.casefold()
        alias = table.alias.casefold() if table.alias else actual
        aliases.setdefault(alias, set()).add(actual)
        aliases.setdefault(actual, set()).add(actual)
    return {key: frozenset(value) for key, value in aliases.items()}


def _same_target(
    actual: _TargetReference,
    expected: _TargetReference,
    *,
    aliases: dict[str, frozenset[str]],
) -> bool:
    if (
        actual.column.casefold() != expected.column.casefold()
        or actual.path != expected.path
    ):
        return False
    expected_table = expected.table.casefold()
    if actual.table:
        actual_table = actual.table.casefold()
        return expected_table in aliases.get(actual_table, frozenset({actual_table}))
    visible_tables = frozenset(
        table for candidates in aliases.values() for table in candidates
    )
    return visible_tables == frozenset({expected_table})


def _expression_contains_target(
    expression: exp.Expression,
    expected: _TargetReference,
    *,
    aliases: dict[str, frozenset[str]],
) -> bool:
    return any(
        _same_target(item, expected, aliases=aliases)
        for item in _candidate_target_nodes(expression)
    )


def _comparison_predicates(statement: exp.Expression) -> Iterable[exp.Expression]:
    predicate_types = (
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.Like,
        exp.ILike,
        exp.In,
        exp.Between,
    )
    for node in statement.walk():
        if isinstance(node, predicate_types):
            yield node

