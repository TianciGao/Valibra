"""Bounded contradiction detection for verified AnswerContract Lite facts.

V0 is intentionally not a SQL correctness validator.  It reports a conflict
only when a parsed candidate contains a directly comparable expression that
contradicts a verified invariant.  Missing, aliased beyond deterministic
resolution, negated, or otherwise complex expressions remain UNKNOWN and pass
through at the runtime boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


GuardStatus = Literal["NO_CONTRADICTION", "CONTRADICTION", "UNKNOWN"]


@dataclass(frozen=True)
class GuardConflict:
    invariant_id: str
    kind: str
    expected: str
    observed: str


@dataclass(frozen=True)
class GuardEvaluation:
    status: GuardStatus
    parsed: bool
    inspected_invariants: int
    conflict: GuardConflict | None = None


_COMPARISONS: dict[type[exp.Expression], str] = {
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.EQ: "=",
    exp.NEQ: "!=",
}
_INVERTED_COMPARISON = {
    ">": "<",
    ">=": "<=",
    "<": ">",
    "<=": ">=",
    "=": "=",
    "!=": "!=",
}
_ARITHMETIC: dict[type[exp.Expression], str] = {
    exp.Add: "add",
    exp.Sub: "sub",
    exp.Mul: "mul",
    exp.Div: "div",
}


def _identifier_key(
    identifier: exp.Identifier | None, *, canonical: bool = False
) -> tuple[str, bool] | None:
    if identifier is None:
        return None
    value = identifier.this
    if not isinstance(value, str) or not value:
        return None
    quoted = bool(identifier.args.get("quoted"))
    # Contract fields are metadata targets rather than executable SQL.  Their
    # original casing is canonical even though the compact carrier omits SQL
    # quotes.  Lowercase canonical identifiers retain normal PostgreSQL
    # unquoted semantics; mixed-case identifiers require an exact quoted SQL
    # match.
    case_sensitive = quoted or (canonical and value != value.casefold())
    return (value if case_sensitive else value.casefold(), case_sensitive)


def _table_name(table: exp.Table) -> tuple[tuple[str, bool], ...] | None:
    parts = [part for part in (table.args.get("catalog"), table.args.get("db"), table.this) if part is not None]
    keys = tuple(_identifier_key(part) for part in parts if isinstance(part, exp.Identifier))
    return keys if keys and len(keys) == len(parts) else None


def _alias_map(statement: exp.Expression) -> dict[tuple[str, bool], tuple[tuple[str, bool], ...]]:
    result: dict[tuple[str, bool], tuple[tuple[str, bool], ...]] = {}
    for table in statement.find_all(exp.Table):
        target = _table_name(table)
        alias = table.args.get("alias")
        if target is None or not isinstance(alias, exp.TableAlias) or not isinstance(alias.this, exp.Identifier):
            continue
        alias_key = _identifier_key(alias.this)
        if alias_key is None:
            continue
        existing = result.get(alias_key)
        if existing is not None and existing != target:
            result.pop(alias_key, None)
            continue
        result[alias_key] = target
    return result


def _decimal(value: str) -> str | None:
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    normalized = number.normalize()
    return format(normalized, "f") if normalized != 0 else "0"


def _json_path_key(path: exp.Expression) -> tuple[str, ...] | None:
    if not isinstance(path, exp.JSONPath):
        return None
    keys: list[str] = []
    for item in path.expressions:
        if isinstance(item, exp.JSONPathRoot):
            continue
        if not isinstance(item, exp.JSONPathKey) or not isinstance(item.this, str):
            return None
        keys.append(item.this)
    return tuple(keys)


def _strip_transparent(node: exp.Expression) -> exp.Expression:
    current = node
    while isinstance(current, (exp.Paren, exp.Cast, exp.TryCast, exp.Alias)):
        child = current.this
        if not isinstance(child, exp.Expression):
            break
        current = child
    return current


def _expression_key(
    node: exp.Expression,
    aliases: dict[tuple[str, bool], tuple[tuple[str, bool], ...]],
    *,
    canonical: bool = False,
) -> tuple[Any, ...] | None:
    node = _strip_transparent(node)
    if isinstance(node, exp.Column):
        name = _identifier_key(
            node.this if isinstance(node.this, exp.Identifier) else None,
            canonical=canonical,
        )
        if name is None:
            return None
        qualifier: tuple[tuple[str, bool], ...] = ()
        table = node.args.get("table")
        db = node.args.get("db")
        catalog = node.args.get("catalog")
        parts = [part for part in (catalog, db, table) if isinstance(part, exp.Identifier)]
        if parts:
            keys = tuple(
                _identifier_key(part, canonical=canonical) for part in parts
            )
            if any(key is None for key in keys):
                return None
            qualifier = tuple(key for key in keys if key is not None)
            if len(qualifier) == 1 and qualifier[0] in aliases:
                qualifier = aliases[qualifier[0]]
        return ("field", qualifier, name)
    if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar)):
        base = node.args.get("this")
        path = node.args.get("expression")
        if not isinstance(base, exp.Expression) or not isinstance(path, exp.Expression):
            return None
        base_key = _expression_key(base, aliases, canonical=canonical)
        path_key = _json_path_key(path)
        if base_key is None or path_key is None:
            return None
        return (
            "json_scalar" if isinstance(node, exp.JSONExtractScalar) else "json",
            base_key,
            path_key,
        )
    if isinstance(node, exp.Literal) and not node.is_string:
        value = _decimal(str(node.this))
        return ("number", value) if value is not None else None
    if isinstance(node, exp.Neg):
        child = node.this
        if isinstance(child, exp.Literal) and not child.is_string:
            value = _decimal(f"-{child.this}")
            return ("number", value) if value is not None else None
        return None
    for node_type, operator in _ARITHMETIC.items():
        if isinstance(node, node_type):
            left = node.args.get("this")
            right = node.args.get("expression")
            if not isinstance(left, exp.Expression) or not isinstance(right, exp.Expression):
                return None
            left_key = _expression_key(left, aliases, canonical=canonical)
            right_key = _expression_key(right, aliases, canonical=canonical)
            if left_key is None or right_key is None:
                return None
            return ("op", operator, left_key, right_key)
    return None


def _parse_contract_expression(
    value: dict[str, Any],
) -> tuple[Any, ...] | None:
    if set(value) == {"field"} and isinstance(value.get("field"), str):
        try:
            parsed = sqlglot.parse_one(value["field"], read="postgres")
        except ParseError:
            return None
        return _expression_key(parsed, {}, canonical=True)
    if set(value) == {"literal"} and isinstance(value.get("literal"), str):
        number = _decimal(value["literal"])
        return ("number", number) if number is not None else None
    if set(value) == {"op", "args"} and value.get("op") in {"add", "sub", "mul", "div"}:
        args = value.get("args")
        if not isinstance(args, list) or len(args) != 2 or not all(isinstance(item, dict) for item in args):
            return None
        left = _parse_contract_expression(args[0])
        right = _parse_contract_expression(args[1])
        if left is None or right is None:
            return None
        return ("op", value["op"], left, right)
    return None


def _numeric_literal(node: exp.Expression, aliases: dict) -> str | None:
    key = _expression_key(node, aliases)
    return key[1] if key is not None and len(key) == 2 and key[0] == "number" else None


def _source_tables(statement: exp.Expression) -> tuple[tuple[tuple[str, bool], ...], ...]:
    return tuple(
        target
        for table in statement.find_all(exp.Table)
        if (target := _table_name(table)) is not None
    )


def _target_equivalent(
    expected: tuple[Any, ...],
    candidate: tuple[Any, ...],
    sources: tuple[tuple[tuple[str, bool], ...], ...],
) -> bool:
    """Compare target identity without treating implementation syntax as policy.

    JSON scalar-vs-JSON extraction is deliberately ignored only while locating
    the target of another protected invariant.  V0 never rejects a candidate
    merely for changing that extraction syntax.
    """

    if expected[0] == "field" and candidate[0] == "field":
        if expected[2] != candidate[2]:
            return False
        expected_qualifier = expected[1]
        candidate_qualifier = candidate[1]
        if expected_qualifier == candidate_qualifier:
            return True
        # An unqualified reference is deterministic here when its verified
        # canonical table is among the SQL sources: that table is known to
        # expose the field; another same-named source field would make the SQL
        # ambiguous rather than bind it to a different valid target.
        return not candidate_qualifier and expected_qualifier in sources
    if expected[0] in {"json", "json_scalar"} and candidate[0] in {
        "json",
        "json_scalar",
    }:
        return expected[2] == candidate[2] and _target_equivalent(
            expected[1], candidate[1], sources
        )
    if expected[0] == "number" and candidate[0] == "number":
        return expected == candidate
    if expected[0] == "op" and candidate[0] == "op":
        return (
            expected[1] == candidate[1]
            and _target_equivalent(expected[2], candidate[2], sources)
            and _target_equivalent(expected[3], candidate[3], sources)
        )
    return expected == candidate


def _under_negation(node: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None and not isinstance(parent, (exp.Select, exp.Where, exp.Having, exp.Join)):
        if isinstance(parent, exp.Not):
            return True
        parent = parent.parent
    return False


def _candidate_comparisons(statement: exp.Expression, aliases: dict) -> list[tuple[tuple[Any, ...], str, str]]:
    result: list[tuple[tuple[Any, ...], str, str]] = []
    for node in statement.walk():
        operator = next((value for kind, value in _COMPARISONS.items() if isinstance(node, kind)), None)
        if operator is None or _under_negation(node):
            continue
        left = node.args.get("this")
        right = node.args.get("expression")
        if not isinstance(left, exp.Expression) or not isinstance(right, exp.Expression):
            continue
        left_key = _expression_key(left, aliases)
        right_key = _expression_key(right, aliases)
        right_literal = _numeric_literal(right, aliases)
        left_literal = _numeric_literal(left, aliases)
        if left_key is not None and right_literal is not None and left_key[0] != "number":
            result.append((left_key, operator, right_literal))
        elif right_key is not None and left_literal is not None and right_key[0] != "number":
            result.append((right_key, _INVERTED_COMPARISON[operator], left_literal))
    return result


def _render_key(key: tuple[Any, ...]) -> str:
    if key and key[0] == "field" and len(key) == 3:
        qualifier = ".".join(str(item[0]) for item in key[1])
        name = str(key[2][0])
        return f"{qualifier}.{name}" if qualifier else name
    if key and key[0] in {"json", "json_scalar"} and len(key) == 3:
        operator = "->>" if key[0] == "json_scalar" else "->"
        suffix = "".join(f" {operator} '{item}'" for item in key[2])
        return f"{_render_key(key[1])}{suffix}"
    if key and key[0] == "number" and len(key) == 2:
        return str(key[1])
    if key and key[0] == "op" and len(key) == 4:
        operator = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(
            str(key[1]), str(key[1])
        )
        return f"({_render_key(key[2])} {operator} {_render_key(key[3])})"
    return "<complex-expression>"


def _predicate_conflict(
    invariant: dict[str, Any],
    comparisons: list[tuple],
    sources: tuple[tuple[tuple[str, bool], ...], ...],
) -> tuple[GuardConflict | None, bool]:
    payload = invariant.get("payload")
    if not isinstance(payload, dict):
        return None
    target_value = payload.get("target_ast", payload.get("target"))
    if not isinstance(target_value, dict):
        return None
    target = _parse_contract_expression(target_value)
    operator = payload.get("operator")
    literal = payload.get("literal")
    literal_value = literal.get("value") if isinstance(literal, dict) else literal
    expected_literal = _decimal(str(literal_value)) if literal_value is not None else None
    if target is None or operator not in _INVERTED_COMPARISON or expected_literal is None:
        return None, False
    matches = [
        (op, value)
        for candidate, op, value in comparisons
        if _target_equivalent(target, candidate, sources)
    ]
    if not matches:
        return None, False
    if (operator, expected_literal) in matches:
        return None, True
    observed = matches[0]
    return (
        GuardConflict(
            invariant_id=str(invariant.get("invariant_id", "unknown")),
            kind="predicate",
            expected=f"{_render_key(target)} {operator} {expected_literal}",
            observed=f"{_render_key(target)} {observed[0]} {observed[1]}",
        ),
        True,
    )


def _band_conflict(
    invariant: dict[str, Any],
    comparisons: list[tuple],
    sources: tuple[tuple[tuple[str, bool], ...], ...],
) -> tuple[GuardConflict | None, bool]:
    payload = invariant.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("target"), dict):
        return None, False
    target = _parse_contract_expression(payload["target"])
    bands = payload.get("bands")
    if target is None or not isinstance(bands, list):
        return None, False
    matches = [
        (op, value)
        for candidate, op, value in comparisons
        if _target_equivalent(target, candidate, sources)
    ]
    expected_pairs: list[tuple[str, str]] = []
    for band in bands:
        if not isinstance(band, dict) or band.get("operator") not in {"<", ">"}:
            continue
        operator = band["operator"]
        expected_literal = _decimal(str(band.get("literal")))
        if expected_literal is None:
            continue
        expected_pairs.append((operator, expected_literal))
        if (operator, expected_literal) in matches:
            continue
        same_direction = [
            item
            for item in matches
            if item[0] in ({"<", "<="} if operator == "<" else {">", ">="})
        ]
        if same_direction:
            observed = same_direction[0]
            return (
                GuardConflict(
                    invariant_id=str(invariant.get("invariant_id", "unknown")),
                    kind="predicate_band_set",
                    expected=f"{_render_key(target)} {operator} {expected_literal}",
                    observed=f"{_render_key(target)} {observed[0]} {observed[1]}",
                ),
                True,
            )
    return None, bool(expected_pairs) and all(item in matches for item in expected_pairs)


def _key_fields(key: tuple[Any, ...]) -> tuple[tuple[Any, ...], ...]:
    if key and key[0] == "field":
        return (key,)
    result: list[tuple[Any, ...]] = []
    for item in key[1:]:
        if isinstance(item, tuple):
            result.extend(_key_fields(item))
    return tuple(sorted(result, key=repr))


def _key_literals(key: tuple[Any, ...]) -> tuple[str, ...]:
    if len(key) == 2 and key[0] == "number":
        return (str(key[1]),)
    result: list[str] = []
    for item in key[1:]:
        if isinstance(item, tuple):
            result.extend(_key_literals(item))
    return tuple(sorted(result))


def _formula_operand_key(
    node: exp.Expression,
    aliases: dict[tuple[str, bool], tuple[tuple[str, bool], ...]],
) -> tuple[Any, ...] | None:
    direct = _expression_key(node, aliases)
    if direct is not None and direct[0] in {"field", "json", "json_scalar", "number"}:
        return direct
    # SQL-only wrappers (casts, numeric parsing, NULL handling, and similar
    # functions) are outside V0's policy surface.  They may be ignored for
    # locating a formula operand only when the subtree contains exactly one
    # field reference.  The wrapper itself is never accepted or rejected.
    fields = {
        key
        for column in node.find_all(exp.Column)
        if (key := _expression_key(column, aliases)) is not None
    }
    return next(iter(fields)) if len(fields) == 1 else None


def _formula_expression_key(
    node: exp.Expression,
    aliases: dict[tuple[str, bool], tuple[tuple[str, bool], ...]],
) -> tuple[Any, ...] | None:
    node = _strip_transparent(node)
    for node_type, operator in _ARITHMETIC.items():
        if not isinstance(node, node_type):
            continue
        left = node.args.get("this")
        right = node.args.get("expression")
        if not isinstance(left, exp.Expression) or not isinstance(right, exp.Expression):
            return None
        left_key = _formula_expression_key(left, aliases)
        right_key = _formula_expression_key(right, aliases)
        if left_key is None or right_key is None:
            return None
        return ("op", operator, left_key, right_key)
    return _formula_operand_key(node, aliases)


def _formula_conflict(
    invariant: dict[str, Any],
    statement: exp.Expression,
    aliases: dict,
    sources: tuple[tuple[tuple[str, bool], ...], ...],
) -> tuple[GuardConflict | None, bool]:
    payload = invariant.get("payload")
    expression_ast = payload.get("expression_ast") if isinstance(payload, dict) else None
    if not isinstance(expression_ast, dict):
        return None, False
    expected = _parse_contract_expression(expression_ast)
    if expected is None:
        return None, False
    arithmetic: list[tuple[Any, ...]] = []
    for node in statement.walk():
        key = _formula_expression_key(node, aliases)
        if key is not None and key[0] == "op":
            arithmetic.append(key)
    if any(_target_equivalent(expected, candidate, sources) for candidate in arithmetic):
        return None, True
    fields = _key_fields(expected)
    literals = _key_literals(expected)
    competing = []
    for key in arithmetic:
        candidate_fields = _key_fields(key)
        remaining = list(candidate_fields)
        for expected_field in fields:
            match = next(
                (
                    index
                    for index, candidate_field in enumerate(remaining)
                    if _target_equivalent(expected_field, candidate_field, sources)
                ),
                None,
            )
            if match is None:
                break
            remaining.pop(match)
        else:
            if not remaining and _key_literals(key) == literals:
                competing.append(key)
    if not competing:
        return None, False
    return (
        GuardConflict(
            invariant_id=str(invariant.get("invariant_id", "unknown")),
            kind="formula",
            expected=_render_key(expected)[:512],
            observed=_render_key(competing[0])[:512],
        ),
        True,
    )


def _aggregate_calls(statement: exp.Expression, aliases: dict) -> list[tuple[tuple[Any, ...], str]]:
    result: list[tuple[tuple[Any, ...], str]] = []
    aggregate_types = ((exp.Avg, "AVG"), (exp.Sum, "SUM"), (exp.Min, "MIN"), (exp.Max, "MAX"), (exp.Count, "COUNT"))
    median_type = getattr(exp, "Median", None)
    for node in statement.walk():
        function = next((name for kind, name in aggregate_types if isinstance(node, kind)), None)
        if median_type is not None and isinstance(node, median_type):
            function = "MEDIAN"
        if function is None:
            continue
        target_node = node.this
        if not isinstance(target_node, exp.Expression):
            continue
        target = _expression_key(target_node, aliases)
        if target is not None:
            result.append((target, function))
    for node in statement.find_all(exp.WithinGroup):
        percentile = node.this
        order = node.args.get("expression")
        if not isinstance(percentile, exp.PercentileCont) or not isinstance(order, exp.Order):
            continue
        quantile = percentile.this
        if not isinstance(quantile, exp.Literal) or _decimal(str(quantile.this)) != "0.5":
            continue
        ordered = order.expressions
        if len(ordered) != 1 or not isinstance(ordered[0], exp.Ordered):
            continue
        target = _expression_key(ordered[0].this, aliases)
        if target is not None:
            result.append((target, "MEDIAN"))
    return result


def _aggregation_conflict(
    invariant: dict[str, Any],
    aggregates: list[tuple],
    sources: tuple[tuple[tuple[str, bool], ...], ...],
) -> tuple[GuardConflict | None, bool]:
    payload = invariant.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("target"), dict):
        return None, False
    target = _parse_contract_expression(payload["target"])
    function = payload.get("function")
    if target is None or function not in {"AVG", "SUM", "MIN", "MAX", "COUNT"}:
        return None, False
    matches = [
        name
        for candidate, name in aggregates
        if _target_equivalent(target, candidate, sources)
    ]
    if not matches:
        return None, False
    if function in matches:
        return None, True
    return (
        GuardConflict(
            invariant_id=str(invariant.get("invariant_id", "unknown")),
            kind="aggregation",
            expected=f"{function}({_render_key(target)})",
            observed=f"{matches[0]}({_render_key(target)})",
        ),
        True,
    )


def _ordering_conflict(
    invariant: dict[str, Any],
    statement: exp.Expression,
    aliases: dict,
    sources: tuple[tuple[tuple[str, bool], ...], ...],
) -> tuple[GuardConflict | None, bool]:
    payload = invariant.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("target"), dict):
        return None, False
    target = _parse_contract_expression(payload["target"])
    direction = payload.get("direction")
    if target is None or direction not in {"ASC", "DESC"}:
        return None, False
    matches: list[str] = []
    for ordered in statement.find_all(exp.Ordered):
        candidate = _expression_key(ordered.this, aliases)
        if candidate is not None and _target_equivalent(target, candidate, sources):
            matches.append("DESC" if ordered.args.get("desc") is True else "ASC")
    if not matches:
        return None, False
    if direction in matches:
        return None, True
    return (
        GuardConflict(
            invariant_id=str(invariant.get("invariant_id", "unknown")),
            kind="ordering",
            expected=f"{_render_key(target)} {direction}",
            observed=f"{_render_key(target)} {matches[0]}",
        ),
        True,
    )


def evaluate_candidate_sql(contract: dict[str, Any], sql: str) -> GuardEvaluation:
    """Return CONTRADICTION only for a directly comparable explicit conflict."""

    if not isinstance(sql, str) or not sql or sql != sql.strip():
        return GuardEvaluation("UNKNOWN", False, 0)
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except ParseError:
        return GuardEvaluation("UNKNOWN", False, 0)
    if len(statements) != 1 or statements[0] is None:
        return GuardEvaluation("UNKNOWN", False, 0)
    statement = statements[0]
    invariants = contract.get("verified_invariants")
    if not isinstance(invariants, list) or not invariants:
        return GuardEvaluation("UNKNOWN", True, 0)
    aliases = _alias_map(statement)
    sources = _source_tables(statement)
    comparisons = _candidate_comparisons(statement, aliases)
    aggregates = _aggregate_calls(statement, aliases)
    inspected = 0
    comparable = False
    for invariant in invariants:
        if not isinstance(invariant, dict):
            continue
        kind = invariant.get("kind")
        conflict: GuardConflict | None = None
        invariant_comparable = False
        if kind == "predicate":
            inspected += 1
            conflict, invariant_comparable = _predicate_conflict(
                invariant, comparisons, sources
            )
        elif kind == "predicate_band_set":
            inspected += 1
            conflict, invariant_comparable = _band_conflict(
                invariant, comparisons, sources
            )
        elif kind == "formula":
            inspected += 1
            conflict, invariant_comparable = _formula_conflict(
                invariant, statement, aliases, sources
            )
        elif kind == "aggregation":
            inspected += 1
            conflict, invariant_comparable = _aggregation_conflict(
                invariant, aggregates, sources
            )
        elif kind == "ordering":
            inspected += 1
            conflict, invariant_comparable = _ordering_conflict(
                invariant, statement, aliases, sources
            )
        comparable = comparable or invariant_comparable
        if conflict is not None:
            return GuardEvaluation("CONTRADICTION", True, inspected, conflict)
    return GuardEvaluation(
        "NO_CONTRADICTION" if comparable else "UNKNOWN",
        True,
        inspected,
    )
