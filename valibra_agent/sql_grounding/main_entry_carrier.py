"""Deterministic, non-semantic carriers for the Main SQL Writer entry."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import sqlglot
from sqlglot import exp

from valibra_agent.sql_grounding.models import SQLGroundingState


MAIN_EXECUTION_ENVELOPES_KEY = "valibra:main_execution_envelopes"
MAIN_EXECUTION_ENVELOPE_VERSION = "main-execution-envelope/v2"
SQL_SAFE_IDENTIFIER_OVERLAY_VERSION = "sql-safe-identifier-overlay/v1"
MAX_SQL_SAFE_IDENTIFIER_ITEMS = 96
MAX_SQL_SAFE_IDENTIFIER_CHARS = 4_096

_OUTPUT_TYPES = frozenset({"scalar", "table"})
_TASK_CATEGORIES = frozenset({"Query", "Management"})
_ACTION_INTENTS = frozenset(
    {
        "RESULT_QUERY",
        "CREATE_TABLE",
        "CREATE_VIEW",
        "CREATE_FUNCTION",
        "SQL_UTILITY",
    }
)

MainActionIntent = Literal[
    "RESULT_QUERY",
    "CREATE_TABLE",
    "CREATE_VIEW",
    "CREATE_FUNCTION",
    "SQL_UTILITY",
]


@dataclass(frozen=True, slots=True)
class MainExecutionEnvelope:
    """Public task response metadata, without any reference-answer fields."""

    response_type: Literal["none", "scalar", "table"]
    task_category: Literal["Query", "Management"]
    decimal_places: int | None
    order_sensitive: bool
    action_intent: MainActionIntent


@dataclass(frozen=True, slots=True)
class RenderedMainEntryCarrier:
    text: str
    sha256: str
    item_count: int
    omitted_item_count: int = 0


def build_main_execution_envelopes(task_data: Mapping[str, Any]) -> dict[str, Any]:
    """Project only non-reference task metadata into phase-local envelopes."""

    phases: dict[str, dict[str, Any]] = {}
    phase_one = _project_task_phase(
        task_data,
        query=task_data.get("amb_user_query"),
    )
    if phase_one is not None:
        phases["1"] = phase_one
    follow_up = task_data.get("follow_up")
    if isinstance(follow_up, Mapping):
        phase_two = _project_task_phase(
            follow_up,
            query=follow_up.get("query"),
        )
        if phase_two is not None:
            phases["2"] = phase_two
    return {
        "version": MAIN_EXECUTION_ENVELOPE_VERSION,
        "phases": phases,
    }


def main_execution_envelope_for_phase(
    state: Mapping[str, Any],
    phase: Literal[1, 2],
) -> MainExecutionEnvelope | None:
    """Load one validated envelope; absence preserves legacy behavior."""

    raw = state.get(MAIN_EXECUTION_ENVELOPES_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Main execution envelopes must be an object")
    if raw.get("version") != MAIN_EXECUTION_ENVELOPE_VERSION:
        raise ValueError("unsupported Main execution envelope version")
    phases = raw.get("phases")
    if not isinstance(phases, Mapping):
        raise ValueError("Main execution envelopes require phases")
    value = phases.get(str(phase))
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("phase Main execution envelope must be an object")
    expected = {
        "response_type",
        "task_category",
        "decimal_places",
        "order_sensitive",
        "action_intent",
    }
    if set(value) != expected:
        raise ValueError("phase Main execution envelope has unexpected fields")
    response_type = value.get("response_type")
    task_category = value.get("task_category")
    decimal_places = value.get("decimal_places")
    order_sensitive = value.get("order_sensitive")
    action_intent = value.get("action_intent")
    if response_type not in {"none", "scalar", "table"}:
        raise ValueError("invalid Main response_type")
    if task_category not in _TASK_CATEGORIES:
        raise ValueError("invalid Main task_category")
    if decimal_places is not None and (
        isinstance(decimal_places, bool)
        or not isinstance(decimal_places, int)
        or not 0 <= decimal_places <= 12
    ):
        raise ValueError("invalid Main decimal_places")
    if not isinstance(order_sensitive, bool):
        raise ValueError("invalid Main order_sensitive")
    if action_intent not in _ACTION_INTENTS:
        raise ValueError("invalid Main action_intent")
    return MainExecutionEnvelope(
        response_type=response_type,
        task_category=task_category,
        decimal_places=decimal_places,
        order_sensitive=order_sensitive,
        action_intent=action_intent,
    )


def render_main_execution_envelope(
    envelope: MainExecutionEnvelope,
) -> RenderedMainEntryCarrier:
    payload = {
        "action_intent": envelope.action_intent,
        "decimal_places": envelope.decimal_places,
        "order_sensitive": envelope.order_sensitive,
        "response_type": envelope.response_type,
        "task_category": envelope.task_category,
    }
    text = "\n".join(
        [
            "[MAIN EXECUTION ENVELOPE]",
            _canonical_json(payload),
            (
                "Authority limits: response_type fixes only result shape; "
                "decimal_places fixes only displayed precision; order_sensitive "
                "only says row order affects evaluation and does not supply a "
                "direction, sort key, aggregation, projection, or business rule; "
                "task_category does not override the Query's requested action. "
                "action_intent is derived only from explicit Query wording. When "
                "it is not RESULT_QUERY, the final SQL must implement that SQL "
                "action; an ordinary SELECT is not an equivalent substitute. "
                "RESULT_QUERY does not authorize inventing a management action. "
                "No object name, function signature, or hidden action detail is "
                "supplied beyond what the Query states."
            ),
        ]
    )
    return RenderedMainEntryCarrier(
        text=text,
        sha256=_sha256(text),
        item_count=1,
    )


def render_sql_safe_identifier_overlay(
    state: SQLGroundingState,
) -> RenderedMainEntryCarrier | None:
    """Render quote-safe forms of already-validated State references only."""

    candidates: list[dict[str, str]] = []
    targets = sorted(
        {
            target
            for mapping in state.column_mapping or ()
            for target in mapping.targets
        }
    )
    for target in targets:
        candidates.append(
            {
                "kind": "target",
                "reference": target,
                "sql": _quote_expression(target),
            }
        )
    for relation in sorted(state.join_keys or ()):
        candidates.append(
            {
                "kind": "relation",
                "reference": relation,
                "sql": _quote_expression(relation),
            }
        )
    for table in sorted(state.tables or ()):
        candidates.append(
            {
                "kind": "table",
                "reference": table,
                "sql": _quote_table(table),
            }
        )
    if not candidates:
        return None

    kept: list[dict[str, str]] = []
    limit = min(len(candidates), MAX_SQL_SAFE_IDENTIFIER_ITEMS)
    for candidate in candidates[:limit]:
        trial = _identifier_overlay_text(kept + [candidate], len(candidates) - len(kept) - 1)
        if len(trial) > MAX_SQL_SAFE_IDENTIFIER_CHARS:
            break
        kept.append(candidate)
    if not kept:
        return None
    omitted = len(candidates) - len(kept)
    text = _identifier_overlay_text(kept, omitted)
    return RenderedMainEntryCarrier(
        text=text,
        sha256=_sha256(text),
        item_count=len(kept),
        omitted_item_count=omitted,
    )


def infer_main_action_intent(query: Any) -> MainActionIntent:
    """Classify only an SQL action explicitly requested in user wording.

    This intentionally does not consult task category, Grounding State, reference
    SQL, or evaluator metadata. Ambiguous requests such as "make a tool" remain
    RESULT_QUERY rather than guessing a hidden SQL object type.
    """

    if not isinstance(query, str) or not query.strip():
        return "RESULT_QUERY"
    normalized = " ".join(query.casefold().replace("`", " ").split())

    if re.search(r"\b(?:vacuum|reindex)\b", normalized):
        return "SQL_UTILITY"
    if re.search(
        r"\brefresh\b(?:\W+\w+){0,5}\W+\bmaterialized\W+view\b",
        normalized,
    ):
        return "SQL_UTILITY"
    if (
        re.search(r"\banaly[sz]e\b", normalized)
        and re.search(r"\btable\b", normalized)
        and re.search(r"\b(?:statistics|stats|up-to-date|tidy)\b", normalized)
    ):
        return "SQL_UTILITY"

    action_verb = r"(?:create|generate|make|build)"
    if re.search(
        rf"\b{action_verb}\b(?:\W+\w+){{0,8}}\W+\b(?:materialized\W+)?view\b",
        normalized,
    ):
        return "CREATE_VIEW"
    if re.search(
        rf"\b{action_verb}\b(?:\W+\w+){{0,8}}\W+\btable\b",
        normalized,
    ):
        return "CREATE_TABLE"
    if re.search(
        r"\b(?:create|generate|make|build|define|write|need)\b"
        r"(?:\W+\w+){0,8}\W+\bfunction\b",
        normalized,
    ):
        return "CREATE_FUNCTION"
    return "RESULT_QUERY"


def _project_task_phase(
    value: Mapping[str, Any],
    *,
    query: Any,
) -> dict[str, Any] | None:
    carrier_fields = frozenset({"output_type", "category", "conditions"})
    present_fields = carrier_fields.intersection(value)
    if not present_fields:
        return None
    if present_fields != carrier_fields:
        raise ValueError("task response metadata is incomplete")

    output_type = value.get("output_type")
    if output_type is None:
        response_type = "none"
    elif output_type in _OUTPUT_TYPES:
        response_type = output_type
    else:
        raise ValueError("unsupported task output_type")

    task_category = value.get("category")
    if task_category not in _TASK_CATEGORIES:
        raise ValueError("unsupported task category")
    conditions = value.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("task conditions must be an object")
    decimal = conditions.get("decimal")
    if isinstance(decimal, bool) or not isinstance(decimal, int):
        raise ValueError("task decimal condition must be an integer")
    if not -1 <= decimal <= 12:
        raise ValueError("task decimal condition is out of range")
    order = conditions.get("order")
    if not isinstance(order, bool):
        raise ValueError("task order condition must be boolean")
    return {
        "action_intent": infer_main_action_intent(query),
        "response_type": response_type,
        "task_category": task_category,
        "decimal_places": None if decimal == -1 else decimal,
        "order_sensitive": order,
    }


def _identifier_overlay_text(
    references: list[dict[str, str]],
    omitted: int,
) -> str:
    payload = {
        "omitted": omitted,
        "references": references,
        "version": SQL_SAFE_IDENTIFIER_OVERLAY_VERSION,
    }
    return "\n".join(
        [
            "[SQL-SAFE CANONICAL REFERENCES]",
            _canonical_json(payload),
            (
                "Authority limits: these are quoting-safe renderings of existing "
                "validated State references only; they add no table, column, JSON "
                "key, formula, predicate, aggregation, ordering, or output rule."
            ),
        ]
    )


def _quote_expression(value: str) -> str:
    expression = sqlglot.parse_one(value, read="postgres")
    return expression.sql(dialect="postgres", identify=True)


def _quote_table(value: str) -> str:
    return exp.to_table(value).sql(dialect="postgres", identify=True)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
