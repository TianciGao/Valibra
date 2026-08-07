"""P4.1 的确定性语言规则，只使用 Python 标准库。

规则宁可漏掉也不乱猜，只产出有限的临时候选；不看 Schema、不绑定字段、不做 I/O，
也不调用模型。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, TypeAlias

HintCategory: TypeAlias = Literal[
    "time",
    "comparison",
    "ordering",
    "top_k",
    "aggregation",
    "negation",
    "range",
    "schema_candidate",
]
HintSlotKind: TypeAlias = Literal["value", "operation", "schema"]
JsonScalar: TypeAlias = str | int | float | bool | None

MAX_HINT_INPUT_CHARS = 4096
MAX_HINTS = 32


@dataclass(frozen=True, slots=True)
class LinguisticHint:
    """从输入文本中识别出的一条稳定临时线索。"""

    hint_id: str
    category: HintCategory
    slot_kind: HintSlotKind
    slot_role: str
    mention: str
    interpretation: str
    start: int
    end: int
    value_type: str | None = None
    operation_type: str | None = None
    parameters: tuple[tuple[str, JsonScalar], ...] = ()


_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|"
    r"(?:this|last|next)\s+(?:day|week|month|quarter|year)|"
    r"(?:past|last|next)\s+[1-9]\d{0,3}\s+"
    r"(?:days?|weeks?|months?|quarters?|years?))\b"
)

_COMPARISON_PATTERNS = (
    (re.compile(r"\b(?:at\s+least|no\s+less\s+than)\s+(-?\d+(?:\.\d+)?)\b"), "gte"),
    (re.compile(r"\b(?:at\s+most|no\s+more\s+than)\s+(-?\d+(?:\.\d+)?)\b"), "lte"),
    (re.compile(r"\b(?:more\s+than|greater\s+than|above|over)\s+(-?\d+(?:\.\d+)?)\b"), "gt"),
    (re.compile(r"\b(?:less\s+than|below|under)\s+(-?\d+(?:\.\d+)?)\b"), "lt"),
)
_ORDER_RE = re.compile(r"\b(ascending|descending|highest|lowest)\b")
_TOP_K_RE = re.compile(r"\b(top|bottom)\s+([1-9]\d{0,3})\b")
_AGGREGATION_RE = re.compile(
    r"\b(average|avg|mean|count|number\s+of|sum|total|maximum|max|minimum|min)\b"
)
_NEGATION_RE = re.compile(
    r"\b(without|excluding|exclude|except|"
    r"(?:do|does)\s+not\s+(?:include|contain|match)|"
    r"not\s+(?:equal\s+to|in|like))\b"
)
_RANGE_RE = re.compile(
    r"\b(?:between\s+(-?\d+(?:\.\d+)?|(?:19|20)\d{2}-\d{2}-\d{2})"
    r"\s+and\s+(-?\d+(?:\.\d+)?|(?:19|20)\d{2}-\d{2}-\d{2})|"
    r"from\s+(-?\d+(?:\.\d+)?|(?:19|20)\d{2}-\d{2}-\d{2})"
    r"\s+to\s+(-?\d+(?:\.\d+)?|(?:19|20)\d{2}-\d{2}-\d{2}))\b"
)
_NOUN_PHRASE_RE = re.compile(
    r"\b(?:show|list|return|display|find)\s+"
    r"(?:me\s+)?(?:the\s+)?"
    r"(?P<phrase>[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,3}?)"
    r"(?=\s+(?:with|where|whose|who|that|by|from|for|in|between|"
    r"ordered|sorted|having|above|below|under|over)\b|[?.!,]|$)"
)

_GENERIC_SCHEMA_PHRASES = {
    "data",
    "information",
    "records",
    "results",
    "entries",
    "things",
    "everything",
    "anything",
}
_AGGREGATION_FUNCTIONS = {
    "average": "avg",
    "avg": "avg",
    "mean": "avg",
    "count": "count",
    "number of": "count",
    "sum": "sum",
    "total": "sum",
    "maximum": "max",
    "max": "max",
    "minimum": "min",
    "min": "min",
}


def normalize_linguistic_text(text: str) -> str:
    """统一 Unicode、空白和大小写，得到稳定文本。"""

    if not isinstance(text, str):
        raise TypeError("linguistic hint input must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    if len(normalized) > MAX_HINT_INPUT_CHARS:
        raise ValueError("linguistic hint input exceeds character limit")
    return normalized


def extract_linguistic_hints(text: str) -> tuple[LinguisticHint, ...]:
    """按原文顺序提取日期、比较、排序、聚合等高精度线索。"""

    normalized = normalize_linguistic_text(text)
    if not normalized:
        return ()

    found: list[LinguisticHint] = []
    occupied_time_spans: list[tuple[int, int]] = []

    # 先识别完整日期和相对时间，避免年份规则重复命中日期中的年份。
    for pattern in (_DATE_RE, _RELATIVE_TIME_RE):
        for match in pattern.finditer(normalized):
            occupied_time_spans.append(match.span())
            _append_hint(
                found,
                category="time",
                slot_kind="value",
                slot_role="time_constraint",
                match=match,
                interpretation=f"temporal constraint: {match.group(0)}",
                value_type="time",
            )
    for match in _YEAR_RE.finditer(normalized):
        if any(_spans_overlap(match.span(), span) for span in occupied_time_spans):
            continue
        _append_hint(
            found,
            category="time",
            slot_kind="value",
            slot_role="time_constraint",
            match=match,
            interpretation=f"temporal constraint: {match.group(0)}",
            value_type="time",
        )

    for pattern, operator in _COMPARISON_PATTERNS:
        for match in pattern.finditer(normalized):
            _append_hint(
                found,
                category="comparison",
                slot_kind="operation",
                slot_role="comparison_filter",
                match=match,
                interpretation=f"comparison filter: {operator} {match.group(1)}",
                operation_type="filter",
                parameters=(("operator", operator), ("threshold", match.group(1))),
            )

    for match in _ORDER_RE.finditer(normalized):
        token = match.group(1)
        direction = "asc" if token in {"ascending", "lowest"} else "desc"
        _append_hint(
            found,
            category="ordering",
            slot_kind="operation",
            slot_role="ordering",
            match=match,
            interpretation=f"ordering direction: {direction}",
            operation_type="order",
            parameters=(("direction", direction),),
        )

    for match in _TOP_K_RE.finditer(normalized):
        direction = "desc" if match.group(1) == "top" else "asc"
        limit = int(match.group(2))
        _append_hint(
            found,
            category="top_k",
            slot_kind="operation",
            slot_role="top_k",
            match=match,
            interpretation=f"ranked limit: {match.group(1)} {limit}",
            operation_type="limit",
            parameters=(("direction", direction), ("limit", limit)),
        )

    for match in _AGGREGATION_RE.finditer(normalized):
        token = match.group(1)
        function = _AGGREGATION_FUNCTIONS[token]
        _append_hint(
            found,
            category="aggregation",
            slot_kind="operation",
            slot_role="aggregation",
            match=match,
            interpretation=f"aggregation function: {function}",
            operation_type="aggregation",
            parameters=(("function", function),),
        )

    for match in _NEGATION_RE.finditer(normalized):
        _append_hint(
            found,
            category="negation",
            slot_kind="operation",
            slot_role="negation_filter",
            match=match,
            interpretation="explicit negative filter",
            operation_type="filter",
            parameters=(("negated", True),),
        )

    for match in _RANGE_RE.finditer(normalized):
        lower = match.group(1) or match.group(3)
        upper = match.group(2) or match.group(4)
        _append_hint(
            found,
            category="range",
            slot_kind="operation",
            slot_role="range_filter",
            match=match,
            interpretation=f"inclusive range: {lower} to {upper}",
            operation_type="filter",
            parameters=(("lower", lower), ("upper", upper)),
        )

    for match in _NOUN_PHRASE_RE.finditer(normalized):
        phrase = match.group("phrase").strip()
        if phrase in _GENERIC_SCHEMA_PHRASES:
            continue
        _append_hint(
            found,
            category="schema_candidate",
            slot_kind="schema",
            slot_role="schema_candidate",
            match=match,
            mention=phrase,
            interpretation=f"candidate schema concept: {phrase}",
        )

    unique: dict[str, LinguisticHint] = {}
    # 排序后按稳定 ID 去重，保证同一输入每次产出完全一致。
    for hint in sorted(found, key=lambda item: (item.start, item.end, item.category, item.hint_id)):
        unique.setdefault(hint.hint_id, hint)
    if len(unique) > MAX_HINTS:
        raise ValueError("linguistic hint output exceeds item limit")
    return tuple(unique.values())


def _append_hint(
    target: list[LinguisticHint],
    *,
    category: HintCategory,
    slot_kind: HintSlotKind,
    slot_role: str,
    match: re.Match[str],
    interpretation: str,
    mention: str | None = None,
    value_type: str | None = None,
    operation_type: str | None = None,
    parameters: tuple[tuple[str, JsonScalar], ...] = (),
) -> None:
    """把一次正则命中转成带稳定 ID 的 Hint。"""

    normalized_mention = mention or match.group(0)
    normalized_parameters = tuple(sorted(parameters))
    identity = {
        "category": category,
        "slot_kind": slot_kind,
        "slot_role": slot_role,
        "mention": normalized_mention,
        "interpretation": interpretation,
        "value_type": value_type,
        "operation_type": operation_type,
        "parameters": normalized_parameters,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    target.append(
        LinguisticHint(
            hint_id=f"hint.{digest[:24]}",
            category=category,
            slot_kind=slot_kind,
            slot_role=slot_role,
            mention=normalized_mention,
            interpretation=interpretation,
            start=match.start(),
            end=match.end(),
            value_type=value_type,
            operation_type=operation_type,
            parameters=normalized_parameters,
        )
    )


def _spans_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    """判断两个半开文本区间是否重叠。"""

    return first[0] < second[1] and second[0] < first[1]
