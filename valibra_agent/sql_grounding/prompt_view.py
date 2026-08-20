"""Stable, bounded renderers for the four-dimensional SQL Grounding State."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken

from valibra_agent.sql_grounding.models import (
    SQLGroundingState,
    UserClarificationRecord,
)

DEFAULT_MAX_VIEW_CHARS = 4_096
DEFAULT_MAX_VIEW_ITEMS = 96
DEFAULT_MAX_VIEW_TOKENS = 1_024
_HEADER = "[VALIBRA DATABASE GROUNDING]"
_SECTIONS = (
    ("Tables", "tables"),
    ("Relations", "join_keys"),
    ("Column mappings", "column_mapping"),
    ("Domain knowledge", "domain_knowledge"),
)
_CLARIFICATION_SECTION = "[USER CLARIFICATIONS]"


@dataclass(frozen=True, slots=True)
class RenderedGroundingView:
    text: str
    sha256: str
    char_count: int
    token_count: int
    included_items: int
    omitted_items: int


@lru_cache(maxsize=1)
def _token_encoding() -> Any:
    return tiktoken.get_encoding("cl100k_base")


def count_grounding_view_tokens(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("Grounding View must be a string")
    return len(_token_encoding().encode_ordinary(text))


def render_grounding_view(
    state: SQLGroundingState,
    *,
    clarifications: tuple[UserClarificationRecord, ...] = (),
    max_chars: int = DEFAULT_MAX_VIEW_CHARS,
    max_items: int = DEFAULT_MAX_VIEW_ITEMS,
    max_tokens: int = DEFAULT_MAX_VIEW_TOKENS,
) -> RenderedGroundingView:
    """Render four State dimensions plus answered, independent overlays."""

    for name, value in (
        ("max_chars", max_chars),
        ("max_items", max_items),
        ("max_tokens", max_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    # Answered user clarifications are the highest-priority model-visible
    # facts.  Select them before any four-dimensional State item so bounded
    # views never discard a clarification merely to retain lower-priority
    # database Grounding content.
    items = _clarification_items(clarifications) + _view_items(state)
    maximum_kept = min(len(items), max_items)
    for kept in range(maximum_kept, -1, -1):
        text = _assemble(items[:kept], omitted=len(items) - kept)
        token_count = count_grounding_view_tokens(text) if text else 0
        if len(text) <= max_chars and token_count <= max_tokens:
            return RenderedGroundingView(
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                char_count=len(text),
                token_count=token_count,
                included_items=kept,
                omitted_items=len(items) - kept,
            )
    return RenderedGroundingView(
        text="",
        sha256=hashlib.sha256(b"").hexdigest(),
        char_count=0,
        token_count=0,
        included_items=0,
        omitted_items=len(items),
    )


def _view_items(state: SQLGroundingState) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for section, attribute in _SECTIONS:
        value = getattr(state, attribute)
        if value is None:
            result.append((section, "null"))
            continue
        if not value:
            result.append((section, "[]"))
            continue
        for item in value:
            payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            result.append((section, _safe_json(payload)))
    section_order = {name: index for index, (name, _) in enumerate(_SECTIONS)}
    return sorted(result, key=lambda item: (section_order[item[0]], item[1]))


def _safe_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _clarification_items(
    records: tuple[UserClarificationRecord, ...],
) -> list[tuple[str, str]]:
    answered = [record for record in records if record.answer is not None]
    return [
        (
            _CLARIFICATION_SECTION,
            _safe_json(
                {
                    "answer": record.answer,
                    "kind": record.kind,
                    "phrase": record.phrase,
                    "question": record.question,
                }
            ),
        )
        for record in sorted(
            answered,
            key=lambda item: (item.phase, item.phrase, item.kind, item.question),
        )
    ]


def _assemble(items: list[tuple[str, str]], *, omitted: int) -> str:
    lines = [_HEADER]
    clarification_items = [
        value for item_section, value in items
        if item_section == _CLARIFICATION_SECTION
    ]
    if clarification_items:
        lines.extend(("", _CLARIFICATION_SECTION))
        lines.extend(f"- {value}" for value in clarification_items)
    for section, _ in _SECTIONS:
        lines.extend(("", f"{section}:"))
        section_items = [value for item_section, value in items if item_section == section]
        lines.extend(f"- {value}" for value in section_items)
    if omitted:
        lines.extend(("", f"... omitted={omitted}"))
    return "\n".join(lines)
