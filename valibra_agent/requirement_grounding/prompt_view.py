"""Render a stable, bounded Agent-facing view of Grounding state.

The view is currently audit-only.  Free text is represented as JSON data so a
slot cannot create new instruction lines in a future model-visible context.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken

from valibra_agent.requirement_grounding.models import RequirementGroundingState

DEFAULT_MAX_CHARS = 2000
DEFAULT_MAX_ITEMS = 50
DEFAULT_MAX_TOKENS = 512

_HEADER = "[CURRENT REQUIREMENT]"
_OMITTED = "... omitted={count}"
_NOTE_LINES = (
    "Note: This is a working requirement state, not database ground truth.",
    "Verify unbound schema concepts with the normal BIRD-Interact tools.",
)
_SECTION_ORDER = {
    "Values": 0,
    "Schema concepts": 1,
    "Operations": 2,
}


@dataclass(frozen=True, slots=True)
class _ViewItem:
    section: str
    sort_key: tuple[str, ...]
    line: str


@lru_cache(maxsize=1)
def _token_encoding() -> Any:
    """Return the fixed research tokenizer; there is intentionally no fallback."""

    return tiktoken.get_encoding("cl100k_base")


def count_prompt_view_tokens(view: str) -> int:
    """Count tokens with the frozen research-only ``cl100k_base`` metric."""

    if not isinstance(view, str):
        raise TypeError("prompt view must be a string")
    return len(_token_encoding().encode_ordinary(view))


def prompt_view_sha256(view: str) -> str:
    """Return the stable SHA256 of the final UTF-8 view."""

    if not isinstance(view, str):
        raise TypeError("prompt view must be a string")
    return hashlib.sha256(view.encode("utf-8")).hexdigest()


def render_prompt_view(
    state: RequirementGroundingState,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Render active Frame slots without internal IDs, evidence, or provenance.

    Limits are applied to the complete result.  A slot is either included in
    full or omitted; JSON strings are never cut in the middle.
    """

    _validate_limit("max_chars", max_chars)
    _validate_limit("max_items", max_items)
    _validate_limit("max_tokens", max_tokens)
    if max_chars == 0 or max_items == 0 or max_tokens == 0:
        return ""

    items = _active_items(state)
    if not items:
        return ""

    maximum_kept = min(len(items), max_items)
    for kept in range(maximum_kept, 0, -1):
        candidate = _assemble_view(items[:kept], omitted=len(items) - kept)
        if (
            len(candidate) <= max_chars
            and count_prompt_view_tokens(candidate) <= max_tokens
        ):
            return candidate
    return ""


def _validate_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _active_items(state: RequirementGroundingState) -> list[_ViewItem]:
    frame = state.requirement_frame
    items: list[_ViewItem] = []

    for slot in frame.value_slots:
        if slot.lifecycle != "active":
            continue
        role = _json_data(slot.slot_role)
        requirement = _json_data(_slot_text(slot))
        line = f"- {role}: {requirement}"
        items.append(
            _ViewItem(
                section="Values",
                sort_key=(role, requirement, line),
                line=line,
            )
        )

    for slot in frame.schema_slots:
        if slot.lifecycle != "active":
            continue
        role = _json_data(slot.slot_role)
        requirement = _json_data(_slot_text(slot))
        if slot.binding_type != "unknown" and slot.bound_identifier:
            binding = _canonical_json(
                {
                    "identifier": slot.bound_identifier,
                    "type": slot.binding_type,
                }
            )
            suffix = f"binding={binding}"
        else:
            suffix = "(schema not yet verified)"
        line = f"- {role}: {requirement} {suffix}"
        items.append(
            _ViewItem(
                section="Schema concepts",
                sort_key=(role, requirement, suffix, line),
                line=line,
            )
        )

    for slot in frame.operation_slots:
        if slot.lifecycle != "active":
            continue
        operation_type = _json_data(slot.operation_type)
        role = _json_data(slot.slot_role)
        requirement = _json_data(_slot_text(slot))
        parameters = _canonical_json(slot.parameters)
        line = (
            f"- {operation_type}: {requirement} "
            f"role={role} params={parameters}"
        )
        items.append(
            _ViewItem(
                section="Operations",
                sort_key=(operation_type, role, requirement, parameters, line),
                line=line,
            )
        )

    return sorted(
        items,
        key=lambda item: (_SECTION_ORDER[item.section], item.sort_key),
    )


def _slot_text(slot: Any) -> str:
    return slot.current_interpretation or slot.mention


def _json_data(value: Any) -> str:
    return _canonical_json(value)


def _canonical_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    # JSON permits these Unicode separators unescaped, but several renderers
    # treat them as physical line breaks.  Keep every free-text field on one
    # structural line.
    return encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _assemble_view(items: list[_ViewItem], *, omitted: int) -> str:
    lines = [_HEADER]
    for section in ("Values", "Schema concepts", "Operations"):
        section_items = [item.line for item in items if item.section == section]
        if not section_items:
            continue
        lines.extend(("", f"{section}:", *section_items))
    if omitted:
        lines.extend(("", _OMITTED.format(count=omitted)))
    lines.extend(("", *_NOTE_LINES))
    return "\n".join(lines)
