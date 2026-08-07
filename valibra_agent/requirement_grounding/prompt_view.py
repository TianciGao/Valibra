"""把 Grounding 状态渲染成稳定、有限的 Prompt View。

当前阶段不会把它注入模型；这里只提前验证未来注入时的数据格式。
"""

from __future__ import annotations

from valibra_agent.requirement_grounding.models import RequirementGroundingState

DEFAULT_MAX_CHARS = 2000
DEFAULT_MAX_ITEMS = 50
_HEADER = "[REQUIREMENT GROUNDING K0]"
_TRUNCATED = "...<truncated>"


def render_prompt_view(
    state: RequirementGroundingState,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """输出当前槽位和歧义，不包含原始证据正文或日志。"""

    if max_chars < 0 or max_items < 0:
        raise ValueError("prompt view limits must be non-negative")
    if max_chars == 0:
        return ""

    slots = sorted(
        (
            *state.requirement_frame.value_slots,
            *state.requirement_frame.schema_slots,
            *state.requirement_frame.operation_slots,
        ),
        key=lambda item: (item.slot_kind, item.slot_id),
    )
    ambiguities = sorted(
        state.ambiguity_index,
        key=lambda item: item.ambiguity_id,
    )
    entries: list[str] = []
    for slot in slots:
        interpretation = slot.current_interpretation or slot.mention or "<missing>"
        entries.append(
            f"SLOT {slot.slot_kind}:{slot.slot_id} "
            f"status={slot.grounding_status} lifecycle={slot.lifecycle} "
            f"interpretation={interpretation!r} "
            f"evidence={','.join(sorted(slot.evidence_refs)) or '-'}"
        )
    for ambiguity in ambiguities:
        candidates = ",".join(
            f"{item.candidate_id}:{item.sql_impact.effect_key}"
            for item in sorted(
                ambiguity.candidate_interpretations,
                key=lambda item: item.candidate_id,
            )
        ) or "-"
        entries.append(
            f"AMBIGUITY {ambiguity.ambiguity_id} status={ambiguity.status} "
            f"primary={ambiguity.primary_slot_id} candidates={candidates} "
            f"resolution={ambiguity.resolution or '-'}"
        )

    # 先限制条目数，再限制总字符数；超出的内容统一显示 omitted 数量。
    selected = entries[:max_items]
    omitted = len(entries) - len(selected)
    lines = [_HEADER]
    for entry in selected:
        candidate = "\n".join((*lines, entry))
        if len(candidate) > max_chars:
            omitted += len(selected) - (len(lines) - 1)
            break
        lines.append(entry)
    text = "\n".join(lines)
    if omitted > 0:
        marker = f"{_TRUNCATED} omitted={omitted}"
        with_marker = f"{text}\n{marker}"
        if len(with_marker) <= max_chars:
            text = with_marker
        elif max_chars >= len(_TRUNCATED):
            prefix = text[: max_chars - len(_TRUNCATED)]
            text = prefix + _TRUNCATED
        else:
            text = _TRUNCATED[:max_chars]
    return text[:max_chars]
