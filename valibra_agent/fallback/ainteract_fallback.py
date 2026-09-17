"""One-way native A-Interact fallback after a Valibra terminal failure.

The module is deliberately isolated from Grounding and Main callbacks.  It
does not reset Official services, phase state, accepted artifacts, or the
Bird-Coin ledger.  The feature is disabled unless explicitly enabled by the
environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from typing import Any, Mapping

from shared.config import settings
from system_agent import callbacks as baseline_callbacks
from system_agent.agent import (
    ADK_AVAILABLE,
    ADK_IMPORT_ERROR,
    AINTERACT_INSTRUCTION,
    Agent,
    _build_model,
    types,
)


FALLBACK_FEATURE_FLAG = "VALIBRA_AINTERACT_FALLBACK"
FALLBACK_MODE = "valibra-ainteract-fallback"
FALLBACK_AUDIT_KEY = "valibra:ainteract_fallback"
FALLBACK_VERSION = "ainteract-fallback-r1"
MINIMUM_REMAINING_BIRD_COIN = 3.0

_CLARIFICATIONS_KEY = "valibra:user_clarifications"
_GROUNDING_RUNTIME_KEY = "valibra:sql_grounding_runtime"
_MAIN_ENVELOPES_KEY = "valibra:main_execution_envelopes"
_SEEN_QUESTIONS_KEY = "_valibra_fallback_seen_question_sha256"
_DUPLICATE_ASK_MARKER = "_valibra_fallback_duplicate_ask"
_DUPLICATE_ASK_AUDITS = "_valibra_fallback_duplicate_ask_audits"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def fallback_enabled() -> bool:
    """Return the live default-off feature-flag value."""

    return os.environ.get(FALLBACK_FEATURE_FLAG, "").strip().lower() in _TRUE_VALUES


def fallback_configuration_report() -> dict[str, Any]:
    return {
        "feature_flag": FALLBACK_FEATURE_FLAG,
        "enabled": fallback_enabled(),
        "minimum_remaining_bird_coin": MINIMUM_REMAINING_BIRD_COIN,
        "mode": FALLBACK_MODE,
        "version": FALLBACK_VERSION,
    }


def _finite_budget(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fallback_eligibility(state: Mapping[str, Any]) -> tuple[bool, str]:
    """Evaluate the terminal checkpoint; caller guarantees the turn ended."""

    if not fallback_enabled():
        return False, "feature_disabled"
    audit = state.get(FALLBACK_AUDIT_KEY)
    if isinstance(audit, Mapping) and bool(audit.get("attempted")):
        return False, "already_attempted"
    if bool(state.get("task_done")):
        return False, "task_already_complete"
    phase = state.get("current_phase", 1)
    if phase not in (1, 2):
        return False, "invalid_phase"
    remaining = _finite_budget(state.get("budget_remaining"))
    if remaining is None:
        return False, "invalid_budget"
    if remaining < MINIMUM_REMAINING_BIRD_COIN:
        return False, "insufficient_budget"
    return True, "terminal_fail_with_legal_budget"


def _question_digest(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _answered_clarifications(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    raw = state.get(_CLARIFICATIONS_KEY)
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping) or item.get("answer") is None:
                continue
            result.append(
                {
                    "phase": item.get("phase"),
                    "phrase": item.get("phrase"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                }
            )
    if result:
        return result

    dialogue = state.get("dialogue_history")
    if not isinstance(dialogue, list):
        return result
    pending_question: str | None = None
    for item in dialogue:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if role == "agent":
            pending_question = content
        elif role == "user" and pending_question is not None:
            result.append(
                {
                    "phase": None,
                    "phrase": None,
                    "question": pending_question,
                    "answer": content,
                }
            )
            pending_question = None
    return result


def _official_follow_up(state: Mapping[str, Any]) -> str | None:
    trajectory = state.get("tool_trajectory")
    if not isinstance(trajectory, list):
        return None
    marker = "Follow-up question: "
    values: list[str] = []
    for event in trajectory:
        if not isinstance(event, Mapping) or event.get("tool") != "submit_sql":
            continue
        result = event.get("result")
        if not isinstance(result, str) or marker not in result:
            continue
        value = result.split(marker, 1)[1].splitlines()[0].strip()
        if value:
            values.append(value)
    return values[-1] if values else None


def _accepted_p1_sql(state: Mapping[str, Any]) -> str | None:
    trajectory = state.get("tool_trajectory")
    if not isinstance(trajectory, list):
        return None
    for event in trajectory:
        if (
            not isinstance(event, Mapping)
            or event.get("tool") != "submit_sql"
            or int(event.get("phase", 1)) != 1
        ):
            continue
        result = event.get("result")
        if not isinstance(result, str) or "Phase 1 correct!" not in result:
            continue
        args = event.get("args")
        if isinstance(args, Mapping) and isinstance(args.get("sql"), str):
            return args["sql"]
    return None


def _grounding_hint(state: Mapping[str, Any]) -> Any:
    runtime = state.get(_GROUNDING_RUNTIME_KEY)
    if not isinstance(runtime, Mapping):
        return None
    value = runtime.get("grounding_state")
    return copy.deepcopy(value) if isinstance(value, Mapping) else None


def _phase_envelope(state: Mapping[str, Any], phase: int) -> Any:
    envelopes = state.get(_MAIN_ENVELOPES_KEY)
    if not isinstance(envelopes, Mapping):
        return None
    phases = envelopes.get("phases")
    if not isinstance(phases, Mapping):
        return None
    value = phases.get(str(phase))
    return copy.deepcopy(value) if isinstance(value, Mapping) else None


def build_fallback_message(state: Mapping[str, Any]) -> str:
    """Build the only new user message for the independent native session."""

    phase = int(state.get("current_phase", 1))
    remaining = _finite_budget(state.get("budget_remaining"))
    payload = {
        "task_id": state.get("task_id"),
        "database": state.get("db_name"),
        "current_phase": phase,
        "original_query": state.get("user_query"),
        "official_follow_up": _official_follow_up(state) if phase == 2 else None,
        "answered_clarifications": _answered_clarifications(state),
        "accepted_phase_1_sql": _accepted_p1_sql(state),
        "public_execution_envelope": _phase_envelope(state, phase),
        "non_authoritative_failed_4d_state_hint": _grounding_hint(state),
        "remaining_bird_coin": remaining,
    }
    return (
        "ONE-TIME NATIVE A-INTERACT FALLBACK\n\n"
        "The Valibra path has ended in a terminal failure. You now have one "
        "independent native A-Interact attempt in the same Official task and DB "
        "session. Solve the current phase and submit the best SQL you can.\n\n"
        "Hard boundaries:\n"
        "- The remaining Bird-Coin shown below is the real inherited ledger; it "
        "has not been reset.\n"
        "- Preserve any accepted phase-1 artifact and the current Official DB state.\n"
        "- Answered clarifications are binding user intent. Do not ask the same "
        "question again.\n"
        "- The four-dimensional State is explicitly non-authoritative because the "
        "Valibra path failed. You may use it as a hint, correct it, or discard it.\n"
        "- Do not infer benchmark/reference-only facts and do not return to Valibra.\n\n"
        "FALLBACK CONTEXT (lawful task/session data):\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    )


def start_fallback_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the live session state and append an auditable one-way marker."""

    copied = copy.deepcopy(dict(state))
    phase = int(copied.get("current_phase", 1))
    clarifications = _answered_clarifications(copied)
    seen = {
        _question_digest(str(item["question"]))
        for item in clarifications
        if isinstance(item.get("question"), str) and item["question"].strip()
    }
    copied[_SEEN_QUESTIONS_KEY] = sorted(seen)
    copied[FALLBACK_AUDIT_KEY] = {
        "version": FALLBACK_VERSION,
        "attempted": True,
        "status": "started",
        "trigger": "terminal_fail_with_legal_budget",
        "phase_before": phase,
        "budget_before": _finite_budget(copied.get("budget_remaining")),
        "phase1_completed_before": bool(copied.get("phase1_completed")),
        "phase2_completed_before": bool(copied.get("phase2_completed")),
        "inherited_clarification_count": len(clarifications),
        "failed_4d_state_authority": "non_authoritative_hint",
        "accepted_p1_sql_carried": _accepted_p1_sql(copied) is not None,
        "message_sha256": hashlib.sha256(
            build_fallback_message(copied).encode("utf-8")
        ).hexdigest(),
    }
    return copied


def finish_fallback_state(
    state: dict[str, Any],
    *,
    tool_count_before: int,
    error_type: str | None = None,
) -> dict[str, Any]:
    audit = dict(state.get(FALLBACK_AUDIT_KEY) or {})
    trajectory = state.get("tool_trajectory")
    tool_count_after = len(trajectory) if isinstance(trajectory, list) else 0
    if error_type is not None:
        status = "fallback_error"
    elif bool(state.get("task_done")):
        status = "task_rescued"
    elif (
        bool(state.get("phase1_completed"))
        and not bool(audit.get("phase1_completed_before"))
    ):
        status = "phase1_rescued_only"
    else:
        status = "not_rescued"
    audit.update(
        {
            "status": status,
            "phase_after": state.get("current_phase"),
            "budget_after": _finite_budget(state.get("budget_remaining")),
            "phase1_completed_after": bool(state.get("phase1_completed")),
            "phase2_completed_after": bool(state.get("phase2_completed")),
            "task_done_after": bool(state.get("task_done")),
            "fallback_tool_calls": max(0, tool_count_after - tool_count_before),
            "duplicate_ask_blocks": len(
                state.get(_DUPLICATE_ASK_AUDITS) or []
            ),
            "error_type": error_type,
        }
    )
    state[FALLBACK_AUDIT_KEY] = audit
    return state


async def _fallback_before_tool_callback(tool, args: dict, tool_context):
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    if tool_name == "ask_user":
        question = args.get("question")
        if isinstance(question, str):
            digest = _question_digest(question)
            seen = set(tool_context.state.get(_SEEN_QUESTIONS_KEY) or [])
            if digest in seen:
                blocked = {"question_sha256": digest, "reason": "already_answered"}
                audits = list(tool_context.state.get(_DUPLICATE_ASK_AUDITS) or [])
                audits.append(blocked)
                tool_context.state[_DUPLICATE_ASK_AUDITS] = audits
                tool_context.state[_DUPLICATE_ASK_MARKER] = digest
                return {
                    "error": "This clarification question was already answered. "
                    "Reuse the inherited answer and do not spend Bird-Coin asking it again."
                }
            seen.add(digest)
            tool_context.state[_SEEN_QUESTIONS_KEY] = sorted(seen)
    return await baseline_callbacks.before_tool_callback(tool, args, tool_context)


def _consume_state_marker(state: Any, key: str) -> Any:
    """Read and clear one callback marker without requiring ``MutableMapping``.

    ADK's live ``State`` supports ``get`` and item assignment, but deliberately
    does not expose ``dict.pop``.  Storing ``None`` is the same marker-cleared
    representation already used by the callback state carriers.
    """

    value = state.get(key)
    if value is not None:
        state[key] = None
    return value


async def _fallback_after_tool_callback(
    tool, args: dict, tool_context, tool_response
):
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    if tool_name == "ask_user" and _consume_state_marker(
        tool_context.state, _DUPLICATE_ASK_MARKER
    ) is not None:
        return None
    return await baseline_callbacks.after_tool_callback(
        tool, args, tool_context, tool_response
    )


def build_fallback_agent() -> Agent:
    """Build the native A-Interact agent with only a duplicate-ask guard."""

    if not ADK_AVAILABLE:
        raise RuntimeError(f"google-adk runtime unavailable: {ADK_IMPORT_ERROR}")
    from system_agent.tools import get_ainteract_tools

    return Agent(
        model=_build_model(settings.system_agent_model),
        name="bird_interact_fallback_agent",
        description="One-time native A-Interact fallback after Valibra terminal failure.",
        instruction=AINTERACT_INSTRUCTION,
        tools=get_ainteract_tools(),
        before_model_callback=baseline_callbacks.before_model_callback,
        after_model_callback=baseline_callbacks.after_model_callback,
        before_tool_callback=_fallback_before_tool_callback,
        after_tool_callback=_fallback_after_tool_callback,
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.system_agent_temperature
        ),
    )
