"""Pure, fail-open export of bounded SQL Grounding evaluation metadata.

The adapter consumes only the current ``valibra:sql_grounding_runtime`` and
bounded telemetry already present in Session State.  It never reads task data,
environment internals, private Provider audits, or network resources, and it
does not alter the Official result, trajectory, or Bird-Coin ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from shared.audit import to_jsonable
from valibra_agent.grounding_callbacks import (
    GROUNDING_CLARIFICATIONS_KEY,
    GROUNDING_PENDING_CHECK_KEY,
    GROUNDING_PHASE_OUTCOMES_KEY,
    GROUNDING_PROVIDER_CALL_COUNT_KEY,
    GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY,
    GROUNDING_RUNTIME_KEY,
    GROUNDING_UPDATE_AUDIT_KEY,
    GROUNDING_VIEW_AUDIT_KEY,
    SHADOW_AUDIT_KEY,
)
from valibra_agent.sql_grounding.models import (
    GroundingRuntime,
    sql_grounding_state_sha256,
)


class MissingGroundingRuntime(ValueError):
    """The run-session state contains no current SQL Grounding Runtime."""


def export_valibra_result(
    state: Mapping[str, Any],
    *,
    user_simulator_audit: Any = None,
) -> dict[str, Any]:
    """Return a JSON-safe research-only block from current Grounding facts.

    Invalid or missing state produces only a bounded error type.  The caller's
    SQL, reward, Official trajectories, and token accounting remain untouched.
    """

    try:
        if not isinstance(state, Mapping):
            raise TypeError("run-session state must be a mapping")
        raw_runtime = state.get(GROUNDING_RUNTIME_KEY)
        if raw_runtime is None:
            raise MissingGroundingRuntime("SQL Grounding Runtime is missing")
        runtime = GroundingRuntime.model_validate(raw_runtime)
        runtime_json = runtime.model_dump(mode="json")

        prompt_flow = _sequence(state.get("system_agent_llm_calls", []), "prompt_flow")
        tool_trajectory = _sequence(state.get("tool_trajectory", []), "tool_trajectory")
        dialogue_history = _sequence(state.get("dialogue_history", []), "dialogue_history")
        adk_events = _sequence(state.get("adk_events", []), "adk_events")
        main_usage = _main_usage_ledger(
            _mapping(
                state.get("system_agent_token_usage", {}),
                "system_agent_token_usage",
            ),
            cost=_reported_main_agent_cost(prompt_flow),
        )
        grounding_audits = _grounding_provider_audits(
            prompt_flow=prompt_flow,
            tool_trajectory=tool_trajectory,
        )
        configured_provider_calls = _provider_call_count(state)
        grounding_usage = _grounding_usage_ledger(
            grounding_audits,
            configured_provider_calls=configured_provider_calls,
        )
        total_cost = (
            main_usage["cost"] + grounding_usage["cost"]
            if main_usage["cost"] is not None
            and grounding_usage["cost"] is not None
            else None
        )
        grounding_state = runtime.grounding_state
        result = {
            "export_status": "succeeded",
            "runtime": runtime_json,
            "grounding_summary": {
                "grounding_revision": runtime.grounding_revision,
                "stage": runtime.stage,
                "focus_dimension": runtime.focus_dimension,
                "state_sha256": sql_grounding_state_sha256(grounding_state),
                "all_dimensions_evaluated": grounding_state.all_dimensions_evaluated,
                "dimensions": {
                    "tables": _dimension_count(grounding_state.tables),
                    "join_keys": _dimension_count(grounding_state.join_keys),
                    "column_mapping": _dimension_count(
                        grounding_state.column_mapping
                    ),
                    "domain_knowledge": _dimension_count(
                        grounding_state.domain_knowledge
                    ),
                },
                "provider_calls": configured_provider_calls,
                "provider_phase_calls": _provider_phase_call_counts(state),
                "provider_audits_captured": len(grounding_audits),
                "input_tokens": grounding_usage["input_tokens"],
                "output_tokens": grounding_usage["output_tokens"],
                "reasoning_tokens": grounding_usage["reasoning_tokens"],
                "total_tokens": grounding_usage["total_tokens"],
                "latency_ms": grounding_usage["latency_ms"],
                "cost": grounding_usage["cost"],
                "clarifications": _clarification_summary(state),
                "pending_check_tool": bool(state.get(GROUNDING_PENDING_CHECK_KEY)),
                "phase_outcome_count": len(_phase_outcomes(state)),
            },
            "model_usage": {
                "main_agent": main_usage,
                "grounding": grounding_usage,
                "total_model_tokens": (
                    main_usage["total_tokens"] + grounding_usage["total_tokens"]
                ),
                "total_model_cost": total_cost,
            },
            "latency": {
                "main_agent_ms": _main_agent_latency_ms(prompt_flow),
                "grounding_ms": grounding_usage["latency_ms"],
            },
            "bird_coin": _bird_coin_snapshot(state),
            "trajectory_manifest": _trajectory_manifest(
                prompt_flow=prompt_flow,
                tool_trajectory=tool_trajectory,
                dialogue_history=dialogue_history,
                adk_events=adk_events,
                user_simulator_audit=user_simulator_audit,
                runtime_json=runtime_json,
            ),
        }
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return result
    except Exception as exc:
        return {
            "export_status": "failed",
            "error_type": type(exc).__name__[:128],
        }


def _trajectory_manifest(
    *,
    prompt_flow: Sequence[Any],
    tool_trajectory: Sequence[Any],
    dialogue_history: Sequence[Any],
    adk_events: Sequence[Any],
    user_simulator_audit: Any,
    runtime_json: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "prompt_flow": _collection_manifest(prompt_flow),
        "tool_trajectory": _collection_manifest(tool_trajectory),
        "dialogue_history": _collection_manifest(dialogue_history),
        "adk_events": _collection_manifest(adk_events),
        "user_simulator_audit": {
            "count": 0 if user_simulator_audit is None else 1,
            "sha256": _stable_sha256(user_simulator_audit),
        },
        "grounding_runtime": {
            "count": 1,
            "sha256": _stable_sha256(runtime_json),
        },
        "locations": _trajectory_locations(prompt_flow, tool_trajectory),
    }


def _trajectory_locations(
    prompt_flow: Sequence[Any],
    tool_trajectory: Sequence[Any],
) -> dict[str, Any]:
    model_updates = []
    views = []
    for index, item in enumerate(prompt_flow):
        if not isinstance(item, Mapping):
            continue
        update = item.get(GROUNDING_UPDATE_AUDIT_KEY)
        if isinstance(update, Mapping):
            model_updates.append(
                {
                    "index": index,
                    "observation_id": update.get("observation_id"),
                    "observation_type": update.get("observation_type"),
                    "stage": update.get("stage"),
                    "service_status": update.get("service_status"),
                    "state_sha256": update.get("state_sha256"),
                }
            )
        view = item.get(GROUNDING_VIEW_AUDIT_KEY)
        if isinstance(view, Mapping):
            views.append(
                {
                    "index": index,
                    "mode": view.get("effective_mode", view.get("mode")),
                    "injected": bool(view.get("injected", False)),
                    "view_sha256": view.get("view_sha256"),
                }
            )

    tool_updates = []
    for index, item in enumerate(tool_trajectory):
        if not isinstance(item, Mapping):
            continue
        audit = item.get(SHADOW_AUDIT_KEY)
        if not isinstance(audit, Mapping):
            continue
        tool_updates.append(
            {
                "index": index,
                "observation_id": audit.get("observation_id"),
                "observation_type": audit.get("observation_type"),
                "tool_name": audit.get("tool_name"),
                "service_status": audit.get("service_status"),
                "state_sha256": audit.get("state_sha256"),
            }
        )
    return {
        "model_grounding_updates": model_updates,
        "tool_grounding_updates": tool_updates,
        "grounding_view_model_calls": views,
        "final_state": {
            "collection": "grounding_runtime",
            "json_pointer": "/grounding_state",
        },
    }


def _grounding_provider_audits(
    *,
    prompt_flow: Sequence[Any],
    tool_trajectory: Sequence[Any],
) -> tuple[Mapping[str, Any], ...]:
    audits: list[Mapping[str, Any]] = []
    for item in prompt_flow:
        if not isinstance(item, Mapping):
            continue
        update = item.get(GROUNDING_UPDATE_AUDIT_KEY)
        if isinstance(update, Mapping) and update.get("provider_attempted") is True:
            audits.append(update)
    for item in tool_trajectory:
        if not isinstance(item, Mapping):
            continue
        update = item.get(SHADOW_AUDIT_KEY)
        if isinstance(update, Mapping) and update.get("provider_attempted") is True:
            audits.append(update)
    return tuple(audits)


def _grounding_usage_ledger(
    audits: Sequence[Mapping[str, Any]],
    *,
    configured_provider_calls: int,
) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    latency_ms = 0.0
    costs: list[float] = []
    complete_cost = len(audits) == configured_provider_calls
    for audit in audits:
        usage = _mapping(audit.get("provider_usage", {}), "provider_usage")
        for name in totals:
            totals[name] += _non_negative_int(usage.get(name, 0), name)
        latency_ms += _non_negative_number(
            audit.get("provider_latency_ms", 0.0),
            "provider_latency_ms",
        )
        cost = audit.get("provider_reported_cost")
        if cost is None:
            complete_cost = False
        else:
            costs.append(_non_negative_number(cost, "provider_reported_cost"))
    return {
        "model_calls": configured_provider_calls,
        **totals,
        "latency_ms": latency_ms,
        "cost": sum(costs) if complete_cost else None,
    }


def _main_usage_ledger(
    usage: Mapping[str, Any],
    *,
    cost: float | None,
) -> dict[str, Any]:
    return {
        "model_calls": _non_negative_int(usage.get("model_calls", 0), "model_calls"),
        "input_tokens": _non_negative_int(
            usage.get("input_tokens", 0), "input_tokens"
        ),
        "output_tokens": _non_negative_int(
            usage.get("output_tokens", 0), "output_tokens"
        ),
        "reasoning_tokens": _non_negative_int(
            usage.get("reasoning_tokens", 0), "reasoning_tokens"
        ),
        "total_tokens": _non_negative_int(
            usage.get("total_tokens", 0), "total_tokens"
        ),
        "cost": cost,
    }


def _provider_call_count(state: Mapping[str, Any]) -> int:
    return _non_negative_int(
        state.get(GROUNDING_PROVIDER_CALL_COUNT_KEY, 0),
        GROUNDING_PROVIDER_CALL_COUNT_KEY,
    )


def _provider_phase_call_counts(state: Mapping[str, Any]) -> dict[str, int]:
    payload = state.get(GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY, {"1": 0, "2": 0})
    value = _mapping(payload, GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY)
    if set(value) != {"1", "2"}:
        raise ValueError("invalid SQL Grounding phase call counters")
    counts = {
        key: _non_negative_int(value[key], f"phase_{key}_provider_calls")
        for key in ("1", "2")
    }
    if sum(counts.values()) != _provider_call_count(state):
        raise ValueError("SQL Grounding total and phase call counters differ")
    return counts


def _clarification_summary(state: Mapping[str, Any]) -> dict[str, int]:
    records = _sequence(
        state.get(GROUNDING_CLARIFICATIONS_KEY, []),
        GROUNDING_CLARIFICATIONS_KEY,
    )
    answered = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("invalid clarification overlay")
        answered += int(record.get("answer") is not None)
    return {"total": len(records), "answered": answered}


def _phase_outcomes(state: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _mapping(
        state.get(GROUNDING_PHASE_OUTCOMES_KEY, {}),
        GROUNDING_PHASE_OUTCOMES_KEY,
    )
    if any(key not in {"1", "2"} for key in payload):
        raise ValueError("invalid SQL Grounding phase outcomes")
    return payload


def _dimension_count(value: Sequence[Any] | None) -> int | None:
    return None if value is None else len(value)


def _collection_manifest(value: Sequence[Any]) -> dict[str, Any]:
    return {"count": len(value), "sha256": _stable_sha256(value)}


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _reported_main_agent_cost(prompt_flow: Sequence[Any]) -> float | None:
    if not prompt_flow:
        return None
    costs = []
    for call in prompt_flow:
        if not isinstance(call, Mapping):
            return None
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            return None
        raw = usage.get("raw")
        if not isinstance(raw, Mapping):
            return None
        value = raw.get("cost", raw.get("response_cost"))
        if value is None:
            return None
        costs.append(_non_negative_number(value, "main_agent_cost"))
    return sum(costs)


def _main_agent_latency_ms(prompt_flow: Sequence[Any]) -> float | None:
    if not prompt_flow:
        return 0.0
    total = 0.0
    for call in prompt_flow:
        if not isinstance(call, Mapping):
            return None
        started = _parse_timestamp(call.get("timestamp"))
        completed = _parse_timestamp(call.get("completed_at"))
        if started is None or completed is None or completed < started:
            return None
        total += (completed - started).total_seconds() * 1000.0
    return total


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _bird_coin_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "initial": _optional_finite_number(state.get("initial_budget")),
        "remaining": _optional_finite_number(state.get("budget_remaining")),
        "included_in_model_usage": False,
    }


def _optional_finite_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value
