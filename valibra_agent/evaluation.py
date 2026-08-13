"""Pure, fail-open export of bounded Valibra evaluation metadata.

The adapter consumes only the state already returned by ``run_session`` plus
the User Simulator audit that the frozen orchestrator has already fetched. It
never reads task data, environment internals, or network resources, and it
does not alter the official result or token ledger.
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
    GROUNDING_LEGACY_INITIALIZATION_UNKNOWN_KEY,
    GROUNDING_RUNTIME_KEY,
    GROUNDING_UPDATE_AUDIT_KEY,
    REQUIREMENT_VIEW_AUDIT_KEY,
    SHADOW_AUDIT_KEY,
)
from valibra_agent.requirement_grounding.models import (
    RequirementGroundingRuntime,
    migrate_requirement_grounding_runtime,
)
from valibra_agent.requirement_grounding.reducer import validate_runtime
from valibra_agent.requirement_grounding.semantic_projection import (
    requirement_semantic_sha256,
)
from valibra_agent.requirement_grounding.telemetry import (
    CombinedModelUsage,
    ModelUsageLedger,
    summarize_model_usage_totals,
)


class MissingGroundingRuntime(ValueError):
    """The run-session state contains no Valibra Runtime."""


def export_valibra_result(
    state: Mapping[str, Any],
    *,
    user_simulator_audit: Any = None,
) -> dict[str, Any]:
    """Return a JSON-safe additive result block without affecting scoring.

    Any invalid or missing Runtime produces only a bounded error type. The
    caller's SQL, reward, official trajectories, and legacy token accounting
    are never mutated.
    """

    try:
        if not isinstance(state, Mapping):
            raise TypeError("run-session state must be a mapping")
        raw_runtime = state.get(GROUNDING_RUNTIME_KEY)
        if raw_runtime is None:
            raise MissingGroundingRuntime("Grounding Runtime is missing")
        migration = migrate_requirement_grounding_runtime(raw_runtime)
        runtime = migration.runtime
        legacy_unknown_history = migration.legacy_unknown_history or bool(
            state.get(GROUNDING_LEGACY_INITIALIZATION_UNKNOWN_KEY, False)
        )
        validate_runtime(runtime)
        runtime_json = runtime.model_dump(mode="json")

        prompt_flow = _sequence(state.get("system_agent_llm_calls", []), "prompt_flow")
        tool_trajectory = _sequence(state.get("tool_trajectory", []), "tool_trajectory")
        dialogue_history = _sequence(state.get("dialogue_history", []), "dialogue_history")
        adk_events = _sequence(state.get("adk_events", []), "adk_events")
        main_usage = _mapping(
            state.get("system_agent_token_usage", {}),
            "system_agent_token_usage",
        )
        model_usage = summarize_model_usage_totals(
            main_usage,
            runtime,
            main_agent_cost=_reported_main_agent_cost(prompt_flow),
        )
        model_usage = _with_complete_grounding_cost(
            model_usage,
            _complete_grounding_cost(
                runtime,
                prompt_flow=prompt_flow,
                tool_trajectory=tool_trajectory,
            ),
        )
        metrics = runtime.metrics.root
        frame = runtime.grounding_state.requirement_frame
        manifest = _trajectory_manifest(
            prompt_flow=prompt_flow,
            tool_trajectory=tool_trajectory,
            dialogue_history=dialogue_history,
            adk_events=adk_events,
            user_simulator_audit=user_simulator_audit,
            runtime_json=runtime_json,
        )
        result = {
            "export_status": "succeeded",
            "runtime": runtime_json,
            "grounding_summary": {
                "grounding_revision": runtime.grounding_revision,
                "requirement_revision": runtime.requirement_revision,
                "requirement_semantic_sha256": (
                    requirement_semantic_sha256(runtime.grounding_state)
                ),
                "frame_initialization_status": (
                    runtime.frame_initialization_status
                ),
                "frame_initialization_reason": (
                    runtime.frame_initialization_reason
                ),
                "frame_initialization_observation_id": (
                    runtime.frame_initialization_observation_id
                ),
                "legacy_unknown_history": legacy_unknown_history,
                "phase": runtime.phase,
                "slots": {
                    "value": len(frame.value_slots),
                    "schema": len(frame.schema_slots),
                    "operation": len(frame.operation_slots),
                    "total": (
                        len(frame.value_slots)
                        + len(frame.schema_slots)
                        + len(frame.operation_slots)
                    ),
                },
                "evidence_count": len(runtime.grounding_state.evidence),
                "ambiguity_count": len(
                    runtime.grounding_state.ambiguity_index
                ),
                "processed_observation_count": len(
                    runtime.processed_observation_ids
                ),
                "pending_count": len(runtime.pending_tool_calls),
                "last_error": (
                    runtime.last_error.model_dump(mode="json")
                    if runtime.last_error is not None
                    else None
                ),
                "llm_calls": _metric_int(metrics, "llm_updater_calls"),
                "llm_errors": _metric_int(metrics, "llm_updater_errors"),
                "llm_timeouts": _metric_int(metrics, "llm_updater_timeouts"),
                "input_tokens": model_usage.grounding.input_tokens,
                "output_tokens": model_usage.grounding.output_tokens,
                "reasoning_tokens": model_usage.grounding.reasoning_tokens,
                "total_tokens": model_usage.grounding.total_tokens,
                "latency_ms": _metric_number(
                    metrics,
                    "llm_updater_latency_ms",
                ),
                "cost": model_usage.grounding.cost,
            },
            "model_usage": model_usage.model_dump(mode="json"),
            "latency": {
                "main_agent_ms": _main_agent_latency_ms(prompt_flow),
                "grounding_ms": _metric_number(
                    metrics,
                    "llm_updater_latency_ms",
                ),
            },
            "bird_coin": _bird_coin_snapshot(state),
            "trajectory_manifest": manifest,
        }
        # A final strict serialization gate prevents SDK objects or NaN values
        # from leaking into the benchmark result.
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
    collections = {
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
    }
    return {
        **collections,
        "locations": _trajectory_locations(
            prompt_flow,
            tool_trajectory,
        ),
    }


def _trajectory_locations(
    prompt_flow: Sequence[Any],
    tool_trajectory: Sequence[Any],
) -> dict[str, Any]:
    initial_queries = []
    views = []
    for index, item in enumerate(prompt_flow):
        if not isinstance(item, Mapping):
            continue
        update = item.get(GROUNDING_UPDATE_AUDIT_KEY)
        if isinstance(update, Mapping):
            initial_queries.append(
                {
                    "index": index,
                    "observation_id": update.get("observation_id"),
                    "status": update.get("status"),
                }
            )
        view = item.get(REQUIREMENT_VIEW_AUDIT_KEY)
        if isinstance(view, Mapping):
            views.append(
                {
                    "index": index,
                    "mode": view.get("effective_mode", view.get("mode")),
                    "injected": bool(view.get("injected", False)),
                    "view_sha256": view.get("view_sha256"),
                }
            )

    ask_user_answers = []
    phase_two_follow_ups = []
    tool_observations = []
    for index, item in enumerate(tool_trajectory):
        if not isinstance(item, Mapping):
            continue
        audit = item.get(SHADOW_AUDIT_KEY)
        if not isinstance(audit, Mapping):
            continue
        observation_type = audit.get("observation_type")
        location = {
            "index": index,
            "observation_id": audit.get("observation_id"),
            "observation_type": observation_type,
            "tool_name": audit.get("tool_name"),
            "status": audit.get("status"),
        }
        tool_observations.append(location)
        if observation_type == "user_answer":
            ask_user_answers.append(location)
        if audit.get("follow_up_observation_id") is not None:
            phase_two_follow_ups.append(
                {
                    "index": index,
                    "observation_id": audit.get("follow_up_observation_id"),
                    "status": audit.get("follow_up_status"),
                }
            )
    return {
        "initial_user_query_grounding": initial_queries,
        "ask_user_answer_grounding": ask_user_answers,
        "phase2_follow_up_grounding": phase_two_follow_ups,
        "tool_observations": tool_observations,
        "requirement_view_model_calls": views,
        "final_frame": {
            "collection": "grounding_runtime",
            "json_pointer": "/grounding_state/requirement_frame",
        },
    }


def _collection_manifest(value: Sequence[Any]) -> dict[str, Any]:
    return {
        "count": len(value),
        "sha256": _stable_sha256(value),
    }


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
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise TypeError(f"{name} must be a sequence")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _metric_int(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name} must be numeric")
    if not math.isfinite(float(value)) or value < 0 or int(value) != value:
        raise ValueError(f"metric {name} must be a non-negative integer")
    return int(value)


def _metric_number(metrics: Mapping[str, Any], name: str) -> int | float:
    value = metrics.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"metric {name} must be finite and non-negative")
    return value


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
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            return None
        costs.append(float(value))
    return sum(costs)


def _complete_grounding_cost(
    runtime: RequirementGroundingRuntime,
    *,
    prompt_flow: Sequence[Any],
    tool_trajectory: Sequence[Any],
) -> float | None:
    """Return cost only when every actual Grounding call reports one."""

    actual_calls = _metric_int(runtime.metrics.root, "llm_updater_calls")
    if actual_calls == 0:
        return 0.0

    audits = _attempted_grounding_audits(prompt_flow, tool_trajectory)
    if len(audits) != actual_calls:
        return None

    costs = []
    for audit in audits:
        value = audit.get("cost")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            return None
        costs.append(float(value))
    return sum(costs)


def _attempted_grounding_audits(
    prompt_flow: Sequence[Any],
    tool_trajectory: Sequence[Any],
) -> list[Mapping[str, Any]]:
    audits: list[Mapping[str, Any]] = []
    for item in prompt_flow:
        if not isinstance(item, Mapping):
            continue
        update = item.get(GROUNDING_UPDATE_AUDIT_KEY)
        if isinstance(update, Mapping):
            _append_attempted_audit(audits, update.get("llm"))
    for item in tool_trajectory:
        if not isinstance(item, Mapping):
            continue
        shadow = item.get(SHADOW_AUDIT_KEY)
        if not isinstance(shadow, Mapping):
            continue
        _append_attempted_audit(audits, shadow.get("llm"))
        _append_attempted_audit(audits, shadow.get("follow_up_llm"))
    return audits


def _append_attempted_audit(
    audits: list[Mapping[str, Any]],
    candidate: Any,
) -> None:
    if isinstance(candidate, Mapping) and candidate.get("attempted") is True:
        audits.append(candidate)


def _with_complete_grounding_cost(
    usage: CombinedModelUsage,
    grounding_cost: float | None,
) -> CombinedModelUsage:
    grounding = ModelUsageLedger.model_validate(
        {
            **usage.grounding.model_dump(mode="json"),
            "cost": grounding_cost,
        }
    )
    total_cost = (
        usage.main_agent.cost + grounding_cost
        if usage.main_agent.cost is not None and grounding_cost is not None
        else None
    )
    return CombinedModelUsage(
        main_agent=usage.main_agent,
        grounding=grounding,
        total_model_tokens=usage.total_model_tokens,
        total_model_cost=total_cost,
    )


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
