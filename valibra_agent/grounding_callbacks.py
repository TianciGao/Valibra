"""P3 NoOp Shadow callbacks composed around the frozen Baseline callbacks.

Shadow tracks only bounded lifecycle observations. It never changes the model
request, tool protocol, Baseline callback return value, or semantic grounding
state. Full tool responses remain solely in the existing Baseline trajectory.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from valibra_agent.requirement_grounding.models import (
    PendingToolCall,
    RequirementGroundingRuntime,
)
from valibra_agent.requirement_grounding.observations import (
    build_observation,
    canonical_json,
)
from valibra_agent.requirement_grounding.reducer import validate_runtime
from valibra_agent.requirement_grounding.service import (
    add_pending_tool_call,
    process_observation,
    process_phase_transition,
    remove_pending_tool_call,
)
from valibra_agent.requirement_grounding.telemetry import record_failure

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.tool_context import ToolContext
else:
    CallbackContext = Any
    LlmRequest = Any
    LlmResponse = Any
    ToolContext = Any


GROUNDING_RUNTIME_KEY = "valibra:grounding_runtime"
GROUNDING_SEQUENCE_KEY = "valibra:grounding_sequence"
SHADOW_AUDIT_KEY = "valibra_shadow"
SHADOW_MODE = "noop"

_MAX_ARGS_SUMMARY_CHARS = 768
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

_OBSERVATION_TYPES = {
    "execute_sql": "sql_execution",
    "get_schema": "schema",
    "get_all_column_meanings": "metadata",
    "get_column_meaning": "metadata",
    "get_all_external_knowledge_names": "knowledge",
    "get_knowledge_definition": "knowledge",
    "get_all_knowledge_definitions": "knowledge",
    "ask_user": "user_answer",
    "submit_sql": "submission",
}


async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Initialize Shadow state fail-open, then delegate without prompt edits."""

    state = getattr(callback_context, "state", None)
    if state is not None:
        try:
            _ensure_runtime(state)
        except Exception:
            # A Grounding initialization failure cannot replace Baseline.
            pass
    from system_agent import callbacks as baseline_callbacks

    return await baseline_callbacks.before_model_callback(
        callback_context,
        llm_request,
    )


async def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Delegate exactly once and never alter the model response."""

    from system_agent import callbacks as baseline_callbacks

    return await baseline_callbacks.after_model_callback(
        callback_context,
        llm_response,
    )


async def before_tool_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
) -> dict | None:
    """Let Baseline decide budget first, then create an exact Pending call."""

    from system_agent import callbacks as baseline_callbacks

    baseline_result = await baseline_callbacks.before_tool_callback(
        tool,
        args,
        tool_context,
    )
    if baseline_result is not None:
        return baseline_result

    state = getattr(tool_context, "state", None)
    if state is None:
        return baseline_result
    sequence = _current_sequence(state)
    try:
        function_call_id = _require_function_call_id(tool_context)
        tool_name = _tool_name(tool)
        phase_before = _phase(state.get("current_phase", 1))
        args_json = canonical_json(args)
        sequence = _next_sequence(state)
        pending = PendingToolCall(
            function_call_id=function_call_id,
            tool_name=tool_name,
            args_summary=_bounded_args_summary(args_json),
            args_digest=hashlib.sha256(args_json.encode("utf-8")).hexdigest(),
            phase_before=phase_before,
            sequence=sequence,
            started_at=None,
        )
        runtime = add_pending_tool_call(_ensure_runtime(state), pending)
        _store_runtime(state, runtime)
    except Exception as exc:
        _record_callback_failure(
            state,
            stage="service",
            exception=exc,
            sequence=sequence,
            function_call_id=_valid_context_identifier(tool_context),
        )
    return baseline_result


async def after_tool_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    tool_response: Any,
) -> Any:
    """Delegate once, consume the original response, and return its override."""

    from system_agent import callbacks as baseline_callbacks

    state = getattr(tool_context, "state", None)
    trajectory_before = (
        _trajectory_length(state) if state is not None else None
    )
    try:
        baseline_override = await baseline_callbacks.after_tool_callback(
            tool,
            args,
            tool_context,
            tool_response,
        )
    except BaseException:
        if state is not None:
            _cleanup_pending_best_effort(state, tool_context)
        raise

    if state is None:
        return baseline_override

    audit_index = _new_trajectory_index(state, trajectory_before)
    raw_log_ref = (
        f"session://tool_trajectory/{audit_index}"
        if audit_index is not None
        else None
    )
    sequence = _current_sequence(state)
    audit_metadata: dict[str, Any] | None = None
    try:
        function_call_id = _require_function_call_id(tool_context)
        runtime = _ensure_runtime(state)
        runtime, pending = remove_pending_tool_call(runtime, function_call_id)
        # Persist cleanup before any normalization/updater/reducer work.
        _store_runtime(state, runtime)
        if pending is None:
            return baseline_override

        tool_name = _tool_name(tool)
        if pending.tool_name != tool_name:
            raise ValueError(
                "pending tool name does not match the exact function call"
            )
        task_id = _task_id(state)
        sequence = _next_sequence(state)
        observation = build_observation(
            task_id=task_id,
            observation_type=_observation_type(tool_name),
            phase=pending.phase_before,
            sequence=sequence,
            source="adk_tool_result",
            raw=tool_response,
            raw_log_ref=raw_log_ref,
            function_call_id=function_call_id,
            invocation_id=_valid_invocation_id(tool_context),
            tool_name=tool_name,
        )
        result = process_observation(runtime, observation)
        runtime = result.runtime

        transition_observation_id: str | None = None
        phase_after = _phase(state.get("current_phase", pending.phase_before))
        if (
            tool_name == "submit_sql"
            and pending.phase_before == 1
            and phase_after == 2
        ):
            transition_sequence = _next_sequence(state)
            transition_observation = build_observation(
                task_id=task_id,
                observation_type="phase_transition",
                phase=2,
                sequence=transition_sequence,
                source="adk_lifecycle",
                raw={
                    "submission_observation_id": observation.observation_id,
                    "phase_before": 1,
                    "phase_after": 2,
                },
                summary="legal submit_sql transitioned Baseline phase 1 to phase 2",
                raw_log_ref=raw_log_ref,
                function_call_id=function_call_id,
                invocation_id=_valid_invocation_id(tool_context),
                tool_name=tool_name,
            )
            transition_result = process_phase_transition(
                runtime,
                transition_observation,
            )
            runtime = transition_result.runtime
            transition_observation_id = transition_observation.observation_id

        _store_runtime(state, runtime)
        audit_metadata = {
            "mode": SHADOW_MODE,
            "status": result.status,
            "function_call_id": function_call_id,
            "tool_name": tool_name,
            "phase_before": pending.phase_before,
            "phase_after": phase_after,
            "observation_id": observation.observation_id,
            "transition_observation_id": transition_observation_id,
            "raw_digest": observation.raw_digest,
            "raw_log_ref": raw_log_ref,
            "summary": observation.summary,
            "grounding_revision": runtime.grounding_revision,
            "runtime_bytes": _runtime_bytes(runtime),
        }
    except Exception as exc:
        function_call_id = _valid_context_identifier(tool_context)
        _cleanup_pending_best_effort(state, tool_context)
        runtime = _record_callback_failure(
            state,
            stage="observation",
            exception=exc,
            sequence=sequence,
            function_call_id=function_call_id,
        )
        audit_metadata = {
            "mode": SHADOW_MODE,
            "status": "failed",
            "function_call_id": function_call_id,
            "tool_name": _safe_tool_name(tool),
            "raw_log_ref": raw_log_ref,
            "error_type": type(exc).__name__[:128],
            "grounding_revision": runtime.grounding_revision,
            "runtime_bytes": _runtime_bytes(runtime),
        }
    finally:
        _cleanup_pending_best_effort(state, tool_context)
        if audit_index is not None and audit_metadata is not None:
            _attach_shadow_audit(state, audit_index, audit_metadata)

    return baseline_override


async def on_tool_error_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    error: Exception,
) -> None:
    """Clean only the exact Pending call, record bounded telemetry, return None."""

    del tool, args
    state = tool_context.state
    sequence = _current_sequence(state)
    function_call_id = _valid_context_identifier(tool_context)
    try:
        exact_id = _require_function_call_id(tool_context)
        runtime = _ensure_runtime(state)
        runtime, _ = remove_pending_tool_call(runtime, exact_id)
        sequence = _next_sequence(state)
        runtime = record_failure(
            runtime,
            stage="service",
            exception=error,
            sequence=sequence,
            function_call_id=exact_id,
        )
        _store_runtime(state, runtime)
    except Exception as callback_error:
        _record_callback_failure(
            state,
            stage="service",
            exception=callback_error,
            sequence=sequence,
            function_call_id=function_call_id,
        )
    return None


def _ensure_runtime(state: Any) -> RequirementGroundingRuntime:
    try:
        raw = state.get(GROUNDING_RUNTIME_KEY)
        runtime = (
            RequirementGroundingRuntime()
            if raw is None
            else RequirementGroundingRuntime.model_validate(raw)
        )
        validate_runtime(runtime)
    except Exception as exc:
        runtime = RequirementGroundingRuntime()
        try:
            runtime = record_failure(
                runtime,
                stage="service",
                exception=exc,
                sequence=_current_sequence(state),
            )
        except Exception:
            runtime = RequirementGroundingRuntime()
    _store_runtime(state, runtime)
    if state.get(GROUNDING_SEQUENCE_KEY) is None:
        state[GROUNDING_SEQUENCE_KEY] = 0
    return runtime


def _store_runtime(state: Any, runtime: RequirementGroundingRuntime) -> None:
    validate_runtime(runtime)
    state[GROUNDING_RUNTIME_KEY] = runtime.model_dump(mode="json")


def _record_callback_failure(
    state: Any,
    *,
    stage: str,
    exception: BaseException,
    sequence: int,
    function_call_id: str | None = None,
) -> RequirementGroundingRuntime:
    runtime = _ensure_runtime(state)
    try:
        runtime = record_failure(
            runtime,
            stage=stage,
            exception=exception,
            sequence=max(0, sequence),
            function_call_id=function_call_id,
        )
        _store_runtime(state, runtime)
    except Exception:
        # Grounding diagnostics must never replace a Baseline result/error.
        pass
    return runtime


def _cleanup_pending_best_effort(state: Any, tool_context: Any) -> None:
    function_call_id = _valid_context_identifier(tool_context)
    if function_call_id is None:
        return
    try:
        runtime = _ensure_runtime(state)
        runtime, _ = remove_pending_tool_call(runtime, function_call_id)
        _store_runtime(state, runtime)
    except Exception:
        return


def _require_function_call_id(tool_context: Any) -> str:
    function_call_id = getattr(tool_context, "function_call_id", None)
    if not isinstance(function_call_id, str) or not _IDENTIFIER_RE.fullmatch(
        function_call_id
    ):
        raise ValueError(
            "google-adk ToolContext.function_call_id is required; no fallback ID"
        )
    return function_call_id


def _valid_context_identifier(tool_context: Any) -> str | None:
    value = getattr(tool_context, "function_call_id", None)
    return value if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) else None


def _valid_invocation_id(tool_context: Any) -> str | None:
    value = getattr(tool_context, "invocation_id", None)
    return value if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) else None


def _tool_name(tool: Any) -> str:
    value = getattr(tool, "name", None)
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError("tool has no bounded official name")
    return value


def _safe_tool_name(tool: Any) -> str:
    value = getattr(tool, "name", None)
    if isinstance(value, str):
        return value[:64]
    return type(tool).__name__[:64]


def _observation_type(tool_name: str) -> str:
    try:
        return _OBSERVATION_TYPES[tool_name]
    except KeyError as exc:
        raise ValueError(f"unsupported a-interact tool: {tool_name}") from exc


def _task_id(state: Any) -> str:
    task_id = state.get("task_id")
    if not isinstance(task_id, str) or not _IDENTIFIER_RE.fullmatch(task_id):
        raise ValueError("bounded task_id is required for Shadow Observation")
    return task_id


def _phase(value: Any) -> int:
    if isinstance(value, bool) or value not in (1, 2):
        raise ValueError("phase must be exactly 1 or 2")
    return int(value)


def _current_sequence(state: Any) -> int:
    value = state.get(GROUNDING_SEQUENCE_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if 0 <= value <= _MAX_SEQUENCE else 0


def _next_sequence(state: Any) -> int:
    current = state.get(GROUNDING_SEQUENCE_KEY, 0)
    if (
        isinstance(current, bool)
        or not isinstance(current, int)
        or current < 0
        or current >= _MAX_SEQUENCE
    ):
        raise ValueError("invalid or exhausted Shadow sequence")
    next_value = current + 1
    state[GROUNDING_SEQUENCE_KEY] = next_value
    return next_value


def _bounded_args_summary(args_json: str) -> str:
    if len(args_json) <= _MAX_ARGS_SUMMARY_CHARS:
        return args_json
    return args_json[: _MAX_ARGS_SUMMARY_CHARS - 1] + "…"


def _trajectory_length(state: Any) -> int | None:
    trajectory = state.get("tool_trajectory", [])
    return len(trajectory) if isinstance(trajectory, list) else None


def _new_trajectory_index(state: Any, before: int | None) -> int | None:
    trajectory = state.get("tool_trajectory", [])
    if before is None or not isinstance(trajectory, list):
        return None
    if len(trajectory) != before + 1:
        return None
    return before


def _attach_shadow_audit(
    state: Any,
    index: int,
    metadata: dict[str, Any],
) -> None:
    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list) or not 0 <= index < len(trajectory):
        return
    event = trajectory[index]
    if not isinstance(event, dict):
        return
    # Fail closed on accidental expansion; this metadata never contains raw
    # tool response, prompt, credentials, or model content.
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 4096:
        return
    updated = list(trajectory)
    updated_event = dict(event)
    updated_event[SHADOW_AUDIT_KEY] = metadata
    updated[index] = updated_event
    state["tool_trajectory"] = updated


def _runtime_bytes(runtime: RequirementGroundingRuntime) -> int:
    return len(
        json.dumps(
            runtime.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
