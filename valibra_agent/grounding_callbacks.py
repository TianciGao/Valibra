"""SQL Grounding V1 shadow callbacks around the frozen B0 callbacks.

SG4 observes the real ADK lifecycle and drives the new four-dimensional
``sql_grounding`` service with an opt-in real Provider or local passthrough.  It
never changes the model request, model response, tool protocol, Bird-Coin,
submit state machine, or official trajectory.  Control, Attempt Gate, and
active View injection are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from shared.audit import to_jsonable
from shared.config import PROJECT_ROOT
from valibra_agent.sql_grounding.models import (
    SQL_GROUNDING_RUNTIME_KEY,
    GroundingLLMResponse,
    GroundingRuntime,
    ValidationContext,
    canonical_json,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.observations import (
    ObservationType,
    SQLGroundingObservation,
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.prompt_view import render_grounding_view
from valibra_agent.sql_grounding.service import (
    SQLGroundingServiceResult,
    process_sql_grounding_observation,
)
from valibra_agent.sql_grounding.telemetry import GroundingLLMTelemetry
from valibra_agent.sql_grounding.updater import (
    SQL_GROUNDING_CONFIGURATION_SHA256,
    SQL_GROUNDING_FORM_SCHEMA_SHA256,
    SQL_GROUNDING_PROMPT_SHA256,
    GroundingUpdaterResult,
    build_real_sql_grounding_updater,
    load_sql_grounding_llm_config,
    requested_sql_grounding_updater_mode,
)

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


GROUNDING_RUNTIME_KEY = SQL_GROUNDING_RUNTIME_KEY
GROUNDING_PENDING_KEY = "valibra:sql_grounding_pending"
GROUNDING_SEQUENCE_KEY = "valibra:sql_grounding_sequence"
SHADOW_AUDIT_KEY = "valibra_sql_grounding_shadow"
GROUNDING_VIEW_AUDIT_KEY = "valibra_sql_grounding_view"
GROUNDING_UPDATE_AUDIT_KEY = "valibra_sql_grounding_update"
GROUNDING_ERROR_AUDIT_KEY = "valibra:sql_grounding_error_audits"
GROUNDING_PROVIDER_CALL_COUNT_KEY = "valibra:sql_grounding_provider_calls"

# The frozen P6 export module imports these names at module load.  They are
# retained only so that historical, read-only export code remains importable;
# SG3 never reads either key and never runs the retired semantic core.
GROUNDING_LEGACY_INITIALIZATION_UNKNOWN_KEY = (
    "valibra:frame_initialization_legacy_unknown"
)
REQUIREMENT_VIEW_AUDIT_KEY = GROUNDING_VIEW_AUDIT_KEY

_MAX_ARGS_SUMMARY_CHARS = 768
_MAX_AUDIT_BYTES = 4_096
_MAX_ERROR_AUDITS = 64
_MAX_PENDING = 64
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")

_TOOL_OBSERVATION_TYPES: Mapping[str, ObservationType] = {
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
_OFFICIAL_TOOL_ERROR_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "execute_sql": ("SQL Error:", "Error calling DB environment:"),
    "get_schema": ("Error:",),
    "get_all_column_meanings": ("Error:",),
    "get_column_meaning": ("Error:",),
    "get_all_external_knowledge_names": ("Error:",),
    "get_knowledge_definition": ("Error:",),
    "get_all_knowledge_definitions": ("Error:",),
    "submit_sql": ("Error:",),
}
_FOLLOW_UP_PREFIX = "Follow-up question: "
_BUDGET_PREFIX = "\nBudget remaining: "


class _PassthroughSQLGroundingUpdater:
    """Default SG4 updater: no I/O, no Provider, and no State guesswork."""

    async def propose(
        self,
        runtime: GroundingRuntime,
        observation: SQLGroundingObservation,
        *,
        original_query: str,
        follow_up_query: str | None = None,
    ) -> GroundingUpdaterResult:
        del observation, original_query, follow_up_query
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=runtime.grounding_state,
                next_focus_dimension=runtime.focus_dimension,
            ),
            telemetry=GroundingLLMTelemetry(
                attempted=False,
                status="succeeded",
                request_sha256="",
                response_sha256="",
                prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
                form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
                configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
            ),
            transport_normalization="none",
        )


_PASSTHROUGH_SQL_GROUNDING_UPDATER = _PassthroughSQLGroundingUpdater()
# Tests may replace this exact object with a deterministic fake.  Production
# resolves the real client only when the explicit mode is exactly ``llm``.
_SQL_GROUNDING_UPDATER: Any = _PASSTHROUGH_SQL_GROUNDING_UPDATER


@dataclass(slots=True)
class _BoundTurnMessage:
    task_id: str
    mode: str
    message: str
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _PendingToolCall:
    function_call_id: str
    tool_name: str
    phase_before: Literal[1, 2]
    args_digest: str
    args_summary: str
    sequence: int

    def to_json(self) -> dict[str, Any]:
        return {
            "function_call_id": self.function_call_id,
            "tool_name": self.tool_name,
            "phase_before": self.phase_before,
            "args_digest": self.args_digest,
            "args_summary": self.args_summary,
            "sequence": self.sequence,
        }

    @classmethod
    def from_json(cls, payload: Any) -> "_PendingToolCall":
        if not isinstance(payload, dict) or set(payload) != {
            "function_call_id",
            "tool_name",
            "phase_before",
            "args_digest",
            "args_summary",
            "sequence",
        }:
            raise ValueError("invalid SQL Grounding pending record")
        record = cls(**payload)
        if not _IDENTIFIER_RE.fullmatch(record.function_call_id):
            raise ValueError("invalid pending function_call_id")
        if not record.tool_name or len(record.tool_name) > 64:
            raise ValueError("invalid pending tool_name")
        if record.phase_before not in (1, 2):
            raise ValueError("invalid pending phase")
        if not re.fullmatch(r"[0-9a-f]{64}", record.args_digest):
            raise ValueError("invalid pending args digest")
        if len(record.args_summary) > _MAX_ARGS_SUMMARY_CHARS:
            raise ValueError("pending args summary is too large")
        if not isinstance(record.sequence, int) or not 1 <= record.sequence <= _MAX_SEQUENCE:
            raise ValueError("invalid pending sequence")
        return record


@dataclass(frozen=True, slots=True)
class _ObservationResult:
    runtime: GroundingRuntime
    service_status: str
    observation: SQLGroundingObservation
    service_result: SQLGroundingServiceResult | None = None


_ACTIVE_TURN_MESSAGE: ContextVar[_BoundTurnMessage | None] = ContextVar(
    "valibra_sql_grounding_active_turn_message",
    default=None,
)


def _bind_turn_message(task_id: str, mode: str, message: str) -> Any:
    return _ACTIVE_TURN_MESSAGE.set(
        _BoundTurnMessage(task_id=task_id, mode=mode, message=message)
    )


def _reset_turn_message(token: Any) -> None:
    _ACTIVE_TURN_MESSAGE.reset(token)


async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Observe one bound query and render a non-injected four-dimensional View."""

    state = getattr(callback_context, "state", None)
    model_call_count = _model_call_count(state)
    request_before: str | None = None
    update_audit: dict[str, Any] | None = None
    view_audit: dict[str, Any] = {
        "mode": "shadow",
        "injected": False,
    }
    try:
        request_before = _request_sha256(llm_request)
        view_audit["request_sha256_before"] = request_before
        if state is not None:
            runtime, degraded = _ensure_runtime(state)
            update_audit = await _consume_bound_user_message(state, runtime)
            runtime, later_degraded = _ensure_runtime(state)
            degraded = degraded or later_degraded
            view = render_grounding_view(runtime.grounding_state)
            view_audit.update(
                {
                    "grounding_revision": runtime.grounding_revision,
                    "stage": runtime.stage,
                    "focus_dimension": runtime.focus_dimension,
                    "state_sha256": sql_grounding_state_sha256(
                        runtime.grounding_state
                    ),
                    "view_sha256": view.sha256,
                    "chars": view.char_count,
                    "tokens_cl100k": view.token_count,
                    "included_items": view.included_items,
                    "omitted_items": view.omitted_items,
                    "runtime_degraded": degraded,
                }
            )
        view_audit["request_sha256_after_shadow"] = _request_sha256(
            llm_request
        )
    except Exception as exc:
        view_audit.update(
            {
                "error_type": type(exc).__name__[:128],
                "runtime_degraded": True,
            }
        )
        if state is not None:
            _append_error_audit(
                state,
                _bounded_error_audit("before_model", exc),
            )

    from system_agent import callbacks as baseline_callbacks

    baseline_result = await baseline_callbacks.before_model_callback(
        callback_context,
        llm_request,
    )
    try:
        request_after = _request_sha256(llm_request)
        view_audit["request_sha256_after"] = request_after
        view_audit["request_unchanged"] = bool(
            request_before is not None
            and request_before
            == view_audit.get("request_sha256_after_shadow")
            == request_after
        )
        if state is not None:
            model_call_index = _new_model_call_index(state, model_call_count)
            if model_call_index is not None:
                _attach_model_call_audit(
                    state,
                    model_call_index,
                    GROUNDING_VIEW_AUDIT_KEY,
                    view_audit,
                )
                if update_audit is not None:
                    _attach_model_call_audit(
                        state,
                        model_call_index,
                        GROUNDING_UPDATE_AUDIT_KEY,
                        update_audit,
                    )
    except Exception as exc:
        if state is not None:
            _append_error_audit(
                state,
                _bounded_error_audit("before_model_audit", exc),
            )
    return baseline_result


async def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Return the frozen Baseline callback result without modification."""

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
    """Let B0 own budget rejection, then register one exact pending call."""

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
    try:
        function_call_id = _require_function_call_id(tool_context)
        tool_name = _tool_name(tool)
        args_json = canonical_json(to_jsonable(args))
        pending = _PendingToolCall(
            function_call_id=function_call_id,
            tool_name=tool_name,
            phase_before=_phase(state.get("current_phase", 1)),
            args_digest=_sha256_text(args_json),
            args_summary=_bounded_args_summary(args_json),
            sequence=_next_sequence(state),
        )
        _add_pending(state, pending)
    except Exception as exc:
        _append_error_audit(
            state,
            _bounded_error_audit(
                "before_tool",
                exc,
                function_call_id=_valid_context_identifier(tool_context),
            ),
        )
    return baseline_result


async def after_tool_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    tool_response: Any,
) -> Any:
    """Delegate once, then observe the original response and return B0 override."""

    from system_agent import callbacks as baseline_callbacks

    state = getattr(tool_context, "state", None)
    trajectory_before = _trajectory_length(state)
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
    private_ref = (
        f"session://tool_trajectory/{audit_index}"
        if audit_index is not None
        else None
    )
    audit: dict[str, Any]
    try:
        function_call_id = _require_function_call_id(tool_context)
        pending = _pop_pending(state, function_call_id)
        if pending is None:
            audit = {
                "service_status": "skipped_missing_pending",
                "function_call_id": function_call_id,
                "tool_name": _safe_tool_name(tool),
                "private_raw_ref": private_ref,
            }
        else:
            tool_name = _tool_name(tool)
            if pending.tool_name != tool_name:
                raise ValueError("pending tool name does not match function_call_id")
            raw_content = to_jsonable(tool_response)
            observation_type = _classify_tool_observation_type(
                tool_name,
                tool_response,
            )
            observation = build_sql_grounding_observation(
                task_id=_task_id(state),
                phase=pending.phase_before,
                sequence=_next_sequence(state),
                observation_type=observation_type,
                content=raw_content,
                summary=f"official {tool_name} result observed",
                tool_name=tool_name,
                function_call_id=function_call_id,
                private_raw_ref=private_ref,
            )
            result = await _handle_observation(state, observation)
            runtime = result.runtime
            follow_up_audit: dict[str, Any] | None = None
            phase_after = _phase(
                state.get("current_phase", pending.phase_before)
            )
            if (
                tool_name == "submit_sql"
                and observation_type == "submission"
                and pending.phase_before == 1
                and phase_after == 2
            ):
                follow_up = _extract_submit_follow_up(tool_response)
                follow_up_observation = build_sql_grounding_observation(
                    task_id=_task_id(state),
                    phase=2,
                    sequence=_next_sequence(state),
                    observation_type="p2_follow_up",
                    content=follow_up,
                    summary="official Phase-2 follow-up observed",
                    private_raw_ref=private_ref,
                )
                follow_up_result = await _handle_observation(
                    state,
                    follow_up_observation,
                )
                runtime = follow_up_result.runtime
                follow_up_audit = _observation_audit(follow_up_result)
            audit = _observation_audit(result)
            audit.update(
                {
                    "function_call_id": function_call_id,
                    "tool_name": tool_name,
                    "phase_before": pending.phase_before,
                    "phase_after": phase_after,
                    "args_digest": pending.args_digest,
                    "private_raw_ref": private_ref,
                    "official_error": observation_type == "tool_error",
                }
            )
            if follow_up_audit is not None:
                audit["p2_follow_up"] = follow_up_audit
            _store_runtime(state, runtime)
    except Exception as exc:
        _cleanup_pending_best_effort(state, tool_context)
        runtime, _ = _ensure_runtime(state)
        audit = {
            "service_status": "failed_open",
            "function_call_id": _valid_context_identifier(tool_context),
            "tool_name": _safe_tool_name(tool),
            "private_raw_ref": private_ref,
            "error_type": type(exc).__name__[:128],
            **_runtime_audit(runtime),
        }
        _append_error_audit(state, audit)
    finally:
        _cleanup_pending_best_effort(state, tool_context)
        if audit_index is not None:
            try:
                _attach_tool_audit(state, audit_index, audit)
            except Exception as exc:
                _append_error_audit(
                    state,
                    _bounded_error_audit(
                        "after_tool_audit",
                        exc,
                        function_call_id=_valid_context_identifier(
                            tool_context
                        ),
                    ),
                )
    return baseline_override


async def on_tool_error_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    error: Exception,
) -> None:
    """Observe an ADK exception, clear its exact pending call, and stay invisible."""

    del args
    state = getattr(tool_context, "state", None)
    if state is None:
        return None
    function_call_id = _valid_context_identifier(tool_context)
    try:
        exact_id = _require_function_call_id(tool_context)
        pending = _pop_pending(state, exact_id)
        if pending is None:
            return None
        tool_name = _tool_name(tool)
        if pending.tool_name != tool_name:
            raise ValueError("pending tool name does not match function_call_id")
        observation = build_sql_grounding_observation(
            task_id=_task_id(state),
            phase=pending.phase_before,
            sequence=_next_sequence(state),
            observation_type="tool_error",
            content={"error_type": type(error).__name__[:128]},
            summary=f"official {tool_name} exception observed",
            tool_name=tool_name,
            function_call_id=exact_id,
        )
        result = await _handle_observation(state, observation)
        _store_runtime(state, result.runtime)
        audit = _observation_audit(result)
        audit.update(
            {
                "function_call_id": exact_id,
                "tool_name": tool_name,
                "official_exception": True,
            }
        )
        _append_error_audit(state, audit)
    except Exception as callback_error:
        _append_error_audit(
            state,
            _bounded_error_audit(
                "tool_error",
                callback_error,
                function_call_id=function_call_id,
            ),
        )
    finally:
        _cleanup_pending_best_effort(state, tool_context)
    return None


async def _consume_bound_user_message(
    state: Any,
    runtime: GroundingRuntime | None = None,
) -> dict[str, Any] | None:
    bound = _ACTIVE_TURN_MESSAGE.get()
    if bound is None or bound.consumed or bound.mode != "a-interact":
        return None
    bound.consumed = True
    if _task_id(state) != bound.task_id:
        raise ValueError("run_turn task_id does not match Session state")
    query = _user_message_query(bound.message)
    observation = build_sql_grounding_observation(
        task_id=bound.task_id,
        phase=_phase(state.get("current_phase", 1)),
        sequence=_next_sequence(state),
        observation_type="user_query",
        content=query,
        summary="current user query observed",
    )
    active_runtime = runtime if runtime is not None else _ensure_runtime(state)[0]
    result = await _handle_observation(state, observation, active_runtime)
    _store_runtime(state, result.runtime)
    return _observation_audit(result)


async def _handle_observation(
    state: Any,
    observation: SQLGroundingObservation,
    runtime: GroundingRuntime | None = None,
) -> _ObservationResult:
    active_runtime = runtime if runtime is not None else _ensure_runtime(state)[0]
    if observation.observation_type in {"submission", "p2_follow_up"}:
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_control_not_active",
            observation=observation,
        )
    if observation.observation_type == "tool_error":
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_tool_error_audit_only",
            observation=observation,
        )
    if observation.observation_type == "user_answer":
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_affected_dimensions_unfrozen",
            observation=observation,
        )
    try:
        updater = _resolve_sql_grounding_updater()
        if _uses_real_provider_adapter():
            llm_config = load_sql_grounding_llm_config(PROJECT_ROOT)
            calls = _provider_call_count(state)
            if calls >= llm_config.max_calls_per_task:
                return _ObservationResult(
                    runtime=active_runtime,
                    service_status="skipped_provider_call_limit",
                    observation=observation,
                )
    except Exception:
        return _ObservationResult(
            runtime=active_runtime,
            service_status="degraded_configuration",
            observation=observation,
        )
    context = _build_validation_context(state, observation)
    service_result = await process_sql_grounding_observation(
        active_runtime,
        observation,
        context,
        updater,
    )
    if service_result.llm_telemetry.attempted and _uses_real_provider_adapter():
        state[GROUNDING_PROVIDER_CALL_COUNT_KEY] = _provider_call_count(state) + 1
    return _ObservationResult(
        runtime=service_result.runtime,
        service_status=service_result.state_update.status,
        observation=observation,
        service_result=service_result,
    )


def _build_validation_context(
    state: Any,
    observation: SQLGroundingObservation,
) -> ValidationContext:
    bound = _ACTIVE_TURN_MESSAGE.get()
    if bound is None or bound.task_id != _task_id(state):
        raise ValueError("current bound query is required for ValidationContext")
    known_tables: set[str] = set()
    known_columns: set[str] = set()
    supported_knowledge: set[tuple[str, str]] = set()
    for event in _completed_sql_grounding_trajectory(state):
        _project_official_evidence(
            tool_name=event["tool_name"],
            observation_type=event["observation_type"],
            content=event["content"],
            known_tables=known_tables,
            known_columns=known_columns,
            supported_knowledge=supported_knowledge,
        )
    _project_official_evidence(
        tool_name=observation.tool_name,
        observation_type=observation.observation_type,
        content=observation.content,
        known_tables=known_tables,
        known_columns=known_columns,
        supported_knowledge=supported_knowledge,
    )
    return ValidationContext(
        current_query=_user_message_query(bound.message),
        follow_up_query=(
            observation.content
            if observation.observation_type == "p2_follow_up"
            and isinstance(observation.content, str)
            else None
        ),
        latest_observation_id=observation.observation_id,
        official_trajectory_observation_ids=_official_trajectory_refs(state),
        known_tables=frozenset(known_tables),
        known_columns=frozenset(known_columns),
        supported_domain_knowledge=frozenset(supported_knowledge),
    )


def _resolve_sql_grounding_updater() -> Any:
    requested = requested_sql_grounding_updater_mode()
    if requested == "":
        return _SQL_GROUNDING_UPDATER
    if requested != "llm":
        raise ValueError("invalid GROUNDING_UPDATER_MODE")
    if _SQL_GROUNDING_UPDATER is not _PASSTHROUGH_SQL_GROUNDING_UPDATER:
        return _SQL_GROUNDING_UPDATER
    return build_real_sql_grounding_updater(PROJECT_ROOT)


def _is_real_provider_mode() -> bool:
    return requested_sql_grounding_updater_mode() == "llm"


def _uses_real_provider_adapter() -> bool:
    return (
        _is_real_provider_mode()
        and _SQL_GROUNDING_UPDATER is _PASSTHROUGH_SQL_GROUNDING_UPDATER
    )


def _provider_call_count(state: Any) -> int:
    value = state.get(GROUNDING_PROVIDER_CALL_COUNT_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid SQL Grounding Provider call counter")
    return value


def _completed_sql_grounding_trajectory(state: Any) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list):
        return ()
    for event in trajectory:
        if not isinstance(event, dict):
            continue
        audit = event.get(SHADOW_AUDIT_KEY)
        tool_name = event.get("tool")
        if not isinstance(audit, dict) or not isinstance(tool_name, str):
            continue
        observation_type = audit.get("observation_type")
        observation_id = audit.get("observation_id")
        if not isinstance(observation_type, str) or not isinstance(observation_id, str):
            continue
        result.append(
            {
                "tool_name": tool_name,
                "observation_type": observation_type,
                "content": event.get("result"),
            }
        )
    return tuple(result[-512:])


def _project_official_evidence(
    *,
    tool_name: str | None,
    observation_type: str,
    content: Any,
    known_tables: set[str],
    known_columns: set[str],
    supported_knowledge: set[tuple[str, str]],
) -> None:
    if observation_type == "schema" and tool_name == "get_schema":
        try:
            tables, columns = _parse_schema_projection(content)
        except ValueError:
            return
        known_tables.update(tables)
        known_columns.update(columns)
        return
    if observation_type == "knowledge" and tool_name == "get_knowledge_definition":
        definition = _exact_knowledge_definition(content)
        if definition is not None:
            supported_knowledge.add(("business_rule", definition))


def _parse_schema_projection(content: Any) -> tuple[frozenset[str], frozenset[str]]:
    """Project only explicit CREATE TABLE DDL; sample rows are never inspected."""

    if not isinstance(content, str) or not content:
        raise ValueError("schema content must be text")
    statements: list[str] = []
    active: list[str] | None = None
    in_sample_rows = False
    for line in content.splitlines():
        if active is None and re.match(
            r'^\s*(?:First|"First")\s+3\s+rows\s*:',
            line,
            re.IGNORECASE,
        ):
            in_sample_rows = True
            continue
        if in_sample_rows:
            if line.strip() == "...":
                in_sample_rows = False
            continue
        if active is None:
            if re.match(r'^\s*(?:CREATE|"CREATE")\s+TABLE\b', line, re.IGNORECASE):
                active = [line]
            continue
        active.append(line)
        if re.match(r"^\s*\);\s*$", line):
            statements.append("\n".join(active))
            active = None
    if active is not None or not statements:
        raise ValueError("schema contains no complete CREATE TABLE DDL")
    tables: set[str] = set()
    columns: set[str] = set()
    for statement in statements:
        normalized = re.sub(
            r'^\s*"CREATE"\s+TABLE\b',
            "CREATE TABLE",
            statement,
            count=1,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r'(?m)^(\s*)"(PRIMARY|FOREIGN)"\s+KEY\b',
            r"\1\2 KEY",
            normalized,
            flags=re.IGNORECASE,
        )
        try:
            parsed = sqlglot.parse_one(normalized, read="postgres")
        except ParseError as exc:
            raise ValueError("schema DDL parse failed") from exc
        if not isinstance(parsed, exp.Create) or str(parsed.args.get("kind", "")).upper() != "TABLE":
            raise ValueError("schema entry is not CREATE TABLE")
        schema = parsed.this
        if not isinstance(schema, exp.Schema) or not isinstance(schema.this, exp.Table):
            raise ValueError("CREATE TABLE has no structured schema")
        table = _ddl_table_identifier(schema.this)
        if table in tables:
            raise ValueError("duplicate CREATE TABLE identifier")
        tables.add(table)
        for definition in schema.expressions:
            if isinstance(definition, exp.ColumnDef):
                name = definition.name
                if not name:
                    raise ValueError("schema column has no identifier")
                columns.add(f"{table}.{name}")
    return frozenset(tables), frozenset(columns)


def _ddl_table_identifier(table: exp.Table) -> str:
    parts = [part for part in (table.db, table.name) if part]
    if not parts or len(parts) > 2:
        raise ValueError("unsupported CREATE TABLE identifier")
    return ".".join(parts)


def _exact_knowledge_definition(content: Any) -> str | None:
    value = content
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict) or set(value).isdisjoint({"definition"}):
        return None
    definition = value.get("definition")
    if not isinstance(definition, str) or not definition or definition != definition.strip():
        return None
    if len(definition) > 2_048:
        return None
    return definition


def _ensure_runtime(state: Any) -> tuple[GroundingRuntime, bool]:
    """Load only the SQL Grounding key; corrupt values reset fail-open."""

    raw = state.get(GROUNDING_RUNTIME_KEY)
    degraded = False
    if raw is None:
        runtime = GroundingRuntime()
    else:
        try:
            runtime = GroundingRuntime.model_validate(raw)
        except Exception:
            runtime = GroundingRuntime()
            degraded = True
    _store_runtime(state, runtime)
    if state.get(GROUNDING_SEQUENCE_KEY) is None:
        state[GROUNDING_SEQUENCE_KEY] = 0
    if state.get(GROUNDING_PENDING_KEY) is None:
        state[GROUNDING_PENDING_KEY] = {}
    return runtime, degraded


def _store_runtime(state: Any, runtime: GroundingRuntime) -> None:
    validated = GroundingRuntime.model_validate(runtime)
    state[GROUNDING_RUNTIME_KEY] = validated.model_dump(mode="json")


def _observation_audit(result: _ObservationResult) -> dict[str, Any]:
    audit = {
        "observation_id": result.observation.observation_id,
        "observation_type": result.observation.observation_type,
        "phase": result.observation.phase,
        "raw_digest": result.observation.raw_digest,
        "private_raw_ref": result.observation.private_raw_ref,
        "service_status": result.service_status,
        **_runtime_audit(result.runtime),
    }
    if result.service_result is not None:
        update = result.service_result.state_update
        llm = result.service_result.llm_telemetry
        audit.update(
            {
                "changed_dimensions": list(update.changed_dimensions),
                "focus_before": update.focus_before,
                "focus_after": update.focus_after,
                "provider_attempted": llm.attempted,
                "provider_status": llm.status,
                "provider_error_type": llm.error_type,
                "provider_model": llm.model,
                "provider_name": llm.provider,
                "credential_source": llm.credential_source,
                "provider_reported_cost": llm.provider_reported_cost,
                "provider_latency_ms": llm.latency_ms,
                "provider_usage": llm.usage.model_dump(mode="json"),
                "provider_request_sha256": llm.request_sha256,
                "provider_response_sha256": llm.response_sha256,
                "raw_private_audit_ref": llm.raw_private_audit_ref,
                "provider_may_continue_after_cancel": (
                    llm.provider_may_continue_after_cancel
                ),
                "provider_may_bill_after_cancel": llm.provider_may_bill_after_cancel,
                "service_error_type": update.error_type,
            }
        )
    return audit


def _runtime_audit(runtime: GroundingRuntime) -> dict[str, Any]:
    return {
        "grounding_revision": runtime.grounding_revision,
        "stage": runtime.stage,
        "focus_dimension": runtime.focus_dimension,
        "state_sha256": sql_grounding_state_sha256(runtime.grounding_state),
        "runtime_bytes": len(canonical_json(runtime).encode("utf-8")),
    }


def _load_pending(state: Any) -> dict[str, _PendingToolCall]:
    payload = state.get(GROUNDING_PENDING_KEY, {})
    if not isinstance(payload, dict) or len(payload) > _MAX_PENDING:
        raise ValueError("invalid SQL Grounding pending store")
    result: dict[str, _PendingToolCall] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("pending key must be a string")
        record = _PendingToolCall.from_json(value)
        if record.function_call_id != key:
            raise ValueError("pending key does not match function_call_id")
        result[key] = record
    return result


def _store_pending(state: Any, pending: Mapping[str, _PendingToolCall]) -> None:
    if len(pending) > _MAX_PENDING:
        raise ValueError("too many pending SQL Grounding calls")
    state[GROUNDING_PENDING_KEY] = {
        key: pending[key].to_json() for key in sorted(pending)
    }


def _add_pending(state: Any, pending: _PendingToolCall) -> None:
    records = _load_pending(state)
    if pending.function_call_id in records:
        raise ValueError("duplicate pending function_call_id")
    records[pending.function_call_id] = pending
    _store_pending(state, records)


def _pop_pending(state: Any, function_call_id: str) -> _PendingToolCall | None:
    records = _load_pending(state)
    record = records.pop(function_call_id, None)
    _store_pending(state, records)
    return record


def _cleanup_pending_best_effort(state: Any, tool_context: Any) -> None:
    function_call_id = _valid_context_identifier(tool_context)
    if function_call_id is None:
        return
    try:
        _pop_pending(state, function_call_id)
    except Exception:
        return


def _official_trajectory_refs(state: Any) -> tuple[str, ...]:
    refs: list[str] = []
    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list):
        return ()
    for event in trajectory:
        if not isinstance(event, dict):
            continue
        audit = event.get(SHADOW_AUDIT_KEY)
        if not isinstance(audit, dict):
            continue
        observation_id = audit.get("observation_id")
        if isinstance(observation_id, str) and observation_id not in refs:
            refs.append(observation_id)
    return tuple(refs[-512:])


def _classify_tool_observation_type(
    tool_name: str,
    tool_response: Any,
) -> ObservationType:
    prefixes = _OFFICIAL_TOOL_ERROR_PREFIXES.get(tool_name, ())
    if isinstance(tool_response, str) and any(
        tool_response.startswith(prefix) for prefix in prefixes
    ):
        return "tool_error"
    try:
        return _TOOL_OBSERVATION_TYPES[tool_name]
    except KeyError as exc:
        raise ValueError(f"unsupported a-interact tool: {tool_name}") from exc


def _extract_submit_follow_up(tool_response: Any) -> str:
    if not isinstance(tool_response, str):
        raise ValueError("submit_sql follow-up response must be a string")
    starts: list[int] = []
    offset = 0
    while True:
        index = tool_response.find(_FOLLOW_UP_PREFIX, offset)
        if index < 0:
            break
        if index == 0 or tool_response[index - 1] == "\n":
            starts.append(index)
        offset = index + len(_FOLLOW_UP_PREFIX)
    if len(starts) != 1:
        raise ValueError("submit_sql response must contain one follow-up marker")
    value_start = starts[0] + len(_FOLLOW_UP_PREFIX)
    value_end = tool_response.find(_BUDGET_PREFIX, value_start)
    if value_end < 0:
        raise ValueError("follow-up must precede official budget line")
    value = tool_response[value_start:value_end].strip()
    if not value or len(value) > 32_768:
        raise ValueError("invalid bounded submit_sql follow-up")
    return value


def _user_message_query(message: str) -> str:
    if not isinstance(message, str) or not message:
        raise ValueError("run_turn message must be a non-empty string")
    marker = "User Query:\n"
    suffix = "\n\nYou have a budget"
    start = message.find(marker)
    if start < 0:
        return message
    start += len(marker)
    end = message.find(suffix, start)
    return message[start:] if end < 0 else message[start:end]


def _task_id(state: Any) -> str:
    task_id = state.get("task_id")
    if not isinstance(task_id, str) or not _IDENTIFIER_RE.fullmatch(task_id):
        raise ValueError("bounded task_id is required")
    return task_id


def _phase(value: Any) -> Literal[1, 2]:
    if isinstance(value, bool) or value not in (1, 2):
        raise ValueError("phase must be exactly 1 or 2")
    return value


def _require_function_call_id(tool_context: Any) -> str:
    value = getattr(tool_context, "function_call_id", None)
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError("ToolContext.function_call_id is required; no fallback")
    return value


def _valid_context_identifier(tool_context: Any) -> str | None:
    value = getattr(tool_context, "function_call_id", None)
    return value if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) else None


def _tool_name(tool: Any) -> str:
    value = getattr(tool, "name", None)
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError("tool has no bounded official name")
    if value not in _TOOL_OBSERVATION_TYPES:
        raise ValueError(f"unsupported a-interact tool: {value}")
    return value


def _safe_tool_name(tool: Any) -> str:
    value = getattr(tool, "name", None)
    return value[:64] if isinstance(value, str) else type(tool).__name__[:64]


def _next_sequence(state: Any) -> int:
    current = state.get(GROUNDING_SEQUENCE_KEY, 0)
    if (
        isinstance(current, bool)
        or not isinstance(current, int)
        or current < 0
        or current >= _MAX_SEQUENCE
    ):
        raise ValueError("invalid or exhausted SQL Grounding sequence")
    value = current + 1
    state[GROUNDING_SEQUENCE_KEY] = value
    return value


def _bounded_args_summary(args_json: str) -> str:
    if len(args_json) <= _MAX_ARGS_SUMMARY_CHARS:
        return args_json
    return args_json[: _MAX_ARGS_SUMMARY_CHARS - 1] + "…"


def _trajectory_length(state: Any) -> int | None:
    if state is None:
        return None
    trajectory = state.get("tool_trajectory", [])
    return len(trajectory) if isinstance(trajectory, list) else None


def _new_trajectory_index(state: Any, before: int | None) -> int | None:
    trajectory = state.get("tool_trajectory", [])
    if before is None or not isinstance(trajectory, list):
        return None
    return before if len(trajectory) == before + 1 else None


def _model_call_count(state: Any) -> int | None:
    if state is None:
        return None
    calls = state.get("system_agent_llm_calls", [])
    return len(calls) if isinstance(calls, list) else None


def _new_model_call_index(state: Any, before: int | None) -> int | None:
    calls = state.get("system_agent_llm_calls", [])
    active = state.get("_active_llm_call_index")
    if before is None or not isinstance(calls, list):
        return None
    if len(calls) != before + 1 or active != before:
        return None
    return before


def _request_sha256(llm_request: Any) -> str:
    return _sha256_text(
        json.dumps(
            to_jsonable(llm_request),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attach_model_call_audit(
    state: Any,
    index: int,
    key: str,
    metadata: dict[str, Any],
) -> None:
    calls = state.get("system_agent_llm_calls", [])
    if not isinstance(calls, list) or not 0 <= index < len(calls):
        return
    _require_bounded_audit(metadata)
    call = calls[index]
    if not isinstance(call, dict):
        return
    updated = list(calls)
    updated_call = dict(call)
    updated_call[key] = metadata
    updated[index] = updated_call
    state["system_agent_llm_calls"] = updated


def _attach_tool_audit(
    state: Any,
    index: int,
    metadata: dict[str, Any],
) -> None:
    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list) or not 0 <= index < len(trajectory):
        return
    _require_bounded_audit(metadata)
    event = trajectory[index]
    if not isinstance(event, dict):
        return
    updated = list(trajectory)
    updated_event = dict(event)
    updated_event[SHADOW_AUDIT_KEY] = metadata
    updated[index] = updated_event
    state["tool_trajectory"] = updated


def _append_error_audit(state: Any, metadata: dict[str, Any]) -> None:
    try:
        _require_bounded_audit(metadata)
        existing = state.get(GROUNDING_ERROR_AUDIT_KEY, [])
        if not isinstance(existing, list):
            existing = []
        state[GROUNDING_ERROR_AUDIT_KEY] = (
            [*existing[-(_MAX_ERROR_AUDITS - 1) :], metadata]
        )
    except Exception:
        return


def _bounded_error_audit(
    stage: str,
    exception: BaseException,
    *,
    function_call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "service_status": "failed_open",
        "stage": stage[:64],
        "function_call_id": function_call_id,
        "error_type": type(exception).__name__[:128],
    }


def _require_bounded_audit(metadata: dict[str, Any]) -> None:
    encoded = canonical_json(metadata).encode("utf-8")
    if len(encoded) > _MAX_AUDIT_BYTES:
        raise ValueError("SQL Grounding audit exceeds bounded size")
