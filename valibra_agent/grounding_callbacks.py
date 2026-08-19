"""SQL Grounding V1 callbacks around the frozen B0 callbacks.

SG6a injects the bounded four-dimensional Grounding View and Control Hint into
the model system instruction.  SG6b hard-blocks only a safely pairable,
premature first submit before B0 can charge it.  A narrowly frozen budget
liveness policy bypasses that block when every current-focus Grounding tool is
unaffordable, so the Baseline forced-exit path remains reachable.  Baseline is
still the only source of Official tool, Bird-Coin, and submit lifecycle facts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
from valibra_agent.sql_grounding.control import (
    StageEvent,
    evaluate_first_submit_gate,
    render_control_hint,
    tool_directions_for_focus,
    transition_grounding_stage,
)
from valibra_agent.sql_grounding.observations import (
    ObservationType,
    SQLGroundingObservation,
    build_sql_grounding_observation,
)
from valibra_agent.sql_grounding.prompt_view import (
    DEFAULT_MAX_VIEW_CHARS,
    DEFAULT_MAX_VIEW_TOKENS,
    count_grounding_view_tokens,
    render_grounding_view,
)
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
GROUNDING_CONTROL_AUDIT_KEY = "valibra_sql_grounding_control"
GROUNDING_ERROR_AUDIT_KEY = "valibra:sql_grounding_error_audits"
GROUNDING_PROVIDER_CALL_COUNT_KEY = "valibra:sql_grounding_provider_calls"
GROUNDING_BLOCKED_SUBMITS_KEY = "valibra:sql_grounding_blocked_submits"
GROUNDING_GATE_AUDITS_KEY = "valibra:sql_grounding_gate_audits"

# The frozen P6 export module imports these names at module load.  They are
# retained only so that historical, read-only export code remains importable;
# SG3 never reads either key and never runs the retired semantic core.
GROUNDING_LEGACY_INITIALIZATION_UNKNOWN_KEY = (
    "valibra:frame_initialization_legacy_unknown"
)
REQUIREMENT_VIEW_AUDIT_KEY = GROUNDING_VIEW_AUDIT_KEY

GROUNDING_VIEW_BEGIN = "[VALIBRA GROUNDING VIEW BEGIN]"
GROUNDING_VIEW_END = "[VALIBRA GROUNDING VIEW END]"
CONTROL_HINT_BEGIN = "[VALIBRA CONTROL HINT BEGIN]"
CONTROL_HINT_END = "[VALIBRA CONTROL HINT END]"
_ACTIVE_CONTEXT_MARKERS = (
    GROUNDING_VIEW_BEGIN,
    GROUNDING_VIEW_END,
    CONTROL_HINT_BEGIN,
    CONTROL_HINT_END,
)
_MAX_CONTROL_HINT_CHARS = 1_024
_MAX_CONTROL_HINT_TOKENS = 256

_MAX_ARGS_SUMMARY_CHARS = 768
_MAX_AUDIT_BYTES = 4_096
_MAX_ERROR_AUDITS = 64
_MAX_PENDING = 64
_MAX_BLOCKED_SUBMITS = 64
_MAX_GATE_AUDITS = 64
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")

_BLOCKED_SUBMIT_STATUS = "VALIBRA_FIRST_SUBMIT_BLOCKED"
_BLOCKED_SUBMIT_GUIDANCE = (
    "Continue Grounding using the current Valibra Control Hint before "
    "retrying submit_sql."
)

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
    control_gate_audit: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "function_call_id": self.function_call_id,
            "tool_name": self.tool_name,
            "phase_before": self.phase_before,
            "args_digest": self.args_digest,
            "args_summary": self.args_summary,
            "sequence": self.sequence,
            "control_gate_audit": self.control_gate_audit,
        }

    @classmethod
    def from_json(cls, payload: Any) -> "_PendingToolCall":
        if not isinstance(payload, dict) or set(payload) not in ({
            "function_call_id",
            "tool_name",
            "phase_before",
            "args_digest",
            "args_summary",
            "sequence",
        }, {
            "function_call_id",
            "tool_name",
            "phase_before",
            "args_digest",
            "args_summary",
            "sequence",
            "control_gate_audit",
        }):
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
        if record.control_gate_audit is not None:
            if not isinstance(record.control_gate_audit, dict):
                raise ValueError("invalid pending Control audit")
            _require_bounded_audit(record.control_gate_audit)
        return record


@dataclass(frozen=True, slots=True)
class _GateExecutionPolicy:
    """Transient SG6b execution facts; never part of Grounding Runtime."""

    action: Literal[
        "blocked",
        "open",
        "budget_liveness_bypass",
        "failed_open",
    ]
    budget_remaining: float | None = None
    focus_directions: tuple[str, ...] = ()
    affordable_directions: tuple[str, ...] = ()
    liveness_bypass_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _BlockedSubmitCall:
    """Bounded Callback glue for one ADK-level synthetic submit denial."""

    function_call_id: str
    tool_name: Literal["submit_sql"]
    args_digest: str
    gate_reason: str
    stage: str
    focus: str
    denial_sha256: str
    sequence: int
    gate_audit: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "function_call_id": self.function_call_id,
            "tool_name": self.tool_name,
            "args_digest": self.args_digest,
            "gate_reason": self.gate_reason,
            "stage": self.stage,
            "focus": self.focus,
            "denial_sha256": self.denial_sha256,
            "sequence": self.sequence,
            "gate_audit": self.gate_audit,
        }

    @classmethod
    def from_json(cls, payload: Any) -> "_BlockedSubmitCall":
        required = {
            "function_call_id",
            "tool_name",
            "args_digest",
            "gate_reason",
            "stage",
            "focus",
            "denial_sha256",
            "sequence",
            "gate_audit",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid blocked-submit record")
        record = cls(**payload)
        if not _IDENTIFIER_RE.fullmatch(record.function_call_id):
            raise ValueError("invalid blocked-submit function_call_id")
        if record.tool_name != "submit_sql":
            raise ValueError("blocked record must be submit_sql")
        for value, label in (
            (record.args_digest, "args digest"),
            (record.denial_sha256, "denial digest"),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"invalid blocked-submit {label}")
        if not _IDENTIFIER_RE.fullmatch(record.gate_reason):
            raise ValueError("invalid blocked-submit Gate reason")
        if record.stage not in {
            "INITIAL_GROUNDING",
            "SQL_ATTEMPT",
            "REPAIR",
            "P2_INCREMENTAL",
            "DONE",
        }:
            raise ValueError("invalid blocked-submit stage")
        if record.focus not in {
            "tables",
            "join_keys",
            "column_mapping",
            "domain_knowledge",
            "none",
        }:
            raise ValueError("invalid blocked-submit focus")
        if (
            isinstance(record.sequence, bool)
            or not isinstance(record.sequence, int)
            or not 1 <= record.sequence <= _MAX_SEQUENCE
        ):
            raise ValueError("invalid blocked-submit sequence")
        if not isinstance(record.gate_audit, dict):
            raise ValueError("invalid blocked-submit Gate audit")
        _require_bounded_audit(record.gate_audit)
        return record


@dataclass(frozen=True, slots=True)
class _ObservationResult:
    runtime: GroundingRuntime
    service_status: str
    observation: SQLGroundingObservation
    service_result: SQLGroundingServiceResult | None = None
    control_events: tuple[dict[str, str], ...] = ()
    control_status: Literal["not_applicable", "succeeded", "failed_open"] = (
        "not_applicable"
    )
    control_error_type: str | None = None
    official_outcome: str | None = None


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
    """Update Shadow State, atomically inject View + Hint, then call B0 once."""

    state = getattr(callback_context, "state", None)
    model_call_count = _model_call_count(state)
    request_before: str | None = None
    update_audit: dict[str, Any] | None = None
    control_audit: dict[str, Any] | None = None
    view_audit: dict[str, Any] = {
        "mode": "active",
        "injected": False,
        "injection_status": "not_attempted",
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
            control_audit = _control_audit_for_runtime(
                runtime,
                control_status="failed_open" if degraded else "succeeded",
                error_type="RuntimeValidationError" if degraded else None,
            )
            hint = render_control_hint(runtime.focus_dimension)
            hint_tokens = count_grounding_view_tokens(hint.text)
            control_audit.update(
                {
                    "mode": "active_hint",
                    "control_hint_tokens_cl100k": hint_tokens,
                    "control_hint_tokens": hint_tokens,
                    "control_hint_injected": False,
                    "injection_status": "not_attempted",
                }
            )
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
                    "view_chars": view.char_count,
                    "tokens_cl100k": view.token_count,
                    "view_tokens": view.token_count,
                    "included_items": view.included_items,
                    "omitted_items": view.omitted_items,
                    "runtime_degraded": degraded,
                }
            )
            if degraded:
                view_audit.update(
                    {
                        "injection_status": "failed_open",
                        "error_type": "RuntimeValidationError",
                    }
                )
                control_audit.update(
                    {
                        "injection_status": "failed_open",
                        "control_hint_injected": False,
                    }
                )
            if not degraded:
                injection = _inject_active_grounding_context(
                    llm_request,
                    view_text=view.text,
                    control_hint_text=hint.text,
                )
                view_audit.update(
                    {
                        "injected": True,
                        "injection_status": "succeeded",
                        "injection_block_sha256": injection[
                            "grounding_view_block_sha256"
                        ],
                        "view_block_sha256": injection[
                            "grounding_view_block_sha256"
                        ],
                        "request_changed_only_system_instruction": True,
                    }
                )
                control_audit.update(
                    {
                        "control_hint_injected": True,
                        "injection_status": "succeeded",
                        "control_hint_block_sha256": injection[
                            "control_hint_block_sha256"
                        ],
                        "request_changed_only_system_instruction": True,
                    }
                )
        view_audit["request_sha256_after_injection"] = _request_sha256(
            llm_request
        )
    except Exception as exc:
        view_audit["error_type"] = type(exc).__name__[:128]
        view_audit["injection_status"] = "failed_open"
        view_audit.setdefault("runtime_degraded", True)
        try:
            view_audit["request_sha256_after_injection"] = _request_sha256(
                llm_request
            )
        except Exception:
            pass
        if control_audit is not None:
            control_audit.update(
                {
                    "control_status": "failed_open",
                    "control_hint_injected": False,
                    "injection_status": "failed_open",
                    "error_type": type(exc).__name__[:128],
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
            == view_audit.get("request_sha256_after_injection")
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
                if control_audit is not None:
                    _attach_model_call_audit(
                        state,
                        model_call_index,
                        GROUNDING_CONTROL_AUDIT_KEY,
                        control_audit,
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
    """Apply the active first-submit policy before B0 cost, then register B0."""

    from system_agent import callbacks as baseline_callbacks

    state = getattr(tool_context, "state", None)
    tool_name = _safe_tool_name(tool)
    control_gate_audit: dict[str, Any] | None = None
    if state is not None and tool_name == "submit_sql":
        try:
            runtime, degraded = _ensure_runtime(state)
            first_submit = _is_first_official_submit(state)
            gate = evaluate_first_submit_gate(
                runtime,
                first_submit=first_submit,
            )
            if degraded:
                policy = _GateExecutionPolicy(action="failed_open")
                control_gate_audit = _active_gate_audit(
                    runtime,
                    gate=gate,
                    first_submit=first_submit,
                    policy=policy,
                    control_status="failed_open",
                    error_type="RuntimeValidationError",
                )
            else:
                try:
                    policy = _evaluate_gate_execution_policy(
                        runtime,
                        gate=gate,
                        state=state,
                        tool_costs=baseline_callbacks.TOOL_COSTS,
                    )
                except Exception as exc:
                    policy = _GateExecutionPolicy(action="failed_open")
                    control_gate_audit = _active_gate_audit(
                        runtime,
                        gate=gate,
                        first_submit=first_submit,
                        policy=policy,
                        control_status="failed_open",
                        error_type=type(exc).__name__[:128],
                    )
                    _append_error_audit(
                        state,
                        _bounded_error_audit(
                            "before_tool_gate_execution_policy",
                            exc,
                            function_call_id=_valid_context_identifier(
                                tool_context
                            ),
                        ),
                    )
                else:
                    control_gate_audit = _active_gate_audit(
                        runtime,
                        gate=gate,
                        first_submit=first_submit,
                        policy=policy,
                        control_status="succeeded",
                    )
            if policy.action == "blocked":
                try:
                    function_call_id = _require_function_call_id(tool_context)
                    args_json = canonical_json(to_jsonable(args))
                    denial = _blocked_submit_denial(gate.reason)
                    denial_sha256 = _sha256_text(canonical_json(denial))
                    record = _BlockedSubmitCall(
                        function_call_id=function_call_id,
                        tool_name="submit_sql",
                        args_digest=_sha256_text(args_json),
                        gate_reason=gate.reason,
                        stage=runtime.stage,
                        focus=runtime.focus_dimension,
                        denial_sha256=denial_sha256,
                        sequence=_next_sequence(state),
                        gate_audit=control_gate_audit,
                    )
                    _add_blocked_submit(state, record)
                    _upsert_gate_audit(
                        state,
                        _blocked_gate_audit(
                            record,
                            budget_remaining=policy.budget_remaining,
                            after_tool_seen=False,
                        ),
                    )
                except Exception as exc:
                    _cleanup_blocked_best_effort(state, tool_context)
                    policy = _GateExecutionPolicy(action="failed_open")
                    control_gate_audit = _active_gate_audit(
                        runtime,
                        gate=gate,
                        first_submit=first_submit,
                        policy=policy,
                        control_status="failed_open",
                        error_type=type(exc).__name__[:128],
                    )
                    _append_error_audit(
                        state,
                        _bounded_error_audit(
                            "before_tool_active_gate",
                            exc,
                            function_call_id=_valid_context_identifier(
                                tool_context
                            ),
                        ),
                    )
                else:
                    return denial
            _require_bounded_audit(control_gate_audit)
        except Exception as exc:
            _cleanup_blocked_best_effort(state, tool_context)
            control_gate_audit = {
                "control_status": "failed_open",
                "stage_before": None,
                "attempt_gate": {
                    "applicable": False,
                    "open": True,
                    "reason": "control_evaluation_failed",
                    "would_block": False,
                    "blocked": False,
                    "first_submit": None,
                    "liveness_bypass": False,
                    "liveness_bypass_reason": None,
                    "budget_remaining": None,
                    "focus_direction_count": None,
                    "affordable_direction_count": None,
                    "affordable_tool_directions": [],
                    "effective_gate_action": "failed_open",
                },
                "error_type": type(exc).__name__[:128],
            }
            _append_error_audit(
                state,
                _bounded_error_audit(
                    "before_tool_control",
                    exc,
                    function_call_id=_valid_context_identifier(tool_context),
                ),
            )

    baseline_result = await baseline_callbacks.before_tool_callback(
        tool,
        args,
        tool_context,
    )
    if baseline_result is not None:
        return baseline_result
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
            control_gate_audit=control_gate_audit,
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
    function_call_id = _valid_context_identifier(tool_context)
    if (
        state is not None
        and function_call_id is not None
        and _blocked_submit_present(state, function_call_id)
    ):
        try:
            blocked = _pop_blocked_submit(state, function_call_id)
            if blocked is None:
                raise ValueError("exact blocked-submit record disappeared")
            denial = _blocked_submit_denial(blocked.gate_reason)
            error_type: str | None = None
            if _safe_tool_name(tool) != "submit_sql":
                error_type = "BlockedToolNameMismatch"
            elif _sha256_text(canonical_json(to_jsonable(tool_response))) != (
                blocked.denial_sha256
            ):
                error_type = "BlockedResponseMismatch"
            _upsert_gate_audit(
                state,
                _blocked_gate_audit(
                    blocked,
                    budget_remaining=_blocked_record_budget(blocked),
                    after_tool_seen=True,
                    error_type=error_type,
                ),
            )
            return denial
        except Exception as exc:
            _cleanup_blocked_best_effort(state, tool_context)
            _append_error_audit(
                state,
                _bounded_error_audit(
                    "after_tool_blocked_submit",
                    exc,
                    function_call_id=function_call_id,
                ),
            )
            # Exact presence proves ADK skipped the tool.  Never create a fake
            # Official result by falling through to Baseline after_tool.
            return tool_response

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
    control_audit: dict[str, Any] | None = None
    pending: _PendingToolCall | None = None
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
            phase_after = _phase(
                state.get("current_phase", pending.phase_before)
            )
            if tool_name == "submit_sql":
                result, follow_up_audit = await _handle_submit_observation(
                    state,
                    observation,
                    pending=pending,
                    phase_after=phase_after,
                    observation_type=observation_type,
                    tool_response=tool_response,
                    private_ref=private_ref,
                )
            else:
                result = await _handle_observation(state, observation)
                follow_up_audit = None
            runtime = result.runtime
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
            control_audit = _control_audit_for_observation(
                result,
                gate_audit=pending.control_gate_audit,
            )
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
        control_audit = _control_audit_for_runtime(
            runtime,
            control_status="failed_open",
            error_type=type(exc).__name__[:128],
            gate_audit=(
                pending.control_gate_audit if pending is not None else None
            ),
        )
        _append_error_audit(state, audit)
    finally:
        _cleanup_pending_best_effort(state, tool_context)
        if audit_index is not None:
            try:
                _attach_tool_audit(state, audit_index, audit)
                if control_audit is not None:
                    _attach_tool_audit(
                        state,
                        audit_index,
                        control_audit,
                        key=GROUNDING_CONTROL_AUDIT_KEY,
                    )
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
    if (
        function_call_id is not None
        and _blocked_submit_present(state, function_call_id)
    ):
        try:
            blocked = _pop_blocked_submit(state, function_call_id)
            if blocked is not None:
                _upsert_gate_audit(
                    state,
                    _blocked_gate_audit(
                        blocked,
                        budget_remaining=_blocked_record_budget(blocked),
                        after_tool_seen=False,
                        error_type=type(error).__name__[:128],
                    ),
                )
        except Exception as callback_error:
            _append_error_audit(
                state,
                _bounded_error_audit(
                    "tool_error_blocked_submit",
                    callback_error,
                    function_call_id=function_call_id,
                ),
            )
        finally:
            _cleanup_blocked_best_effort(state, tool_context)
        return None
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


async def _handle_submit_observation(
    state: Any,
    observation: SQLGroundingObservation,
    *,
    pending: _PendingToolCall,
    phase_after: Literal[1, 2],
    observation_type: ObservationType,
    tool_response: Any,
    private_ref: str | None,
) -> tuple[_ObservationResult, dict[str, Any] | None]:
    """Apply only official submit facts; a shadow-closed gate never transitions."""

    runtime, _ = _ensure_runtime(state)
    outcome, event, outcome_error = _official_submit_control_event(
        state,
        phase_before=pending.phase_before,
        phase_after=phase_after,
        observation_type=observation_type,
    )
    gate = pending.control_gate_audit or {}
    attempt_gate = gate.get("attempt_gate", {})
    blocked = bool(
        isinstance(attempt_gate, dict) and attempt_gate.get("blocked") is True
    )
    if blocked:
        return (
            _ObservationResult(
                runtime=runtime,
                service_status="skipped_active_gate_blocked_submit",
                observation=observation,
                control_status="succeeded",
                official_outcome=outcome,
            ),
            None,
        )
    if outcome_error is not None or event is None:
        return (
            _ObservationResult(
                runtime=runtime,
                service_status="skipped_inconsistent_official_submit_state",
                observation=observation,
                control_status="failed_open",
                control_error_type=outcome_error or "OfficialSubmitStateError",
                official_outcome=outcome,
            ),
            None,
        )

    liveness_bypass = bool(
        isinstance(attempt_gate, dict)
        and attempt_gate.get("effective_gate_action")
        == "budget_liveness_bypass"
    )
    transitioned, transition = _apply_control_event(
        runtime,
        event,
        allow_initial_forced_exit=liveness_bypass,
    )
    if event == "official_submit_failed":
        repaired = await _handle_observation(state, observation, transitioned)
        return (
            _ObservationResult(
                runtime=repaired.runtime,
                service_status=repaired.service_status,
                observation=observation,
                service_result=repaired.service_result,
                control_events=(transition, *repaired.control_events),
                control_status=repaired.control_status,
                control_error_type=repaired.control_error_type,
                official_outcome=outcome,
            ),
            None,
        )

    follow_up_audit: dict[str, Any] | None = None
    final_runtime = transitioned
    if event == "official_p2_follow_up":
        try:
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
                transitioned,
            )
            final_runtime = follow_up_result.runtime
            follow_up_audit = _observation_audit(follow_up_result)
        except Exception as exc:
            return (
                _ObservationResult(
                    runtime=transitioned,
                    service_status="skipped_control_lifecycle_only",
                    observation=observation,
                    control_events=(transition,),
                    control_status="failed_open",
                    control_error_type=type(exc).__name__[:128],
                    official_outcome=outcome,
                ),
                None,
            )
    return (
        _ObservationResult(
            runtime=final_runtime,
            service_status="skipped_control_lifecycle_only",
            observation=observation,
            control_events=(transition,),
            control_status="succeeded",
            official_outcome=outcome,
        ),
        follow_up_audit,
    )


async def _handle_observation(
    state: Any,
    observation: SQLGroundingObservation,
    runtime: GroundingRuntime | None = None,
) -> _ObservationResult:
    active_runtime = runtime if runtime is not None else _ensure_runtime(state)[0]
    if observation.observation_type == "p2_follow_up":
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_affected_dimensions_unfrozen",
            observation=observation,
        )
    if (
        observation.observation_type == "submission"
        and active_runtime.stage != "REPAIR"
    ):
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_control_lifecycle_only",
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
                    control_status="failed_open",
                    control_error_type="ProviderCallLimit",
                )
    except Exception as exc:
        return _ObservationResult(
            runtime=active_runtime,
            service_status="degraded_configuration",
            observation=observation,
            control_status="failed_open",
            control_error_type=type(exc).__name__[:128],
        )
    try:
        context = _build_validation_context(state, observation)
        service_result = await process_sql_grounding_observation(
            active_runtime,
            observation,
            context,
            updater,
        )
    except Exception as exc:
        return _ObservationResult(
            runtime=active_runtime,
            service_status="degraded_service_boundary",
            observation=observation,
            control_status="failed_open",
            control_error_type=type(exc).__name__[:128],
        )
    if service_result.llm_telemetry.attempted and _uses_real_provider_adapter():
        state[GROUNDING_PROVIDER_CALL_COUNT_KEY] = _provider_call_count(state) + 1
    candidate = service_result.runtime
    control_events: tuple[dict[str, str], ...] = ()
    control_status: Literal["not_applicable", "succeeded", "failed_open"]
    control_error_type: str | None = None
    if service_result.state_update.status in {"accepted", "noop"}:
        control_status = "succeeded"
        try:
            candidate, control_events = _advance_ready_control_stage(
                state,
                candidate,
            )
        except Exception as exc:
            control_status = "failed_open"
            control_error_type = type(exc).__name__[:128]
    else:
        control_status = "failed_open"
        control_error_type = service_result.state_update.error_type
    return _ObservationResult(
        runtime=candidate,
        service_status=service_result.state_update.status,
        observation=observation,
        service_result=service_result,
        control_events=control_events,
        control_status=control_status,
        control_error_type=control_error_type,
    )


def _apply_control_event(
    runtime: GroundingRuntime,
    event: StageEvent,
    *,
    allow_initial_forced_exit: bool = False,
) -> tuple[GroundingRuntime, dict[str, str]]:
    candidate = transition_grounding_stage(
        runtime,
        event,
        allow_initial_forced_exit=allow_initial_forced_exit,
    )
    return candidate, {
        "event": event,
        "stage_before": runtime.stage,
        "stage_after": candidate.stage,
    }


def _advance_ready_control_stage(
    state: Any,
    runtime: GroundingRuntime,
) -> tuple[GroundingRuntime, tuple[dict[str, str], ...]]:
    event: StageEvent | None = None
    if (
        runtime.stage == "INITIAL_GROUNDING"
        and runtime.grounding_state.all_dimensions_evaluated
        and runtime.focus_dimension == "none"
    ):
        event = "grounding_completed"
    elif (
        runtime.stage == "REPAIR"
        and runtime.grounding_state.all_dimensions_evaluated
        and runtime.focus_dimension == "none"
    ):
        event = (
            "repair_completed_p1"
            if _phase(state.get("current_phase", 1)) == 1
            else "repair_completed_p2"
        )
    if event is None:
        return runtime, ()
    candidate, record = _apply_control_event(runtime, event)
    return candidate, (record,)


def _official_submit_control_event(
    state: Any,
    *,
    phase_before: Literal[1, 2],
    phase_after: Literal[1, 2],
    observation_type: ObservationType,
) -> tuple[str, StageEvent | None, str | None]:
    """Classify only frozen Official Session facts; never parse reward or prose."""

    if observation_type == "tool_error":
        return "tool_error", None, "OfficialToolError"
    phase1_completed = _official_bool(state, "phase1_completed")
    phase2_completed = _official_bool(state, "phase2_completed")
    task_done = _official_bool(state, "task_done")
    if phase_before == 1:
        if (
            task_done
            and phase1_completed
            and not phase2_completed
            and phase_after == 2
        ):
            return "task_completed_p1", "official_task_completed", None
        if (
            not task_done
            and phase1_completed
            and not phase2_completed
            and phase_after == 2
        ):
            return "p1_follow_up", "official_p2_follow_up", None
        if (
            not task_done
            and not phase1_completed
            and not phase2_completed
            and phase_after == 1
        ):
            return "p1_failed", "official_submit_failed", None
    else:
        if task_done and phase2_completed and phase_after == 2:
            return "task_completed_p2", "official_task_completed", None
        if not task_done and not phase2_completed and phase_after == 2:
            return "p2_failed", "official_submit_failed", None
    return "inconsistent_official_state", None, "OfficialSubmitStateError"


def _official_bool(state: Any, key: str) -> bool:
    value = state.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"official {key} must be boolean")
    return value


def _is_first_official_submit(state: Any) -> bool:
    """Use only completed Official trajectory entries before the current call."""

    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list):
        raise ValueError("official tool_trajectory must be a list")
    return not any(
        isinstance(event, dict) and event.get("tool") == "submit_sql"
        for event in trajectory
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


def _control_audit_for_observation(
    result: _ObservationResult,
    *,
    gate_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    return _control_audit_for_runtime(
        result.runtime,
        control_status=result.control_status,
        error_type=result.control_error_type,
        events=result.control_events,
        gate_audit=gate_audit,
        official_outcome=result.official_outcome,
    )


def _control_audit_for_runtime(
    runtime: GroundingRuntime,
    *,
    control_status: Literal["not_applicable", "succeeded", "failed_open"],
    error_type: str | None = None,
    events: tuple[dict[str, str], ...] = (),
    gate_audit: dict[str, Any] | None = None,
    official_outcome: str | None = None,
) -> dict[str, Any]:
    hint = render_control_hint(runtime.focus_dimension)
    gate = gate_audit or {}
    attempt_gate = gate.get("attempt_gate")
    if not isinstance(attempt_gate, dict):
        attempt_gate = {
            "applicable": False,
            "open": True,
            "reason": "not_applicable",
            "would_block": False,
            "blocked": False,
            "first_submit": None,
            "liveness_bypass": False,
            "liveness_bypass_reason": None,
            "budget_remaining": None,
            "focus_direction_count": None,
            "affordable_direction_count": None,
            "affordable_tool_directions": [],
            "effective_gate_action": "open",
        }
    stage_before = gate.get("stage_before")
    if not isinstance(stage_before, str):
        stage_before = events[0]["stage_before"] if events else runtime.stage
    effective_status = (
        "failed_open"
        if gate.get("control_status") == "failed_open"
        else control_status
    )
    audit: dict[str, Any] = {
        "mode": "active_first_submit",
        "control_status": effective_status,
        "stage_before": stage_before,
        "stage_after": runtime.stage,
        "event": events[-1]["event"] if events else None,
        "events": [dict(item) for item in events],
        "focus_dimension": runtime.focus_dimension,
        "tool_directions": list(tool_directions_for_focus(runtime.focus_dimension)),
        "control_hint_sha256": hint.sha256,
        "control_hint_chars": len(hint.text),
        "control_hint_injected": False,
        "attempt_gate": dict(attempt_gate),
        "official_outcome": official_outcome,
    }
    effective_error = error_type or gate.get("error_type")
    if isinstance(effective_error, str):
        audit["error_type"] = effective_error[:128]
    _require_bounded_audit(audit)
    return audit


def _evaluate_gate_execution_policy(
    runtime: GroundingRuntime,
    *,
    gate: Any,
    state: Any,
    tool_costs: Mapping[str, Any],
) -> _GateExecutionPolicy:
    """Resolve the approved SG6b policy without changing pure readiness."""

    if not gate.applicable or gate.open:
        return _GateExecutionPolicy(action="open")
    if runtime.stage != "INITIAL_GROUNDING" or runtime.focus_dimension == "none":
        return _GateExecutionPolicy(action="blocked")

    directions = tuple(tool_directions_for_focus(runtime.focus_dimension))
    if not directions:
        raise ValueError("current focus has no frozen tool directions")
    budget = state.get("budget_remaining")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
    ):
        raise ValueError("budget_remaining must be a finite number")
    if not isinstance(tool_costs, Mapping):
        raise ValueError("Baseline TOOL_COSTS is unavailable")

    affordable: list[str] = []
    for direction in directions:
        if direction not in tool_costs:
            raise ValueError("focus direction has no Baseline cost")
        cost = tool_costs[direction]
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0
        ):
            raise ValueError("Baseline tool cost must be finite and non-negative")
        if float(cost) <= float(budget):
            affordable.append(direction)

    if affordable:
        return _GateExecutionPolicy(
            action="blocked",
            budget_remaining=float(budget),
            focus_directions=directions,
            affordable_directions=tuple(affordable),
        )
    return _GateExecutionPolicy(
        action="budget_liveness_bypass",
        budget_remaining=float(budget),
        focus_directions=directions,
        affordable_directions=(),
        liveness_bypass_reason="no_affordable_focus_direction",
    )


def _active_gate_audit(
    runtime: GroundingRuntime,
    *,
    gate: Any,
    first_submit: bool,
    policy: _GateExecutionPolicy,
    control_status: Literal["succeeded", "failed_open"],
    error_type: str | None = None,
) -> dict[str, Any]:
    attempt_gate = {
        "applicable": bool(gate.applicable),
        "open": bool(gate.open),
        "reason": str(gate.reason)[:128],
        "would_block": bool(gate.applicable and not gate.open),
        "blocked": policy.action == "blocked",
        "first_submit": first_submit,
        "liveness_bypass": policy.action == "budget_liveness_bypass",
        "liveness_bypass_reason": policy.liveness_bypass_reason,
        "budget_remaining": policy.budget_remaining,
        "focus_direction_count": len(policy.focus_directions),
        "affordable_direction_count": len(policy.affordable_directions),
        "affordable_tool_directions": list(policy.affordable_directions),
        "effective_gate_action": policy.action,
    }
    audit: dict[str, Any] = {
        "control_status": control_status,
        "stage_before": runtime.stage,
        "grounding_revision": runtime.grounding_revision,
        "state_sha256": sql_grounding_state_sha256(runtime.grounding_state),
        "attempt_gate": attempt_gate,
    }
    if error_type is not None:
        audit["error_type"] = error_type[:128]
    _require_bounded_audit(audit)
    return audit


def _blocked_submit_denial(gate_reason: str) -> dict[str, str]:
    if not isinstance(gate_reason, str) or not _IDENTIFIER_RE.fullmatch(gate_reason):
        raise ValueError("invalid Gate reason for blocked response")
    return {
        "status": _BLOCKED_SUBMIT_STATUS,
        "reason": gate_reason,
        "guidance": _BLOCKED_SUBMIT_GUIDANCE,
    }


def _load_blocked_submits(state: Any) -> dict[str, _BlockedSubmitCall]:
    payload = state.get(GROUNDING_BLOCKED_SUBMITS_KEY, {})
    if not isinstance(payload, dict) or len(payload) > _MAX_BLOCKED_SUBMITS:
        raise ValueError("invalid blocked-submit store")
    records: dict[str, _BlockedSubmitCall] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("blocked-submit key must be a string")
        record = _BlockedSubmitCall.from_json(value)
        if record.function_call_id != key:
            raise ValueError("blocked-submit key does not match function_call_id")
        records[key] = record
    return records


def _store_blocked_submits(
    state: Any,
    records: Mapping[str, _BlockedSubmitCall],
) -> None:
    if len(records) > _MAX_BLOCKED_SUBMITS:
        raise ValueError("too many blocked-submit records")
    state[GROUNDING_BLOCKED_SUBMITS_KEY] = {
        key: _BlockedSubmitCall.from_json(records[key].to_json()).to_json()
        for key in sorted(records)
    }


def _add_blocked_submit(state: Any, record: _BlockedSubmitCall) -> None:
    records = _load_blocked_submits(state)
    if record.function_call_id in records:
        raise ValueError("duplicate blocked-submit function_call_id")
    records[record.function_call_id] = _BlockedSubmitCall.from_json(
        record.to_json()
    )
    _store_blocked_submits(state, records)


def _pop_blocked_submit(
    state: Any,
    function_call_id: str,
) -> _BlockedSubmitCall | None:
    records = _load_blocked_submits(state)
    record = records.pop(function_call_id, None)
    _store_blocked_submits(state, records)
    return record


def _blocked_submit_present(state: Any, function_call_id: str) -> bool:
    payload = state.get(GROUNDING_BLOCKED_SUBMITS_KEY, {})
    return isinstance(payload, dict) and function_call_id in payload


def _cleanup_blocked_best_effort(state: Any, tool_context: Any) -> None:
    function_call_id = _valid_context_identifier(tool_context)
    if function_call_id is None:
        return
    payload = state.get(GROUNDING_BLOCKED_SUBMITS_KEY)
    if not isinstance(payload, dict) or function_call_id not in payload:
        return
    cleaned = dict(payload)
    cleaned.pop(function_call_id, None)
    state[GROUNDING_BLOCKED_SUBMITS_KEY] = cleaned


def _load_gate_audits(state: Any) -> list[dict[str, Any]]:
    payload = state.get(GROUNDING_GATE_AUDITS_KEY, [])
    if not isinstance(payload, list) or len(payload) > _MAX_GATE_AUDITS:
        raise ValueError("invalid active Gate audit store")
    records: list[dict[str, Any]] = []
    for value in payload:
        if not isinstance(value, dict):
            raise ValueError("active Gate audit must be an object")
        _require_bounded_audit(value)
        records.append(dict(value))
    return records


def _upsert_gate_audit(state: Any, audit: dict[str, Any]) -> None:
    _require_bounded_audit(audit)
    function_call_id = audit.get("function_call_id")
    if not isinstance(function_call_id, str) or not _IDENTIFIER_RE.fullmatch(
        function_call_id
    ):
        raise ValueError("active Gate audit requires exact function_call_id")
    records = _load_gate_audits(state)
    indexes = [
        index
        for index, item in enumerate(records)
        if item.get("function_call_id") == function_call_id
    ]
    if len(indexes) > 1:
        raise ValueError("duplicate active Gate audits")
    if indexes:
        records[indexes[0]] = dict(audit)
    else:
        if len(records) >= _MAX_GATE_AUDITS:
            raise ValueError("active Gate audit store is full")
        records.append(dict(audit))
    state[GROUNDING_GATE_AUDITS_KEY] = records


def _blocked_gate_audit(
    record: _BlockedSubmitCall,
    *,
    budget_remaining: float | None,
    after_tool_seen: bool,
    error_type: str | None = None,
) -> dict[str, Any]:
    audit = record.gate_audit
    result: dict[str, Any] = {
        "function_call_id": record.function_call_id,
        "mode": "active_first_submit",
        "pure_gate_reason": record.gate_reason,
        "effective_gate_action": "blocked",
        "stage": record.stage,
        "focus": record.focus,
        "state_sha256": audit.get("state_sha256"),
        "grounding_revision": audit.get("grounding_revision"),
        "budget_remaining": budget_remaining,
        "denial_sha256": record.denial_sha256,
        "after_tool_seen": after_tool_seen,
        "actual_tool_executed": False,
        "baseline_before_tool_called": False,
        "baseline_after_tool_called": False,
        "sequence": record.sequence,
    }
    if error_type is not None:
        result["error_type"] = error_type[:128]
    _require_bounded_audit(result)
    return result


def _blocked_record_budget(record: _BlockedSubmitCall) -> float | None:
    attempt = record.gate_audit.get("attempt_gate")
    if not isinstance(attempt, dict):
        return None
    value = attempt.get("budget_remaining")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


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


def _active_context_block(begin: str, text: str, end: str) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("active Valibra context text must be non-empty")
    if any(marker in text for marker in _ACTIVE_CONTEXT_MARKERS):
        raise ValueError("active Valibra context contains a reserved marker")
    return f"{begin}\n{text}\n{end}"


def _strip_active_grounding_context(value: str) -> str:
    """Remove one complete trailing View/Hint pair; reject all ambiguity."""

    if not isinstance(value, str):
        raise TypeError("LlmRequest system_instruction must be a string")
    counts = {marker: value.count(marker) for marker in _ACTIVE_CONTEXT_MARKERS}
    if not any(counts.values()):
        return value
    if any(count != 1 for count in counts.values()):
        raise ValueError("duplicate or incomplete Valibra active context markers")

    view_start = value.find(GROUNDING_VIEW_BEGIN)
    suffix = value[view_start:]
    pattern = re.compile(
        rf"{re.escape(GROUNDING_VIEW_BEGIN)}\n"
        rf"(?P<view>.*?)\n{re.escape(GROUNDING_VIEW_END)}\n\n"
        rf"{re.escape(CONTROL_HINT_BEGIN)}\n"
        rf"(?P<hint>.*?)\n{re.escape(CONTROL_HINT_END)}",
        re.DOTALL,
    )
    match = pattern.fullmatch(suffix)
    if match is None:
        raise ValueError("malformed or cross-nested Valibra active context")
    if any(
        marker in match.group(name)
        for name in ("view", "hint")
        for marker in _ACTIVE_CONTEXT_MARKERS
    ):
        raise ValueError("Valibra active context contains a marker collision")
    if view_start == 0:
        return ""
    if value[view_start - 2 : view_start] != "\n\n":
        raise ValueError("Valibra active context is not an appended block pair")
    return value[: view_start - 2]


def _config_without_system_instruction(config: Any) -> Any:
    cloned = copy.deepcopy(config)
    setattr(cloned, "system_instruction", None)
    return to_jsonable(cloned)


def _request_without_injection_fields(llm_request: Any) -> Any:
    cloned = copy.deepcopy(llm_request)
    cloned.config = None
    cloned.contents = []
    return to_jsonable(cloned)


def _restore_llm_request(llm_request: Any, original: Any) -> None:
    fields = getattr(type(llm_request), "model_fields", None)
    if not isinstance(fields, dict):
        raise TypeError("LlmRequest must expose Pydantic model fields")
    for field in fields:
        setattr(llm_request, field, copy.deepcopy(getattr(original, field)))


def _inject_active_grounding_context(
    llm_request: Any,
    *,
    view_text: str,
    control_hint_text: str,
) -> dict[str, str]:
    """Atomically leave exactly one current View followed by one Hint block."""

    append = getattr(llm_request, "append_instructions", None)
    config = getattr(llm_request, "config", None)
    if not callable(append) or config is None:
        raise TypeError("LlmRequest.append_instructions is required")
    system_instruction = getattr(config, "system_instruction", None)
    if system_instruction is not None and not isinstance(system_instruction, str):
        raise TypeError("LlmRequest system_instruction must be a string")
    if len(view_text) > DEFAULT_MAX_VIEW_CHARS:
        raise ValueError("Grounding View exceeds the character limit")
    if count_grounding_view_tokens(view_text) > DEFAULT_MAX_VIEW_TOKENS:
        raise ValueError("Grounding View exceeds the token limit")
    if len(control_hint_text) > _MAX_CONTROL_HINT_CHARS:
        raise ValueError("Control Hint exceeds the character limit")
    if count_grounding_view_tokens(control_hint_text) > _MAX_CONTROL_HINT_TOKENS:
        raise ValueError("Control Hint exceeds the token limit")

    view_block = _active_context_block(
        GROUNDING_VIEW_BEGIN,
        view_text,
        GROUNDING_VIEW_END,
    )
    hint_block = _active_context_block(
        CONTROL_HINT_BEGIN,
        control_hint_text,
        CONTROL_HINT_END,
    )
    original_request = copy.deepcopy(llm_request)
    original_contents = copy.deepcopy(getattr(llm_request, "contents", None))
    original_non_system = _config_without_system_instruction(config)
    original_non_injection = _request_without_injection_fields(llm_request)
    try:
        base_instruction = _strip_active_grounding_context(system_instruction or "")
        config.system_instruction = base_instruction or None
        returned_contents = append([view_block, hint_block])
        if returned_contents:
            raise RuntimeError("append_instructions returned unexpected user contents")
        if getattr(llm_request, "contents", None) != original_contents:
            raise RuntimeError("append_instructions modified request contents")
        if _config_without_system_instruction(config) != original_non_system:
            raise RuntimeError("active context changed non-system generation config")
        if _request_without_injection_fields(llm_request) != original_non_injection:
            raise RuntimeError("active context changed another model request field")

        final_instruction = getattr(config, "system_instruction", None)
        if not isinstance(final_instruction, str):
            raise RuntimeError("append_instructions produced no string instruction")
        expected = (
            f"{base_instruction}\n\n{view_block}\n\n{hint_block}"
            if base_instruction
            else f"{view_block}\n\n{hint_block}"
        )
        if final_instruction != expected:
            raise RuntimeError("active context injection did not preserve exact order")
        if _strip_active_grounding_context(final_instruction) != base_instruction:
            raise RuntimeError("active context injection was not reversible")
        return {
            "grounding_view_block_sha256": _sha256_text(view_block),
            "control_hint_block_sha256": _sha256_text(hint_block),
        }
    except BaseException:
        _restore_llm_request(llm_request, original_request)
        raise


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
    *,
    key: str = SHADOW_AUDIT_KEY,
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
    updated_event[key] = metadata
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
