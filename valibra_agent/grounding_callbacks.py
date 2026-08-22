"""SQL Grounding callbacks around the frozen Official BIRD lifecycle.

Version 1.3 interleaves the one-time bootstrap with Structure, Mapping, and
Knowledge forms, then runs a bounded unified Check loop.  Only a complete Check
enters the SQL Writer surface, where the Main Agent sees the final four-
dimensional view plus answered clarifications and exactly execute/submit tools.
Baseline remains the only source of Official tools, Bird-Coin, and submit facts.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager
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
    GroundingCheckResponse,
    GroundingCheckToolRequest,
    GroundingLLMResponse,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    UserClarificationRecord,
    UserClarificationRequest,
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
    GroundingCallKind,
    build_real_sql_grounding_updater,
    classify_grounding_input,
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
GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY = (
    "valibra:sql_grounding_provider_phase_calls"
)
GROUNDING_TOOL_AUDITS_KEY = "valibra:sql_grounding_tool_audits"
GROUNDING_BLOCKED_SUBMITS_KEY = "valibra:sql_grounding_blocked_submits"
GROUNDING_GATE_AUDITS_KEY = "valibra:sql_grounding_gate_audits"
GROUNDING_SUPPRESSED_BOOTSTRAP_KEY = (
    "valibra:sql_grounding_suppressed_bootstrap"
)
GROUNDING_CLARIFICATIONS_KEY = "valibra:user_clarifications"
GROUNDING_PHASE_OUTCOMES_KEY = "valibra:sql_grounding_phase_outcomes"
GROUNDING_FAILED_CLOSED_CALLS_KEY = "valibra:sql_grounding_failed_closed_calls"
GROUNDING_PENDING_CHECK_KEY = "valibra:sql_grounding_pending_check"
GROUNDING_CHECK_AUDITS_KEY = "valibra:sql_grounding_check_audits"

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
SQL_WRITER_CONTEXT_BEGIN = "[VALIBRA SQL WRITER CONTEXT BEGIN]"
SQL_WRITER_CONTEXT_END = "[VALIBRA SQL WRITER CONTEXT END]"
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
_MAX_TOOL_AUDITS = 64
_MAX_SUPPRESSED_BOOTSTRAP = 64
_MAX_CLARIFICATION_RECORDS = 16
_MAX_PHASE_OUTCOMES = 2
_MAX_FAILED_CLOSED_CALLS = 64
_MAX_TOOL_AUDIT_RECORD_BYTES = 8_448
_MAX_ACTIVE_TASK_SYNCHRONIZERS = 512
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")

_BLOCKED_SUBMIT_STATUS = "VALIBRA_FIRST_SUBMIT_BLOCKED"
_BLOCKED_SUBMIT_GUIDANCE = (
    "Complete the pending Valibra user clarification before retrying submit_sql."
)
_GROUNDING_FAILED_CLOSED_STATUS = "VALIBRA_SQL_GROUNDING_FAILED_CLOSED"
_GROUNDING_FAILED_CLOSED_GUIDANCE = (
    "A valid SQL Grounding State was not established for this phase. "
    "Main SQL generation and Official tools are disabled without retry or fallback."
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

# Stage 1 bootstrap is deliberately a fixed Official-tool sequence.  Raw
# results remain in Baseline's Official trajectory; they are never copied into
# SQLGroundingState or another persistent evidence store.
_BOOTSTRAP_TOOL_SEQUENCE: tuple[str, ...] = (
    "get_schema",
    "get_all_column_meanings",
    "get_all_knowledge_definitions",
)
_BOOTSTRAP_MODEL_VISIBLE_PREFIX = "VALIBRA_BOOTSTRAP_EVIDENCE_STORED"
_BOOTSTRAP_ALREADY_VISIBLE_PREFIX = "VALIBRA_BOOTSTRAP_EVIDENCE_ALREADY_STORED"
_MAX_PROVIDER_CALLS_PER_TASK = 32
_MIN_BUDGET_AFTER_CHECK_TOOL = 6.0
_MAX_CHECK_AUDITS = 32
_SQL_WRITER_TOOL_NAMES = ("execute_sql", "submit_sql")
_SQL_WRITER_PROMPT = """Grounding 已完成，数据库语义结果视为本 phase 的最终结果。
你的任务是根据 Original Query、可选 Follow-up、Final Grounding State 和已回答澄清写 PostgreSQL SQL。
你可以处理 JOIN、JSON、CAST、NULL、聚合、排序、DISTINCT、latest record 等 SQL 实现问题；
必要时只用 execute_sql 做小范围验证和修正，最终用 submit_sql 提交。
不要重新探索 schema、column meaning 或 knowledge；不要查询 information_schema / pg_catalog；
不要 SELECT *；不要提出新问题；不要修改或重新解释 Grounding State。
State.tables 是允许使用的候选表范围，不代表每张表都必须出现在最终 SQL。"""
_BULK_KNOWLEDGE_VISIBLE_FIELDS = frozenset(
    {"id", "knowledge", "description", "definition"}
)


class _PassthroughSQLGroundingUpdater:
    """Default SG4 updater: no I/O, no Provider, and no State guesswork."""

    async def propose(
        self,
        runtime: GroundingRuntime,
        observation: SQLGroundingObservation,
        *,
        original_query: str,
        follow_up_query: str | None = None,
        grounding_input: Mapping[str, Any] | None = None,
    ) -> GroundingUpdaterResult:
        del observation, original_query, follow_up_query, grounding_input
        return GroundingUpdaterResult(
            response=GroundingLLMResponse(
                sql_grounding_state=runtime.grounding_state,
                user_clarification_requests=(),
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
class _TaskGroundingSynchronization:
    """Process-local glue; never persisted in Runtime or Session State."""

    lock: asyncio.Lock
    users: int = 0
    provider_slot_reserved: bool = False


_TASK_SYNCHRONIZERS: dict[
    tuple[int, str], _TaskGroundingSynchronization
] = {}
_TASK_SYNCHRONIZERS_GUARD = threading.Lock()


@asynccontextmanager
async def _serialized_task_grounding(state: Any):
    """Serialize Grounding updates per task while allowing task-level parallelism."""

    task_id = _task_id(state)
    key = (id(asyncio.get_running_loop()), task_id)
    with _TASK_SYNCHRONIZERS_GUARD:
        synchronizer = _TASK_SYNCHRONIZERS.get(key)
        if synchronizer is None:
            if len(_TASK_SYNCHRONIZERS) >= _MAX_ACTIVE_TASK_SYNCHRONIZERS:
                raise RuntimeError("SQL Grounding task synchronizer capacity exhausted")
            synchronizer = _TaskGroundingSynchronization(lock=asyncio.Lock())
            _TASK_SYNCHRONIZERS[key] = synchronizer
        synchronizer.users += 1
    try:
        async with synchronizer.lock:
            yield synchronizer
    finally:
        with _TASK_SYNCHRONIZERS_GUARD:
            synchronizer.users -= 1
            if synchronizer.users == 0:
                _TASK_SYNCHRONIZERS.pop(key, None)


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
class _SuppressedBootstrapCall:
    """Short-lived glue for one duplicate bootstrap call skipped before cost."""

    function_call_id: str
    tool_name: str
    args_digest: str
    response_sha256: str
    sequence: int

    def to_json(self) -> dict[str, Any]:
        return {
            "function_call_id": self.function_call_id,
            "tool_name": self.tool_name,
            "args_digest": self.args_digest,
            "response_sha256": self.response_sha256,
            "sequence": self.sequence,
        }

    @classmethod
    def from_json(cls, payload: Any) -> "_SuppressedBootstrapCall":
        required = {
            "function_call_id",
            "tool_name",
            "args_digest",
            "response_sha256",
            "sequence",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid suppressed-bootstrap record")
        record = cls(**payload)
        if not _IDENTIFIER_RE.fullmatch(record.function_call_id):
            raise ValueError("invalid suppressed-bootstrap function_call_id")
        if record.tool_name not in _BOOTSTRAP_TOOL_SEQUENCE:
            raise ValueError("invalid suppressed-bootstrap tool")
        for value in (record.args_digest, record.response_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("invalid suppressed-bootstrap digest")
        if (
            isinstance(record.sequence, bool)
            or not isinstance(record.sequence, int)
            or not 1 <= record.sequence <= _MAX_SEQUENCE
        ):
            raise ValueError("invalid suppressed-bootstrap sequence")
        return record


@dataclass(frozen=True, slots=True)
class _PhaseGroundingOutcome:
    """One immutable phase terminal outside the four-dimensional Runtime."""

    phase: Literal[1, 2]
    status: Literal["succeeded", "failed"]
    observation_id: str
    error_type: str | None
    provider_attempted: bool
    grounding_revision: int
    state_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "observation_id": self.observation_id,
            "error_type": self.error_type,
            "provider_attempted": self.provider_attempted,
            "grounding_revision": self.grounding_revision,
            "state_sha256": self.state_sha256,
        }

    @classmethod
    def from_json(cls, payload: Any) -> "_PhaseGroundingOutcome":
        required = {
            "phase",
            "status",
            "observation_id",
            "error_type",
            "provider_attempted",
            "grounding_revision",
            "state_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid phase Grounding outcome")
        record = cls(**payload)
        if record.phase not in (1, 2):
            raise ValueError("invalid phase Grounding outcome phase")
        if record.status not in {"succeeded", "failed"}:
            raise ValueError("invalid phase Grounding outcome status")
        if not _IDENTIFIER_RE.fullmatch(record.observation_id):
            raise ValueError("invalid phase Grounding observation_id")
        if record.status == "succeeded" and record.error_type is not None:
            raise ValueError("successful phase Grounding cannot have an error")
        if record.status == "failed":
            if not isinstance(record.error_type, str) or not _IDENTIFIER_RE.fullmatch(
                record.error_type
            ):
                raise ValueError("failed phase Grounding requires a bounded error")
        if not isinstance(record.provider_attempted, bool):
            raise ValueError("invalid phase Grounding attempted flag")
        if (
            isinstance(record.grounding_revision, bool)
            or not isinstance(record.grounding_revision, int)
            or record.grounding_revision < 0
        ):
            raise ValueError("invalid phase Grounding revision")
        if not re.fullmatch(r"[0-9a-f]{64}", record.state_sha256):
            raise ValueError("invalid phase Grounding State SHA")
        return record


@dataclass(frozen=True, slots=True)
class _PendingCheckTool:
    """One Check-selected Official action, outside four-dimensional State."""

    phase: Literal[1, 2]
    function_call_id: str
    missing_information: str
    tool_name: str
    arguments: dict[str, str]
    request_digest: str

    def to_json(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "function_call_id": self.function_call_id,
            "missing_information": self.missing_information,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "request_digest": self.request_digest,
        }

    @classmethod
    def from_json(cls, payload: Any) -> "_PendingCheckTool":
        required = {
            "phase",
            "function_call_id",
            "missing_information",
            "tool_name",
            "arguments",
            "request_digest",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid pending Check tool")
        record = cls(**payload)
        if record.phase not in (1, 2):
            raise ValueError("invalid pending Check phase")
        if not _IDENTIFIER_RE.fullmatch(record.function_call_id):
            raise ValueError("invalid pending Check function_call_id")
        if record.tool_name not in {
            "ask_user",
            "get_column_meaning",
            "get_all_external_knowledge_names",
            "get_knowledge_definition",
            "execute_sql",
        }:
            raise ValueError("invalid pending Check tool name")
        if not isinstance(record.arguments, dict):
            raise ValueError("invalid pending Check arguments")
        if record.tool_name == "ask_user":
            if set(record.arguments) != {"question"}:
                raise ValueError("invalid pending ask_user arguments")
            question = record.arguments["question"]
            if not isinstance(question, str) or not question or len(question) > 1_024:
                raise ValueError("invalid pending ask_user question")
        else:
            GroundingCheckToolRequest(
                tool_name=record.tool_name,
                arguments=record.arguments,
                user_clarification_request=None,
            )
        if not record.missing_information or len(record.missing_information) > 1_024:
            raise ValueError("invalid pending Check gap")
        if not re.fullmatch(r"[0-9a-f]{64}", record.request_digest):
            raise ValueError("invalid pending Check digest")
        return record


@dataclass(frozen=True, slots=True)
class _FailedClosedToolCall:
    """Exact ADK pairing for one tool denied by a failed phase terminal."""

    function_call_id: str
    tool_name: str
    phase: Literal[1, 2]
    response_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "function_call_id": self.function_call_id,
            "tool_name": self.tool_name,
            "phase": self.phase,
            "response_sha256": self.response_sha256,
        }

    @classmethod
    def from_json(cls, payload: Any) -> "_FailedClosedToolCall":
        required = {"function_call_id", "tool_name", "phase", "response_sha256"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid failed-closed tool record")
        record = cls(**payload)
        if not _IDENTIFIER_RE.fullmatch(record.function_call_id):
            raise ValueError("invalid failed-closed function_call_id")
        if not record.tool_name or len(record.tool_name) > 64:
            raise ValueError("invalid failed-closed tool name")
        if record.phase not in (1, 2):
            raise ValueError("invalid failed-closed phase")
        if not re.fullmatch(r"[0-9a-f]{64}", record.response_sha256):
            raise ValueError("invalid failed-closed response SHA")
        return record


@dataclass(frozen=True, slots=True)
class _ObservationResult:
    runtime: GroundingRuntime
    service_status: str
    observation: SQLGroundingObservation
    service_result: SQLGroundingServiceResult | None = None
    control_events: tuple[dict[str, str], ...] = ()
    control_status: Literal[
        "not_applicable", "succeeded", "failed_open", "failed_closed"
    ] = "not_applicable"
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
    bootstrap_tool: str | None = None
    pending_check_tool: _PendingCheckTool | None = None
    phase_failure: _PhaseGroundingOutcome | None = None
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
            phase_failure = _phase_grounding_failure(state)
            if not degraded and phase_failure is None:
                bootstrap_tool = _next_bootstrap_tool(
                    state,
                    runtime,
                    llm_request,
                )
                if bootstrap_tool is None:
                    pending_check_tool = _pending_check_tool(state)
            view = render_grounding_view(
                runtime.grounding_state,
                clarifications=_clarification_records(state),
            )
            control_audit = _control_audit_for_runtime(
                runtime,
                control_status="failed_open" if degraded else "succeeded",
                error_type="RuntimeValidationError" if degraded else None,
            )
            phase_ready = _phase_grounding_succeeded(state)
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
            if phase_failure is not None:
                view_audit.update(
                    {
                        "injection_status": "failed_closed",
                        "error_type": phase_failure.error_type,
                        "failed_closed_phase": phase_failure.phase,
                        "phase_grounding_observation_id": (
                            phase_failure.observation_id
                        ),
                    }
                )
                control_audit.update(
                    {
                        "control_status": "failed_closed",
                        "control_hint_injected": False,
                        "injection_status": "failed_closed",
                        "error_type": phase_failure.error_type,
                    }
                )
            if not degraded and phase_failure is None:
                if phase_ready and bootstrap_tool is None and pending_check_tool is None:
                    bound = _ACTIVE_TURN_MESSAGE.get()
                    if bound is None or bound.task_id != _task_id(state):
                        raise ValueError("current bound query is required for SQL Writer")
                    injection = _inject_sql_writer_context(
                        llm_request,
                        phase=_phase(state.get("current_phase", 1)),
                        original_query=_user_message_query(bound.message),
                        follow_up=(
                            _official_p2_follow_up(state)
                            if _phase(state.get("current_phase", 1)) == 2
                            else None
                        ),
                        view_text=view.text,
                        budget_remaining=state.get("budget_remaining"),
                    )
                    control_audit.update(
                        {
                            "mode": "sql_writer",
                            "control_hint_injected": False,
                            "writer_tools": list(_SQL_WRITER_TOOL_NAMES),
                        }
                    )
                else:
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
                        "request_changed_only_system_instruction": not phase_ready,
                    }
                )
                control_audit.update(
                    {
                        "control_hint_injected": not phase_ready,
                        "injection_status": "succeeded",
                        "control_hint_block_sha256": injection.get(
                            "control_hint_block_sha256"
                        ),
                        "request_changed_only_system_instruction": not phase_ready,
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
    if bootstrap_tool is not None:
        view_audit["bootstrap_tool_forced"] = bootstrap_tool
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
    if phase_failure is not None:
        return _failed_closed_model_response(phase_failure)
    if baseline_result is not None:
        return baseline_result
    if bootstrap_tool is not None:
        try:
            return _bootstrap_function_call_response(state, bootstrap_tool)
        except Exception as exc:
            if state is not None:
                _append_error_audit(
                    state,
                    _bounded_error_audit("before_model_bootstrap", exc),
                )
            return None
    if pending_check_tool is not None:
        try:
            return _check_tool_function_call_response(
                state,
                pending_check_tool,
            )
        except Exception as exc:
            if state is not None:
                _append_error_audit(
                    state,
                    _bounded_error_audit("before_model_clarification", exc),
                )
            return None
    return None


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
    if state is not None:
        failure = _phase_grounding_failure(state)
        if failure is not None:
            response = _failed_closed_response(failure)
            try:
                function_call_id = _require_function_call_id(tool_context)
                _add_failed_closed_call(
                    state,
                    _FailedClosedToolCall(
                        function_call_id=function_call_id,
                        tool_name=tool_name,
                        phase=failure.phase,
                        response_sha256=_sha256_text(canonical_json(response)),
                    ),
                )
                _upsert_tool_callback_audit(
                    state,
                    function_call_id,
                    shadow_audit={
                        "service_status": "failed_closed_phase_grounding",
                        "function_call_id": function_call_id,
                        "tool_name": tool_name,
                        "phase_before": failure.phase,
                        "provider_attempted": False,
                        "official_tool_executed": False,
                        "error_type": failure.error_type,
                        "grounding_revision": failure.grounding_revision,
                        "state_sha256": failure.state_sha256,
                    },
                    control_audit=None,
                )
            except Exception as exc:
                _append_error_audit(
                    state,
                    _bounded_error_audit(
                        "before_tool_failed_closed",
                        exc,
                        function_call_id=_valid_context_identifier(tool_context),
                    ),
                )
            return response
    if (
        state is not None
        and tool_name in _BOOTSTRAP_TOOL_SEQUENCE
        and _bootstrap_tool_already_attempted(state, tool_name)
    ):
        try:
            function_call_id = _require_function_call_id(tool_context)
            response = _duplicate_bootstrap_response(tool_name)
            record = _SuppressedBootstrapCall(
                function_call_id=function_call_id,
                tool_name=tool_name,
                args_digest=_sha256_text(canonical_json(to_jsonable(args))),
                response_sha256=_sha256_text(canonical_json(response)),
                sequence=_next_sequence(state),
            )
            _add_suppressed_bootstrap(state, record)
            _upsert_tool_callback_audit(
                state,
                function_call_id,
                shadow_audit={
                    "service_status": "skipped_duplicate_bootstrap_no_charge",
                    "function_call_id": function_call_id,
                    "tool_name": tool_name,
                    "provider_attempted": False,
                    "official_tool_executed": False,
                },
                control_audit=None,
            )
            return response
        except Exception as exc:
            _cleanup_suppressed_bootstrap_best_effort(state, tool_context)
            _append_error_audit(
                state,
                _bounded_error_audit(
                    "before_tool_duplicate_bootstrap",
                    exc,
                    function_call_id=_valid_context_identifier(tool_context),
                ),
            )
            return _duplicate_bootstrap_response(tool_name)
    if state is not None and tool_name == "submit_sql":
        try:
            runtime, degraded = _ensure_runtime(state)
            first_submit = _is_first_official_submit(state)
            gate = evaluate_first_submit_gate(
                runtime,
                first_submit=first_submit,
                pending_clarifications=_pending_clarification_count(state),
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
        and _failed_closed_call_present(state, function_call_id)
    ):
        try:
            blocked = _pop_failed_closed_call(state, function_call_id)
            if blocked is None:
                raise ValueError("exact failed-closed tool record disappeared")
            if _safe_tool_name(tool) != blocked.tool_name:
                raise ValueError("failed-closed tool name mismatch")
            if _sha256_text(canonical_json(to_jsonable(tool_response))) != (
                blocked.response_sha256
            ):
                raise ValueError("failed-closed response mismatch")
            return tool_response
        except Exception as exc:
            _append_error_audit(
                state,
                _bounded_error_audit(
                    "after_tool_failed_closed",
                    exc,
                    function_call_id=function_call_id,
                ),
            )
            return tool_response
    if (
        state is not None
        and function_call_id is not None
        and _suppressed_bootstrap_present(state, function_call_id)
    ):
        try:
            suppressed = _pop_suppressed_bootstrap(state, function_call_id)
            if suppressed is None:
                raise ValueError("exact suppressed-bootstrap record disappeared")
            response = _duplicate_bootstrap_response(suppressed.tool_name)
            if _safe_tool_name(tool) != suppressed.tool_name:
                raise ValueError("suppressed bootstrap tool name mismatch")
            if _sha256_text(canonical_json(to_jsonable(tool_response))) != (
                suppressed.response_sha256
            ):
                raise ValueError("suppressed bootstrap response mismatch")
            return response
        except Exception as exc:
            _cleanup_suppressed_bootstrap_best_effort(state, tool_context)
            _append_error_audit(
                state,
                _bounded_error_audit(
                    "after_tool_duplicate_bootstrap",
                    exc,
                    function_call_id=function_call_id,
                ),
            )
            return tool_response
    if state is not None and _safe_tool_name(tool) in _BOOTSTRAP_TOOL_SEQUENCE:
        # A duplicate is already proven by the frozen Official trajectory.  If
        # exact-ID bookkeeping was itself degraded, the deterministic response
        # still proves ADK skipped execution; never manufacture an Official
        # event by falling through to Baseline after_tool.
        try:
            tool_name = _safe_tool_name(tool)
            if (
                _bootstrap_tool_already_attempted(state, tool_name)
                and tool_response == _duplicate_bootstrap_response(tool_name)
            ):
                _cleanup_suppressed_bootstrap_best_effort(state, tool_context)
                return tool_response
        except Exception as exc:
            _append_error_audit(
                state,
                _bounded_error_audit(
                    "after_tool_duplicate_bootstrap_fallback",
                    exc,
                    function_call_id=function_call_id,
                ),
            )
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
    bootstrap_tool_result = False
    bootstrap_tool_succeeded = False
    grounding_clarification_answered = False
    pending_check: _PendingCheckTool | None = None
    phase_grounding_observation: SQLGroundingObservation | None = None
    grounding_input: Mapping[str, Any] | None = None
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
            if _is_check_function_call(
                state,
                function_call_id=function_call_id,
                tool_name=tool_name,
            ):
                pending_check = _pending_check_tool(state)
            raw_content = to_jsonable(tool_response)
            observation_type = _classify_tool_observation_type(
                tool_name,
                tool_response,
            )
            if tool_name == "ask_user" and observation_type != "tool_error":
                question = args.get("question") if isinstance(args, dict) else None
                if (
                    isinstance(question, str)
                    and _is_clarification_function_call(
                        state,
                        function_call_id=function_call_id,
                        phase=pending.phase_before,
                        question=question,
                    )
                ):
                    _record_clarification_answer(
                        state,
                        phase=pending.phase_before,
                        question=question,
                        answer=tool_response,
                    )
                    grounding_clarification_answered = True
            phase_after = _phase(
                state.get("current_phase", pending.phase_before)
            )
            if _is_bootstrap_function_call(
                state,
                function_call_id=function_call_id,
                tool_name=tool_name,
            ):
                bootstrap_tool_result = True
                bootstrap_tool_succeeded = observation_type != "tool_error"
                runtime, degraded = _ensure_runtime(state)
                phase_grounding_observation = build_sql_grounding_observation(
                    task_id=_task_id(state),
                    phase=pending.phase_before,
                    sequence=_next_sequence(state),
                    observation_type=observation_type,
                    content=raw_content,
                    summary=f"Official {tool_name} evidence observed for staged Grounding",
                    tool_name=tool_name,
                    function_call_id=function_call_id,
                    private_raw_ref=private_ref,
                )
                if (
                    bootstrap_tool_succeeded
                    and not degraded
                ):
                    observation = phase_grounding_observation
                    bound = _ACTIVE_TURN_MESSAGE.get()
                    if bound is None or bound.task_id != _task_id(state):
                        raise ValueError(
                            "current bound query is required for staged Grounding"
                        )
                    call_kind: GroundingCallKind = {
                        "get_schema": "structure",
                        "get_all_column_meanings": "mapping",
                        "get_all_knowledge_definitions": "knowledge",
                    }[tool_name]
                    grounding_input = _build_staged_grounding_request(
                        state,
                        call_kind=call_kind,
                        query=_user_message_query(bound.message),
                        runtime=runtime,
                        phase=1,
                    )
                    result = await _handle_observation(
                        state,
                        observation,
                        grounding_input=grounding_input,
                    )
                    runtime = result.runtime
                    audit = _observation_audit(result)
                    audit["staged_grounding_kind"] = call_kind
                    audit["staged_grounding_triggered"] = True
                    audit["bootstrap_evidence_complete"] = (
                        tool_name == _BOOTSTRAP_TOOL_SEQUENCE[-1]
                    )
                    if (
                        call_kind == "knowledge"
                        and result.service_status in {"accepted", "noop"}
                        and _phase_grounding_failure(state, 1) is None
                        and result.service_result is not None
                        and isinstance(
                            result.service_result.response,
                            KnowledgeGroundingResponse,
                        )
                    ):
                        stage_audit = dict(audit)
                        check_observation = build_sql_grounding_observation(
                            task_id=_task_id(state),
                            phase=1,
                            sequence=_next_sequence(state),
                            observation_type="knowledge",
                            content=raw_content,
                            summary="initial unified Grounding Check",
                            tool_name=tool_name,
                            function_call_id=function_call_id,
                            private_raw_ref=private_ref,
                        )
                        check_input = _build_check_grounding_request(
                            state,
                            query=_user_message_query(bound.message),
                            runtime=runtime,
                            phase=1,
                            initial=True,
                        )
                        result = await _handle_observation(
                            state,
                            check_observation,
                            runtime,
                            grounding_input=check_input,
                        )
                        runtime = result.runtime
                        audit = _observation_audit(result)
                        audit["staged_grounding_kind"] = "check"
                        audit["preceding_knowledge"] = stage_audit
                    control_audit = _control_audit_for_observation(
                        result,
                        gate_audit=pending.control_gate_audit,
                    )
                else:
                    if _is_real_provider_mode():
                        _record_phase_grounding_outcome(
                            state,
                            observation=phase_grounding_observation,
                            runtime=runtime,
                            status="failed",
                            error_type=(
                                "RuntimeValidationError"
                                if degraded
                                else "BootstrapToolError"
                            ),
                            provider_attempted=False,
                        )
                    audit = {
                        "service_status": (
                            "stored_bootstrap_tool_error"
                            if observation_type == "tool_error"
                            else "stored_bootstrap_evidence"
                        ),
                        "function_call_id": function_call_id,
                        "tool_name": tool_name,
                        "observation_type": observation_type,
                        "phase_before": pending.phase_before,
                        "phase_after": phase_after,
                        "args_digest": pending.args_digest,
                        "raw_digest": _sha256_text(canonical_json(raw_content)),
                        "private_raw_ref": private_ref,
                        "official_error": observation_type == "tool_error",
                        "provider_attempted": False,
                        "bootstrap_evidence_complete": False,
                        **_runtime_audit(runtime),
                    }
                    control_audit = _control_audit_for_runtime(
                        runtime,
                        control_status="failed_open" if degraded else "succeeded",
                        error_type="RuntimeValidationError" if degraded else None,
                        gate_audit=pending.control_gate_audit,
                    )
                _store_runtime(state, runtime)
            else:
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
                    grounding_input = None
                    if pending_check is not None:
                        bound = _ACTIVE_TURN_MESSAGE.get()
                        if bound is None or bound.task_id != _task_id(state):
                            raise ValueError(
                                "current bound query is required for Check"
                            )
                        follow_up = (
                            _official_p2_follow_up(state)
                            if pending.phase_before == 2
                            else None
                        )
                        patch_runtime = _ensure_runtime(state)[0]
                        grounding_input = _build_check_grounding_request(
                            state,
                            query=_user_message_query(bound.message),
                            runtime=patch_runtime,
                            phase=pending.phase_before,
                            follow_up=follow_up,
                            latest_tool_name=(
                                None if grounding_clarification_answered else tool_name
                            ),
                            latest_tool_arguments=(
                                None if grounding_clarification_answered else args
                            ),
                            latest_tool_result=(
                                None
                                if grounding_clarification_answered
                                else raw_content
                            ),
                            latest_user_answer=(
                                tool_response
                                if tool_name == "ask_user"
                                and grounding_clarification_answered
                                else None
                            ),
                        )
                        _store_pending_check_tool(state, None)
                    result = await _handle_observation(
                        state,
                        observation,
                        grounding_input=grounding_input,
                    )
                    follow_up_audit = None
                runtime = result.runtime
                audit = _observation_audit(result)
                if grounding_input is not None:
                    audit["staged_grounding_kind"] = "check"
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
        if (
            phase_grounding_observation is not None
            and _is_real_provider_mode()
            and _phase_grounding_outcomes(state).get(
                str(phase_grounding_observation.phase)
            )
            is None
        ):
            _record_phase_grounding_outcome(
                state,
                observation=phase_grounding_observation,
                runtime=runtime,
                status="failed",
                error_type=type(exc).__name__[:128],
                provider_attempted=False,
            )
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
        exact_id = _valid_context_identifier(tool_context)
        if exact_id is not None:
            try:
                _upsert_tool_callback_audit(
                    state,
                    exact_id,
                    shadow_audit=audit,
                    control_audit=control_audit,
                )
            except Exception as exc:
                _append_error_audit(
                    state,
                    _bounded_error_audit(
                        "after_tool_exact_audit",
                        exc,
                        function_call_id=exact_id,
                    ),
                )
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
    if bootstrap_tool_result:
        return _bootstrap_model_visible_result(
            _safe_tool_name(tool),
            baseline_override,
            succeeded=bootstrap_tool_succeeded,
        )
    return baseline_override


async def on_tool_error_callback(
    tool: Any,
    args: dict,
    tool_context: ToolContext,
    error: Exception,
) -> None:
    """Observe an ADK exception, clear its exact pending call, and stay invisible."""

    state = getattr(tool_context, "state", None)
    if state is None:
        return None
    function_call_id = _valid_context_identifier(tool_context)
    if (
        function_call_id is not None
        and _failed_closed_call_present(state, function_call_id)
    ):
        _pop_failed_closed_call(state, function_call_id)
        return None
    if (
        function_call_id is not None
        and _suppressed_bootstrap_present(state, function_call_id)
    ):
        _pop_suppressed_bootstrap(state, function_call_id)
        return None
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
        grounding_input = None
        if _is_check_function_call(
            state,
            function_call_id=exact_id,
            tool_name=tool_name,
        ):
            bound = _ACTIVE_TURN_MESSAGE.get()
            if bound is None or bound.task_id != _task_id(state):
                raise ValueError("current bound query is required for Check")
            runtime = _ensure_runtime(state)[0]
            phase = pending.phase_before
            grounding_input = _build_check_grounding_request(
                state,
                query=_user_message_query(bound.message),
                follow_up=_official_p2_follow_up(state) if phase == 2 else None,
                runtime=runtime,
                phase=phase,
                latest_tool_name=tool_name,
                latest_tool_arguments=args,
                latest_tool_result={"error_type": type(error).__name__[:128]},
            )
            _store_pending_check_tool(state, None)
        result = await _handle_observation(
            state,
            observation,
            grounding_input=grounding_input,
        )
        _store_runtime(state, result.runtime)
        audit = _observation_audit(result)
        audit.update(
            {
                "function_call_id": exact_id,
                "tool_name": tool_name,
                "official_exception": True,
            }
        )
        _upsert_tool_callback_audit(
            state,
            exact_id,
            shadow_audit=audit,
            control_audit=None,
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
        return (
            _ObservationResult(
                runtime=transitioned,
                service_status="skipped_submit_failure_no_repair",
                observation=observation,
                control_events=(transition,),
                control_status="succeeded",
                official_outcome=outcome,
            ),
            None,
        )

    follow_up_audit: dict[str, Any] | None = None
    final_runtime = transitioned
    if event == "official_p2_follow_up":
        follow_up_observation: SQLGroundingObservation | None = None
        try:
            follow_up = _extract_submit_follow_up(tool_response)
            bound = _ACTIVE_TURN_MESSAGE.get()
            if bound is None or bound.task_id != _task_id(state):
                raise ValueError(
                    "current bound query is required for P2 Grounding"
                )
            staged_audits: list[dict[str, Any]] = []
            for call_kind in ("structure", "mapping", "knowledge"):
                follow_up_observation = build_sql_grounding_observation(
                    task_id=_task_id(state),
                    phase=2,
                    sequence=_next_sequence(state),
                    observation_type="p2_follow_up",
                    content=follow_up,
                    summary=f"official Phase-2 follow-up for {call_kind} Grounding",
                    private_raw_ref=private_ref,
                )
                grounding_input = _build_staged_grounding_request(
                    state,
                    call_kind=call_kind,
                    query=_user_message_query(bound.message),
                    follow_up=follow_up,
                    runtime=final_runtime,
                    phase=2,
                )
                follow_up_result = await _handle_observation(
                    state,
                    follow_up_observation,
                    final_runtime,
                    grounding_input=grounding_input,
                )
                final_runtime = follow_up_result.runtime
                stage_audit = _observation_audit(follow_up_result)
                stage_audit["staged_grounding_kind"] = call_kind
                staged_audits.append(stage_audit)
                if _phase_grounding_failure(state, 2) is not None:
                    break
            if (
                _phase_grounding_failure(state, 2) is None
                and follow_up_result.service_result is not None
                and isinstance(
                    follow_up_result.service_result.response,
                    KnowledgeGroundingResponse,
                )
            ):
                check_observation = build_sql_grounding_observation(
                    task_id=_task_id(state),
                    phase=2,
                    sequence=_next_sequence(state),
                    observation_type="p2_follow_up",
                    content=follow_up,
                    summary="initial Phase-2 unified Grounding Check",
                    private_raw_ref=private_ref,
                )
                check_input = _build_check_grounding_request(
                    state,
                    query=_user_message_query(bound.message),
                    follow_up=follow_up,
                    runtime=final_runtime,
                    phase=2,
                    initial=True,
                )
                check_result = await _handle_observation(
                    state,
                    check_observation,
                    final_runtime,
                    grounding_input=check_input,
                )
                final_runtime = check_result.runtime
                check_audit = _observation_audit(check_result)
                check_audit["staged_grounding_kind"] = "check"
                staged_audits.append(check_audit)
            follow_up_audit = {"staged_grounding": staged_audits}
        except Exception as exc:
            if follow_up_observation is not None and _is_real_provider_mode():
                _record_phase_grounding_outcome(
                    state,
                    observation=follow_up_observation,
                    runtime=transitioned,
                    status="failed",
                    error_type=type(exc).__name__[:128],
                    provider_attempted=False,
                )
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
    *,
    grounding_input: Mapping[str, Any] | None = None,
) -> _ObservationResult:
    async with _serialized_task_grounding(state) as synchronization:
        # Ordinary sibling callbacks must re-read after waiting.  An explicit
        # Runtime is reserved for a control transition that has not yet been
        # persisted (submit failure/P2 lifecycle).
        active_runtime = runtime if runtime is not None else _ensure_runtime(state)[0]
        result = await _handle_observation_serialized(
            state,
            observation,
            active_runtime,
            synchronization,
            grounding_input=grounding_input,
        )
        _store_runtime(state, result.runtime)
        return result


def _failed_phase_grounding_result(
    state: Any,
    *,
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    service_status: str,
    error_type: str,
    provider_attempted: bool,
    service_result: SQLGroundingServiceResult | None = None,
) -> _ObservationResult:
    if _is_real_provider_mode():
        _record_phase_grounding_outcome(
            state,
            observation=observation,
            runtime=runtime,
            status="failed",
            error_type=error_type,
            provider_attempted=provider_attempted,
        )
    return _ObservationResult(
        runtime=runtime,
        service_status=service_status,
        observation=observation,
        service_result=service_result,
        control_status="failed_closed",
        control_error_type=error_type,
    )


async def _handle_observation_serialized(
    state: Any,
    observation: SQLGroundingObservation,
    active_runtime: GroundingRuntime,
    synchronization: _TaskGroundingSynchronization,
    *,
    grounding_input: Mapping[str, Any] | None = None,
) -> _ObservationResult:
    if (
        observation.observation_type == "p2_follow_up"
        and grounding_input is None
    ):
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_affected_dimensions_unfrozen",
            observation=observation,
        )
    if observation.observation_type == "submission":
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_control_lifecycle_only",
            observation=observation,
        )
    if observation.observation_type == "tool_error" and grounding_input is None:
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_tool_error_audit_only",
            observation=observation,
        )
    if observation.observation_type == "user_answer" and grounding_input is None:
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_affected_dimensions_unfrozen",
            observation=observation,
        )
    if observation.observation_type == "user_query":
        return _ObservationResult(
            runtime=active_runtime,
            service_status="skipped_provider_no_state_evidence",
            observation=observation,
        )
    if grounding_input is None:
        return _ObservationResult(
            runtime=active_runtime,
            service_status="stored_official_evidence_only",
            observation=observation,
        )
    try:
        call_kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
    except Exception as exc:
        return _failed_phase_grounding_result(
            state,
            runtime=active_runtime,
            observation=observation,
            service_status="rejected_staged_grounding_input",
            error_type=type(exc).__name__[:128],
            provider_attempted=False,
        )
    provider_mode = _is_real_provider_mode()
    if provider_mode:
        existing_outcome = _phase_grounding_outcomes(state).get(
            str(observation.phase)
        )
        if existing_outcome is not None:
            return _ObservationResult(
                runtime=active_runtime,
                service_status="skipped_phase_grounding_terminal",
                observation=observation,
                control_status=(
                    "failed_closed"
                    if existing_outcome.status == "failed"
                    else "failed_open"
                ),
                control_error_type=(
                    existing_outcome.error_type
                    or "PhaseGroundingAlreadySucceeded"
                ),
            )
    try:
        updater = _resolve_sql_grounding_updater()
        if provider_mode:
            llm_config = load_sql_grounding_llm_config(PROJECT_ROOT)
            calls = _provider_call_count(state)
            phase_calls = _provider_phase_call_count(state, observation.phase)
            sequence_ok = (
                (call_kind == "structure" and phase_calls == 0)
                or (call_kind == "mapping" and phase_calls == 1)
                or (call_kind == "knowledge" and phase_calls == 2)
                or (call_kind == "check" and phase_calls >= 3)
            )
            if not sequence_ok:
                return _failed_phase_grounding_result(
                    state,
                    runtime=active_runtime,
                    observation=observation,
                    service_status="skipped_bundled_call_sequence",
                    error_type="BundledProviderCallSequenceError",
                    provider_attempted=False,
                )
            if calls >= llm_config.max_calls_per_task:
                return _failed_phase_grounding_result(
                    state,
                    runtime=active_runtime,
                    observation=observation,
                    service_status="skipped_provider_call_limit",
                    error_type="ProviderSafetyLimit",
                    provider_attempted=False,
                )
            if synchronization.provider_slot_reserved:
                raise RuntimeError("duplicate same-task Provider reservation")
            # The reservation is process-local and occurs while holding the
            # task lock, before the Provider await.  Persistent call count is
            # committed only if telemetry confirms that an HTTP attempt began.
            synchronization.provider_slot_reserved = True
    except Exception as exc:
        return _failed_phase_grounding_result(
            state,
            runtime=active_runtime,
            observation=observation,
            service_status="degraded_configuration",
            error_type=type(exc).__name__[:128],
            provider_attempted=False,
        )
    try:
        try:
            context = _build_validation_context(state, observation)
            service_result = await process_sql_grounding_observation(
                active_runtime,
                observation,
                context,
                updater,
                grounding_input=grounding_input,
            )
        finally:
            synchronization.provider_slot_reserved = False
    except Exception as exc:
        return _failed_phase_grounding_result(
            state,
            runtime=active_runtime,
            observation=observation,
            service_status="degraded_service_boundary",
            error_type=type(exc).__name__[:128],
            provider_attempted=False,
        )
    if service_result.llm_telemetry.attempted and provider_mode:
        _record_provider_call(state, observation.phase)
    candidate = service_result.runtime
    control_events: tuple[dict[str, str], ...] = ()
    control_status: Literal[
        "not_applicable", "succeeded", "failed_open", "failed_closed"
    ]
    control_error_type: str | None = None
    if service_result.state_update.status in {"accepted", "noop"}:
        control_status = "succeeded"
        try:
            if call_kind == "check":
                if not isinstance(service_result.response, GroundingCheckResponse):
                    raise ValueError("Check returned the wrong typed response")
                if service_result.response.status == "complete":
                    _append_check_audit(
                        state,
                        {
                            "phase": observation.phase,
                            "status": "complete",
                            "missing_information": None,
                            "tool_name": None,
                            "budget_before": _finite_budget(state),
                            "tool_cost": 0.0,
                            "budget_after": _finite_budget(state),
                            "blocked_reason": None,
                        },
                    )
                    candidate, control_events = _advance_ready_control_stage(
                        state,
                        candidate,
                    )
                    if provider_mode:
                        _record_phase_grounding_outcome(
                            state,
                            observation=observation,
                            runtime=candidate,
                            status="succeeded",
                            provider_attempted=service_result.llm_telemetry.attempted,
                        )
                else:
                    _schedule_check_tool(
                        state,
                        phase=observation.phase,
                        response=service_result.response,
                    )
            elif call_kind == "knowledge" and isinstance(
                service_result.response,
                GroundingLLMResponse,
            ):
                # Archived injected-client fixtures use the retired whole-State
                # response.  Real 1.3 Provider calls cannot reach this branch.
                candidate, control_events = _advance_ready_control_stage(
                    state,
                    candidate,
                )
                if provider_mode:
                    _record_phase_grounding_outcome(
                        state,
                        observation=observation,
                        runtime=candidate,
                        status="succeeded",
                        provider_attempted=service_result.llm_telemetry.attempted,
                    )
        except Exception as exc:
            return _failed_phase_grounding_result(
                state,
                runtime=active_runtime,
                observation=observation,
                service_status="rejected_phase_finalization",
                error_type=_grounding_control_error_type(exc),
                provider_attempted=service_result.llm_telemetry.attempted,
                service_result=service_result,
            )
    else:
        return _failed_phase_grounding_result(
            state,
            runtime=active_runtime,
            observation=observation,
            service_status=service_result.state_update.status,
            error_type=(
                service_result.state_update.error_type
                or "GroundingStateNotEstablished"
            ),
            provider_attempted=service_result.llm_telemetry.attempted,
            service_result=service_result,
        )
    return _ObservationResult(
        runtime=candidate,
        service_status=service_result.state_update.status,
        observation=observation,
        service_result=service_result,
        control_events=control_events,
        control_status=control_status,
        control_error_type=control_error_type,
    )


def _grounding_control_error_type(exc: Exception) -> str:
    if str(exc) == "budget_exhausted":
        return "budget_exhausted"
    return type(exc).__name__[:128]


def _finite_budget(state: Any) -> float:
    value = state.get("budget_remaining")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("budget_remaining must be finite")
    return float(value)


def _schedule_check_tool(
    state: Any,
    *,
    phase: Literal[1, 2],
    response: GroundingCheckResponse,
) -> _PendingCheckTool:
    """Apply duplicate/gap/budget guardrails before any Official tool call."""

    request = response.next_tool
    gap = response.missing_information
    if response.status != "incomplete" or request is None or gap is None:
        raise ValueError("only an incomplete Check can schedule one tool")
    if _pending_check_tool(state) is not None:
        raise ValueError("a Check tool is already pending")
    phase_audits = [
        item for item in _check_audits(state) if item.get("phase") == phase
    ]
    if any(item.get("missing_information") == gap for item in phase_audits):
        raise ValueError("Check did not identify a new concrete gap")
    args_json = canonical_json(request.arguments)
    request_digest = _sha256_text(f"{request.tool_name}:{args_json}")
    if any(item.get("request_digest") == request_digest for item in phase_audits):
        raise ValueError("Check repeated an Official tool with identical arguments")

    from system_agent import callbacks as baseline_callbacks

    cost = baseline_callbacks.TOOL_COSTS.get(request.tool_name)
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise ValueError("Check tool has no valid Official Bird-Coin cost")
    budget_before = _finite_budget(state)
    budget_after = budget_before - float(cost)
    if budget_after < _MIN_BUDGET_AFTER_CHECK_TOOL:
        _append_check_audit(
            state,
            {
                "phase": phase,
                "status": "incomplete",
                "missing_information": gap,
                "tool_name": request.tool_name,
                "request_digest": request_digest,
                "budget_before": budget_before,
                "tool_cost": float(cost),
                "budget_after": budget_after,
                "blocked_reason": "budget_exhausted",
            },
        )
        raise ValueError("budget_exhausted")

    if request.tool_name == "ask_user":
        clarification = request.user_clarification_request
        if clarification is None:
            raise ValueError("ask_user Check is missing clarification metadata")
        _register_clarification_requests(
            state,
            phase=phase,
            requests=(clarification,),
        )
        record = next(
            item
            for item in _clarification_records(state)
            if item.phase == phase and item.question == clarification.question
        )
        function_call_id = _clarification_function_call_id(state, record)
    else:
        function_call_id = (
            f"valibra-check-{phase}-{len(phase_audits) + 1}-"
            f"{request_digest[:16]}"
        )
    pending = _PendingCheckTool(
        phase=phase,
        function_call_id=function_call_id,
        missing_information=gap,
        tool_name=request.tool_name,
        arguments=dict(request.arguments),
        request_digest=request_digest,
    )
    _store_pending_check_tool(state, pending)
    _append_check_audit(
        state,
        {
            "phase": phase,
            "status": "incomplete",
            "missing_information": gap,
            "tool_name": request.tool_name,
            "arguments_sha256": _sha256_text(args_json),
            "request_digest": request_digest,
            "budget_before": budget_before,
            "tool_cost": float(cost),
            "budget_after": budget_after,
            "blocked_reason": None,
        },
    )
    return pending


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


def _official_submit_count(state: Any) -> int:
    """Count only actual submit entries written by the frozen Baseline."""

    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list):
        raise ValueError("official tool_trajectory must be a list")
    return sum(
        1
        for event in trajectory
        if isinstance(event, dict) and event.get("tool") == "submit_sql"
    )


def _official_submit_count_for_phase(
    state: Any,
    phase: Literal[1, 2],
) -> int:
    """Count actual submits only within the current Official phase."""

    return sum(
        1
        for event in _completed_sql_grounding_trajectory(state)
        if event["tool_name"] == "submit_sql" and event["phase"] == phase
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
    supported_json_paths: set[tuple[str, tuple[str, ...]]] = set()
    supported_knowledge: set[tuple[str, str]] = set()
    for event in _completed_sql_grounding_trajectory(state):
        _project_official_evidence(
            tool_name=event["tool_name"],
            observation_type=event["observation_type"],
            content=event["content"],
            known_tables=known_tables,
            known_columns=known_columns,
            supported_json_paths=supported_json_paths,
            supported_knowledge=supported_knowledge,
        )
    for clarification in _clarification_records(state):
        if (
            clarification.kind == "missing_knowledge"
            and clarification.answer is not None
            and clarification.answer == clarification.answer.strip()
            and len(clarification.answer) <= 2_048
        ):
            supported_knowledge.add(("business_rule", clarification.answer))
    _project_official_evidence(
        tool_name=observation.tool_name,
        observation_type=observation.observation_type,
        content=observation.content,
        known_tables=known_tables,
        known_columns=known_columns,
        supported_json_paths=supported_json_paths,
        supported_knowledge=supported_knowledge,
    )
    return ValidationContext(
        current_query=_user_message_query(bound.message),
        follow_up_query=_validation_follow_up_query(state, observation),
        latest_observation_id=observation.observation_id,
        official_trajectory_observation_ids=_official_trajectory_refs(state),
        known_tables=frozenset(known_tables),
        known_columns=frozenset(known_columns),
        supported_json_paths=frozenset(supported_json_paths),
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


def _provider_phase_call_counts(state: Any) -> dict[str, int]:
    """Load the bounded per-phase ledger, migrating the old P1-only counter."""

    total = _provider_call_count(state)
    payload = state.get(GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY)
    if payload is None:
        if total > _MAX_PROVIDER_CALLS_PER_TASK:
            raise ValueError("legacy Provider counter exceeds the P1 phase limit")
        return {"1": total, "2": 0}
    if not isinstance(payload, dict) or set(payload) != {"1", "2"}:
        raise ValueError("invalid SQL Grounding phase call counters")
    counts: dict[str, int] = {}
    for key in ("1", "2"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= _MAX_PROVIDER_CALLS_PER_TASK
        ):
            raise ValueError("invalid SQL Grounding phase call counter")
        counts[key] = value
    if sum(counts.values()) != total or total > _MAX_PROVIDER_CALLS_PER_TASK:
        raise ValueError("SQL Grounding total and phase call counters differ")
    return counts


def _provider_phase_call_count(state: Any, phase: Literal[1, 2]) -> int:
    return _provider_phase_call_counts(state)[str(phase)]


def _record_provider_call(state: Any, phase: Literal[1, 2]) -> None:
    counts = _provider_phase_call_counts(state)
    total = _provider_call_count(state)
    key = str(phase)
    if (
        total >= _MAX_PROVIDER_CALLS_PER_TASK
    ):
        raise ValueError("SQL Grounding Provider call limit exceeded")
    counts[key] += 1
    state[GROUNDING_PROVIDER_PHASE_CALL_COUNTS_KEY] = counts
    state[GROUNDING_PROVIDER_CALL_COUNT_KEY] = total + 1


def _request_exposes_bootstrap_tools(llm_request: Any) -> bool:
    """Require the real ADK request to expose all three Official tools."""

    tools = getattr(llm_request, "tools_dict", None)
    return isinstance(tools, dict) and all(
        tool_name in tools for tool_name in _BOOTSTRAP_TOOL_SEQUENCE
    )


def _next_bootstrap_tool(
    state: Any,
    runtime: GroundingRuntime,
    llm_request: Any,
) -> str | None:
    """Return the next fixed Official bootstrap tool without persisting state."""

    if (
        not _is_real_provider_mode()
        or runtime.stage != "INITIAL_GROUNDING"
        or _phase(state.get("current_phase", 1)) != 1
        or not _request_exposes_bootstrap_tools(llm_request)
    ):
        return None
    observed = tuple(
        event["tool_name"]
        for event in _completed_sql_grounding_trajectory(state)
        if event["tool_name"] in _BOOTSTRAP_TOOL_SEQUENCE
    )
    expected_prefix = _BOOTSTRAP_TOOL_SEQUENCE[: len(observed)]
    if observed != expected_prefix:
        raise ValueError("Official bootstrap trajectory is duplicate or out of order")
    if len(observed) == len(_BOOTSTRAP_TOOL_SEQUENCE):
        return None
    if _provider_phase_call_count(state, 1) != len(observed):
        raise ValueError("bootstrap evidence and staged Grounding calls differ")
    return _BOOTSTRAP_TOOL_SEQUENCE[len(observed)]


def _bootstrap_function_call_response(state: Any, tool_name: str) -> Any:
    """Return one deterministic ADK function call; Baseline runs the tool."""

    if tool_name not in _BOOTSTRAP_TOOL_SEQUENCE:
        raise ValueError("unsupported bootstrap tool")
    from google.adk.models.llm_response import LlmResponse as AdkLlmResponse
    from google.genai import types as genai_types

    return AdkLlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=_bootstrap_function_call_id(state, tool_name),
                        name=tool_name,
                        args={},
                    )
                )
            ],
        )
    )


def _clarification_function_call_response(
    state: Any,
    record: UserClarificationRecord,
) -> Any:
    """Force one Grounding-owned clarification through the Official tool."""

    if record.answer is not None:
        raise ValueError("answered clarification cannot be requested again")
    from google.adk.models.llm_response import LlmResponse as AdkLlmResponse
    from google.genai import types as genai_types

    return AdkLlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=_clarification_function_call_id(state, record),
                        name="ask_user",
                        args={"question": record.question},
                    )
                )
            ],
        )
    )


def _check_tool_function_call_response(
    state: Any,
    record: _PendingCheckTool,
) -> Any:
    """Force exactly the bounded Official action selected by Check."""

    current = _pending_check_tool(state)
    if current != record:
        raise ValueError("pending Check tool changed before dispatch")
    from google.adk.models.llm_response import LlmResponse as AdkLlmResponse
    from google.genai import types as genai_types

    return AdkLlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=record.function_call_id,
                        name=record.tool_name,
                        args=record.arguments,
                    )
                )
            ],
        )
    )


def _pending_check_tool(state: Any) -> _PendingCheckTool | None:
    payload = state.get(GROUNDING_PENDING_CHECK_KEY)
    if payload is None:
        return None
    return _PendingCheckTool.from_json(payload)


def _store_pending_check_tool(state: Any, record: _PendingCheckTool | None) -> None:
    if record is None:
        state.pop(GROUNDING_PENDING_CHECK_KEY, None)
        return
    if _pending_check_tool(state) is not None:
        raise ValueError("a Check tool is already pending")
    state[GROUNDING_PENDING_CHECK_KEY] = record.to_json()


def _is_check_function_call(
    state: Any,
    *,
    function_call_id: str,
    tool_name: str,
) -> bool:
    record = _pending_check_tool(state)
    return bool(
        record is not None
        and record.function_call_id == function_call_id
        and record.tool_name == tool_name
    )


def _check_audits(state: Any) -> tuple[dict[str, Any], ...]:
    payload = state.get(GROUNDING_CHECK_AUDITS_KEY, [])
    if not isinstance(payload, list) or len(payload) > _MAX_CHECK_AUDITS:
        raise ValueError("invalid Check audit ledger")
    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("invalid Check audit record")
        _require_bounded_audit(item)
        records.append(dict(item))
    return tuple(records)


def _append_check_audit(state: Any, audit: dict[str, Any]) -> None:
    _require_bounded_audit(audit)
    records = list(_check_audits(state))
    if len(records) >= _MAX_CHECK_AUDITS:
        raise ValueError("Check audit ledger is full")
    records.append(dict(audit))
    state[GROUNDING_CHECK_AUDITS_KEY] = records


def _clarification_records(state: Any) -> tuple[UserClarificationRecord, ...]:
    raw = state.get(GROUNDING_CLARIFICATIONS_KEY, [])
    if not isinstance(raw, list) or len(raw) > _MAX_CLARIFICATION_RECORDS:
        raise ValueError("invalid clarification overlay")
    records = tuple(UserClarificationRecord.model_validate(item) for item in raw)
    keys = [(item.phase, item.question) for item in records]
    if len(keys) != len(set(keys)):
        raise ValueError("clarification overlay contains duplicates")
    return records


def _store_clarification_records(
    state: Any,
    records: tuple[UserClarificationRecord, ...],
) -> None:
    if len(records) > _MAX_CLARIFICATION_RECORDS:
        raise ValueError("clarification overlay exceeds its task bound")
    state[GROUNDING_CLARIFICATIONS_KEY] = [
        item.model_dump(mode="json") for item in records
    ]


def _register_clarification_requests(
    state: Any,
    *,
    phase: Literal[1, 2],
    requests: tuple[UserClarificationRequest, ...],
) -> None:
    records = _clarification_records(state)
    proposed = tuple(
        UserClarificationRecord(
            phase=phase,
            phrase=item.phrase,
            kind=item.kind,
            question=item.question,
        )
        for item in requests
    )
    proposed_questions = [item.question for item in proposed]
    if len(proposed_questions) != len(set(proposed_questions)):
        raise ValueError("clarification questions must be unique within a phase")
    existing_questions = {
        item.question for item in records if item.phase == phase
    }
    if existing_questions.intersection(proposed_questions):
        raise ValueError("clarification question already exists in this phase")
    _store_clarification_records(state, (*records, *proposed))


def _next_pending_clarification(state: Any) -> UserClarificationRecord | None:
    return next(
        (item for item in _clarification_records(state) if item.answer is None),
        None,
    )


def _pending_clarification_count(state: Any) -> int:
    return sum(
        item.answer is None for item in _clarification_records(state)
    )


def _clarification_function_call_id(
    state: Any,
    record: UserClarificationRecord,
) -> str:
    records = _clarification_records(state)
    try:
        position = records.index(record) + 1
    except ValueError as exc:
        raise ValueError("clarification is absent from Session overlay") from exc
    suffix = _sha256_text(
        f"{_task_id(state)}:{record.phase}:{record.question}"
    )[:16]
    return f"valibra-clarification-{position}-{suffix}"


def _is_clarification_function_call(
    state: Any,
    *,
    function_call_id: str,
    phase: Literal[1, 2],
    question: str,
) -> bool:
    for record in _clarification_records(state):
        if record.phase == phase and record.question == question:
            return function_call_id == _clarification_function_call_id(
                state,
                record,
            )
    return False


def _record_clarification_answer(
    state: Any,
    *,
    phase: Literal[1, 2],
    question: str,
    answer: Any,
) -> None:
    if not isinstance(answer, str):
        raise ValueError("clarification answer must be a string")
    records = list(_clarification_records(state))
    matches = [
        index
        for index, record in enumerate(records)
        if record.phase == phase and record.question == question
    ]
    if len(matches) != 1:
        raise ValueError("clarification answer does not pair uniquely")
    index = matches[0]
    current = records[index]
    if current.answer is not None:
        if current.answer == answer:
            return
        raise ValueError("clarification answer is immutable")
    records[index] = UserClarificationRecord(
        phase=current.phase,
        phrase=current.phrase,
        kind=current.kind,
        question=current.question,
        answer=answer,
    )
    _store_clarification_records(state, tuple(records))


def _phase_grounding_outcomes(
    state: Any,
) -> dict[str, _PhaseGroundingOutcome]:
    payload = state.get(GROUNDING_PHASE_OUTCOMES_KEY, {})
    if not isinstance(payload, dict) or len(payload) > _MAX_PHASE_OUTCOMES:
        raise ValueError("invalid phase Grounding outcome store")
    records: dict[str, _PhaseGroundingOutcome] = {}
    for key, value in payload.items():
        if key not in {"1", "2"}:
            raise ValueError("invalid phase Grounding outcome key")
        record = _PhaseGroundingOutcome.from_json(value)
        if str(record.phase) != key:
            raise ValueError("phase Grounding outcome key mismatch")
        records[key] = record
    return records


def _store_phase_grounding_outcomes(
    state: Any,
    records: Mapping[str, _PhaseGroundingOutcome],
) -> None:
    if len(records) > _MAX_PHASE_OUTCOMES or any(
        key not in {"1", "2"} for key in records
    ):
        raise ValueError("invalid phase Grounding outcomes")
    state[GROUNDING_PHASE_OUTCOMES_KEY] = {
        key: _PhaseGroundingOutcome.from_json(records[key].to_json()).to_json()
        for key in sorted(records)
    }


def _record_phase_grounding_outcome(
    state: Any,
    *,
    observation: SQLGroundingObservation,
    runtime: GroundingRuntime,
    status: Literal["succeeded", "failed"],
    error_type: str | None = None,
    provider_attempted: bool,
) -> _PhaseGroundingOutcome:
    if status == "succeeded" and (
        not runtime.grounding_state.all_dimensions_evaluated
        or runtime.focus_dimension != "none"
    ):
        raise ValueError("successful phase Grounding requires a complete State")
    record = _PhaseGroundingOutcome(
        phase=observation.phase,
        status=status,
        observation_id=observation.observation_id,
        error_type=error_type,
        provider_attempted=provider_attempted,
        grounding_revision=runtime.grounding_revision,
        state_sha256=sql_grounding_state_sha256(runtime.grounding_state),
    )
    record = _PhaseGroundingOutcome.from_json(record.to_json())
    records = _phase_grounding_outcomes(state)
    key = str(observation.phase)
    existing = records.get(key)
    if existing is not None:
        if existing == record:
            return existing
        raise ValueError("phase Grounding outcome is immutable")
    records[key] = record
    _store_phase_grounding_outcomes(state, records)
    return record


def _phase_grounding_failure(
    state: Any,
    phase: Literal[1, 2] | None = None,
) -> _PhaseGroundingOutcome | None:
    selected_phase = phase or _phase(state.get("current_phase", 1))
    record = _phase_grounding_outcomes(state).get(str(selected_phase))
    return record if record is not None and record.status == "failed" else None


def _phase_grounding_succeeded(
    state: Any,
    phase: Literal[1, 2] | None = None,
) -> bool:
    selected_phase = phase or _phase(state.get("current_phase", 1))
    record = _phase_grounding_outcomes(state).get(str(selected_phase))
    return record is not None and record.status == "succeeded"


def _failed_closed_response(
    failure: _PhaseGroundingOutcome,
) -> dict[str, Any]:
    return {
        "status": _GROUNDING_FAILED_CLOSED_STATUS,
        "phase": failure.phase,
        "error_type": failure.error_type,
        "guidance": _GROUNDING_FAILED_CLOSED_GUIDANCE,
    }


def _failed_closed_model_response(
    failure: _PhaseGroundingOutcome,
) -> Any:
    from google.adk.models.llm_response import LlmResponse as AdkLlmResponse
    from google.genai import types as genai_types

    return AdkLlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part.from_text(
                    text=(
                        f"{_GROUNDING_FAILED_CLOSED_STATUS}: phase={failure.phase}. "
                        f"{_GROUNDING_FAILED_CLOSED_GUIDANCE}"
                    )
                )
            ],
        )
    )


def _load_failed_closed_calls(
    state: Any,
) -> dict[str, _FailedClosedToolCall]:
    payload = state.get(GROUNDING_FAILED_CLOSED_CALLS_KEY, {})
    if not isinstance(payload, dict) or len(payload) > _MAX_FAILED_CLOSED_CALLS:
        raise ValueError("invalid failed-closed call store")
    records: dict[str, _FailedClosedToolCall] = {}
    for key, value in payload.items():
        record = _FailedClosedToolCall.from_json(value)
        if key != record.function_call_id:
            raise ValueError("failed-closed call key mismatch")
        records[key] = record
    return records


def _store_failed_closed_calls(
    state: Any,
    records: Mapping[str, _FailedClosedToolCall],
) -> None:
    if len(records) > _MAX_FAILED_CLOSED_CALLS:
        raise ValueError("failed-closed call store is full")
    state[GROUNDING_FAILED_CLOSED_CALLS_KEY] = {
        key: _FailedClosedToolCall.from_json(records[key].to_json()).to_json()
        for key in sorted(records)
    }


def _add_failed_closed_call(
    state: Any,
    record: _FailedClosedToolCall,
) -> None:
    records = _load_failed_closed_calls(state)
    if record.function_call_id in records:
        raise ValueError("duplicate failed-closed function_call_id")
    records[record.function_call_id] = record
    _store_failed_closed_calls(state, records)


def _pop_failed_closed_call(
    state: Any,
    function_call_id: str,
) -> _FailedClosedToolCall | None:
    records = _load_failed_closed_calls(state)
    record = records.pop(function_call_id, None)
    _store_failed_closed_calls(state, records)
    return record


def _failed_closed_call_present(state: Any, function_call_id: str) -> bool:
    payload = state.get(GROUNDING_FAILED_CLOSED_CALLS_KEY, {})
    return isinstance(payload, dict) and function_call_id in payload


def _bootstrap_function_call_id(state: Any, tool_name: str) -> str:
    if tool_name not in _BOOTSTRAP_TOOL_SEQUENCE:
        raise ValueError("unsupported bootstrap tool")
    position = _BOOTSTRAP_TOOL_SEQUENCE.index(tool_name) + 1
    call_suffix = _sha256_text(f"{_task_id(state)}:{position}")[:16]
    return f"valibra-bootstrap-{position}-{call_suffix}"


def _is_bootstrap_function_call(
    state: Any,
    *,
    function_call_id: str,
    tool_name: str,
) -> bool:
    return (
        tool_name in _BOOTSTRAP_TOOL_SEQUENCE
        and function_call_id == _bootstrap_function_call_id(state, tool_name)
    )


def _bootstrap_model_visible_result(
    tool_name: str,
    baseline_override: Any,
    *,
    succeeded: bool,
) -> str:
    """Expose only completion status and Baseline's budget note to Main."""

    if tool_name not in _BOOTSTRAP_TOOL_SEQUENCE:
        raise ValueError("unsupported bootstrap tool")
    budget_note = ""
    if isinstance(baseline_override, str):
        marker = "\n\n[SYSTEM NOTE:"
        start = baseline_override.rfind(marker)
        candidate = baseline_override[start:] if start >= 0 else ""
        if candidate.endswith("]") and len(candidate) <= 256:
            budget_note = candidate
    status = "stored" if succeeded else "tool_error_stored"
    return (
        f"{_BOOTSTRAP_MODEL_VISIBLE_PREFIX}: {tool_name}; status={status}. "
        "Full content is retained only in the Official tool trajectory."
        f"{budget_note}"
    )


def _duplicate_bootstrap_response(tool_name: str) -> dict[str, str]:
    if tool_name not in _BOOTSTRAP_TOOL_SEQUENCE:
        raise ValueError("unsupported duplicate bootstrap tool")
    return {
        "status": "already_stored",
        "tool": tool_name,
        "message": (
            f"{_BOOTSTRAP_ALREADY_VISIBLE_PREFIX}: {tool_name}. "
            "Reuse the existing Official evidence; do not call this tool again."
        ),
    }


def _bootstrap_tool_already_attempted(state: Any, tool_name: str) -> bool:
    if tool_name not in _BOOTSTRAP_TOOL_SEQUENCE:
        return False
    return any(
        event["tool_name"] == tool_name
        for event in _completed_sql_grounding_trajectory(state)
    )


def _bootstrap_evidence_events(state: Any) -> tuple[dict[str, Any], ...]:
    """Require one successful Official result for each fixed bootstrap tool."""

    events = tuple(
        event
        for event in _completed_sql_grounding_trajectory(state)
        if event["tool_name"] in _BOOTSTRAP_TOOL_SEQUENCE
    )
    observed = tuple(event["tool_name"] for event in events)
    if observed != _BOOTSTRAP_TOOL_SEQUENCE:
        raise ValueError("complete ordered bootstrap evidence is required")
    if any(event["observation_type"] == "tool_error" for event in events):
        raise ValueError("bootstrap evidence contains an Official tool error")
    return events


def _bootstrap_evidence_prefix(state: Any) -> tuple[dict[str, Any], ...]:
    """Return the unique successful Official bootstrap prefix seen so far."""

    events = tuple(
        event
        for event in _completed_sql_grounding_trajectory(state)
        if event["tool_name"] in _BOOTSTRAP_TOOL_SEQUENCE
    )
    observed = tuple(event["tool_name"] for event in events)
    if observed != _BOOTSTRAP_TOOL_SEQUENCE[: len(observed)]:
        raise ValueError("bootstrap evidence is duplicate or out of order")
    if any(event["observation_type"] == "tool_error" for event in events):
        raise ValueError("bootstrap evidence contains an Official tool error")
    return events


def _official_p2_follow_up(state: Any) -> str:
    """Recover the unique Official P2 question from a successful P1 submit."""

    follow_ups: list[str] = []
    for event in _completed_sql_grounding_trajectory(state):
        if event["tool_name"] != "submit_sql" or event["phase"] != 1:
            continue
        try:
            follow_ups.append(_extract_submit_follow_up(event["content"]))
        except ValueError:
            continue
    if len(follow_ups) != 1:
        raise ValueError("exactly one Official P2 follow-up is required")
    return follow_ups[0]


def _validation_follow_up_query(
    state: Any,
    observation: SQLGroundingObservation,
) -> str | None:
    if observation.observation_type == "p2_follow_up":
        if not isinstance(observation.content, str):
            raise ValueError("P2 follow-up Observation must be text")
        return observation.content
    if observation.phase == 2:
        return _official_p2_follow_up(state)
    return None


def _phase_request_common(
    state: Any,
    *,
    query: str,
    runtime: GroundingRuntime,
    phase: Literal[1, 2],
    follow_up: str | None = None,
) -> dict[str, Any]:
    """Build only the stable query/State fields shared by one staged call."""

    if not isinstance(query, str) or not query or query != query.strip():
        raise ValueError("bounded original query is required")
    result: dict[str, Any] = {
        "query": query,
        "current_state": runtime.grounding_state.model_dump(mode="json"),
    }
    if phase == 2:
        if (
            runtime.stage != "P2_INCREMENTAL"
            or not isinstance(follow_up, str)
            or not follow_up
            or follow_up != follow_up.strip()
        ):
            raise ValueError("P2 staged Grounding requires a bounded follow-up")
        result.update(
            {
                "follow_up": follow_up,
                "user_clarifications": [
                    item.model_dump(mode="json")
                    for item in _clarification_records(state)
                    if item.phase == 1 and item.answer is not None
                ],
            }
        )
    elif phase != 1:
        raise ValueError("Grounding phase must be 1 or 2")
    return result


def _build_staged_grounding_request(
    state: Any,
    *,
    call_kind: GroundingCallKind,
    query: str,
    runtime: GroundingRuntime,
    phase: Literal[1, 2],
    follow_up: str | None = None,
) -> dict[str, Any]:
    """Project only the evidence authorized for one fixed 1.2 call."""

    events = _bootstrap_evidence_prefix(state)
    by_tool = {event["tool_name"]: event["content"] for event in events}
    base = _phase_request_common(
        state,
        query=query,
        runtime=runtime,
        phase=phase,
        follow_up=follow_up,
    )
    if call_kind == "structure":
        schema = by_tool.get("get_schema")
        _parse_schema_projection(schema)
        return {**base, "schema": schema}
    if call_kind == "mapping":
        meanings = by_tool.get("get_all_column_meanings")
        if runtime.grounding_state.tables is None:
            raise ValueError("Mapping Grounding requires evaluated candidate tables")
        return {
            **base,
            "column_meanings": _project_column_meanings(
                meanings,
                tables=runtime.grounding_state.tables,
            ),
        }
    if call_kind == "knowledge":
        definitions = by_tool.get("get_all_knowledge_definitions")
        if definitions is None:
            raise ValueError("Knowledge Grounding requires bootstrap knowledge")
        return {
            **base,
            "knowledge_definitions": _normalized_knowledge_definitions(definitions),
            "relevant_column_meanings": _relevant_mapping_column_meanings(
                by_tool.get("get_all_column_meanings"),
                runtime,
            ),
        }
    raise ValueError("unsupported staged Grounding call kind")


def _build_check_grounding_request(
    state: Any,
    *,
    query: str,
    runtime: GroundingRuntime,
    phase: Literal[1, 2],
    follow_up: str | None = None,
    initial: bool = False,
    latest_tool_name: str | None = None,
    latest_tool_arguments: Any = None,
    latest_tool_result: Any = None,
    latest_user_answer: Any = None,
) -> dict[str, Any]:
    """Build one Check request with no accumulated raw Check history."""

    base = _phase_request_common(
        state,
        query=query,
        runtime=runtime,
        phase=phase,
        follow_up=follow_up,
    )
    modes = sum((initial, latest_tool_name is not None, latest_user_answer is not None))
    if modes != 1:
        raise ValueError("Check requires exactly one bounded latest-input mode")
    if initial:
        return {**base, "check_context": {"kind": "initial"}}
    if latest_user_answer is not None:
        pending = _pending_check_tool(state)
        # after_tool clears pending only after this payload is built
        if pending is None or pending.tool_name != "ask_user":
            raise ValueError("latest user answer has no paired Check request")
        return {
            **base,
            "latest_user_answer": {
                "question": pending.arguments["question"],
                "answer": latest_user_answer,
            },
        }
    if latest_tool_name not in {
        "ask_user",
        "get_column_meaning",
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "execute_sql",
    }:
        raise ValueError("Check result uses an unapproved tool")
    return {
        **base,
        "latest_tool": {
            "name": latest_tool_name,
            "arguments": to_jsonable(latest_tool_arguments),
            "result": to_jsonable(latest_tool_result),
        },
    }


def _completed_sql_grounding_trajectory(state: Any) -> tuple[dict[str, Any], ...]:
    """Reconstruct trusted evidence from Official events, never shadow presence."""

    result: list[dict[str, Any]] = []
    trajectory = state.get("tool_trajectory", [])
    if not isinstance(trajectory, list):
        return ()
    for index, event in enumerate(trajectory):
        if not isinstance(event, dict):
            continue
        tool_name = event.get("tool")
        if not isinstance(tool_name, str):
            continue
        content = event.get("result")
        try:
            observation_type = _classify_tool_observation_type(tool_name, content)
        except ValueError:
            continue
        result.append(
            {
                "tool_name": tool_name,
                "observation_type": observation_type,
                "content": content,
                "phase": event.get("phase"),
                "args": event.get("args"),
                "trajectory_index": index,
                "private_raw_ref": f"session://tool_trajectory/{index}",
                "raw_digest": _sha256_text(canonical_json(to_jsonable(content))),
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
    supported_json_paths: set[tuple[str, tuple[str, ...]]] | None = None,
) -> None:
    if observation_type == "schema" and tool_name == "get_schema":
        try:
            tables, columns = _parse_schema_projection(content)
        except ValueError:
            return
        known_tables.update(tables)
        known_columns.update(columns)
        return
    if observation_type == "metadata" and tool_name == "get_all_column_meanings":
        if supported_json_paths is None:
            return
        supported_json_paths.update(
            _column_meaning_json_paths(content, known_columns=known_columns)
        )
        return
    if observation_type == "knowledge" and tool_name == "get_knowledge_definition":
        definition = _exact_knowledge_definition(content)
        if definition is not None:
            supported_knowledge.add(("business_rule", definition))
        return
    if (
        observation_type == "knowledge"
        and tool_name == "get_all_knowledge_definitions"
    ):
        try:
            definitions = _exact_knowledge_definitions(content)
        except ValueError:
            return
        supported_knowledge.update(
            ("business_rule", definition) for definition in definitions
        )
        return
    if tool_name == "execute_sql" and isinstance(content, str):
        bounded = content.strip()
        if bounded and len(bounded) <= 2_048:
            kind = (
                "database_capability"
                if observation_type == "tool_error"
                else "runtime_state"
            )
            supported_knowledge.add((kind, bounded))


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


def _exact_all_column_meanings(content: Any) -> dict[str, Any]:
    """Validate the Official bulk metadata envelope without persisting it."""

    value = content
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("bulk column meanings must be strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("bulk column meanings must be a JSON object")
    try:
        normalized = json.loads(canonical_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("bulk column meanings must be finite JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError("bulk column meanings must be a JSON object")
    return normalized


def _exact_knowledge_definition_items(
    content: Any,
) -> tuple[tuple[int, str], ...]:
    """Recognize unique integer IDs and exact definitions in Official bulk KB."""

    value = content
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("bulk knowledge definitions must be strict JSON") from exc
    if not isinstance(value, list):
        raise ValueError("bulk knowledge definitions must be a JSON list")
    try:
        canonical_json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("bulk knowledge definitions must be finite JSON") from exc
    definitions: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each bulk knowledge entry must be an object")
        if not set(item).issubset(_BULK_KNOWLEDGE_VISIBLE_FIELDS):
            raise ValueError("bulk knowledge entry contains an unsupported field")
        knowledge_id = item.get("id")
        if isinstance(knowledge_id, bool) or not isinstance(knowledge_id, int):
            raise ValueError("bulk knowledge entry has no integer id")
        if knowledge_id in seen_ids:
            raise ValueError("bulk knowledge ids must be unique")
        definition = _exact_knowledge_definition(item)
        if definition is None:
            raise ValueError("bulk knowledge entry has no canonical definition")
        seen_ids.add(knowledge_id)
        definitions.append((knowledge_id, definition))
    return tuple(definitions)


def _exact_knowledge_definitions(content: Any) -> tuple[str, ...]:
    """Recognize every exact definition in the Official bulk response."""

    return tuple(
        definition
        for _, definition in _exact_knowledge_definition_items(content)
    )


def _normalized_knowledge_definitions(content: Any) -> list[dict[str, Any]]:
    """Return the validated Official list as finite JSON, never raw history."""

    value = content
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("bulk knowledge definitions must be strict JSON") from exc
    _exact_knowledge_definition_items(value)
    normalized = json.loads(canonical_json(value))
    if not isinstance(normalized, list):
        raise ValueError("bulk knowledge definitions must be a JSON list")
    return normalized


def _project_column_meanings(
    content: Any,
    *,
    tables: tuple[str, ...],
    columns: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Keep only metadata for candidate tables and, optionally, exact fields."""

    meanings = _exact_all_column_meanings(content)
    table_names = set(tables)
    short_tables = {table.rsplit(".", 1)[-1] for table in tables}
    result: dict[str, Any] = {}
    for raw_key, value in meanings.items():
        if not isinstance(raw_key, str):
            continue
        parts = raw_key.split("|")
        if len(parts) >= 3:
            table = parts[-2]
            column = f"{table}.{parts[-1]}"
            if table not in table_names and table not in short_tables:
                continue
            if columns is not None and not any(
                candidate == column
                or candidate.endswith(f".{column}")
                for candidate in columns
            ):
                continue
            result[raw_key] = value
            continue
        if raw_key not in table_names and raw_key not in short_tables:
            continue
        if not isinstance(value, dict) or columns is None:
            result[raw_key] = value
            continue
        selected = {
            name: meaning
            for name, meaning in value.items()
            if isinstance(name, str)
            and any(
                candidate == f"{raw_key}.{name}"
                or candidate.endswith(f".{raw_key}.{name}")
                for candidate in columns
            )
        }
        if selected:
            result[raw_key] = selected
    return json.loads(canonical_json(result))


def _relevant_mapping_column_meanings(
    content: Any,
    runtime: GroundingRuntime,
) -> dict[str, Any]:
    """Expose all Official meanings inside the candidate-table boundary."""

    tables = runtime.grounding_state.tables
    if tables is None:
        raise ValueError("relevant metadata requires evaluated tables")
    return _project_column_meanings(content, tables=tables)


def _relevant_clarification_column_meanings(
    content: Any,
    runtime: GroundingRuntime,
    records: tuple[UserClarificationRecord, ...],
) -> dict[str, Any]:
    """Expose candidate-table meanings so a clarification can replace A with B."""

    if not records:
        raise ValueError("clarification metadata requires answered records")
    tables = runtime.grounding_state.tables
    if tables is None:
        raise ValueError("clarification metadata requires evaluated tables")
    return _project_column_meanings(content, tables=tables)


def _relevant_clarification_knowledge(
    content: Any,
    runtime: GroundingRuntime,
    records: tuple[UserClarificationRecord, ...],
) -> list[dict[str, Any]]:
    definitions = _normalized_knowledge_definitions(content)
    retained = {
        item.content for item in runtime.grounding_state.domain_knowledge or ()
    }
    needles = _clarification_needles(records)
    result: list[dict[str, Any]] = []
    for item in definitions:
        definition = _exact_knowledge_definition(item)
        if definition is None:
            continue
        folded = definition.casefold()
        if definition in retained or any(needle in folded for needle in needles):
            result.append(item)
    return json.loads(canonical_json(result))


def _clarification_needles(
    records: tuple[UserClarificationRecord, ...],
) -> frozenset[str]:
    values = " ".join(
        part
        for item in records
        for part in (item.phrase, item.question, item.answer or "")
    ).casefold()
    return frozenset(
        token
        for token in re.findall(r"[\w$]+", values, flags=re.UNICODE)
        if len(token) >= 3
    )


def _column_meaning_json_paths(
    content: Any,
    *,
    known_columns: set[str],
) -> frozenset[tuple[str, tuple[str, ...]]]:
    """Project only explicit fields_meaning paths for known schema columns."""

    value = _exact_all_column_meanings(content)
    result: set[tuple[str, tuple[str, ...]]] = set()
    for raw_column, meaning in value.items():
        if not isinstance(raw_column, str):
            continue
        parts = raw_column.split("|")
        if len(parts) >= 3:
            column = ".".join(parts[-2:])
            _collect_fields_meaning_paths(
                column,
                meaning,
                known_columns=known_columns,
                result=result,
            )
            continue
        if raw_column in {item.rsplit(".", 1)[0] for item in known_columns} and isinstance(
            meaning, dict
        ):
            for column_name, column_meaning in meaning.items():
                if isinstance(column_name, str):
                    _collect_fields_meaning_paths(
                        f"{raw_column}.{column_name}",
                        column_meaning,
                        known_columns=known_columns,
                        result=result,
                    )
    return frozenset(result)


def _collect_fields_meaning_paths(
    column: str,
    meaning: Any,
    *,
    known_columns: set[str],
    result: set[tuple[str, tuple[str, ...]]],
) -> None:
    if column not in known_columns or not isinstance(meaning, dict):
        return
    fields = meaning.get("fields_meaning")
    if not isinstance(fields, dict):
        return

    def visit(node: dict[str, Any], prefix: tuple[str, ...]) -> None:
        for key, child in node.items():
            if not isinstance(key, str) or not key or key != key.strip():
                continue
            path = prefix + (key,)
            result.add((column, path))
            if isinstance(child, dict):
                visit(child, path)

    visit(fields, ())


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


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
    control_status: Literal[
        "not_applicable", "succeeded", "failed_open", "failed_closed"
    ],
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


def _load_suppressed_bootstrap(
    state: Any,
) -> dict[str, _SuppressedBootstrapCall]:
    payload = state.get(GROUNDING_SUPPRESSED_BOOTSTRAP_KEY, {})
    if not isinstance(payload, dict) or len(payload) > _MAX_SUPPRESSED_BOOTSTRAP:
        raise ValueError("invalid suppressed-bootstrap store")
    records: dict[str, _SuppressedBootstrapCall] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("suppressed-bootstrap key must be a string")
        record = _SuppressedBootstrapCall.from_json(value)
        if record.function_call_id != key:
            raise ValueError(
                "suppressed-bootstrap key does not match function_call_id"
            )
        records[key] = record
    return records


def _store_suppressed_bootstrap(
    state: Any,
    records: Mapping[str, _SuppressedBootstrapCall],
) -> None:
    if len(records) > _MAX_SUPPRESSED_BOOTSTRAP:
        raise ValueError("too many suppressed-bootstrap records")
    state[GROUNDING_SUPPRESSED_BOOTSTRAP_KEY] = {
        key: _SuppressedBootstrapCall.from_json(records[key].to_json()).to_json()
        for key in sorted(records)
    }


def _add_suppressed_bootstrap(
    state: Any,
    record: _SuppressedBootstrapCall,
) -> None:
    records = _load_suppressed_bootstrap(state)
    if record.function_call_id in records:
        raise ValueError("duplicate suppressed-bootstrap function_call_id")
    records[record.function_call_id] = _SuppressedBootstrapCall.from_json(
        record.to_json()
    )
    _store_suppressed_bootstrap(state, records)


def _pop_suppressed_bootstrap(
    state: Any,
    function_call_id: str,
) -> _SuppressedBootstrapCall | None:
    records = _load_suppressed_bootstrap(state)
    record = records.pop(function_call_id, None)
    _store_suppressed_bootstrap(state, records)
    return record


def _suppressed_bootstrap_present(state: Any, function_call_id: str) -> bool:
    payload = state.get(GROUNDING_SUPPRESSED_BOOTSTRAP_KEY, {})
    return isinstance(payload, dict) and function_call_id in payload


def _cleanup_suppressed_bootstrap_best_effort(
    state: Any,
    tool_context: Any,
) -> None:
    function_call_id = _valid_context_identifier(tool_context)
    if function_call_id is None:
        return
    payload = state.get(GROUNDING_SUPPRESSED_BOOTSTRAP_KEY)
    if not isinstance(payload, dict) or function_call_id not in payload:
        return
    cleaned = dict(payload)
    cleaned.pop(function_call_id, None)
    state[GROUNDING_SUPPRESSED_BOOTSTRAP_KEY] = cleaned


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


def _load_tool_callback_audits(state: Any) -> dict[str, dict[str, Any]]:
    payload = state.get(GROUNDING_TOOL_AUDITS_KEY, {})
    if not isinstance(payload, dict) or len(payload) > _MAX_TOOL_AUDITS:
        raise ValueError("invalid exact tool callback audit store")
    records: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
            raise ValueError("invalid exact tool callback audit key")
        if not isinstance(value, dict) or value.get("function_call_id") != key:
            raise ValueError("invalid exact tool callback audit record")
        encoded = canonical_json(value).encode("utf-8")
        if len(encoded) > _MAX_TOOL_AUDIT_RECORD_BYTES:
            raise ValueError("exact tool callback audit record exceeds bound")
        records[key] = dict(value)
    return records


def _upsert_tool_callback_audit(
    state: Any,
    function_call_id: str,
    *,
    shadow_audit: dict[str, Any],
    control_audit: dict[str, Any] | None,
) -> None:
    """Persist exact-ID callback glue independently of trajectory timing."""

    if not _IDENTIFIER_RE.fullmatch(function_call_id):
        raise ValueError("exact tool callback audit requires function_call_id")
    _require_bounded_audit(shadow_audit)
    if control_audit is not None:
        _require_bounded_audit(control_audit)
    records = _load_tool_callback_audits(state)
    record: dict[str, Any] = {
        "function_call_id": function_call_id,
        SHADOW_AUDIT_KEY: dict(shadow_audit),
    }
    if control_audit is not None:
        record[GROUNDING_CONTROL_AUDIT_KEY] = dict(control_audit)
    if len(canonical_json(record).encode("utf-8")) > _MAX_TOOL_AUDIT_RECORD_BYTES:
        raise ValueError("exact tool callback audit record exceeds bound")
    if function_call_id not in records and len(records) >= _MAX_TOOL_AUDITS:
        raise ValueError("exact tool callback audit store is full")
    records[function_call_id] = record
    state[GROUNDING_TOOL_AUDITS_KEY] = records


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
    for index, event in enumerate(trajectory):
        if not isinstance(event, dict):
            continue
        audit = event.get(SHADOW_AUDIT_KEY)
        observation_id = (
            audit.get("observation_id") if isinstance(audit, dict) else None
        )
        if not isinstance(observation_id, str):
            try:
                digest = hashlib.sha256(
                    canonical_json(
                        {
                            "tool": event.get("tool"),
                            "phase": event.get("phase"),
                            "args": event.get("args"),
                            "result": event.get("result"),
                        }
                    ).encode("utf-8")
                ).hexdigest()
            except Exception:
                continue
            observation_id = f"official-trajectory-{index + 1}-{digest[:16]}"
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


def _filter_writer_tools(llm_request: Any) -> None:
    tools_dict = getattr(llm_request, "tools_dict", None)
    if not isinstance(tools_dict, dict):
        raise TypeError("LlmRequest.tools_dict is required")
    missing = [name for name in _SQL_WRITER_TOOL_NAMES if name not in tools_dict]
    if missing:
        raise ValueError("SQL Writer tools are unavailable")
    llm_request.tools_dict = {
        name: tools_dict[name] for name in _SQL_WRITER_TOOL_NAMES
    }
    config = getattr(llm_request, "config", None)
    declarations_groups = getattr(config, "tools", None)
    if not isinstance(declarations_groups, list):
        raise TypeError("LlmRequest.config.tools is required")
    retained_groups: list[Any] = []
    for group in declarations_groups:
        declarations = getattr(group, "function_declarations", None)
        if declarations is None:
            continue
        retained = [
            item
            for item in declarations
            if getattr(item, "name", None) in _SQL_WRITER_TOOL_NAMES
        ]
        if retained:
            cloned = copy.deepcopy(group)
            cloned.function_declarations = retained
            retained_groups.append(cloned)
    names = [
        getattr(item, "name", None)
        for group in retained_groups
        for item in (getattr(group, "function_declarations", None) or [])
    ]
    if names != list(_SQL_WRITER_TOOL_NAMES):
        raise ValueError("SQL Writer declaration set must be exactly two tools")
    config.tools = retained_groups


def _filter_writer_contents(llm_request: Any) -> None:
    """Hide Grounding/bootstrap transport while preserving SQL execution history."""

    contents = getattr(llm_request, "contents", None)
    if not isinstance(contents, list):
        raise TypeError("LlmRequest.contents is required")
    retained_contents: list[Any] = []
    for content in contents:
        parts = getattr(content, "parts", None)
        if not isinstance(parts, list):
            retained_contents.append(copy.deepcopy(content))
            continue
        retained_parts: list[Any] = []
        for part in parts:
            function_call = getattr(part, "function_call", None)
            function_response = getattr(part, "function_response", None)
            call_name = getattr(function_call, "name", None)
            response_name = getattr(function_response, "name", None)
            if call_name is not None and call_name not in _SQL_WRITER_TOOL_NAMES:
                continue
            if response_name is not None and response_name not in _SQL_WRITER_TOOL_NAMES:
                continue
            retained_parts.append(copy.deepcopy(part))
        if retained_parts:
            cloned = copy.deepcopy(content)
            cloned.parts = retained_parts
            retained_contents.append(cloned)
    llm_request.contents = retained_contents


def _inject_sql_writer_context(
    llm_request: Any,
    *,
    phase: Literal[1, 2],
    original_query: str,
    follow_up: str | None,
    view_text: str,
    budget_remaining: Any,
) -> dict[str, str]:
    """Atomically replace exploratory instructions with the SQL Writer view."""

    if not isinstance(original_query, str) or not original_query:
        raise ValueError("SQL Writer requires Original Query")
    budget = float(budget_remaining)
    if not math.isfinite(budget):
        raise ValueError("SQL Writer requires finite Bird-Coin")
    if phase == 2 and (not isinstance(follow_up, str) or not follow_up):
        raise ValueError("Phase 2 SQL Writer requires follow-up")
    context = "\n".join(
        [
            SQL_WRITER_CONTEXT_BEGIN,
            f"Phase: {phase}",
            f"Original Query: {original_query}",
            f"Follow-up: {follow_up if follow_up is not None else 'none'}",
            "Final Grounding State and answered Clarifications:",
            view_text,
            f"Remaining Bird-Coin: {budget:g}",
            SQL_WRITER_CONTEXT_END,
        ]
    )
    original = copy.deepcopy(llm_request)
    original_tools_dict = copy.deepcopy(getattr(llm_request, "tools_dict", None))
    try:
        config = getattr(llm_request, "config", None)
        if config is None:
            raise TypeError("LlmRequest.config is required")
        config.system_instruction = f"{_SQL_WRITER_PROMPT}\n\n{context}"
        _filter_writer_tools(llm_request)
        _filter_writer_contents(llm_request)
        final_names = tuple(getattr(llm_request, "tools_dict", {}).keys())
        if final_names != _SQL_WRITER_TOOL_NAMES:
            raise RuntimeError("SQL Writer exposed an unexpected tool")
        return {
            "grounding_view_block_sha256": _sha256_text(context),
            "writer_prompt_sha256": _sha256_text(_SQL_WRITER_PROMPT),
        }
    except BaseException:
        _restore_llm_request(llm_request, original)
        llm_request.tools_dict = original_tools_dict
        raise


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
