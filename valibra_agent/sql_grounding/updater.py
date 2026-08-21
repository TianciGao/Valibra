"""Strict SQL Grounding updater and opt-in LiteLLM Provider adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator

from shared.audit import to_jsonable
from shared.model_presets import load_model_preset

from valibra_agent.sql_grounding.models import (
    ContractModel,
    GroundingLLMResponse,
    GroundingRuntime,
    UserClarificationRecord,
    canonical_json,
)
from valibra_agent.sql_grounding.observations import (
    SQLGroundingObservation,
    observation_for_updater,
)
from valibra_agent.sql_grounding.telemetry import (
    GroundingLLMTelemetry,
    GroundingTokenUsage,
)

MAX_GROUNDING_REQUEST_CHARS = 262_144
MAX_GROUNDING_RESPONSE_CHARS = 65_536
DEFAULT_GROUNDING_TIMEOUT_SECONDS = 600.0
DEFAULT_GROUNDING_MAX_CALLS_PER_TASK = 8
GroundingCallKind: TypeAlias = Literal[
    "structure",
    "mapping",
    "knowledge",
    "clarification_patch",
]
GROUNDING_LLM_ENV_NAMES = (
    "GROUNDING_UPDATER_MODE",
    "GROUNDING_MODEL_PRESET",
    "GROUNDING_TIMEOUT_SECONDS",
    "GROUNDING_MAX_TOKENS",
    "GROUNDING_MAX_CALLS_PER_TASK",
    "GROUNDING_PROMPT_SHA256",
)
_SENSITIVE_AUDIT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)
_BEARER_MARKER_RE = re.compile(r"Authorization\s*:\s*Bearer\b", re.IGNORECASE)
_BEARER_TOKEN_RE = re.compile(
    r"Authorization\s*:\s*Bearer\s+([^\s\"'\\]+)(?=$|\s|[\"'])",
    re.IGNORECASE,
)
_RAW_TOKEN_RE = re.compile(r"^[^\s\"'\\=:\{\}\[\]]+$")
_OPAQUE_RAW_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9._-]{16,}$")
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)

SQL_GROUNDING_PROMPT = """请根据有界 JSON 输入填写固定的 Valibra SQL Grounding 表单。
只返回一个 JSON 对象。不要返回说明文字、注释、调用方允许的单个传输代码围栏之外的
Markdown，也不要添加额外字段。

输出字段只能是：
- sql_grounding_state：tables、join_keys、column_mapping、domain_knowledge
- user_clarification_requests
- next_focus_dimension

精确输出结构：
- sql_grounding_state 中只能包含：
  - tables：null 或由表名字符串组成的数组
  - join_keys：null 或由关联表达式字符串组成的数组
  - column_mapping：null 或由对象组成的数组；每个对象只能包含：
    - phrase：一个字符串
    - targets：由一个或多个字符串组成的数组
  - domain_knowledge：null 或由对象组成的数组；每个对象只能包含：
    - kind：business_rule、runtime_state 或 database_capability
    - content：一个字符串
- user_clarification_requests：由零个或多个对象组成的数组；每个对象只能包含：
  - phrase：当前 phase 用户 query / follow_up 中逐字连续的非空片段
  - kind：user_intent 或 missing_knowledge
  - question：只向用户确认意图或用户掌握但当前缺失的业务知识
- next_focus_dimension 只能是 tables、join_keys、column_mapping、
  domain_knowledge 或 none。

重要结构规则：
- 字段名必须是复数形式 "targets"。
- 即使只有一个元素，"targets" 也必须是 JSON 数组。
- 不存在名为 "target" 的字段。
- 不要添加以上未列出的任何字段。

分阶段持久化四维状态的规则：
1. 调度固定为 Structure → Mapping → Knowledge；不能根据 focus 跳步，也不能等待
   execute_sql 或 submit_sql 后重新 Grounding。每个请求都只包含 query、current_state、
   当前阶段 evidence；Phase 2 另外包含 follow_up 和 Phase 1 已回答的
   user_clarifications。不得要求或假装看见前一阶段的完整原始 request。
2. Structure Grounding 的阶段 evidence 只有 schema。主要更新 tables、join_keys；响应必须让
   tables、join_keys 都成为数组，column_mapping、domain_knowledge 保持原值（Phase 1 初始时
   仍为 null），user_clarification_requests 必须为空，next_focus_dimension 必须为
   column_mapping。优先保证候选表 recall，但不得使用 schema 不支持的名字。
3. Mapping Grounding 的阶段 evidence 只有候选表相关的 column_meanings。主要更新
   column_mapping；只有 metadata 明确证明前一轮有误时，才能小范围修正 tables、join_keys。
   响应必须让 tables、join_keys、column_mapping 都成为数组，domain_knowledge 保持原值，
   user_clarification_requests 必须为空，next_focus_dimension 必须为 domain_knowledge。
4. Knowledge Grounding 的阶段 evidence 只有 knowledge_definitions 和当前 mapping 相关的
   relevant_column_meanings。主要更新 domain_knowledge，必要时可定向修正 column_mapping；
   tables、join_keys 必须保持不变。响应必须把四维都评估为数组，next_focus_dimension 必须
   为 none。只有这一阶段可以产生 user_clarification_requests。
5. 如果同一 phase 有澄清问题，调用方会先收集全部回答，再至多调用一次
   Clarification Patch。该请求只含本 phase clarification_qa 及受影响 phrase 的
   relevant_column_meanings / relevant_knowledge_definitions。它只能修改受回答影响的
   column_mapping / domain_knowledge；不得修改 tables、join_keys 或无关 mapping；不得再次
   产生澄清问题，next_focus_dimension 必须为 none。
6. P2 从 P1 最终 State 增量开始并复用 task-level bootstrap evidence，不从空 State 重建，
   也不重新调用三个 bootstrap 工具。每 phase 无澄清时恰好三次 Provider 调用，有澄清时
   最多四次；整个 task 最多八次。任何阶段技术或合同失败都 fail-closed，不 retry、不 fallback。
7. tables 只能使用 schema 支持的真实、完全限定的数据库标识符。join_keys 中的表和字段也必须
   得到 schema 支持。绝不能持久化简写别名或臆造名称。
8. column_mapping 中的普通字段必须得到 schema 支持；JSON / JSONB target 的真实列必须来自
   schema，其路径 key 必须来自 column_meanings 中该列的 fields_meaning。
9. domain_knowledge.content 必须逐字复制自 knowledge_definitions 中的 Official BIRD
   definition，或来自当前 missing_knowledge 澄清回答的逐字内容。不能从 schema、
   column_meanings 或模型常识中改写或臆造知识。只有名字相近、但公式或业务定义不同的邻近
   knowledge 不能代替精确规则；完成问题依赖缺失或 masked knowledge 时，应提出澄清而不是猜。
10. 对非分阶段的兼容离线调用，只能使用 latest_observation 已经支持的真实、完全限定数据库
   标识符；该兼容入口不会扩大任何维度授权。
11. join_keys 和 column_mapping 的 targets 必须是符合 Valibra 受限字段/关联表达式合同的
   规范化 PostgreSQL 表达式。它们不是自由 SQL，不能包含语句、注释、任意函数或未经
   批准的 AST 结构。
   输出的每个 join_keys 表达式和 column_mapping target，都必须已经采用 sqlglot 26.16.4
   按 PostgreSQL 解析后、通过 expression.sql(dialect="postgres") 渲染所得的精确词法
   形式。不要输出仅空格或格式不同的等价表达式。对于 PostgreSQL JSON / JSONB 运算符，
   -> 和 ->> 的两侧都必须各有一个 ASCII 空格。仅用于展示语法的示例：
   t.c -> 'key' ->> 'leaf'。这个示例只展示格式；除非当前合法观测支持其中的
   标识符或字面量，否则绝不能复制它们。
12. 每个 column_mapping.phrase 都必须是原始用户问题 query / original_query 或追问 follow_up 中
   非空、逐字连续的子串。
   不要改写 phrase。
13. domain_knowledge.kind 只能是 business_rule、runtime_state 或 database_capability。
   每个新增或修改后的非空 domain_knowledge.content，都必须逐字复制自 knowledge_definitions
   或最新合法观测中的 Official BIRD 规范知识陈述。已有且通过验证的 content 只能原样保留。
   绝不能改写或臆造知识。
14. P2 与 Clarification Patch 必须保留所有未受 follow_up / 当前澄清回答影响的现有维度和条目。
15. null 表示尚未评估；[] 表示已经评估且不需要；非空数组包含通过验证的结果。
16. user_clarification_requests=[] 表示不需要用户确认。只有当前 query / follow_up 中确实存在
   无法安全消解的 user_intent，或完成任务必须依赖但用户可能掌握的 missing_knowledge，才可以
   提出问题。不得询问 schema、表名、列名、join、SQL 写法、identifier 或 runtime error；这些
   必须由 Official BIRD 工具证据和 Main Agent 自行处理。不得为了保险而要求用户确认数据库事实。

不要输出 score、confidence、ambiguity、reasoning、Evidence 摘要、SQL plan、Bird-Coin、
final SQL、tool call 或具体 tool choice。响应只能提出当前阶段允许的四维状态、有限澄清请求和
下一个聚焦维度。
"""

SQL_GROUNDING_PROMPT_SHA256 = hashlib.sha256(
    SQL_GROUNDING_PROMPT.encode("utf-8")
).hexdigest()
SQL_GROUNDING_FORM_SCHEMA = GroundingLLMResponse.model_json_schema()
SQL_GROUNDING_FORM_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json(SQL_GROUNDING_FORM_SCHEMA).encode("utf-8")
).hexdigest()
SQL_GROUNDING_CONFIGURATION = {
    "form_schema_sha256": SQL_GROUNDING_FORM_SCHEMA_SHA256,
    "max_calls_per_task": DEFAULT_GROUNDING_MAX_CALLS_PER_TASK,
    "max_request_chars": MAX_GROUNDING_REQUEST_CHARS,
    "max_response_chars": MAX_GROUNDING_RESPONSE_CHARS,
    "prompt_sha256": SQL_GROUNDING_PROMPT_SHA256,
    "timeout_seconds": DEFAULT_GROUNDING_TIMEOUT_SECONDS,
    "transport": "bare_json_or_single_json_fence",
}
SQL_GROUNDING_CONFIGURATION_SHA256 = hashlib.sha256(
    canonical_json(SQL_GROUNDING_CONFIGURATION).encode("utf-8")
).hexdigest()

_SINGLE_JSON_FENCE_RE = re.compile(
    r"\A```(?P<label>[A-Za-z]*)\r?\n(?P<body>[\s\S]*?)\r?\n```\Z"
)


class SQLGroundingLLMConfig(ContractModel):
    """Frozen non-secret model configuration for one SQL Grounding call."""

    mode: Literal["llm"] = "llm"
    model_preset: str = Field(min_length=1, max_length=128)
    preset_config: dict[str, Any]
    preset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeout_seconds: float = Field(gt=0.0, le=600.0)
    max_tokens: int = Field(ge=1, le=131_072)
    max_calls_per_task: int = Field(ge=1, le=128)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    form_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kernel_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_frozen_contract(self) -> "SQLGroundingLLMConfig":
        if self.timeout_seconds != DEFAULT_GROUNDING_TIMEOUT_SECONDS:
            raise ValueError("GROUNDING_TIMEOUT_SECONDS must equal 600 seconds")
        if self.prompt_sha256 != SQL_GROUNDING_PROMPT_SHA256:
            raise ValueError("GROUNDING_PROMPT_SHA256 mismatch")
        if self.form_schema_sha256 != SQL_GROUNDING_FORM_SCHEMA_SHA256:
            raise ValueError("SQL Grounding form schema SHA mismatch")
        if self.kernel_configuration_sha256 != SQL_GROUNDING_CONFIGURATION_SHA256:
            raise ValueError("SQL Grounding kernel configuration SHA mismatch")
        if self.preset_config.get("max_tokens") != self.max_tokens:
            raise ValueError("GROUNDING_MAX_TOKENS must equal the frozen preset value")
        if self.max_calls_per_task != DEFAULT_GROUNDING_MAX_CALLS_PER_TASK:
            raise ValueError("GROUNDING_MAX_CALLS_PER_TASK must equal 8")
        return self


class SQLGroundingProviderConfig(ContractModel):
    """Validated SQL Grounding connection settings without credential contents."""

    api_base: str = Field(min_length=1, max_length=2048)
    model_id: str = Field(min_length=1, max_length=256)
    credential_source: Literal["direct", "file"]
    api_key_file: str | None = Field(default=None, max_length=2048)
    use_bearer_for_custom_base: bool = False
    retry_count: Literal[0] = 0
    raw_audit_dir: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_connection(self) -> "SQLGroundingProviderConfig":
        parsed = urlsplit(self.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GROUNDING_API_BASE must be an absolute HTTP(S) URL")
        if self.credential_source == "file" and not self.api_key_file:
            raise ValueError("file credential source requires api_key_file")
        if self.credential_source == "direct" and self.api_key_file is not None:
            raise ValueError("direct credential source cannot retain api_key_file")
        return self


class GroundingLLMRequest(ContractModel):
    """The complete request passed to an injected offline client."""

    prompt: str = Field(min_length=1, max_length=32_768)
    input_json: str = Field(min_length=1, max_length=MAX_GROUNDING_REQUEST_CHARS)
    response_schema: dict[str, Any]
    observation_id: str = Field(default="", max_length=256)
    observation_type: str = Field(default="", max_length=64)


class GroundingClientResponse(ContractModel):
    """One Provider result with bounded non-secret audit metadata."""

    content: str
    usage: GroundingTokenUsage = Field(default_factory=GroundingTokenUsage)
    provider_reported_cost: float | None = Field(default=None, ge=0.0)
    model: str = Field(default="", max_length=256)
    provider: str = Field(default="", max_length=128)
    credential_source: Literal["", "direct", "file"] = ""
    request_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    response_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    raw_private_audit_ref: str = Field(default="", max_length=1024)
    provider_may_continue_after_cancel: bool | None = None
    provider_may_bill_after_cancel: bool | None = None


class AsyncGroundingClient(Protocol):
    async def complete(self, request: GroundingLLMRequest) -> GroundingClientResponse:
        """Return one fixed response without tools or retry behavior."""


class SQLGroundingProviderError(RuntimeError):
    """Sanitized real Provider failure; raw exception text is never retained."""

    def __init__(
        self,
        reason: str,
        *,
        request_sha256: str = "",
        raw_private_audit_ref: str = "",
        attempted: bool = False,
    ) -> None:
        super().__init__(reason)
        self.request_sha256 = request_sha256
        self.raw_private_audit_ref = raw_private_audit_ref
        self.attempted = attempted


class LiteLLMSQLGroundingClient:
    """Native async LiteLLM adapter with one request and private raw audit."""

    provider_may_continue_after_cancel = True
    provider_may_bill_after_cancel = True

    def __init__(
        self,
        llm_config: SQLGroundingLLMConfig,
        provider_config: SQLGroundingProviderConfig,
        *,
        environment: Mapping[str, str] | None = None,
        completion: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.llm_config = SQLGroundingLLMConfig.model_validate(llm_config)
        self.config = SQLGroundingProviderConfig.model_validate(provider_config)
        self._environment = environment if environment is not None else os.environ
        self._completion = completion
        self._last_request_sha256 = ""
        self._last_private_audit_ref = ""
        if self.llm_config.preset_config.get("model") != self.config.model_id:
            raise ValueError("Grounding model preset and Provider model differ")

    async def complete(self, request: GroundingLLMRequest) -> GroundingClientResponse:
        api_key = _read_grounding_api_key(self.config, self._environment)
        request_audit, provider_kwargs = _build_provider_request(
            request,
            self.llm_config,
            self.config,
            api_key=api_key,
        )
        request_sha = _stable_json_sha256(request_audit)
        audit_path = _new_private_audit_path(self.config, request_sha)
        audit_ref = _private_audit_ref(self.config, audit_path)
        self._last_request_sha256 = request_sha
        self._last_private_audit_ref = audit_ref
        completion = self._completion
        if completion is None:
            try:
                import litellm
            except Exception:
                raise SQLGroundingProviderError(
                    "SQL Grounding Provider client unavailable",
                    request_sha256=request_sha,
                    raw_private_audit_ref=audit_ref,
                    attempted=False,
                ) from None
            completion = litellm.acompletion
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        observation_id, observation_type = _request_observation_identity(request)
        initial_audit = {
            "schema_version": "1.0",
            "status": "started",
            "started_at": started_at,
            "request": request_audit,
            "request_sha256": request_sha,
            "prompt_sha256": SQL_GROUNDING_PROMPT_SHA256,
            "form_schema_sha256": SQL_GROUNDING_FORM_SCHEMA_SHA256,
            "configuration_sha256": SQL_GROUNDING_CONFIGURATION_SHA256,
            "model": self.config.model_id,
            "provider": _provider_endpoint_identity(self.config.api_base),
            "credential_source": self.config.credential_source,
            "observation_id": observation_id,
            "observation_type": observation_type,
        }
        try:
            _write_private_provider_audit(
                audit_path,
                initial_audit,
                api_key=api_key,
            )
        except SQLGroundingProviderError:
            raise
        except Exception:
            raise SQLGroundingProviderError(
                "SQL Grounding private request audit preflight failed",
                request_sha256=request_sha,
                raw_private_audit_ref=audit_ref,
                attempted=False,
            ) from None
        try:
            response = await completion(**provider_kwargs)
        except asyncio.CancelledError:
            _write_private_provider_audit(
                audit_path,
                {
                    **initial_audit,
                    "status": "timed_out",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": _elapsed_ms(started),
                    "provider_may_continue_after_cancel": (
                        self.provider_may_continue_after_cancel
                    ),
                    "provider_may_bill_after_cancel": (
                        self.provider_may_bill_after_cancel
                    ),
                },
                api_key=api_key,
            )
            raise
        except Exception as exc:
            _write_private_provider_audit(
                audit_path,
                {
                    **initial_audit,
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": _elapsed_ms(started),
                    "error_type": type(exc).__name__[:128],
                },
                api_key=api_key,
            )
            raise SQLGroundingProviderError(
                f"SQL Grounding Provider request failed ({type(exc).__name__[:128]})",
                request_sha256=request_sha,
                raw_private_audit_ref=audit_ref,
                attempted=True,
            ) from None

        raw_response = _sanitize_audit_value(to_jsonable(response))
        response_sha = _stable_json_sha256(raw_response)
        usage, cost = _provider_usage(response)
        _write_private_provider_audit(
            audit_path,
            {
                **initial_audit,
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "latency_ms": _elapsed_ms(started),
                "response": raw_response,
                "response_sha256": response_sha,
                "usage": usage.model_dump(mode="json"),
                "provider_reported_cost": cost,
            },
            api_key=api_key,
        )
        try:
            content = _provider_response_content(response)
        except SQLGroundingProviderError as exc:
            raise SQLGroundingProviderError(
                str(exc),
                request_sha256=request_sha,
                raw_private_audit_ref=audit_ref,
                attempted=True,
            ) from None
        return GroundingClientResponse(
            content=content,
            usage=usage,
            provider_reported_cost=cost,
            model=self.config.model_id,
            provider=_provider_name(response),
            credential_source=self.config.credential_source,
            request_sha256=request_sha,
            response_sha256=response_sha,
            raw_private_audit_ref=audit_ref,
            provider_may_continue_after_cancel=self.provider_may_continue_after_cancel,
            provider_may_bill_after_cancel=self.provider_may_bill_after_cancel,
        )


class GroundingUpdaterResult(ContractModel):
    response: GroundingLLMResponse
    telemetry: GroundingLLMTelemetry
    transport_normalization: Literal["none", "single_json_fence"]


class GroundingUpdaterError(RuntimeError):
    """A bounded updater failure with telemetry but no raw response body."""

    def __init__(self, reason: str, telemetry: GroundingLLMTelemetry) -> None:
        super().__init__(reason)
        self.reason = reason
        self.telemetry = telemetry


class SQLGroundingUpdater:
    """Build one bounded request and strictly parse one injected-client response."""

    def __init__(
        self,
        client: AsyncGroundingClient,
    ) -> None:
        self._client = client

    async def propose(
        self,
        runtime: GroundingRuntime,
        observation: SQLGroundingObservation,
        *,
        original_query: str,
        follow_up_query: str | None = None,
        grounding_input: Mapping[str, Any] | None = None,
    ) -> GroundingUpdaterResult:
        """Return a typed proposal without mutating Runtime or Observation."""

        request = _build_request(
            runtime,
            observation,
            original_query=original_query,
            follow_up_query=follow_up_query,
            grounding_input=grounding_input,
        )
        request_sha = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
        started = time.perf_counter()
        try:
            client_response = await asyncio.wait_for(
                self._client.complete(request),
                timeout=DEFAULT_GROUNDING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            metadata = _client_telemetry_metadata(self._client)
            telemetry = _telemetry(
                attempted=True,
                status="timed_out",
                latency_ms=_elapsed_ms(started),
                request_sha=(
                    str(getattr(self._client, "_last_request_sha256", ""))
                    or request_sha
                ),
                error_type="timeout",
                **metadata,
            )
            raise GroundingUpdaterError("timeout", telemetry) from exc
        except SQLGroundingProviderError as exc:
            metadata = _client_telemetry_metadata(self._client)
            metadata["raw_private_audit_ref"] = exc.raw_private_audit_ref
            telemetry = _telemetry(
                attempted=exc.attempted,
                status="failed" if exc.attempted else "rejected",
                latency_ms=_elapsed_ms(started),
                request_sha=exc.request_sha256 or request_sha,
                error_type=(
                    "provider_error"
                    if exc.attempted
                    else "provider_preflight_error"
                ),
                **metadata,
            )
            raise GroundingUpdaterError(
                telemetry.error_type or "provider_error",
                telemetry,
            ) from exc
        except Exception as exc:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                latency_ms=_elapsed_ms(started),
                request_sha=request_sha,
                error_type="client_error",
                **_client_telemetry_metadata(self._client),
            )
            raise GroundingUpdaterError("client_error", telemetry) from exc

        response_sha = client_response.response_sha256 or hashlib.sha256(
            client_response.content.encode("utf-8")
        ).hexdigest()
        effective_request_sha = client_response.request_sha256 or request_sha
        provider_metadata = _response_telemetry_metadata(client_response)
        if len(client_response.content) > MAX_GROUNDING_RESPONSE_CHARS:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                usage=client_response.usage,
                latency_ms=_elapsed_ms(started),
                request_sha=effective_request_sha,
                response_sha=response_sha,
                error_type="response_too_large",
                **provider_metadata,
            )
            raise GroundingUpdaterError("response_too_large", telemetry)
        try:
            payload, normalization = normalize_grounding_transport(
                client_response.content
            )
            response = parse_grounding_response(payload)
        except _StrictResponseError as exc:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                usage=client_response.usage,
                latency_ms=_elapsed_ms(started),
                request_sha=effective_request_sha,
                response_sha=response_sha,
                error_type=exc.reason,
                **provider_metadata,
            )
            raise GroundingUpdaterError(exc.reason, telemetry) from exc

        telemetry = _telemetry(
            attempted=True,
            status="succeeded",
            usage=client_response.usage,
            latency_ms=_elapsed_ms(started),
            request_sha=effective_request_sha,
            response_sha=response_sha,
            **provider_metadata,
        )
        return GroundingUpdaterResult(
            response=response,
            telemetry=telemetry,
            transport_normalization=normalization,
        )


class _StrictResponseError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _build_request(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    *,
    original_query: str,
    follow_up_query: str | None,
    grounding_input: Mapping[str, Any] | None,
) -> GroundingLLMRequest:
    if grounding_input is None:
        input_payload = {
            "current_sql_grounding_state": runtime.grounding_state.model_dump(mode="json"),
            "current_stage": runtime.stage,
            "follow_up": follow_up_query,
            "latest_observation": observation_for_updater(observation),
            "original_query": original_query,
        }
    else:
        input_payload = _validated_bundled_grounding_input(
            grounding_input,
            runtime=runtime,
            original_query=original_query,
            follow_up_query=follow_up_query,
            observation=observation,
        )
    input_json = canonical_json(input_payload)
    if len(input_json) > MAX_GROUNDING_REQUEST_CHARS:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="request_too_large",
        )
        raise GroundingUpdaterError("request_too_large", telemetry)
    request = GroundingLLMRequest(
        prompt=SQL_GROUNDING_PROMPT,
        input_json=input_json,
        response_schema=SQL_GROUNDING_FORM_SCHEMA,
        observation_id=observation.observation_id,
        observation_type=observation.observation_type,
    )
    if len(canonical_json(request)) > MAX_GROUNDING_REQUEST_CHARS:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="request_too_large",
        )
        raise GroundingUpdaterError("request_too_large", telemetry)
    return request


def _validated_bundled_grounding_input(
    grounding_input: Mapping[str, Any],
    *,
    runtime: GroundingRuntime,
    original_query: str,
    follow_up_query: str | None,
    observation: SQLGroundingObservation,
) -> dict[str, Any]:
    """Copy and validate one exact, stage-local Grounding bundle."""

    try:
        call_kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
    except (TypeError, ValueError) as exc:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry) from exc
    payload = dict(grounding_input)
    if payload["query"] != original_query or payload["query"] != original_query.strip():
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if payload["current_state"] != runtime.grounding_state.model_dump(mode="json"):
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if observation.phase == 2:
        if (
            not isinstance(follow_up_query, str)
            or not follow_up_query
            or payload["follow_up"] != follow_up_query
            or follow_up_query != follow_up_query.strip()
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        clarifications = payload["user_clarifications"]
        if not isinstance(clarifications, list):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        try:
            records = tuple(
                UserClarificationRecord.model_validate(item)
                for item in clarifications
            )
        except ValidationError as exc:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError(
                "grounding_bundle_invalid", telemetry
            ) from exc
        if any(item.phase != 1 or item.answer is None for item in records):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        questions = [item.question for item in records]
        if len(questions) != len(set(questions)):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if call_kind == "clarification_patch":
        clarification_qa = payload["clarification_qa"]
        if not isinstance(clarification_qa, list) or not clarification_qa:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        try:
            records = tuple(
                UserClarificationRecord.model_validate(item)
                for item in clarification_qa
            )
        except ValidationError as exc:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry) from exc
        if any(
            item.phase != observation.phase or item.answer is None
            for item in records
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        questions = [item.question for item in records]
        if len(questions) != len(set(questions)):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    try:
        canonical_json(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry) from exc
    return payload


_CALL_KIND_EVIDENCE_FIELDS: Mapping[GroundingCallKind, frozenset[str]] = {
    "structure": frozenset({"schema"}),
    "mapping": frozenset({"column_meanings"}),
    "knowledge": frozenset(
        {"knowledge_definitions", "relevant_column_meanings"}
    ),
    "clarification_patch": frozenset(
        {
            "clarification_qa",
            "relevant_column_meanings",
            "relevant_knowledge_definitions",
        }
    ),
}


def classify_grounding_input(
    grounding_input: Mapping[str, Any],
    *,
    phase: Literal[1, 2],
) -> GroundingCallKind:
    """Classify one 1.2 request only by its exact finite field set."""

    if not isinstance(grounding_input, Mapping):
        raise TypeError("Grounding input must be a mapping")
    common = {"query", "current_state"}
    if phase == 2:
        common.update({"follow_up", "user_clarifications"})
    elif phase != 1:
        raise ValueError("Grounding phase must be 1 or 2")
    fields = set(grounding_input)
    matches = [
        call_kind
        for call_kind, evidence_fields in _CALL_KIND_EVIDENCE_FIELDS.items()
        if fields == common | set(evidence_fields)
    ]
    if len(matches) != 1:
        raise ValueError("Grounding input has an invalid stage-local field set")
    return matches[0]


def normalize_grounding_transport(
    content: str,
) -> tuple[str, Literal["none", "single_json_fence"]]:
    """Strip only one complete JSON/unlabelled Markdown transport fence."""

    if not isinstance(content, str):
        raise _StrictResponseError("transport_format_invalid")
    stripped = content.strip()
    if "```" not in stripped:
        return content, "none"
    if not stripped.startswith("```"):
        if re.search(r"(?m)^[ \t]*```", stripped) or _contains_unquoted_fence(stripped):
            raise _StrictResponseError("transport_format_invalid")
        return content, "none"
    match = _SINGLE_JSON_FENCE_RE.fullmatch(stripped)
    fence_lines = re.findall(r"(?m)^[ \t]*```", stripped)
    if match is None or len(fence_lines) != 2:
        raise _StrictResponseError("transport_format_invalid")
    label = match.group("label")
    if label and label.lower() != "json":
        raise _StrictResponseError("transport_format_invalid")
    return match.group("body"), "single_json_fence"


def parse_grounding_response(payload: str) -> GroundingLLMResponse:
    """Run strict JSON, duplicate-key, finite-value, and Pydantic validation."""

    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except _StrictResponseError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise _StrictResponseError("json_invalid") from exc
    if not isinstance(raw, dict):
        raise _StrictResponseError("form_validation_failed")
    try:
        return GroundingLLMResponse.model_validate(raw)
    except ValidationError as exc:
        raise _StrictResponseError("form_validation_failed") from exc


def _contains_unquoted_fence(text: str) -> bool:
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif text.startswith("```", index):
            return True
        index += 1
    return False


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictResponseError("duplicate_json_key")
        result[key] = value
    return result


def _reject_non_finite_constant(_: str) -> None:
    raise ValueError("non-finite JSON constants are forbidden")


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def requested_sql_grounding_updater_mode(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the exact requested mode; empty is the SG3 passthrough default."""

    source = environment if environment is not None else os.environ
    return str(source.get("GROUNDING_UPDATER_MODE", "")).strip()


def load_sql_grounding_llm_config(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
) -> SQLGroundingLLMConfig:
    """Read only explicit ``GROUNDING_*`` model settings without activation."""

    source = environment if environment is not None else os.environ
    missing = [name for name in GROUNDING_LLM_ENV_NAMES if not source.get(name)]
    if missing:
        raise ValueError(f"missing frozen SQL Grounding configuration: {missing}")
    if str(source["GROUNDING_UPDATER_MODE"]).strip() != "llm":
        raise ValueError("GROUNDING_UPDATER_MODE must be exactly 'llm'")
    if str(source["GROUNDING_PROMPT_SHA256"]).strip() != SQL_GROUNDING_PROMPT_SHA256:
        raise ValueError("GROUNDING_PROMPT_SHA256 does not match SQL Grounding prompt")
    preset = load_model_preset(
        project_root,
        str(source["GROUNDING_MODEL_PRESET"]).strip(),
    )
    timeout = _parse_positive_float(
        str(source["GROUNDING_TIMEOUT_SECONDS"]),
        "GROUNDING_TIMEOUT_SECONDS",
    )
    max_tokens = _parse_positive_int(
        str(source["GROUNDING_MAX_TOKENS"]),
        "GROUNDING_MAX_TOKENS",
    )
    max_calls = _parse_positive_int(
        str(source["GROUNDING_MAX_CALLS_PER_TASK"]),
        "GROUNDING_MAX_CALLS_PER_TASK",
    )
    return SQLGroundingLLMConfig(
        model_preset=preset.name,
        preset_config=preset.normalized_config,
        preset_sha256=preset.normalized_sha256,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        max_calls_per_task=max_calls,
        prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
        form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
        kernel_configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


def load_sql_grounding_provider_config(
    project_root: Path,
    llm_config: SQLGroundingLLMConfig,
    environment: Mapping[str, str] | None = None,
) -> SQLGroundingProviderConfig:
    """Validate an independent Provider endpoint and exactly one credential source."""

    source = environment if environment is not None else os.environ
    api_base = str(source.get("GROUNDING_API_BASE", "")).strip()
    direct_present = bool(str(source.get("GROUNDING_API_KEY", "")).strip())
    key_file_raw = str(source.get("GROUNDING_API_KEY_FILE", "")).strip()
    if not api_base:
        raise ValueError("missing Grounding Provider variable: GROUNDING_API_BASE")
    if direct_present == bool(key_file_raw):
        raise ValueError(
            "configure exactly one of GROUNDING_API_KEY or GROUNDING_API_KEY_FILE"
        )
    credential_source: Literal["direct", "file"]
    resolved_key_file: str | None = None
    if key_file_raw:
        credential_source = "file"
        key_path = Path(key_file_raw).expanduser()
        if not key_path.is_absolute():
            key_path = project_root / key_path
        key_path = key_path.resolve()
        try:
            if not key_path.is_file() or key_path.stat().st_size <= 0:
                raise ValueError("GROUNDING_API_KEY_FILE must be a non-empty file")
            with key_path.open("rb") as handle:
                if not handle.readable():
                    raise ValueError("GROUNDING_API_KEY_FILE must be readable")
        except OSError as exc:
            raise ValueError("GROUNDING_API_KEY_FILE is not readable") from exc
        resolved_key_file = str(key_path)
    else:
        credential_source = "direct"
    use_bearer = _parse_bool(
        str(source.get("GROUNDING_USE_BEARER_FOR_CUSTOM_BASE", "false")),
        "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE",
    )
    model_id = str(llm_config.preset_config.get("model", "")).strip()
    if not model_id:
        raise ValueError("Grounding model preset has no model ID")
    return SQLGroundingProviderConfig(
        api_base=api_base,
        model_id=model_id,
        credential_source=credential_source,
        api_key_file=resolved_key_file,
        use_bearer_for_custom_base=use_bearer,
        retry_count=0,
        raw_audit_dir=str(
            (project_root / "research-runtime" / "sql-grounding-llm").resolve()
        ),
    )


def build_real_sql_grounding_updater(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
    *,
    completion: Callable[..., Awaitable[Any]] | None = None,
) -> SQLGroundingUpdater:
    """Build the opt-in real updater after both non-secret configs validate."""

    llm_config = load_sql_grounding_llm_config(project_root, environment)
    provider_config = load_sql_grounding_provider_config(
        project_root,
        llm_config,
        environment,
    )
    return SQLGroundingUpdater(
        LiteLLMSQLGroundingClient(
            llm_config,
            provider_config,
            environment=environment,
            completion=completion,
        )
    )


def sql_grounding_provider_health_report(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a bounded secret-free requested/effective-mode report."""

    requested = requested_sql_grounding_updater_mode(environment)
    if requested == "":
        return {
            "requested_updater_mode": "passthrough",
            "effective_updater_mode": "passthrough",
            "provider_configuration_valid": True,
            "provider_enabled": False,
            "configuration_error_type": None,
        }
    if requested != "llm":
        return {
            "requested_updater_mode": requested[:64],
            "effective_updater_mode": "invalid",
            "provider_configuration_valid": False,
            "provider_enabled": False,
            "configuration_error_type": "invalid_updater_mode",
        }
    try:
        llm_config = load_sql_grounding_llm_config(project_root, environment)
        load_sql_grounding_provider_config(project_root, llm_config, environment)
    except Exception as exc:
        return {
            "requested_updater_mode": "llm",
            "effective_updater_mode": "invalid",
            "provider_configuration_valid": False,
            "provider_enabled": False,
            "configuration_error_type": type(exc).__name__[:128],
        }
    return {
        "requested_updater_mode": "llm",
        "effective_updater_mode": "llm",
        "provider_configuration_valid": True,
        "provider_enabled": True,
        "configuration_error_type": None,
    }


def _parse_positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_positive_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _parse_bool(raw: str, name: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _read_grounding_api_key(
    config: SQLGroundingProviderConfig,
    environment: Mapping[str, str],
) -> str:
    if config.credential_source == "direct":
        value = str(environment.get("GROUNDING_API_KEY", "")).strip()
        if not value:
            raise SQLGroundingProviderError("Grounding direct credential unavailable")
        return value
    try:
        content = Path(config.api_key_file or "").read_text(encoding="utf-8")
    except OSError:
        raise SQLGroundingProviderError("Grounding credential file unavailable") from None
    return _parse_grounding_credential_file(content)


def _parse_grounding_credential_file(content: str) -> str:
    """Accept one raw token or exactly one Bearer header, and nothing ambiguous."""

    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(nonempty_lines) == 1 and _RAW_TOKEN_RE.fullmatch(nonempty_lines[0]):
        return nonempty_lines[0]

    stripped = content.strip()
    if (
        not stripped
        or _PRIVATE_KEY_BLOCK_RE.search(content)
        or (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        raise SQLGroundingProviderError("Grounding credential file format invalid")

    markers = tuple(_BEARER_MARKER_RE.finditer(content))
    matches = tuple(_BEARER_TOKEN_RE.finditer(content))
    if len(markers) != 1 or len(matches) != 1:
        raise SQLGroundingProviderError("Grounding credential file format invalid")

    bearer_line_index = content[: matches[0].start()].count("\n")
    for index, raw_line in enumerate(content.splitlines()):
        if index == bearer_line_index:
            continue
        candidate = raw_line.strip()
        if (
            _OPAQUE_RAW_CANDIDATE_RE.fullmatch(candidate)
            and any(character.isdigit() for character in candidate)
            and any(character in "._-" for character in candidate)
        ):
            raise SQLGroundingProviderError("Grounding credential file format invalid")
    return matches[0].group(1)


def _build_provider_request(
    request: GroundingLLMRequest,
    llm_config: SQLGroundingLLMConfig,
    provider_config: SQLGroundingProviderConfig,
    *,
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if request.prompt != SQL_GROUNDING_PROMPT:
        raise SQLGroundingProviderError("SQL Grounding prompt mismatch")
    if request.response_schema != SQL_GROUNDING_FORM_SCHEMA:
        raise SQLGroundingProviderError("SQL Grounding form schema mismatch")
    messages = [
        {"role": "system", "content": request.prompt},
        {"role": "user", "content": request.input_json},
    ]
    # The Provider transport guarantees only one JSON object.  The frozen
    # Pydantic form and SQL Grounding semantic validators remain the sole
    # authoritative shape and business-state gates after transport parsing.
    response_format = {"type": "json_object"}
    preset = llm_config.preset_config
    generation: dict[str, Any] = {
        "temperature": preset.get("temperature", 0.0),
        "max_tokens": llm_config.max_tokens,
    }
    if "top_p" in preset:
        generation["top_p"] = preset["top_p"]
    extra_body: dict[str, Any] = {}
    if "thinking" in preset:
        extra_body["thinking"] = preset["thinking"]
    if "reasoning_effort" in preset:
        extra_body["reasoning_effort"] = preset["reasoning_effort"]
    provider_kwargs: dict[str, Any] = {
        "model": provider_config.model_id,
        "messages": messages,
        **generation,
        "stream": False,
        "timeout": llm_config.timeout_seconds,
        "num_retries": 0,
        "max_retries": 0,
        "api_base": provider_config.api_base,
        "api_key": api_key,
        "response_format": response_format,
    }
    if extra_body:
        provider_kwargs["extra_body"] = extra_body
    if provider_config.use_bearer_for_custom_base:
        provider_kwargs["use_bearer_for_custom_base"] = True
    request_audit = {
        "model": provider_config.model_id,
        "messages": messages,
        "generation": generation,
        "extra_body": extra_body,
        "timeout_seconds": llm_config.timeout_seconds,
        "provider_retry_count": 0,
        "tools": [],
        "tool_choice": None,
        "response_format": response_format,
        "connection": {
            "api_base": provider_config.api_base,
            "credential_source": provider_config.credential_source,
            "use_bearer_for_custom_base": (
                provider_config.use_bearer_for_custom_base
            ),
        },
        "prompt_sha256": SQL_GROUNDING_PROMPT_SHA256,
        "form_schema_sha256": SQL_GROUNDING_FORM_SCHEMA_SHA256,
        "kernel_configuration_sha256": SQL_GROUNDING_CONFIGURATION_SHA256,
        "model_preset": llm_config.model_preset,
        "model_preset_sha256": llm_config.preset_sha256,
    }
    return request_audit, provider_kwargs


def _request_observation_identity(request: GroundingLLMRequest) -> tuple[str, str]:
    """Extract only bounded audit identity from the internally built input."""

    observation_id = request.observation_id
    observation_type = request.observation_type
    if not observation_id or not observation_type:
        try:
            payload = json.loads(request.input_json)
            latest = payload["latest_observation"]
            observation_id = latest["observation_id"]
            observation_type = latest["observation_type"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SQLGroundingProviderError(
                "SQL Grounding request observation identity invalid"
            ) from exc
    if (
        not isinstance(observation_id, str)
        or not observation_id
        or len(observation_id) > 256
        or not isinstance(observation_type, str)
        or not observation_type
        or len(observation_type) > 64
    ):
        raise SQLGroundingProviderError(
            "SQL Grounding request observation identity invalid"
        )
    return observation_id, observation_type


def _provider_endpoint_identity(api_base: str) -> str:
    parsed = urlsplit(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SQLGroundingProviderError("SQL Grounding Provider endpoint invalid")
    return f"{parsed.scheme}://{parsed.netloc}"


def _provider_response_content(response: Any) -> str:
    try:
        choices = _value(response, "choices")
        content = _value(_value(choices[0], "message"), "content")
    except Exception:
        raise SQLGroundingProviderError("Provider response has no text content") from None
    if not isinstance(content, str):
        raise SQLGroundingProviderError("Provider response content invalid")
    return content.strip()


def _provider_usage(response: Any) -> tuple[GroundingTokenUsage, float | None]:
    raw = to_jsonable(_value(response, "usage", default={})) or {}
    if not isinstance(raw, dict):
        raw = {}
    input_tokens = _usage_token(raw, "prompt_tokens", "input_tokens")
    output_tokens = _usage_token(raw, "completion_tokens", "output_tokens")
    reasoning_tokens = _usage_token(raw, "reasoning_tokens")
    if not reasoning_tokens:
        for name in ("completion_tokens_details", "output_tokens_details"):
            details = raw.get(name)
            if isinstance(details, dict):
                reasoning_tokens = _usage_token(details, "reasoning_tokens")
                if reasoning_tokens:
                    break
    reasoning_tokens = min(reasoning_tokens, output_tokens)
    cost = _provider_reported_cost(raw)
    return (
        GroundingTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        cost,
    )


def _usage_token(raw: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)) and value >= 0 and int(value) == value:
            return int(value)
    return 0


def _provider_reported_cost(raw: Mapping[str, Any]) -> float | None:
    for name in ("cost", "response_cost"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric) and numeric >= 0:
            return numeric
    return None


def _provider_name(response: Any) -> str:
    hidden = _value(response, "_hidden_params", default={}) or {}
    if isinstance(hidden, Mapping):
        value = hidden.get("custom_llm_provider")
        if isinstance(value, str):
            return value[:128]
    return ""


def _value(value: Any, name: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sanitize_audit_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_AUDIT_KEYS:
                continue
            sanitized[str(key)] = _sanitize_audit_value(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_audit_value(item) for item in value]
    return value


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _new_private_audit_path(
    config: SQLGroundingProviderConfig,
    request_sha256: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(config.raw_audit_dir) / (
        f"sql-grounding-{timestamp}-{request_sha256[:12]}-{uuid.uuid4().hex[:8]}.json"
    )


def _write_private_provider_audit(
    path: Path,
    payload: Mapping[str, Any],
    *,
    api_key: str,
) -> None:
    encoded = json.dumps(
        _sanitize_audit_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    if api_key and api_key in encoded:
        raise SQLGroundingProviderError("private audit credential check failed")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _private_audit_ref(config: SQLGroundingProviderConfig, path: Path) -> str:
    project_root = Path(config.raw_audit_dir).parents[1]
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.name


def _telemetry(
    *,
    attempted: bool,
    status: Literal["succeeded", "failed", "timed_out", "rejected"],
    usage: GroundingTokenUsage | None = None,
    latency_ms: float = 0.0,
    request_sha: str = "",
    response_sha: str = "",
    error_type: str | None = None,
    provider_reported_cost: float | None = None,
    model: str = "",
    provider: str = "",
    credential_source: Literal["", "direct", "file"] = "",
    raw_private_audit_ref: str = "",
    provider_may_continue_after_cancel: bool | None = None,
    provider_may_bill_after_cancel: bool | None = None,
) -> GroundingLLMTelemetry:
    return GroundingLLMTelemetry(
        attempted=attempted,
        status=status,
        usage=usage or GroundingTokenUsage(),
        latency_ms=latency_ms,
        timed_out=status == "timed_out",
        error_type=error_type,
        provider_reported_cost=provider_reported_cost,
        model=model,
        provider=provider,
        credential_source=credential_source,
        raw_private_audit_ref=raw_private_audit_ref,
        provider_may_continue_after_cancel=provider_may_continue_after_cancel,
        provider_may_bill_after_cancel=provider_may_bill_after_cancel,
        request_sha256=request_sha,
        response_sha256=response_sha,
        prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
        form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


def _response_telemetry_metadata(response: GroundingClientResponse) -> dict[str, Any]:
    return {
        "provider_reported_cost": response.provider_reported_cost,
        "model": response.model,
        "provider": response.provider,
        "credential_source": response.credential_source,
        "raw_private_audit_ref": response.raw_private_audit_ref,
        "provider_may_continue_after_cancel": (
            response.provider_may_continue_after_cancel
        ),
        "provider_may_bill_after_cancel": response.provider_may_bill_after_cancel,
    }


def _client_telemetry_metadata(client: Any) -> dict[str, Any]:
    config = getattr(client, "config", None)
    return {
        "model": str(getattr(config, "model_id", ""))[:256],
        "credential_source": getattr(config, "credential_source", ""),
        "raw_private_audit_ref": str(
            getattr(client, "_last_private_audit_ref", "")
        )[:1024],
        "provider_may_continue_after_cancel": getattr(
            client, "provider_may_continue_after_cancel", None
        ),
        "provider_may_bill_after_cancel": getattr(
            client, "provider_may_bill_after_cancel", None
        ),
    }
