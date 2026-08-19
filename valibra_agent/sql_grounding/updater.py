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
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator

from shared.audit import to_jsonable
from shared.model_presets import load_model_preset

from valibra_agent.sql_grounding.models import (
    ContractModel,
    GroundingLLMResponse,
    GroundingRuntime,
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
DEFAULT_GROUNDING_TIMEOUT_SECONDS = 30.0
DEFAULT_GROUNDING_MAX_CALLS_PER_TASK = 2
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

SQL_GROUNDING_PROMPT = """Fill the fixed Valibra SQL Grounding form from the bounded JSON input.
Return exactly one JSON object and no prose, comments, Markdown other than the
single transport fence allowed by the caller, or additional fields.

The output fields are exactly:
- sql_grounding_state: tables, join_keys, column_mapping, domain_knowledge
- next_focus_dimension

Exact output shape:
- sql_grounding_state contains exactly:
  - tables: null or an array of table-name strings
  - join_keys: null or an array of relation-expression strings
  - column_mapping: null or an array of objects containing exactly:
    - phrase: one string
    - targets: an array of one or more strings
  - domain_knowledge: null or an array of objects containing exactly:
    - kind: business_rule, runtime_state, or database_capability
    - content: one string
- next_focus_dimension is exactly one of tables, join_keys, column_mapping,
  domain_knowledge, or none.

Important structural rules:
- The field name is exactly "targets", plural.
- "targets" is always a JSON array, even when it contains one item.
- There is no field named "target".
- Do not add any fields not listed above.

Rules for the persisted four-dimensional State:
1. Use only real, fully qualified database identifiers already supported by the
   latest legal Observation. Never persist shorthand aliases or invented names.
2. join_keys and column_mapping targets are canonical PostgreSQL expressions
   under Valibra's restricted field/relation expression contract. They are not
   free SQL and cannot contain statements, comments, arbitrary functions, or
   unapproved AST shapes.
3. Every column_mapping.phrase is a non-empty verbatim substring of the original
   Query or follow-up. Do not paraphrase the phrase.
4. domain_knowledge.kind is exactly one of business_rule, runtime_state, or
   database_capability. Every newly added or changed non-empty
   domain_knowledge.content is copied verbatim from a canonical Official BIRD
   knowledge statement present in the latest legal Observation. Existing
   validated content may only be preserved unchanged. Never paraphrase or
   invent knowledge.
5. Preserve every existing dimension and item not affected by the latest legal
   Observation. Do not clear or replace an unaffected dimension.
6. null means not evaluated. [] means evaluated and not needed. A non-empty
   array contains the validated result.
7. In INITIAL_GROUNDING, if any dimension is null, next_focus_dimension must be
   tables, join_keys, column_mapping, or domain_knowledge. When all four
   dimensions are evaluated, next_focus_dimension must be none.
8. In REPAIR, a completed State may focus any one dimension again. Use none only
   when the latest legal Observation gives no clear need for more Grounding.

Do not output score, confidence, ambiguity, reasoning, an Evidence summary, an
SQL plan, Bird-Coin, final SQL, a tool call, or a specific tool choice. The
response proposes only the complete four-dimensional State and the next focus.
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
            raise ValueError("GROUNDING_TIMEOUT_SECONDS differs from frozen SG2 value")
        if self.prompt_sha256 != SQL_GROUNDING_PROMPT_SHA256:
            raise ValueError("GROUNDING_PROMPT_SHA256 mismatch")
        if self.form_schema_sha256 != SQL_GROUNDING_FORM_SCHEMA_SHA256:
            raise ValueError("SQL Grounding form schema SHA mismatch")
        if self.kernel_configuration_sha256 != SQL_GROUNDING_CONFIGURATION_SHA256:
            raise ValueError("SQL Grounding kernel configuration SHA mismatch")
        if self.preset_config.get("max_tokens") != self.max_tokens:
            raise ValueError("GROUNDING_MAX_TOKENS must equal the frozen preset value")
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
    ) -> None:
        super().__init__(reason)
        self.request_sha256 = request_sha256
        self.raw_private_audit_ref = raw_private_audit_ref


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
        completion = self._completion
        if completion is None:
            import litellm

            completion = litellm.acompletion
        try:
            response = await completion(**provider_kwargs)
        except Exception as exc:
            _write_private_provider_audit(
                audit_path,
                {
                    "schema_version": "1.0",
                    "status": "failed",
                    "request": request_audit,
                    "request_sha256": request_sha,
                    "error_type": type(exc).__name__[:128],
                },
                api_key=api_key,
            )
            raise SQLGroundingProviderError(
                f"SQL Grounding Provider request failed ({type(exc).__name__[:128]})",
                request_sha256=request_sha,
                raw_private_audit_ref=_private_audit_ref(self.config, audit_path),
            ) from None

        raw_response = _sanitize_audit_value(to_jsonable(response))
        response_sha = _stable_json_sha256(raw_response)
        _write_private_provider_audit(
            audit_path,
            {
                "schema_version": "1.0",
                "status": "succeeded",
                "request": request_audit,
                "request_sha256": request_sha,
                "response": raw_response,
                "response_sha256": response_sha,
            },
            api_key=api_key,
        )
        usage, cost = _provider_usage(response)
        try:
            content = _provider_response_content(response)
        except SQLGroundingProviderError as exc:
            raise SQLGroundingProviderError(
                str(exc),
                request_sha256=request_sha,
                raw_private_audit_ref=_private_audit_ref(self.config, audit_path),
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
            raw_private_audit_ref=_private_audit_ref(self.config, audit_path),
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
    ) -> GroundingUpdaterResult:
        """Return a typed proposal without mutating Runtime or Observation."""

        request = _build_request(
            runtime,
            observation,
            original_query=original_query,
            follow_up_query=follow_up_query,
        )
        request_sha = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
        started = time.perf_counter()
        try:
            client_response = await asyncio.wait_for(
                self._client.complete(request),
                timeout=DEFAULT_GROUNDING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            telemetry = _telemetry(
                attempted=True,
                status="timed_out",
                latency_ms=_elapsed_ms(started),
                request_sha=request_sha,
                error_type="timeout",
                **_client_telemetry_metadata(self._client),
            )
            raise GroundingUpdaterError("timeout", telemetry) from exc
        except SQLGroundingProviderError as exc:
            metadata = _client_telemetry_metadata(self._client)
            metadata["raw_private_audit_ref"] = exc.raw_private_audit_ref
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                latency_ms=_elapsed_ms(started),
                request_sha=exc.request_sha256 or request_sha,
                error_type="provider_error",
                **metadata,
            )
            raise GroundingUpdaterError("provider_error", telemetry) from exc
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
) -> GroundingLLMRequest:
    input_payload = {
        "current_sql_grounding_state": runtime.grounding_state.model_dump(mode="json"),
        "current_stage": runtime.stage,
        "follow_up": follow_up_query,
        "latest_observation": observation_for_updater(observation),
        "original_query": original_query,
    }
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
    )
    if len(canonical_json(request)) > MAX_GROUNDING_REQUEST_CHARS:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="request_too_large",
        )
        raise GroundingUpdaterError("request_too_large", telemetry)
    return request


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
        "provider_may_continue_after_cancel": getattr(
            client, "provider_may_continue_after_cancel", None
        ),
        "provider_may_bill_after_cancel": getattr(
            client, "provider_may_bill_after_cancel", None
        ),
    }
