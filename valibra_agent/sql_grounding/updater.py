"""Strict, provider-agnostic SQL Grounding updater for offline SG2 tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError

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

SQL_GROUNDING_PROMPT = """Fill the fixed Valibra SQL Grounding form from the bounded JSON input.
Return exactly one JSON object and no prose, comments, Markdown other than the
single transport fence allowed by the caller, or additional fields.

The output fields are exactly:
- sql_grounding_state: tables, join_keys, column_mapping, domain_knowledge
- next_focus_dimension

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


class GroundingLLMRequest(ContractModel):
    """The complete request passed to an injected offline client."""

    prompt: str = Field(min_length=1, max_length=32_768)
    input_json: str = Field(min_length=1, max_length=MAX_GROUNDING_REQUEST_CHARS)
    response_schema: dict[str, Any]


class GroundingClientResponse(ContractModel):
    """A fake-client result; no Provider metadata or credential is accepted."""

    content: str
    usage: GroundingTokenUsage = Field(default_factory=GroundingTokenUsage)


class AsyncGroundingClient(Protocol):
    async def complete(self, request: GroundingLLMRequest) -> GroundingClientResponse:
        """Return one fixed response without tools or retry behavior."""


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
            )
            raise GroundingUpdaterError("timeout", telemetry) from exc
        except Exception as exc:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                latency_ms=_elapsed_ms(started),
                request_sha=request_sha,
                error_type="client_error",
            )
            raise GroundingUpdaterError("client_error", telemetry) from exc

        response_sha = hashlib.sha256(
            client_response.content.encode("utf-8")
        ).hexdigest()
        if len(client_response.content) > MAX_GROUNDING_RESPONSE_CHARS:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                usage=client_response.usage,
                latency_ms=_elapsed_ms(started),
                request_sha=request_sha,
                response_sha=response_sha,
                error_type="response_too_large",
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
                request_sha=request_sha,
                response_sha=response_sha,
                error_type=exc.reason,
            )
            raise GroundingUpdaterError(exc.reason, telemetry) from exc

        telemetry = _telemetry(
            attempted=True,
            status="succeeded",
            usage=client_response.usage,
            latency_ms=_elapsed_ms(started),
            request_sha=request_sha,
            response_sha=response_sha,
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


def _telemetry(
    *,
    attempted: bool,
    status: Literal["succeeded", "failed", "timed_out", "rejected"],
    usage: GroundingTokenUsage | None = None,
    latency_ms: float = 0.0,
    request_sha: str = "",
    response_sha: str = "",
    error_type: str | None = None,
) -> GroundingLLMTelemetry:
    return GroundingLLMTelemetry(
        attempted=attempted,
        status=status,
        usage=usage or GroundingTokenUsage(),
        latency_ms=latency_ms,
        timed_out=status == "timed_out",
        error_type=error_type,
        request_sha256=request_sha,
        response_sha256=response_sha,
        prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
        form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )
