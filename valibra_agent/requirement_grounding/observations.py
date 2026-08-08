"""把 Agent 已获准看到的输入规范化为有大小上限的 Observation。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from valibra_agent.requirement_grounding.models import (
    MAX_SUMMARY_CHARS,
    Observation,
    ObservationType,
    Phase,
)

MAX_RAW_CANONICAL_BYTES = 1_048_576
MAX_COLLECTION_ITEMS = 512
MAX_NESTING_DEPTH = 12
MAX_RAW_STRING_CHARS = 262_144

_OFFICIAL_TOOL_ERROR_PREFIXES: dict[str, tuple[str, ...]] = {
    "execute_sql": (
        "SQL Error:",
        "Error calling DB environment:",
    ),
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


class ObservationNormalizationError(ValueError):
    """输入无法安全转成 JSON，或超过明确的大小限制。"""


def canonical_json(value: Any) -> str:
    """严格检查后生成键顺序稳定的 JSON，不偷偷做类型转换。"""

    count = [0]
    _validate_json_value(value, depth=0, count=count)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ObservationNormalizationError(str(exc)) from exc
    if len(encoded.encode("utf-8")) > MAX_RAW_CANONICAL_BYTES:
        raise ObservationNormalizationError("canonical input exceeds total byte limit")
    return encoded


def stable_digest(value: Any) -> str:
    """对规范 JSON 计算稳定 SHA-256。"""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_observation(
    *,
    task_id: str,
    observation_type: ObservationType,
    phase: Phase,
    sequence: int,
    source: str,
    raw: Any,
    summary: str | None = None,
    raw_log_ref: str | None = None,
    function_call_id: str | None = None,
    invocation_id: str | None = None,
    tool_name: str | None = None,
) -> Observation:
    """只根据传入内容构造 Observation，不查询任何外部数据。"""

    raw_json = canonical_json(raw)
    raw_digest = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    summary_text = _bounded_summary(raw_json if summary is None else summary)
    identity = {
        "task_id": task_id,
        "observation_type": observation_type,
        "phase": phase,
        "sequence": sequence,
        "source": source,
        "raw_digest": raw_digest,
        "function_call_id": function_call_id,
        "invocation_id": invocation_id,
        "tool_name": tool_name,
    }
    # ID 同时包含来源和内容摘要，可用于安全去重。
    observation_id = hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return Observation(
        observation_id=observation_id,
        observation_type=observation_type,
        task_id=task_id,
        phase=phase,
        sequence=sequence,
        source=source,
        summary=summary_text,
        raw_digest=raw_digest,
        raw_log_ref=raw_log_ref,
        function_call_id=function_call_id,
        invocation_id=invocation_id,
        tool_name=tool_name,
    )


def classify_tool_observation_type(
    *,
    tool_name: str,
    tool_response: Any,
    success_type: ObservationType,
) -> ObservationType:
    """只按冻结官方工具的精确错误前缀区分成功与失败。"""

    prefixes = _OFFICIAL_TOOL_ERROR_PREFIXES.get(tool_name, ())
    if isinstance(tool_response, str) and any(
        tool_response.startswith(prefix) for prefix in prefixes
    ):
        return "tool_error"
    return success_type


def extract_submit_follow_up(tool_response: Any) -> str:
    """从本次合法 submit_sql 原始返回中提取唯一、有界的 Phase-2 问题。"""

    if not isinstance(tool_response, str):
        raise ObservationNormalizationError(
            "submit_sql follow-up response must be a string"
        )
    starts = []
    offset = 0
    while True:
        index = tool_response.find(_FOLLOW_UP_PREFIX, offset)
        if index < 0:
            break
        if index == 0 or tool_response[index - 1] == "\n":
            starts.append(index)
        offset = index + len(_FOLLOW_UP_PREFIX)
    if len(starts) != 1:
        raise ObservationNormalizationError(
            "submit_sql response must contain exactly one legal follow-up marker"
        )

    value_start = starts[0] + len(_FOLLOW_UP_PREFIX)
    value_end = tool_response.find(_BUDGET_PREFIX, value_start)
    if value_end < 0:
        raise ObservationNormalizationError(
            "submit_sql follow-up must precede the official budget line"
        )
    value = tool_response[value_start:value_end]
    if not value or not value.strip():
        raise ObservationNormalizationError("submit_sql follow-up is empty")
    if len(value) > MAX_RAW_STRING_CHARS:
        raise ObservationNormalizationError("submit_sql follow-up exceeds length limit")
    return value.strip()


def _validate_json_value(value: Any, *, depth: int, count: list[int]) -> None:
    """递归检查 JSON 类型、深度、数量和字符串长度。"""

    if depth > MAX_NESTING_DEPTH:
        raise ObservationNormalizationError("JSON input exceeds nesting limit")
    count[0] += 1
    if count[0] > MAX_COLLECTION_ITEMS:
        raise ObservationNormalizationError("JSON input exceeds item-count limit")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ObservationNormalizationError("non-finite floats are not JSON-safe")
        return
    if isinstance(value, str):
        if len(value) > MAX_RAW_STRING_CHARS:
            raise ObservationNormalizationError("JSON string exceeds length limit")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth=depth + 1, count=count)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ObservationNormalizationError("JSON object keys must be strings")
            if len(key) > 256:
                raise ObservationNormalizationError("JSON object key exceeds length limit")
            _validate_json_value(item, depth=depth + 1, count=count)
        return
    raise ObservationNormalizationError(
        f"unsupported non-JSON input type: {type(value).__name__}"
    )


def _bounded_summary(value: str) -> str:
    """压平空白，并把摘要截到状态允许的长度。"""

    if not isinstance(value, str):
        raise ObservationNormalizationError("summary must be a string")
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= MAX_SUMMARY_CHARS:
        return normalized
    return normalized[: MAX_SUMMARY_CHARS - 1] + "…"
