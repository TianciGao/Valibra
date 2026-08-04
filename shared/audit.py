"""Lossless, JSON-safe audit helpers for benchmark model/tool traces."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from typing import Any, Dict, Iterable


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert Pydantic/SDK objects to JSON-safe Python values without truncation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(
                mode="json",
                exclude_none=True,
                warnings=False,
                fallback=str,
            )
        except TypeError:
            return value.model_dump(mode="json", exclude_none=True)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return repr(value)


def json_text(value: Any) -> str:
    """Render a value as stable, readable JSON for prompt-flow exports."""
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, default=str)


def count_tokens(value: Any) -> int:
    """Count tokens using the reference implementation's cl100k_base tokenizer."""
    text = value if isinstance(value, str) else json_text(value)
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        # Deterministic fallback used only when tiktoken is unavailable.
        return max(1, len(text) // 4) if text else 0


def normalize_usage(value: Any) -> Dict[str, Any]:
    """Normalize LiteLLM and Google GenAI token-usage field names."""
    raw = to_jsonable(value) or {}
    if not isinstance(raw, dict):
        raw = {"raw": raw}

    def _first(*names: str) -> int:
        for name in names:
            item = raw.get(name)
            if isinstance(item, (int, float)):
                return int(item)
        return 0

    input_tokens = _first("prompt_tokens", "prompt_token_count", "input_tokens")
    output_tokens = _first(
        "completion_tokens",
        "candidates_token_count",
        "response_token_count",
        "output_tokens",
    )
    total_tokens = _first("total_tokens", "total_token_count")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": _first(
            "cached_tokens",
            "cached_content_token_count",
            "cache_read_input_tokens",
        ),
        "reasoning_tokens": _first(
            "reasoning_tokens",
            "thoughts_token_count",
        ),
        "tool_prompt_tokens": _first("tool_use_prompt_token_count"),
        "raw": raw,
    }


def summarize_usage(calls: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum normalized usage records from model-call audit entries."""
    summary: Dict[str, Any] = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "tool_prompt_tokens": 0,
        "reported_calls": 0,
    }
    for call in calls:
        summary["model_calls"] += 1
        usage = call.get("usage") or {}
        if not usage:
            continue
        summary["reported_calls"] += 1
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "tool_prompt_tokens",
        ):
            item = usage.get(field, 0)
            if isinstance(item, (int, float)):
                summary[field] += int(item)
    return summary
