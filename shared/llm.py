"""Unified LLM call interface. 统一大模型调用适配层

Release: uses LiteLlm (supports any provider).
Local override: place _local_provider.py in this directory (gitignored)
to use a custom backend.
"""

import logging
from pathlib import Path
import re
import time

from shared.config import PROJECT_ROOT, normalized_system_agent_config, settings
from shared.audit import normalize_usage, to_jsonable, utc_now

logger = logging.getLogger(__name__)

MAX_RETRIES = 5 # 最大重试次数，默认值为 5
_BEARER_TOKEN_RE = re.compile( # 定义一个正则表达式，用于匹配 Bearer 令牌的模式
    r"Authorization:\s*Bearer\s+([^\"'\\\s]+)", # 匹配 Authorization: Bearer 后面的非空白字符，忽略大小写
    flags=re.IGNORECASE, # 设置正则表达式的标志为忽略大小写
)


def _read_api_key_file(path_value: str) -> str: # 用于读取 API 密钥文件的内容，并返回密钥字符串
    """Read a raw key or extract a Bearer token from a saved curl example."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read API key file: {path}") from exc

    match = _BEARER_TOKEN_RE.search(content)
    if match:
        return match.group(1)

    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(nonempty_lines) == 1 and not any(char.isspace() for char in nonempty_lines[0]):
        return nonempty_lines[0]

    raise ValueError(
        f"API key file must contain one raw key or an Authorization: Bearer header: {path}"
    )


def _connection_kwargs(model_name: str) -> dict: # 会根据模型名称选择连接参数，包括 API 基础 URL、API 密钥和是否使用 Bearer 认证，并返回一个字典
    """Return the endpoint/auth settings belonging to the selected model role."""
    if model_name == settings.system_agent_model:
        api_base = settings.system_agent_api_base
        api_key = settings.system_agent_api_key
        api_key_file = settings.system_agent_api_key_file
        use_bearer = settings.system_agent_use_bearer_for_custom_base
    elif model_name == settings.user_sim_model:
        api_base = settings.user_sim_api_base
        api_key = settings.user_sim_api_key
        api_key_file = settings.user_sim_api_key_file
        use_bearer = settings.user_sim_use_bearer_for_custom_base
    else:
        api_base = ""
        api_key = ""
        api_key_file = ""
        use_bearer = False

    if not api_key and api_key_file:
        api_key = _read_api_key_file(api_key_file)

    kwargs = {}
    resolved_api_base = api_base or settings.litellm_api_base
    resolved_api_key = api_key or settings.litellm_api_key
    if resolved_api_base:
        kwargs["api_base"] = resolved_api_base
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if use_bearer:
        # Needed by Anthropic-compatible third-party endpoints that authenticate
        # with Authorization: Bearer instead of x-api-key.
        kwargs["use_bearer_for_custom_base"] = True
    return kwargs


def _generation_kwargs(model_name: str) -> dict:
    """Return role-specific generation compatibility settings."""
    if model_name == settings.system_agent_model:
        normalized = normalized_system_agent_config()
        extra_body = {
            "thinking": normalized["thinking"],
        }
        if "reasoning_effort" in normalized:
            extra_body["reasoning_effort"] = normalized["reasoning_effort"]
        kwargs = {
            # The OpenAI SDK merges extra_body into the final top-level JSON.
            # GLM's OpenAI-compatible docs require provider-specific fields
            # such as `thinking` to use this transport path.
            "extra_body": extra_body,
            "tool_choice": settings.system_agent_tool_choice,
        }
        if settings.system_agent_top_p is not None:
            kwargs["top_p"] = settings.system_agent_top_p
        return kwargs
    if model_name == settings.user_sim_model and settings.user_sim_disable_thinking:
        # Some Anthropic-compatible gateways enable extended thinking by
        # default. The official user simulator needs its small output budget
        # for the visible <think>/<s> protocol, not hidden provider thinking.
        return {"thinking": {"type": "disabled"}}
    return {}


def build_adk_model_kwargs(model_name: str | None = None) -> dict:
    """Build the exact LiteLLM constructor kwargs used by the ADK model."""
    model_name = model_name or settings.system_agent_model
    kwargs = {
        "model": model_name,
        "max_tokens": settings.system_agent_max_tokens,
        "temperature": settings.system_agent_temperature,
        "num_retries": MAX_RETRIES,
    }
    kwargs.update(_connection_kwargs(model_name))
    kwargs.update(_generation_kwargs(model_name))
    return kwargs


def system_agent_request_preview() -> dict:
    """Return a secret-free flattened preview of system-agent request fields."""
    kwargs = build_adk_model_kwargs(settings.system_agent_model)
    preview = {
        "model": kwargs["model"],
        "thinking": dict(kwargs.get("extra_body", {}).get("thinking", {})),
    }
    reasoning_effort = kwargs.get("extra_body", {}).get("reasoning_effort")
    if reasoning_effort is not None:
        preview["reasoning_effort"] = reasoning_effort
    preview["max_tokens"] = kwargs["max_tokens"]
    preview["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        preview["top_p"] = kwargs["top_p"]
    preview["tool_choice"] = kwargs.get("tool_choice")
    return preview

# Try local override first (gitignored, not in release)
try:
    from shared._local_provider import call_llm, build_adk_model
    try:
        from shared._local_provider import call_llm_with_details
    except ImportError:
        def call_llm_with_details(
            messages: list,
            model_name: str = None,
            temperature: float = 0,
            max_tokens: int = 1024,
        ) -> dict:
            """Best-effort audit wrapper for legacy local providers."""
            started = time.perf_counter()
            content = call_llm(
                messages,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {
                "timestamp": utc_now(),
                "model": model_name or settings.system_agent_model,
                "content": content,
                "usage": {},
                "latency_seconds": time.perf_counter() - started,
                "raw_response": {"content": content, "local_provider": True},
            }
except ImportError:
    # Default: LiteLlm
    def _completion(messages: list, model_name: str, temperature: float, max_tokens: int):
        import litellm
        kwargs = dict(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=MAX_RETRIES,
        )
        kwargs.update(_connection_kwargs(model_name))
        kwargs.update(_generation_kwargs(model_name))

        return litellm.completion(**kwargs)

    def call_llm(messages: list, model_name: str = None, temperature: float = 0, max_tokens: int = 1024) -> str:
        """Call LLM via LiteLlm. Retries on rate limit / transient errors."""
        model_name = model_name or settings.system_agent_model
        resp = _completion(messages, model_name, temperature, max_tokens)
        return resp.choices[0].message.content.strip()

    def call_llm_with_details(
        messages: list,
        model_name: str = None,
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> dict:
        """Call LiteLLM and retain the complete response plus provider token usage."""
        import litellm

        model_name = model_name or settings.system_agent_model
        started_at = utc_now()
        started = time.perf_counter()
        resp = _completion(messages, model_name, temperature, max_tokens)
        latency = time.perf_counter() - started
        message = resp.choices[0].message
        content = (message.content or "").strip()
        usage = normalize_usage(getattr(resp, "usage", None))
        hidden = getattr(resp, "_hidden_params", {}) or {}
        response_cost = hidden.get("response_cost")
        if response_cost is None:
            try:
                response_cost = litellm.completion_cost(completion_response=resp)
            except Exception:
                response_cost = None

        return {
            "timestamp": started_at,
            "model": model_name,
            "content": content,
            "usage": usage,
            "latency_seconds": latency,
            "response_cost": response_cost,
            "provider": hidden.get("custom_llm_provider", ""),
            "raw_response": to_jsonable(resp),
        }

    def build_adk_model(model_name: str = None): # 用于通过 LiteLlm 构建 ADK 兼容模型，并设置重试配置
        """Build ADK-compatible model via LiteLlm with retry config."""
        from google.adk.models.lite_llm import LiteLlm
        return LiteLlm(**build_adk_model_kwargs(model_name))
