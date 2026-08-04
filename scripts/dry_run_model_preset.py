"""Offline model-preset and frozen a-interact invariant validation."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    active_model_preset_report,
    normalized_system_agent_config,
    settings,
)
from shared.llm import (
    build_adk_model_kwargs,
    system_agent_request_preview,
)
from system_agent.agent import AINTERACT_INSTRUCTION, build_agent
from system_agent.callbacks import MAX_MODEL_TURNS, TOOL_COSTS


EXPECTED_TOOL_NAMES = [
    "execute_sql",
    "get_schema",
    "get_all_column_meanings",
    "get_column_meaning",
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
    "ask_user",
    "submit_sql",
]
EXPECTED_TOOL_COSTS = {
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
    "ask_user": 2.0,
    "submit_sql": 3.0,
}
EXPECTED_AINTERACT_PROMPT_SHA256 = (
    "a43338edfe4679f538c1c747029a866eb2facf8e02dd419762879dc4e0738ce0"
)
EXPECTED_TOOLS_FILE_SHA256 = (
    "a1d7c79b4bbdad4d4ff8bfd718a511b7cb5ecabdb824ecc1074fb76c14a1c760"
)
EXPECTED_AGENT_FILE_SHA256 = (
    "680bca80a39848afc62211f3ee5efc5cbe06f6e0a80d95444c535c9aad52f478"
)
EXPECTED_CALLBACKS_FILE_SHA256 = (
    "9f572d9eb3744a0619531fc59ddaf04fd99b482e590c4f6b6a6f48b0990b191c"
)
EXPECTED_USER_SIMULATOR_FILE_SHA256 = {
    "user_simulator/prompts.py": (
        "55b771f68845bcb18490defc55a430d365ec4caaa5162708bd3ec92155bf25f1"
    ),
    "user_simulator/server.py": (
        "8fad047dd88c35ef2f6b103b9056d0a4dd0148d21ebea271fd8e7b2a66fb4de5"
    ),
}
EXPECTED_CALLBACK_NAMES = {
    "before_model_callback": "before_model_callback",
    "after_model_callback": "after_model_callback",
    "before_tool_callback": "before_tool_callback",
    "after_tool_callback": "after_tool_callback",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callback_name(agent, attribute: str) -> str:
    value = getattr(agent, attribute, None)
    return getattr(value, "__name__", "")


def validate() -> dict:
    preset = active_model_preset_report()
    if preset is None:
        raise RuntimeError("MODEL_PRESET is required for a frozen dry-run")

    normalized = normalized_system_agent_config()
    request_preview = system_agent_request_preview()
    constructor_kwargs = build_adk_model_kwargs()
    safe_constructor_kwargs = {
        key: value
        for key, value in constructor_kwargs.items()
        if key not in {"api_key"}
    }

    if request_preview != normalized:
        raise RuntimeError(
            "Request preview differs from normalized frozen configuration"
        )
    if (
        constructor_kwargs.get("extra_body", {}).get("thinking")
        != normalized["thinking"]
    ):
        raise RuntimeError("LiteLLM thinking payload differs from preset")

    if normalized["model"] == "openai/glm-4.7":
        expected = {
            "model": "openai/glm-4.7",
            "thinking": {
                "type": "enabled",
                "clear_thinking": False,
            },
            "max_tokens": 32768,
            "temperature": 0.0,
            "top_p": 0.95,
            "tool_choice": "auto",
        }
        if request_preview != expected:
            raise RuntimeError(
                f"GLM-4.7 request mismatch: {request_preview!r}"
            )
        if "reasoning_effort" in json.dumps(
            {
                "normalized": normalized,
                "request": request_preview,
                "constructor": safe_constructor_kwargs,
            },
            ensure_ascii=False,
        ):
            raise RuntimeError(
                "GLM-4.7 dry-run contains forbidden reasoning_effort"
            )

    agent = build_agent("a-interact")
    tool_names = [tool.name for tool in agent.tools]
    if tool_names != EXPECTED_TOOL_NAMES:
        raise RuntimeError(f"Frozen tool list changed: {tool_names}")
    if TOOL_COSTS != EXPECTED_TOOL_COSTS:
        raise RuntimeError(f"Frozen tool costs changed: {TOOL_COSTS}")

    prompt_sha = hashlib.sha256(AINTERACT_INSTRUCTION.encode("utf-8")).hexdigest()
    if prompt_sha != EXPECTED_AINTERACT_PROMPT_SHA256:
        raise RuntimeError(f"Frozen a-interact prompt changed: {prompt_sha}")
    tools_file_sha = _sha256_file(PROJECT_ROOT / "system_agent" / "tools.py")
    if tools_file_sha != EXPECTED_TOOLS_FILE_SHA256:
        raise RuntimeError(f"Frozen tool implementation changed: {tools_file_sha}")
    agent_file_sha = _sha256_file(PROJECT_ROOT / "system_agent" / "agent.py")
    if agent_file_sha != EXPECTED_AGENT_FILE_SHA256:
        raise RuntimeError(f"Frozen Agent implementation changed: {agent_file_sha}")
    callbacks_file_sha = _sha256_file(
        PROJECT_ROOT / "system_agent" / "callbacks.py"
    )
    if callbacks_file_sha != EXPECTED_CALLBACKS_FILE_SHA256:
        raise RuntimeError(
            f"Frozen Callback implementation changed: {callbacks_file_sha}"
        )
    if MAX_MODEL_TURNS != 60:
        raise RuntimeError(f"Frozen model turn cap changed: {MAX_MODEL_TURNS}")

    user_simulator_file_sha = {
        relative: _sha256_file(PROJECT_ROOT / relative)
        for relative in EXPECTED_USER_SIMULATOR_FILE_SHA256
    }
    if user_simulator_file_sha != EXPECTED_USER_SIMULATOR_FILE_SHA256:
        raise RuntimeError(
            "Frozen User Simulator implementation changed: "
            f"{user_simulator_file_sha}"
        )
    user_sim_profile = os.environ.get(
        "USER_SIM_PROFILE", "claude_haiku_4_5_gptsapi"
    )
    expected_user_simulators = {
        "claude_haiku_4_5_gptsapi": {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "prompt_version": "v2",
            "provider_hidden_thinking_disabled": True,
            "use_bearer_for_custom_base": True,
            "protocol_policy": "official",
            "protocol_max_attempts": 1,
        },
        "claude_haiku_4_5_official": {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "prompt_version": "v2",
            "provider_hidden_thinking_disabled": True,
            "use_bearer_for_custom_base": False,
            "protocol_policy": "official",
            "protocol_max_attempts": 1,
        },
        "gpt4o_gptsapi": {
            "model": "openai/gpt-4o",
            "prompt_version": "v2",
            "provider_hidden_thinking_disabled": False,
            "use_bearer_for_custom_base": False,
            "protocol_policy": "strict_retry",
            "protocol_max_attempts": 3,
        },
    }
    if user_sim_profile not in expected_user_simulators:
        raise RuntimeError(f"Unknown USER_SIM_PROFILE: {user_sim_profile}")
    user_simulator = {
        "profile": user_sim_profile,
        "model": settings.user_sim_model,
        "prompt_version": settings.prompt_version,
        "provider_hidden_thinking_disabled": (
            settings.user_sim_disable_thinking
        ),
        "api_base": settings.user_sim_api_base.rstrip("/"),
        "use_bearer_for_custom_base": (
            settings.user_sim_use_bearer_for_custom_base
        ),
        "provider_failure_policy": "fail_fast_uncheckpointed",
        "protocol_policy": settings.user_sim_protocol_policy,
        "protocol_max_attempts": settings.user_sim_protocol_max_attempts,
        "file_sha256": user_simulator_file_sha,
    }
    expected_user_simulator = {
        "profile": user_sim_profile,
        **expected_user_simulators[user_sim_profile],
        "api_base": (
            "https://api.anthropic.com"
            if user_sim_profile == "claude_haiku_4_5_official"
            else "https://api.gptsapi.net"
        ),
        "provider_failure_policy": "fail_fast_uncheckpointed",
        "protocol_policy": expected_user_simulators[user_sim_profile][
            "protocol_policy"
        ],
        "protocol_max_attempts": expected_user_simulators[user_sim_profile][
            "protocol_max_attempts"
        ],
        "file_sha256": EXPECTED_USER_SIMULATOR_FILE_SHA256,
    }
    if user_simulator != expected_user_simulator:
        raise RuntimeError(f"Frozen User Simulator changed: {user_simulator}")
    if not (settings.user_sim_api_key or settings.user_sim_api_key_file):
        raise RuntimeError("User Simulator API credential is not configured")
    if version("sqlglot") != "26.16.4":
        raise RuntimeError(
            f"sqlglot must be 26.16.4, found {version('sqlglot')}"
        )

    callback_names = {
        key: _callback_name(agent, key)
        for key in EXPECTED_CALLBACK_NAMES
    }
    if callback_names != EXPECTED_CALLBACK_NAMES:
        raise RuntimeError(f"Frozen callback bindings changed: {callback_names}")

    return {
        "status": "passed",
        "network_or_provider_calls": 0,
        "model_preset": preset,
        "normalized_request": request_preview,
        "litellm_adk_constructor_kwargs": safe_constructor_kwargs,
        "checks": {
            "sqlglot_version": version("sqlglot"),
            "reasoning_effort_absent_for_glm47": (
                "reasoning_effort" not in request_preview
            ),
            "tool_count": len(tool_names),
            "tool_names": tool_names,
            "tool_source_sha256": tools_file_sha,
            "agent_source_sha256": agent_file_sha,
            "prompt_unchanged": True,
            "ainteract_prompt_sha256": prompt_sha,
            "callbacks_unchanged": True,
            "callbacks_source_sha256": callbacks_file_sha,
            "max_model_turns": MAX_MODEL_TURNS,
            "callback_bindings": callback_names,
            "tool_costs": TOOL_COSTS,
            "user_simulator_frozen": True,
            "user_simulator": user_simulator,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
