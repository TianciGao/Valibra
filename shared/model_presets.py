"""Frozen system-agent model presets and conflict-safe activation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


PRESET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MODEL_ENV_BY_FIELD = {
    "model": "SYSTEM_AGENT_MODEL",
    "thinking.type": "SYSTEM_AGENT_THINKING",
    "thinking.clear_thinking": "SYSTEM_AGENT_CLEAR_THINKING",
    "reasoning_effort": "SYSTEM_AGENT_REASONING_EFFORT",
    "max_tokens": "SYSTEM_AGENT_MAX_TOKENS",
    "temperature": "SYSTEM_AGENT_TEMPERATURE",
    "top_p": "SYSTEM_AGENT_TOP_P",
    "tool_choice": "SYSTEM_AGENT_TOOL_CHOICE",
}
CONTROLLED_MODEL_ENV = frozenset(MODEL_ENV_BY_FIELD.values())
MODEL_CONFIG_KEYS = frozenset(
    {
        "model",
        "thinking",
        "reasoning_effort",
        "max_tokens",
        "temperature",
        "top_p",
        "tool_choice",
    }
)
METADATA_KEYS = frozenset({"history"})


class ModelPresetError(RuntimeError):
    """Raised when a frozen model preset is invalid or conflicts with the shell."""


@dataclass(frozen=True)
class ModelPreset:
    name: str
    path: Path
    normalized_config: dict[str, Any]
    normalized_sha256: str
    file_sha256: str
    metadata: dict[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "normalized_config": self.normalized_config,
            "normalized_sha256": self.normalized_sha256,
            "file_sha256": self.file_sha256,
            "metadata": self.metadata,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def capture_explicit_model_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = environment if environment is not None else os.environ
    return {
        key: source[key]
        for key in CONTROLLED_MODEL_ENV
        if key in source
    }


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelPresetError(f"{field} must be a number")
    return float(value)


def _normalize_config(raw: dict[str, Any], preset_name: str) -> dict[str, Any]:
    unknown = set(raw) - MODEL_CONFIG_KEYS - METADATA_KEYS
    if unknown:
        raise ModelPresetError(
            f"Preset {preset_name} contains unsupported keys: {sorted(unknown)}"
        )

    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ModelPresetError(f"Preset {preset_name} requires a non-empty model")

    thinking = raw.get("thinking")
    if not isinstance(thinking, dict) or set(thinking) != {
        "type",
        "clear_thinking",
    }:
        raise ModelPresetError(
            f"Preset {preset_name} thinking must contain exactly "
            "type and clear_thinking"
        )
    thinking_type = thinking.get("type")
    clear_thinking = thinking.get("clear_thinking")
    if not isinstance(thinking_type, str) or not thinking_type:
        raise ModelPresetError(f"Preset {preset_name} has invalid thinking.type")
    if not isinstance(clear_thinking, bool):
        raise ModelPresetError(
            f"Preset {preset_name} thinking.clear_thinking must be boolean"
        )

    max_tokens = raw.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ModelPresetError(
            f"Preset {preset_name} max_tokens must be a positive integer"
        )

    temperature = _require_number(
        raw.get("temperature"), f"Preset {preset_name} temperature"
    )
    tool_choice = raw.get("tool_choice")
    if not isinstance(tool_choice, str) or not tool_choice:
        raise ModelPresetError(f"Preset {preset_name} has invalid tool_choice")

    normalized: dict[str, Any] = {
        "model": model,
        "thinking": {
            "type": thinking_type,
            "clear_thinking": clear_thinking,
        },
    }

    if "reasoning_effort" in raw:
        reasoning_effort = raw["reasoning_effort"]
        if not isinstance(reasoning_effort, str) or not reasoning_effort:
            raise ModelPresetError(
                f"Preset {preset_name} has invalid reasoning_effort"
            )
        normalized["reasoning_effort"] = reasoning_effort

    normalized["max_tokens"] = max_tokens
    normalized["temperature"] = temperature

    if "top_p" in raw:
        top_p = _require_number(raw["top_p"], f"Preset {preset_name} top_p")
        if not 0.0 <= top_p <= 1.0:
            raise ModelPresetError(
                f"Preset {preset_name} top_p must be between 0 and 1"
            )
        normalized["top_p"] = top_p

    normalized["tool_choice"] = tool_choice

    if model == "openai/glm-4.7" and "reasoning_effort" in normalized:
        raise ModelPresetError(
            "GLM-4.7 presets must omit reasoning_effort completely"
        )
    return normalized


def _validate_history(history: Any, preset_name: str) -> dict[str, Any]:
    if history is None:
        return {}
    if not isinstance(history, dict):
        raise ModelPresetError(f"Preset {preset_name} history must be an object")
    for key in ("configuration_sha256", "prediction_sha256"):
        value = history.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ModelPresetError(
                f"Preset {preset_name} history.{key} must be a SHA256"
            )
    return {"history": history}


def load_model_preset(project_root: Path, name: str) -> ModelPreset:
    if not PRESET_NAME_RE.fullmatch(name):
        raise ModelPresetError(f"Invalid MODEL_PRESET name: {name!r}")
    preset_dir = (project_root / "configs" / "model_presets").resolve()
    path = (preset_dir / f"{name}.json").resolve()
    if path.parent != preset_dir:
        raise ModelPresetError(f"MODEL_PRESET escapes preset directory: {name!r}")
    if not path.is_file():
        available = sorted(item.stem for item in preset_dir.glob("*.json"))
        raise ModelPresetError(
            f"Unknown MODEL_PRESET {name!r}; available presets: {available}"
        )
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ModelPresetError(f"Invalid JSON in preset {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelPresetError(f"Preset {path} must contain a JSON object")

    normalized = _normalize_config(raw, name)
    metadata = _validate_history(raw.get("history"), name)
    return ModelPreset(
        name=name,
        path=path,
        normalized_config=normalized,
        normalized_sha256=sha256_bytes(canonical_json(normalized).encode("utf-8")),
        file_sha256=sha256_bytes(raw_bytes),
        metadata=metadata,
    )


def _field_value(config: dict[str, Any], field: str) -> Any:
    if field == "thinking.type":
        return config["thinking"]["type"]
    if field == "thinking.clear_thinking":
        return config["thinking"]["clear_thinking"]
    return config.get(field)


def _encode_environment_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _environment_value_matches(raw: str, expected: Any) -> bool:
    if isinstance(expected, bool):
        return raw.strip().lower() in (
            {"1", "true", "yes", "on"} if expected else {"0", "false", "no", "off"}
        )
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(raw) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            return float(raw) == expected
        except ValueError:
            return False
    return raw == str(expected)


def activate_model_preset(
    project_root: Path,
    name: str | None,
    explicit_model_environment: Mapping[str, str],
) -> ModelPreset | None:
    if not name:
        return None
    preset = load_model_preset(project_root, name)
    config = preset.normalized_config

    field_by_env = {value: key for key, value in MODEL_ENV_BY_FIELD.items()}
    conflicts: list[str] = []
    for env_name, raw_value in explicit_model_environment.items():
        field = field_by_env[env_name]
        expected = _field_value(config, field)
        if expected is None or not _environment_value_matches(raw_value, expected):
            conflicts.append(env_name)
    if conflicts:
        raise ModelPresetError(
            f"MODEL_PRESET={name} conflicts with explicit model environment "
            f"variables: {sorted(conflicts)}. Remove them; preset fields are frozen."
        )

    # Remove values loaded from .env, including optional parameters deliberately
    # omitted by the preset, then install the frozen values.
    for env_name in CONTROLLED_MODEL_ENV:
        os.environ.pop(env_name, None)
    for field, env_name in MODEL_ENV_BY_FIELD.items():
        value = _field_value(config, field)
        if value is not None:
            os.environ[env_name] = _encode_environment_value(value)
    return preset
