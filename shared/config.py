"""Centralized configuration.

Settings are loaded in this priority (highest wins):
  1. Frozen MODEL_PRESET fields, when MODEL_PRESET is set
  2. Environment variables (for fields not frozen by a preset)
  3. .env file in project root (user-specific, gitignored)
  4. Defaults defined below
也就是说项目的加载优先级是
1. MODEL_PRESET 中冻结的模型参数  最高（冲突的显式环境变量会报错）
2. 启动命令或系统环境变量
3. BIRD-Interact-ADK/.env
4. config.py 中定义的默认值      最低
Users: copy .env.example to .env and edit.
See .env.example for all available settings.

"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv # 加载 .env 文件的库
from pydantic_settings import BaseSettings
from shared.model_presets import (
    ModelPreset,
    activate_model_preset,
    capture_explicit_model_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent # 获取项目根目录的路径，即 BIRD-Interact-ADK 的父目录

_EXPLICIT_MODEL_ENVIRONMENT = capture_explicit_model_environment()

# Load the project-local .env even when the process is started from another
# working directory.
load_dotenv(PROJECT_ROOT / ".env") # 加载项目根目录下的 .env 文件，设置环境变量，

active_model_preset: ModelPreset | None = activate_model_preset(
    PROJECT_ROOT,
    os.environ.get("MODEL_PRESET"),
    _EXPLICIT_MODEL_ENVIRONMENT,
)


class Settings(BaseSettings): # 定义一个 Settings 类，继承自 BaseSettings，用于存储项目的配置参数
    # LLM provider
    llm_provider: str = "litellm" # 大模型调用适配层

    # PostgreSQL
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_user: str = "root"
    pg_password: str = "123123"
    pg_minconn: int = 1
    pg_maxconn: int = 5

    # Service ports
    system_agent_port: int = 6000
    user_sim_port: int = 6001
    db_env_port: int = 6002

    # Models (LiteLlm format: provider/model-name)
    user_sim_model: str = "anthropic/claude-haiku-4-5-20251001"
    system_agent_model: str = "anthropic/claude-sonnet-4-20250514"

    # Per-role LiteLLM connections. These allow the system agent and user
    # simulator to use different providers/endpoints in the same evaluation.
    system_agent_api_base: str = ""
    system_agent_api_key: str = ""
    system_agent_api_key_file: str = ""
    system_agent_use_bearer_for_custom_base: bool = False # 布尔值，表示是否为自定义 API 基础 URL 使用 Bearer 令牌进行身份验证
    # GLM-5.2 generation settings. `clear_thinking=False` enables preserved
    # thinking; ADK/LiteLLM must replay the complete reasoning_content.
    system_agent_thinking: str = "enabled"
    system_agent_reasoning_effort: str | None = "max"
    system_agent_clear_thinking: bool = False
    system_agent_max_tokens: int = 65536
    system_agent_temperature: float = 0.0
    system_agent_top_p: float | None = None
    system_agent_tool_choice: str = "auto"
    user_sim_api_base: str = ""
    user_sim_api_key: str = ""
    user_sim_api_key_file: str = ""
    user_sim_use_bearer_for_custom_base: bool = False # 也是这样，为了配适服务商的要求
    user_sim_disable_thinking: bool = False
    # Keep the upstream parser by default.  A named custom User Simulator may
    # explicitly opt into bounded retries for malformed <s>...</s> transport.
    user_sim_protocol_policy: str = "official"
    user_sim_protocol_max_attempts: int = 1

    # LiteLlm proxy (optional — set if using a LiteLlm proxy server)
    # These remain as fallbacks for models without a per-role connection. 
    litellm_api_base: str = ""
    litellm_api_key: str = ""

    # Dataset: "lite" or "full"
    dataset: str = "lite"

    # User simulator prompt version: "v1" (旧版，用于复现论文实验) or "v2" (新版，官方推荐，优化过的提示词)
    prompt_version: str = "v2"

    # Budget / turns
    patience: int = 3 # 用户模拟器的耐心值，表示在评测过程中允许的额外探索次数，默认值为 3

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / f"bird-interact-{self.dataset}"      # 自动寻找数据集

    @property
    def data_path(self) -> str:
        return str(self.data_dir / "bird_interact_data.jsonl") # 自动寻找数据集中的 JSONL 文件

    @property
    def db_data_path(self) -> str: # 自动寻找数据集中的数据库文件夹
        return str(self.data_dir)

    class Config:
        env_file = str(PROJECT_ROOT / ".env")  # 自动寻找项目根目录下的 .env 文件
        env_file_encoding = "utf-8"
        extra = "ignore" 


_settings_overrides = {}
if (
    active_model_preset is not None
    and "reasoning_effort" not in active_model_preset.normalized_config
):
    _settings_overrides["system_agent_reasoning_effort"] = None

settings = Settings( # 创建一个 Settings 类的实例 settings，用于访问项目的配置参数
    _env_file=None if active_model_preset is not None else str(PROJECT_ROOT / ".env"),
    **_settings_overrides,
)


def normalized_system_agent_config() -> dict:
    """Return exactly the generation fields sent for the system-agent role."""
    config = {
        "model": settings.system_agent_model,
        "thinking": {
            "type": settings.system_agent_thinking,
            "clear_thinking": settings.system_agent_clear_thinking,
        },
    }
    if settings.system_agent_reasoning_effort is not None:
        config["reasoning_effort"] = settings.system_agent_reasoning_effort
    config["max_tokens"] = settings.system_agent_max_tokens
    config["temperature"] = settings.system_agent_temperature
    if settings.system_agent_top_p is not None:
        config["top_p"] = settings.system_agent_top_p
    config["tool_choice"] = settings.system_agent_tool_choice
    return config


def active_model_preset_report() -> dict | None:
    if active_model_preset is None:
        return None
    return active_model_preset.report()


if (
    active_model_preset is not None
    and normalized_system_agent_config()
    != active_model_preset.normalized_config
):
    raise RuntimeError(
        "Loaded Settings differ from frozen MODEL_PRESET: "
        + json.dumps(normalized_system_agent_config(), ensure_ascii=False)
    )
