#!/usr/bin/env bash

# Source this file from a shell before running research commands.
RESEARCH_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$RESEARCH_PROJECT_DIR/.research-environment" ]]; then
    echo "Research marker missing: $RESEARCH_PROJECT_DIR" >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="$RESEARCH_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MODEL_PRESET="${MODEL_PRESET:-glm52_high_32768}"
export DATASET=full
export PROMPT_VERSION=v2
export PATIENCE=3

export PG_HOST=127.0.0.1
export PG_PORT=6433
export PG_USER=root
export PG_PASSWORD=123123
export SYSTEM_AGENT_PORT=6100
export USER_SIM_PORT=6101
export DB_ENV_PORT=6102

# Never inherit production credentials into the research shell.
unset OPENAI_API_KEY ANTHROPIC_API_KEY ZHIPUAI_API_KEY GOOGLE_API_KEY GEMINI_API_KEY
unset AZURE_API_KEY AZURE_OPENAI_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
unset SYSTEM_AGENT_API_KEY_FILE USER_SIM_API_KEY_FILE
# Frozen preset fields must come only from MODEL_PRESET, never from the shell
# that may also be supervising the production evaluation.
unset SYSTEM_AGENT_MODEL SYSTEM_AGENT_THINKING SYSTEM_AGENT_CLEAR_THINKING
unset SYSTEM_AGENT_REASONING_EFFORT SYSTEM_AGENT_MAX_TOKENS
unset SYSTEM_AGENT_TEMPERATURE SYSTEM_AGENT_TOP_P SYSTEM_AGENT_TOOL_CHOICE
export SYSTEM_AGENT_API_BASE=http://127.0.0.1:9
export SYSTEM_AGENT_API_KEY=research-offline
export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
export USER_SIM_API_BASE=http://127.0.0.1:9
export USER_SIM_API_KEY=research-offline
export USER_SIM_USE_BEARER_FOR_CUSTOM_BASE=false
export USER_SIM_DISABLE_THINKING=true
export USER_SIM_PROTOCOL_POLICY=official
export USER_SIM_PROTOCOL_MAX_ATTEMPTS=1
export LITELLM_API_BASE=http://127.0.0.1:9
export LITELLM_API_KEY=research-offline
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

if [[ -f "$RESEARCH_PROJECT_DIR/.venv-research/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$RESEARCH_PROJECT_DIR/.venv-research/bin/activate"
fi

cd "$RESEARCH_PROJECT_DIR"
