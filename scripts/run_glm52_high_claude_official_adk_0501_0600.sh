#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/research_production_guard.sh"
research_block_production_script "${BASH_SOURCE[0]}"

PROJECT_DIR="/home/user/code/BIRD-Interact/BIRD-Interact-ADK"
WORKSPACE_DIR="/home/user/code/BIRD-Interact"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PRESET=glm52_high_32768
export USER_SIM_PROFILE=claude_haiku_4_5_official
export CONCURRENCY=1
export RANGE_START=501
export RANGE_END=600

export SYSTEM_AGENT_API_KEY_FILE="${SYSTEM_AGENT_API_KEY_FILE:-$WORKSPACE_DIR/GLM_API_key_1.txt}"
export USER_SIM_API_KEY_FILE="${USER_SIM_API_KEY_FILE:-$WORKSPACE_DIR/Claude_official_API_key.txt}"

for credential in "$SYSTEM_AGENT_API_KEY_FILE" "$USER_SIM_API_KEY_FILE"; do
    if [[ ! -f "$credential" || ! -s "$credential" ]]; then
        echo "Missing or empty credential file: $credential" >&2
        exit 1
    fi
    credential_mode="$(stat -c '%a' "$credential")"
    if (( (8#$credential_mode & 8#077) != 0 )); then
        echo "Credential file must not be group/world accessible: $credential" >&2
        exit 1
    fi
done

PYTHON_BIN="$PROJECT_DIR/.venv-adk/bin/python"
SQLGLOT_VERSION="$($PYTHON_BIN -c 'import sqlglot; print(sqlglot.__version__)')"
if [[ "$SQLGLOT_VERSION" != "26.16.4" ]]; then
    echo "sqlglot must be 26.16.4, found $SQLGLOT_VERSION" >&2
    exit 2
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="${RUN_DIR:-$PROJECT_DIR/results/official_full_glm52_high32768_claude_official_s0501_0600_${RUN_STAMP}}"

echo "Experiment: GLM-5.2 High/32768 + Claude Haiku 4.5 official API"
echo "Mode: Full / Stress / a-interact / concurrency=1"
echo "Range: records[500:600] (global_index 501-600)"
echo "sqlglot: $SQLGLOT_VERSION"
echo "Result directory: $RUN_DIR"

bash scripts/run_frozen_range.sh
