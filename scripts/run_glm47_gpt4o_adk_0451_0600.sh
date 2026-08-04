#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/research_production_guard.sh"
research_block_production_script "${BASH_SOURCE[0]}"

PROJECT_DIR="/home/user/code/BIRD-Interact/BIRD-Interact-ADK"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export MODEL_PRESET=glm47_matched_32768
export USER_SIM_PROFILE=gpt4o_gptsapi
export CONCURRENCY=1
# The original GLM credential is exhausted.  Freeze the replacement credential
# explicitly for this experiment instead of inheriting the stale .env path.
export SYSTEM_AGENT_API_KEY_FILE="${SYSTEM_AGENT_API_KEY_FILE:-/home/user/code/BIRD-Interact/GLM_API_2.txt}"

PYTHON_BIN="$PROJECT_DIR/.venv-adk/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment missing: $PYTHON_BIN" >&2
    exit 1
fi

SQLGLOT_VERSION="$($PYTHON_BIN -c 'import sqlglot; print(sqlglot.__version__)')"
if [[ "$SQLGLOT_VERSION" != "26.16.4" ]]; then
    echo "sqlglot must be 26.16.4, found $SQLGLOT_VERSION" >&2
    exit 2
fi

ARCHIVED_PREDICTIONS="$PROJECT_DIR/results/final_full_glm47_matched_32768_adk_0001_0600_20260803T090235Z/submission_predictions.jsonl"
ARCHIVED_SHA="$(sha256sum "$ARCHIVED_PREDICTIONS" | awk '{print $1}')"
if [[ "$ARCHIVED_SHA" != "dd8e9ea47738f7e561908e30d915d51bf1b24e78d45e7246c4371b4646f0c104" ]]; then
    echo "Historical prediction archive hash changed; refusing new run." >&2
    exit 2
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR/results/glm47_gpt4o_adk_full_0451_0600_${RUN_STAMP}}"
if [[ "$EXPERIMENT_DIR" != /* ]]; then
    EXPERIMENT_DIR="$PROJECT_DIR/$EXPERIMENT_DIR"
fi
mkdir -p "$EXPERIMENT_DIR"

run_shard() {
    local range_start="$1"
    local range_end="$2"
    local shard_dir="$EXPERIMENT_DIR/shard_$(printf '%04d_%04d' "$range_start" "$range_end")"
    RANGE_START="$range_start" \
    RANGE_END="$range_end" \
    RUN_DIR="$shard_dir" \
    MODEL_PRESET="$MODEL_PRESET" \
    USER_SIM_PROFILE="$USER_SIM_PROFILE" \
    CONCURRENCY=1 \
        bash scripts/run_frozen_range.sh
}

echo "Experiment directory: $EXPERIMENT_DIR"
echo "System Agent preset: $MODEL_PRESET"
echo "User Simulator profile: $USER_SIM_PROFILE"
echo "Full source slice: records[450:600] (global_index 451-600)"
echo "sqlglot: $SQLGLOT_VERSION"
echo "Historical prediction archive SHA256: $ARCHIVED_SHA"

run_shard 451 500
run_shard 501 600

MERGED_DIR="$EXPERIMENT_DIR/merged_0451_0600"
if [[ ! -e "$MERGED_DIR" ]]; then
    "$PYTHON_BIN" scripts/merge_official_shards.py \
        --source "$EXPERIMENT_DIR/shard_0451_0500" \
        --source "$EXPERIMENT_DIR/shard_0501_0600" \
        --output "$MERGED_DIR" \
        --input "$PROJECT_DIR/bird-interact-full/bird_interact_data.jsonl" \
        --expected-start 451 \
        --expected-end 600
fi

echo "Completed GLM-4.7 + GPT-4o User Simulator ADK 451-600 run."
echo "Merged result: $MERGED_DIR"
