#!/usr/bin/env bash
set -Eeuo pipefail

ADK_ROOT="/home/user/code/BIRD-Interact/BIRD-Interact-ADK"
OFFICIAL_ROOT="/home/user/code/BIRD-Interact/bird_interact_agent"
PYTHON_BIN="$ADK_ROOT/.venv-adk/bin/python"

cd "$ADK_ROOT"
if [[ "${MODEL_PRESET:-}" != "glm47_matched_32768" ]]; then
    echo "MODEL_PRESET must be glm47_matched_32768" >&2
    exit 2
fi
if [[ "${CONCURRENCY:-1}" != "1" ]]; then
    echo "CONCURRENCY must remain 1" >&2
    exit 2
fi

export PYTHONPATH="$OFFICIAL_ROOT:$ADK_ROOT"
export PYTHONUNBUFFERED=1
export BIRD_ADK_ROOT="$ADK_ROOT"
export BIRD_NONADK_FROZEN_PROVIDER=1
export DATASET=full
export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
export USER_SIM_DISABLE_THINKING=true
export PROMPT_VERSION=v2
export PATIENCE=3
export PG_HOST=127.0.0.1
export PG_PORT=5433
export CONCURRENCY=1
export P1_GATE_AFTER=100
export P1_GATE_THRESHOLD=0.15

if ! docker inspect -f '{{.State.Running}}' bird_interact_postgresql_full \
    2>/dev/null | grep -qx true; then
    echo "Full PostgreSQL container is not running" >&2
    exit 1
fi
if ! pg_isready -h 127.0.0.1 -p 5433 -U root >/dev/null; then
    echo "Full PostgreSQL is not ready on 127.0.0.1:5433" >&2
    exit 1
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$ADK_ROOT/results/nonadk_official_glm47_0301_0600_${RUN_STAMP}}"
mkdir -p "$RUN_DIR/logs"
"$PYTHON_BIN" -m pip freeze >"$RUN_DIR/dependencies.freeze.txt"

echo "Framework: official text-ReAct non-ADK"
echo "Range: records[300:600] / global_index 301-600"
echo "MODEL_PRESET=$MODEL_PRESET"
echo "Concurrency: 1"
echo "P1 gate: pause after 100 tasks when rate < 15%"
echo "Result directory: $RUN_DIR"

exec "$PYTHON_BIN" scripts/run_official_nonadk_range.py \
    --run-dir "$RUN_DIR" \
    --input "$ADK_ROOT/bird-interact-full/bird_interact_data.jsonl" \
    --official-root "$OFFICIAL_ROOT" \
    --start 301 \
    --end 600
