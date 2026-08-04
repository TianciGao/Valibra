#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/home/user/code/BIRD-Interact/BIRD-Interact-ADK"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv-adk/bin/activate"
export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1

# Pin every process to the same verified leaderboard configuration.
export DATASET=full
export SYSTEM_AGENT_MODEL=openai/glm-5.2
export SYSTEM_AGENT_THINKING=enabled
export SYSTEM_AGENT_CLEAR_THINKING=false
export SYSTEM_AGENT_REASONING_EFFORT=max
export SYSTEM_AGENT_MAX_TOKENS=65536
export SYSTEM_AGENT_TEMPERATURE=0
export SYSTEM_AGENT_TOOL_CHOICE=auto
export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
export USER_SIM_DISABLE_THINKING=true
export PROMPT_VERSION=v2
export PATIENCE=3
export PG_HOST=127.0.0.1
export PG_PORT=5433
export SYSTEM_AGENT_PORT=6000
export USER_SIM_PORT=6001
export DB_ENV_PORT=6002
# This max/65K audit may legitimately keep one provider request open for
# longer than the repository's default 1800-second orchestrator wait.
export SYSTEM_AGENT_RUN_TIMEOUT_SECONDS=7200

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [existing-or-new-run-directory]" >&2
    exit 2
fi

SHARD_START="${SHARD_START:-551}"
SHARD_END="${SHARD_END:-600}"
printf -v SHARD_LABEL 's%04d_%04d' "$SHARD_START" "$SHARD_END"

if [[ $# -eq 1 ]]; then
    RUN_DIR="$1"
else
    RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_DIR="$PROJECT_DIR/results/official_full_glm52_max65536_${SHARD_LABEL}_${RUN_TIMESTAMP}"
fi
if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$PROJECT_DIR/$RUN_DIR"
fi

mkdir -p "$RUN_DIR/logs"
RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SERVICE_PIDS=()

collect_logs_and_stop() {
    local exit_code=$?
    trap - EXIT INT TERM
    docker logs --since "$RUN_STARTED_AT" bird_interact_postgresql_full \
        >>"$RUN_DIR/logs/postgresql_full.log" 2>&1 || true
    for pid in "${SERVICE_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${SERVICE_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$exit_code"
}
trap collect_logs_and_stop EXIT INT TERM

if ! docker inspect -f '{{.State.Running}}' bird_interact_postgresql_full \
    2>/dev/null | grep -qx true; then
    echo "Full PostgreSQL container is not running." >&2
    exit 1
fi

if ! pg_isready -h 127.0.0.1 -p 5433 -U root >/dev/null; then
    echo "Full PostgreSQL is not accepting connections on 127.0.0.1:5433." >&2
    exit 1
fi

for port in 6000 6001 6002; do
    if curl --noproxy '*' -fsS --max-time 2 \
        "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "Port ${port} already has a healthy service; refusing mixed-process run." >&2
        exit 1
    fi
done

python -m uvicorn db_environment.server:app \
    --host 127.0.0.1 --port 6002 --log-level info \
    >"$RUN_DIR/logs/db_environment.log" 2>&1 &
SERVICE_PIDS+=("$!")

python -m uvicorn user_simulator.server:app \
    --host 127.0.0.1 --port 6001 --log-level info \
    >"$RUN_DIR/logs/user_simulator.log" 2>&1 &
SERVICE_PIDS+=("$!")

python -m uvicorn system_agent.server:app \
    --host 127.0.0.1 --port 6000 --log-level info \
    >"$RUN_DIR/logs/system_agent.log" 2>&1 &
SERVICE_PIDS+=("$!")

for port in 6002 6001 6000; do
    healthy=false
    for _ in $(seq 1 90); do
        if curl --noproxy '*' -fsS --max-time 2 \
            "http://127.0.0.1:${port}/health" \
            >"$RUN_DIR/logs/health_${port}.json"; then
            healthy=true
            break
        fi
        sleep 1
    done
    if [[ "$healthy" != true ]]; then
        echo "Service on port ${port} did not become healthy." >&2
        exit 1
    fi
done

echo "Run directory: $RUN_DIR"
echo "Strict slice: records[$((SHARD_START - 1)):${SHARD_END}] (global_index ${SHARD_START}-${SHARD_END})"
echo "Concurrency: 1"
echo "Orchestrator system-agent wait timeout: ${SYSTEM_AGENT_RUN_TIMEOUT_SECONDS}s"

python scripts/run_official_shard_with_timeout.py \
    --run-dir "$RUN_DIR" \
    --start "$SHARD_START" \
    --end "$SHARD_END" \
    --input "$PROJECT_DIR/bird-interact-full/bird_interact_data.jsonl" \
    2>&1 | tee -a "$RUN_DIR/logs/orchestrator.log"
