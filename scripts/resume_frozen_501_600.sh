#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/home/user/code/BIRD-Interact/BIRD-Interact-ADK"
cd "$PROJECT_DIR"

PYTHON_BIN="$PROJECT_DIR/.venv-adk/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment missing: $PYTHON_BIN" >&2
    exit 1
fi
if [[ "${MODEL_PRESET:-}" != "glm47_matched_32768" ]]; then
    echo "MODEL_PRESET must be glm47_matched_32768" >&2
    exit 2
fi
if [[ -z "${RUN_DIR:-}" ]]; then
    echo "RUN_DIR must identify the existing frozen run" >&2
    exit 2
fi
if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$PROJECT_DIR/$RUN_DIR"
fi
if [[ ! -d "$RUN_DIR" ]]; then
    echo "RUN_DIR does not exist: $RUN_DIR" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export DATASET=full
export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
export USER_SIM_DISABLE_THINKING=true
export PROMPT_VERSION=v2
export PATIENCE=3
export PG_HOST=127.0.0.1
export PG_PORT=5433
export SYSTEM_AGENT_PORT=6000
export USER_SIM_PORT=6001
export DB_ENV_PORT=6002

if ! docker inspect -f '{{.State.Running}}' bird_interact_postgresql_full \
    2>/dev/null | grep -qx true; then
    echo "Full PostgreSQL container is not running." >&2
    exit 1
fi
if ! pg_isready -h 127.0.0.1 -p 5433 -U root >/dev/null; then
    echo "Full PostgreSQL is not ready on 127.0.0.1:5433." >&2
    exit 1
fi
for port in 6000 6001 6002; do
    if curl --noproxy '*' -fsS --max-time 2 \
        "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "Port $port already has a healthy service; refusing mixed run." >&2
        exit 1
    fi
done

RESUME_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESUME_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_PREFIX="$RUN_DIR/logs/resume_${RESUME_STAMP}"
SERVICE_PIDS=()

collect_logs_and_stop() {
    local exit_code=$?
    trap - EXIT INT TERM
    docker logs --since "$RESUME_STARTED_AT" bird_interact_postgresql_full \
        >>"${LOG_PREFIX}_postgresql_full.log" 2>&1 || true
    for pid in "${SERVICE_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${SERVICE_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$exit_code"
}
trap collect_logs_and_stop EXIT INT TERM

"$PYTHON_BIN" -m uvicorn db_environment.server:app \
    --host 127.0.0.1 --port 6002 --log-level info \
    >"${LOG_PREFIX}_db_environment.log" 2>&1 &
SERVICE_PIDS+=("$!")
"$PYTHON_BIN" -m uvicorn user_simulator.server:app \
    --host 127.0.0.1 --port 6001 --log-level info \
    >"${LOG_PREFIX}_user_simulator.log" 2>&1 &
SERVICE_PIDS+=("$!")
"$PYTHON_BIN" -m uvicorn system_agent.server:app \
    --host 127.0.0.1 --port 6000 --log-level info \
    >"${LOG_PREFIX}_system_agent.log" 2>&1 &
SERVICE_PIDS+=("$!")

for port in 6002 6001 6000; do
    healthy=false
    for _ in $(seq 1 90); do
        if curl --noproxy '*' -fsS --max-time 2 \
            "http://127.0.0.1:${port}/health" \
            >"${LOG_PREFIX}_health_${port}.json"; then
            healthy=true
            break
        fi
        sleep 1
    done
    if [[ "$healthy" != "true" ]]; then
        echo "Service on port $port did not become healthy." >&2
        exit 1
    fi
done

echo "Resuming frozen GLM-4.7 range in: $RUN_DIR"
echo "Full task range: records[500:600], global_index 501-600"
echo "Concurrency: 1"

set +e
"$PYTHON_BIN" -m orchestrator.official_shard_runner \
    --run-dir "$RUN_DIR" \
    --start 501 \
    --end 600 \
    --input "$PROJECT_DIR/bird-interact-full/bird_interact_data.jsonl" \
    2>&1 | tee "${LOG_PREFIX}_official_runner.log"
runner_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$runner_status" >"${LOG_PREFIX}_runner_exit_status.txt"
exit "$runner_status"
