#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv-adk/bin/python"
RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${1:-$PROJECT_ROOT/results/audit_glm52_full_${RUN_STAMP}}"
LOG_DIR="$RUN_DIR/logs"
RESULT_JSON="$RUN_DIR/result.json"
DATA_PATH="$PROJECT_ROOT/bird-interact-full/bird_interact_data.jsonl"

mkdir -p "$LOG_DIR"

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export DATASET=full
export PG_PORT=5433
export SYSTEM_AGENT_PORT=6000
export USER_SIM_PORT=6001
export DB_ENV_PORT=6002
export SYSTEM_AGENT_MODEL=openai/glm-5.2
export SYSTEM_AGENT_THINKING=enabled
export SYSTEM_AGENT_REASONING_EFFORT=max
export SYSTEM_AGENT_CLEAR_THINKING=false
export SYSTEM_AGENT_MAX_TOKENS=65536
export SYSTEM_AGENT_TEMPERATURE=0
export SYSTEM_AGENT_TOOL_CHOICE=auto
export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
export USER_SIM_DISABLE_THINKING=true
export PROMPT_VERSION=v2
export PATIENCE=3

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Full dataset missing: $DATA_PATH" >&2
  exit 1
fi
if ! pg_isready -h 127.0.0.1 -p 5433 -U root >/dev/null; then
  echo "Full PostgreSQL is not accepting connections on 127.0.0.1:5433" >&2
  exit 1
fi

for port in 6000 6001 6002; do
  if ss -ltn "sport = :$port" | sed -n '2p' | grep -q .; then
    echo "Port $port is already in use; refusing to disturb an existing service." >&2
    exit 1
  fi
done

"$PYTHON_BIN" -m scripts.write_audit_manifest \
  --run-dir "$RUN_DIR" \
  --data "$DATA_PATH" \
  --limit 3 \
  --concurrency 1 \
  --started-at "$RUN_STARTED_AT"

docker logs --timestamps bird_interact_postgresql_full \
  >"$LOG_DIR/postgresql_full_before.log" 2>&1 || true

system_pid=""
user_pid=""
db_pid=""

cleanup() {
  for pid in "$system_pid" "$user_pid" "$db_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$system_pid" "$user_pid" "$db_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  docker logs --timestamps bird_interact_postgresql_full \
    >"$LOG_DIR/postgresql_full_after.log" 2>&1 || true
}
trap cleanup EXIT INT TERM

"$PYTHON_BIN" -m uvicorn db_environment.server:app \
  --host 127.0.0.1 --port 6002 --log-level debug \
  >"$LOG_DIR/db_environment.log" 2>&1 &
db_pid=$!

"$PYTHON_BIN" -m uvicorn user_simulator.server:app \
  --host 127.0.0.1 --port 6001 --log-level debug \
  >"$LOG_DIR/user_simulator.log" 2>&1 &
user_pid=$!

"$PYTHON_BIN" -m uvicorn system_agent.server:app \
  --host 127.0.0.1 --port 6000 --log-level debug \
  >"$LOG_DIR/system_agent.log" 2>&1 &
system_pid=$!

wait_for_health() {
  local name="$1"
  local url="$2"
  local output="$3"
  for _ in $(seq 1 60); do
    if curl --noproxy '*' --fail --silent --show-error "$url" >"$output"; then
      return 0
    fi
    sleep 1
  done
  echo "$name did not become healthy: $url" >&2
  return 1
}

wait_for_health "DB Environment" "http://127.0.0.1:6002/health" "$RUN_DIR/health_db_environment.json"
wait_for_health "User Simulator" "http://127.0.0.1:6001/health" "$RUN_DIR/health_user_simulator.json"
wait_for_health "System Agent" "http://127.0.0.1:6000/health" "$RUN_DIR/health_system_agent.json"

set +e
"$PYTHON_BIN" -m orchestrator.runner \
  --mode a-interact \
  --data "$DATA_PATH" \
  --limit 3 \
  --concurrency 1 \
  --output "$RESULT_JSON" \
  2>&1 | tee "$LOG_DIR/orchestrator.log"
runner_status=${PIPESTATUS[0]}
set -e

if [[ -f "$RESULT_JSON" ]]; then
  "$PYTHON_BIN" -m orchestrator.export_submission \
    "$RESULT_JSON" "$RUN_DIR/submission_predictions.jsonl"
  "$PYTHON_BIN" -m orchestrator.audit_summary \
    "$RESULT_JSON" "$RUN_DIR/analysis_summary.md"
  "$PYTHON_BIN" -m orchestrator.report \
    "$RESULT_JSON" \
    >"$LOG_DIR/report.log" 2>&1 || true
fi

printf '%s\n' "$runner_status" >"$RUN_DIR/runner_exit_status.txt"
exit "$runner_status"
