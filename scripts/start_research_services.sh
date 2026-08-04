#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/research_env.sh"

PYTHON_BIN="$PROJECT_DIR/.venv-research/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Research virtual environment missing: $PYTHON_BIN" >&2
    exit 1
fi
if ! pg_isready -h 127.0.0.1 -p 6433 -U root >/dev/null; then
    echo "Research PostgreSQL is not ready. Run scripts/start_research_db.sh first." >&2
    exit 1
fi
for port in 6100 6101 6102; do
    if ss -ltn | grep -q ":${port} "; then
        echo "Research port already occupied: $port" >&2
        exit 1
    fi
done

RUNTIME_DIR="$PROJECT_DIR/research-runtime"
mkdir -p "$RUNTIME_DIR/logs"
PID_FILE="$RUNTIME_DIR/service_pids"
if [[ -f "$PID_FILE" ]]; then
    echo "Existing research PID file found; run scripts/stop_research_services.sh first." >&2
    exit 1
fi

"$PYTHON_BIN" -m uvicorn db_environment.server:app \
    --host 127.0.0.1 --port 6102 --log-level info \
    >"$RUNTIME_DIR/logs/db_environment.log" 2>&1 &
db_pid=$!
"$PYTHON_BIN" -m uvicorn user_simulator.server:app \
    --host 127.0.0.1 --port 6101 --log-level info \
    >"$RUNTIME_DIR/logs/user_simulator.log" 2>&1 &
user_pid=$!
"$PYTHON_BIN" -m uvicorn system_agent.server:app \
    --host 127.0.0.1 --port 6100 --log-level info \
    >"$RUNTIME_DIR/logs/system_agent.log" 2>&1 &
agent_pid=$!
printf '%s\n' "$db_pid" "$user_pid" "$agent_pid" >"$PID_FILE"

for port in 6102 6101 6100; do
    ready=false
    for _ in $(seq 1 60); do
        if curl --noproxy '*' -fsS --max-time 2 \
            "http://127.0.0.1:${port}/health" >/dev/null; then
            ready=true
            break
        fi
        sleep 1
    done
    if [[ "$ready" != true ]]; then
        echo "Research service failed health check on port $port" >&2
        "$PROJECT_DIR/scripts/stop_research_services.sh" || true
        exit 1
    fi
done

echo "Research services ready: System Agent 6100, User Simulator 6101, DB Environment 6102"
