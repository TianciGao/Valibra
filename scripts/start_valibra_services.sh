#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/research_env.sh"

# V0 has its own HTTP port; the two downstream Research services retain their
# frozen isolation ports from research_env.sh.
export SYSTEM_AGENT_PORT=6110

PYTHON_BIN="$PROJECT_DIR/.venv-research/bin/python"
RUNTIME_DIR="$PROJECT_DIR/research-runtime/valibra"
PID_FILE="$RUNTIME_DIR/service_pids"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Research virtual environment missing: $PYTHON_BIN" >&2
    exit 1
fi
if ! pg_isready -h 127.0.0.1 -p 6433 -U root >/dev/null; then
    echo "Research PostgreSQL is not ready. Run scripts/start_research_db.sh first." >&2
    exit 1
fi
for port in 6110 6101 6102; do
    if ss -ltnH "sport = :$port" | grep -q .; then
        echo "Research port already occupied: $port" >&2
        exit 1
    fi
done
if [[ -e "$PID_FILE" ]]; then
    echo "Existing Valibra PID file found; run scripts/stop_valibra_services.sh first." >&2
    exit 1
fi

mkdir -p "$RUNTIME_DIR/logs"

start_service() {
    local service="$1"
    local module="$2"
    local port="$3"
    local log_file="$4"

    "$PYTHON_BIN" -m uvicorn "$module" \
        --host 127.0.0.1 --port "$port" --log-level info \
        >"$log_file" 2>&1 &
    printf '%s\t%s\t%s\t%s\n' "$service" "$!" "$module" "$port" >>"$PID_FILE"
}

start_service \
    "db_environment" "db_environment.server:app" "6102" \
    "$RUNTIME_DIR/logs/db_environment.log"
start_service \
    "user_simulator" "user_simulator.server:app" "6101" \
    "$RUNTIME_DIR/logs/user_simulator.log"
start_service \
    "valibra_agent" "valibra_agent.server:app" "6110" \
    "$RUNTIME_DIR/logs/valibra_agent.log"

cleanup_on_error() {
    "$PROJECT_DIR/scripts/stop_valibra_services.sh" || true
}
trap cleanup_on_error ERR

for port in 6102 6101 6110; do
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
        echo "Valibra service failed health check on port $port" >&2
        false
    fi
done

trap - ERR
echo "Valibra V0 services ready: Valibra 6110, User Simulator 6101, DB Environment 6102"
