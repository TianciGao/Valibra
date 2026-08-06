#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv-research/bin/python"
PID_FILE="$PROJECT_DIR/research-runtime/valibra/service_pids"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No Valibra service PID file; nothing to stop."
    exit 0
fi

declare -A expected_modules=(
    [db_environment]="db_environment.server:app"
    [user_simulator]="user_simulator.server:app"
    [valibra_agent]="valibra_agent.server:app"
)
declare -A expected_ports=(
    [db_environment]="6102"
    [user_simulator]="6101"
    [valibra_agent]="6110"
)
declare -a validated_pids=()
declare -A seen_services=()
validation_failed=false

while IFS=$'\t' read -r service pid module port extra; do
    if [[ -z "$service" || -z "$pid" || -z "$module" || -z "$port" || -n "${extra:-}" ]]; then
        echo "Malformed Valibra PID entry: $service $pid $module $port ${extra:-}" >&2
        validation_failed=true
        continue
    fi
    if [[ -z "${expected_modules[$service]+x}" || "${seen_services[$service]:-}" == true ]]; then
        echo "Unexpected or duplicate Valibra PID service: $service" >&2
        validation_failed=true
        continue
    fi
    seen_services[$service]=true
    if [[ "$module" != "${expected_modules[$service]}" || "$port" != "${expected_ports[$service]}" ]]; then
        echo "Valibra PID metadata mismatch for $service" >&2
        validation_failed=true
        continue
    fi
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        echo "Invalid Valibra PID for $service: $pid" >&2
        validation_failed=true
        continue
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Valibra process already stopped: $service PID $pid"
        continue
    fi

    cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    if [[ "$state" == Z* ]]; then
        echo "Valibra process is defunct: $service PID $pid"
        continue
    fi
    if [[ "$cmdline" != *"$PYTHON_BIN -m uvicorn $module"* || "$cmdline" != *"--port $port"* ]]; then
        echo "Refusing to stop PID $pid with unexpected command: $cmdline" >&2
        validation_failed=true
        continue
    fi
    if [[ "$cwd" != "$PROJECT_DIR" ]]; then
        echo "Refusing to stop PID $pid with unexpected cwd: $cwd" >&2
        validation_failed=true
        continue
    fi
    validated_pids+=("$pid")
done <"$PID_FILE"

for service in "${!expected_modules[@]}"; do
    if [[ "${seen_services[$service]:-}" != true ]]; then
        echo "Missing Valibra PID entry for $service" >&2
        validation_failed=true
    fi
done
if [[ "$validation_failed" == true ]]; then
    echo "Valibra stop aborted before signaling any process." >&2
    exit 1
fi

for pid in "${validated_pids[@]}"; do
    kill -TERM "$pid"
done
for pid in "${validated_pids[@]}"; do
    for _ in $(seq 1 100); do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
        [[ "$state" == Z* ]] && break
        sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
        state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
        if [[ "$state" != Z* ]]; then
            echo "Valibra PID did not stop after SIGTERM: $pid" >&2
            exit 1
        fi
    fi
done

rm -f "$PID_FILE"
echo "Valibra V0 services stopped; production services were not touched."
