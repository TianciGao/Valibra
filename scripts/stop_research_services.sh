#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_DIR/research-runtime/service_pids"
if [[ ! -f "$PID_FILE" ]]; then
    echo "No research service PID file; nothing to stop."
    exit 0
fi

mapfile -t pids <"$PID_FILE"
for pid in "${pids[@]}"; do
    cmdline="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    if [[ "$cmdline" == *"$PROJECT_DIR/.venv-research/bin/python"*uvicorn* ]]; then
        kill "$pid" 2>/dev/null || true
    elif [[ -n "$cmdline" ]]; then
        echo "Refusing to stop unrelated PID $pid: $cmdline" >&2
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done
rm -f "$PID_FILE"
echo "Research services stopped; production services were not touched."
