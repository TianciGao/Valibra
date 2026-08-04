#!/usr/bin/env bash
# Run BIRD-Interact evaluation.
# Usage:
#   bash scripts/run_eval.sh --mode a-interact --concurrency 3
#   bash scripts/run_eval.sh --mode c-interact --concurrency 5
#   bash scripts/run_eval.sh --mode oracle --concurrency 5
#   bash scripts/run_eval.sh --mode a-interact --limit 10

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

PYTHON_BIN="python"
if [ -x "$PROJECT_DIR/.conda-py310/bin/python" ]; then
    PYTHON_BIN="$PROJECT_DIR/.conda-py310/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "$PROJECT_DIR/.venv-adk/bin/python" ]; then
    PYTHON_BIN="$PROJECT_DIR/.venv-adk/bin/python"
fi

# Start services if not running
if ! curl --noproxy '*' -s "http://127.0.0.1:6000/health" > /dev/null 2>&1; then
    echo "Starting services..."
    bash "$PROJECT_DIR/scripts/start_services.sh"
fi

# Run evaluation
"$PYTHON_BIN" -m orchestrator.runner "$@"
