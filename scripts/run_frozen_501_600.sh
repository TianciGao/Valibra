#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/research_production_guard.sh"
research_block_production_script "${BASH_SOURCE[0]}"

PROJECT_DIR="/home/user/code/BIRD-Interact/BIRD-Interact-ADK"
cd "$PROJECT_DIR"

PYTHON_BIN="$PROJECT_DIR/.venv-adk/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment missing: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -z "${MODEL_PRESET:-}" ]]; then
    echo "MODEL_PRESET is required." >&2
    exit 2
fi
case "$MODEL_PRESET" in
    glm52_high_32768|glm52_max_65536|glm47_matched_32768) ;;
    *)
        echo "Unsupported MODEL_PRESET: $MODEL_PRESET" >&2
        exit 2
        ;;
esac

CONCURRENCY="${CONCURRENCY:-5}"
if [[ ! "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONCURRENCY must be a positive integer: $CONCURRENCY" >&2
    exit 2
fi
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1

# Freeze every non-model benchmark setting. Model fields come exclusively from
# MODEL_PRESET and are conflict-checked by shared.config.
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

PREPARED_DIR="$PROJECT_DIR/prepared_tasks/full_0501_0600"
TASK_JSONL="$PREPARED_DIR/tasks_0501_0600.jsonl"
TASK_MANIFEST="$PREPARED_DIR/manifest.json"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/results/frozen_501_600_${MODEL_PRESET}_c${CONCURRENCY}_${RUN_STAMP}}"
if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$PROJECT_DIR/$RUN_DIR"
fi
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty RUN_DIR: $RUN_DIR" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/logs"
RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON_BIN" scripts/prepare_tasks_0501_0600.py \
    --output-dir "$PREPARED_DIR" \
    >"$RUN_DIR/logs/task_preparation.json"

cp "$PROJECT_DIR/configs/model_presets/${MODEL_PRESET}.json" \
    "$RUN_DIR/model_preset.json"
"$PYTHON_BIN" -m pip freeze >"$RUN_DIR/dependencies.freeze.txt"

# This is offline: it constructs the ADK model and agent but makes no provider
# or service request. It prints the normalized config and both preset SHAs.
"$PYTHON_BIN" scripts/dry_run_model_preset.py \
    --output "$RUN_DIR/model_dry_run.json" \
    | tee "$RUN_DIR/logs/model_dry_run.log"

metadata_args=(
    --run-dir "$RUN_DIR"
    --task-manifest "$TASK_MANIFEST"
    --dependency-freeze "$RUN_DIR/dependencies.freeze.txt"
    --concurrency "$CONCURRENCY"
    --started-at "$RUN_STARTED_AT"
)
if [[ "$DRY_RUN" == "1" ]]; then
    metadata_args+=(--dry-run)
fi
"$PYTHON_BIN" scripts/write_frozen_run_metadata.py "${metadata_args[@]}" \
    >"$RUN_DIR/logs/run_manifest_stdout.json"

echo "MODEL_PRESET=$MODEL_PRESET"
"$PYTHON_BIN" - <<'PY'
import json
from shared.config import active_model_preset_report
report = active_model_preset_report()
print("Normalized model config:")
print(json.dumps(report["normalized_config"], ensure_ascii=False, sort_keys=True))
print(f"Normalized config SHA256: {report['normalized_sha256']}")
print(f"Preset file SHA256: {report['file_sha256']}")
PY
echo "Concurrency: $CONCURRENCY"
echo "Result directory: $RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1: preparation complete; no services or provider were called."
    exit 0
fi

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

"$PYTHON_BIN" -m uvicorn db_environment.server:app \
    --host 127.0.0.1 --port 6002 --log-level info \
    >"$RUN_DIR/logs/db_environment.log" 2>&1 &
SERVICE_PIDS+=("$!")

"$PYTHON_BIN" -m uvicorn user_simulator.server:app \
    --host 127.0.0.1 --port 6001 --log-level info \
    >"$RUN_DIR/logs/user_simulator.log" 2>&1 &
SERVICE_PIDS+=("$!")

"$PYTHON_BIN" -m uvicorn system_agent.server:app \
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
    if [[ "$healthy" != "true" ]]; then
        echo "Service on port $port did not become healthy." >&2
        exit 1
    fi
done

set +e
"$PYTHON_BIN" -m orchestrator.runner \
    --mode a-interact \
    --data "$TASK_JSONL" \
    --concurrency "$CONCURRENCY" \
    --output "$RUN_DIR/result_private.json" \
    2>&1 | tee "$RUN_DIR/logs/orchestrator.log"
runner_status=${PIPESTATUS[0]}
set -e

if [[ -f "$RUN_DIR/result_private.json" ]]; then
    "$PYTHON_BIN" -m orchestrator.export_submission \
        "$RUN_DIR/result_private.json" \
        "$RUN_DIR/submission_predictions.jsonl"
    "$PYTHON_BIN" -m orchestrator.audit_summary \
        "$RUN_DIR/result_private.json" \
        "$RUN_DIR/analysis_summary.md"
fi
printf '%s\n' "$runner_status" >"$RUN_DIR/runner_exit_status.txt"
exit "$runner_status"
