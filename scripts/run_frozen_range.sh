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
case "${MODEL_PRESET:-}" in
    glm47_matched_32768|glm52_high_32768|glm52_max_65536) ;;
    *)
        echo "Unsupported frozen MODEL_PRESET: ${MODEL_PRESET:-<unset>}" >&2
        exit 2
        ;;
esac
if [[ "${CONCURRENCY:-1}" != "1" ]]; then
    echo "CONCURRENCY must remain 1 for the frozen comparison" >&2
    exit 2
fi
if [[ -z "${RANGE_START:-}" || -z "${RANGE_END:-}" ]]; then
    echo "RANGE_START and RANGE_END are required" >&2
    exit 2
fi
if [[ ! "$RANGE_START" =~ ^[0-9]+$ || ! "$RANGE_END" =~ ^[0-9]+$ ]]; then
    echo "RANGE_START and RANGE_END must be integers" >&2
    exit 2
fi
RANGE_COUNT=$((RANGE_END - RANGE_START + 1))
if (( RANGE_START < 1 || RANGE_END > 600 )) || \
   [[ "$RANGE_COUNT" != "50" && "$RANGE_COUNT" != "100" ]]; then
    echo "The frozen runner requires a contiguous 50- or 100-task range" >&2
    exit 2
fi

USER_SIM_PROFILE="${USER_SIM_PROFILE:-claude_haiku_4_5_gptsapi}"
case "$USER_SIM_PROFILE" in
    claude_haiku_4_5_gptsapi)
        export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
        export USER_SIM_API_BASE=https://api.gptsapi.net
        export USER_SIM_DISABLE_THINKING=true
        export USER_SIM_USE_BEARER_FOR_CUSTOM_BASE=true
        export USER_SIM_PROTOCOL_POLICY=official
        export USER_SIM_PROTOCOL_MAX_ATTEMPTS=1
        ;;
    claude_haiku_4_5_official)
        export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
        export USER_SIM_API_BASE=https://api.anthropic.com
        export USER_SIM_DISABLE_THINKING=true
        export USER_SIM_USE_BEARER_FOR_CUSTOM_BASE=false
        export USER_SIM_PROTOCOL_POLICY=official
        export USER_SIM_PROTOCOL_MAX_ATTEMPTS=1
        ;;
    gpt4o_gptsapi)
        export USER_SIM_MODEL=openai/gpt-4o
        export USER_SIM_API_BASE=https://api.gptsapi.net
        # GPT-4o has no provider hidden-thinking field; omit the Anthropic-only
        # compatibility payload while retaining the unchanged v2 prompt.
        export USER_SIM_DISABLE_THINKING=false
        export USER_SIM_USE_BEARER_FOR_CUSTOM_BASE=false
        # GPT-4o occasionally returns the correct semantic response without
        # the official <s>...</s> envelope.  Retry only that transport error;
        # never replace it silently with a scored fallback answer.
        export USER_SIM_PROTOCOL_POLICY=strict_retry
        export USER_SIM_PROTOCOL_MAX_ATTEMPTS=3
        ;;
    *)
        echo "Unsupported USER_SIM_PROFILE: $USER_SIM_PROFILE" >&2
        exit 2
        ;;
esac
export USER_SIM_PROFILE

export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export DATASET=full
export PROMPT_VERSION=v2
export PATIENCE=3
export PG_HOST=127.0.0.1
export PG_PORT=5433
export SYSTEM_AGENT_PORT=6000
export USER_SIM_PORT=6001
export DB_ENV_PORT=6002
export CONCURRENCY=1

RANGE_LABEL="$(printf '%04d_%04d' "$RANGE_START" "$RANGE_END")"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/results/frozen_${RANGE_LABEL}_${MODEL_PRESET}_c1_${RUN_STAMP}}"
if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$PROJECT_DIR/$RUN_DIR"
fi

if [[ -f "$RUN_DIR/checkpoint.json" ]]; then
    IS_RESUME=1
else
    IS_RESUME=0
    if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing non-empty RUN_DIR without checkpoint: $RUN_DIR" >&2
        exit 1
    fi
fi
mkdir -p "$RUN_DIR/logs"

if [[ "$IS_RESUME" == "0" ]]; then
    PREPARED_DIR="$RUN_DIR/prepared_tasks"
    "$PYTHON_BIN" scripts/prepare_task_range.py \
        --start "$RANGE_START" \
        --end "$RANGE_END" \
        --output-dir "$PREPARED_DIR" \
        >"$RUN_DIR/logs/task_preparation.json"

    cp "$PROJECT_DIR/configs/model_presets/${MODEL_PRESET}.json" \
        "$RUN_DIR/model_preset.json"
    "$PYTHON_BIN" -m pip freeze >"$RUN_DIR/dependencies.freeze.txt"
    "$PYTHON_BIN" scripts/dry_run_model_preset.py \
        --output "$RUN_DIR/model_dry_run.json" \
        | tee "$RUN_DIR/logs/model_dry_run.log"
    "$PYTHON_BIN" scripts/write_frozen_run_metadata.py \
        --run-dir "$RUN_DIR" \
        --task-manifest "$PREPARED_DIR/manifest.json" \
        --dependency-freeze "$RUN_DIR/dependencies.freeze.txt" \
        --concurrency 1 \
        --started-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$RUN_DIR/logs/run_manifest_stdout.json"
else
    "$PYTHON_BIN" - "$RUN_DIR" "$RANGE_START" "$RANGE_END" \
        "$USER_SIM_PROFILE" "$MODEL_PRESET" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_range = [int(sys.argv[2]), int(sys.argv[3])]
expected_user_sim_profile = sys.argv[4]
expected_model_preset = sys.argv[5]
checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
summary = json.loads((run_dir / "summary.json").read_text())
if checkpoint.get("global_index_range") != expected_range:
    raise SystemExit("Existing checkpoint range does not match requested range")
if summary.get("configuration", {}).get("model_preset", {}).get("name") != (
    expected_model_preset
):
    raise SystemExit("Existing checkpoint uses a different frozen model preset")
if summary.get("configuration", {}).get("user_simulator", {}).get("profile") != (
    expected_user_sim_profile
):
    raise SystemExit("Existing checkpoint uses a different User Simulator profile")
PY
fi

echo "MODEL_PRESET=$MODEL_PRESET"
echo "USER_SIM_PROFILE=$USER_SIM_PROFILE"
"$PYTHON_BIN" - <<'PY'
import json
from shared.config import active_model_preset_report

report = active_model_preset_report()
print("Normalized model config:")
print(json.dumps(report["normalized_config"], ensure_ascii=False, sort_keys=True))
print(f"Normalized config SHA256: {report['normalized_sha256']}")
print(f"Preset file SHA256: {report['file_sha256']}")
PY
echo "Range: global_index $RANGE_START-$RANGE_END"
echo "Concurrency: 1"
echo "Result directory: $RUN_DIR"

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

RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_PREFIX="$RUN_DIR/logs/run_$(date -u +%Y%m%dT%H%M%SZ)"
SERVICE_PIDS=()

collect_logs_and_stop() {
    local exit_code=$?
    trap - EXIT INT TERM
    docker logs --since "$RUN_STARTED_AT" bird_interact_postgresql_full \
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

set +e
"$PYTHON_BIN" -m orchestrator.official_shard_runner \
    --run-dir "$RUN_DIR" \
    --start "$RANGE_START" \
    --end "$RANGE_END" \
    --input "$PROJECT_DIR/bird-interact-full/bird_interact_data.jsonl" \
    2>&1 | tee "${LOG_PREFIX}_official_runner.log"
runner_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$runner_status" >"${LOG_PREFIX}_runner_exit_status.txt"
exit "$runner_status"
