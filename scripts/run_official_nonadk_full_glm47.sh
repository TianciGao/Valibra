#!/usr/bin/env bash
set -Eeuo pipefail

ADK_ROOT="/home/user/code/BIRD-Interact/BIRD-Interact-ADK"
OFFICIAL_ROOT="/home/user/code/BIRD-Interact/bird_interact_agent"
PYTHON_BIN="$ADK_ROOT/.venv-adk/bin/python"

cd "$ADK_ROOT"
if [[ "${MODEL_PRESET:-}" != "glm47_matched_32768" ]]; then
    echo "MODEL_PRESET must be glm47_matched_32768" >&2
    exit 2
fi
if [[ "${CONCURRENCY:-1}" != "1" ]]; then
    echo "CONCURRENCY must remain 1 for this frozen run" >&2
    exit 2
fi

export PYTHONPATH="$OFFICIAL_ROOT:$ADK_ROOT"
export PYTHONUNBUFFERED=1
export BIRD_ADK_ROOT="$ADK_ROOT"
export BIRD_NONADK_FROZEN_PROVIDER=1
export DATASET=full
export USER_SIM_MODEL=anthropic/claude-haiku-4-5-20251001
export USER_SIM_DISABLE_THINKING=true
export PROMPT_VERSION=v2
export PATIENCE=3
export PG_HOST=127.0.0.1
export PG_PORT=5433
export CONCURRENCY=1
export P1_GATE_AFTER="${P1_GATE_AFTER:-100}"
export P1_GATE_THRESHOLD="${P1_GATE_THRESHOLD:-0.15}"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$ADK_ROOT/results/nonadk_official_glm47_full_0001_0600_${RUN_STAMP}}"
mkdir -p "$RUN_DIR/logs"

"$PYTHON_BIN" scripts/preflight_official_nonadk.py \
    --input "$ADK_ROOT/bird-interact-full/bird_interact_data.jsonl" \
    --official-root "$OFFICIAL_ROOT" \
    --output "$RUN_DIR/preflight.json"
FREEZE_PATH="$RUN_DIR/dependencies.freeze.txt"
FREEZE_TMP="$RUN_DIR/.dependencies.freeze.current"
"$PYTHON_BIN" -m pip freeze >"$FREEZE_TMP"
if [[ -f "$FREEZE_PATH" ]] && ! cmp -s "$FREEZE_PATH" "$FREEZE_TMP"; then
    rm -f "$FREEZE_TMP"
    echo "Dependency set changed; refusing to resume the frozen run" >&2
    exit 2
fi
if [[ ! -f "$FREEZE_PATH" ]]; then
    mv "$FREEZE_TMP" "$FREEZE_PATH"
else
    rm -f "$FREEZE_TMP"
fi

echo "Framework: official text-ReAct non-ADK with audited GLM transport"
echo "Range: records[0:600] / global_index 1-600"
echo "MODEL_PRESET=$MODEL_PRESET"
echo "Concurrency: 1"
echo "Result directory: $RUN_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: preflight and dependency freeze complete; no model was called."
    exit 0
fi

"$PYTHON_BIN" scripts/run_official_nonadk_range.py \
    --run-dir "$RUN_DIR" \
    --input "$ADK_ROOT/bird-interact-full/bird_interact_data.jsonl" \
    --official-root "$OFFICIAL_ROOT" \
    --start 1 \
    --end 600

RUN_STATUS="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/checkpoint.json")"
if [[ "$RUN_STATUS" == "complete" ]]; then
    "$PYTHON_BIN" scripts/validate_nonadk_submission.py \
        --submission "$RUN_DIR/submission_official.jsonl" \
        --manifest "$RUN_DIR/shard_manifest.jsonl" \
        --expected-count 600 \
        --start 1 \
        --end 600 \
        --report "$RUN_DIR/submission_validation.json"
else
    echo "Run stopped with checkpoint status: $RUN_STATUS"
    echo "Formal submission validation will run only after all 600 tasks complete."
fi
