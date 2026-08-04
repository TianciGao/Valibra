#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/research_env.sh"
PYTHON_BIN="$PROJECT_DIR/.venv-research/bin/python"

[[ -x "$PYTHON_BIN" ]] || { echo "Research venv missing" >&2; exit 1; }
[[ "$(git -C "$PROJECT_DIR" rev-parse --show-toplevel)" == "$PROJECT_DIR" ]] || {
    echo "Research Git root mismatch" >&2; exit 1;
}

"$PYTHON_BIN" - <<'PY'
from shared.config import PROJECT_ROOT, active_model_preset_report, settings

assert PROJECT_ROOT.name == "BIRD-Interact-ADK-agent-research", PROJECT_ROOT
assert settings.pg_port == 6433
assert settings.system_agent_port == 6100
assert settings.user_sim_port == 6101
assert settings.db_env_port == 6102
assert settings.system_agent_api_base == "http://127.0.0.1:9"
assert settings.user_sim_api_base == "http://127.0.0.1:9"
assert settings.system_agent_api_key == "research-offline"
assert settings.user_sim_api_key == "research-offline"
report = active_model_preset_report()
assert report and report["name"] == "glm52_high_32768"
print("python_config_isolation=passed")
print(f"project_root={PROJECT_ROOT}")
print(f"model_preset={report['name']}")
print(f"ports={settings.system_agent_port},{settings.user_sim_port},{settings.db_env_port}")
print(f"postgres={settings.pg_host}:{settings.pg_port}")
PY

compose_config="$(docker compose -f "$PROJECT_DIR/docker-compose.research.yml" --profile full config)"
grep -q 'bird_interact_postgresql_full_research' <<<"$compose_config"
grep -q 'published: "6433"' <<<"$compose_config"
if grep -q 'published: "5433"' <<<"$compose_config"; then
    echo "Research compose unexpectedly publishes production DB port" >&2
    exit 1
fi

for port in 6000 6001 6002; do
    owner="$(ss -ltnp 2>/dev/null | grep ":${port} " || true)"
    if [[ "$owner" == *"BIRD-Interact-ADK-agent-research"* ]]; then
        echo "Research process found on production port $port" >&2
        exit 1
    fi
done

echo "docker_isolation=passed"
echo "production_port_ownership=passed"
echo "research_preflight=passed"
