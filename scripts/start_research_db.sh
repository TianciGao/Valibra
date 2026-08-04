#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
docker compose \
    -p bird_interact_adk_agent_research \
    -f docker-compose.research.yml \
    --profile full up -d postgresql_full_research
echo "Research PostgreSQL: 127.0.0.1:6433"
