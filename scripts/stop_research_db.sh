#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
docker compose \
    -p bird_interact_adk_agent_research \
    -f docker-compose.research.yml \
    --profile full down
echo "Research PostgreSQL stopped; its named volume was preserved."
