#!/usr/bin/env bash

research_block_production_script() {
    local project_dir
    project_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
    if [[ -f "$project_dir/.research-environment" ]]; then
        echo "Blocked production runner inside isolated research copy: $1" >&2
        echo "Use scripts/start_research_services.sh or an explicitly reviewed research command." >&2
        exit 86
    fi
}
