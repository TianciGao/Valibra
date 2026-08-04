# Isolated Agent Research Environment

This working tree is a byte-verified source snapshot of the active
`BIRD-Interact-ADK` framework. Its baseline is Git commit
`ff6226a7a082ee1f8f49c1abc3328a42b0da2d69`.

## Isolation boundaries

- Production evaluation directory: `/home/user/code/BIRD-Interact/BIRD-Interact-ADK`
- Research directory: `/home/user/code/BIRD-Interact/BIRD-Interact-ADK-agent-research`
- Production HTTP ports: `6000`, `6001`, `6002`
- Research HTTP ports: `6100`, `6101`, `6102`
- Production Full PostgreSQL: `127.0.0.1:5433`
- Research Full PostgreSQL: `127.0.0.1:6433`
- Research container: `bird_interact_postgresql_full_research`
- Research volume: `bird_interact_adk_agent_research_postgresql_data_full`
- Full task data: independent local copy under `bird-interact-full/`

The research `.env` contains dummy credentials and sends both model roles to
`127.0.0.1:9`. Paid provider calls are therefore disabled by default.
Provider credentials and frozen model-field variables inherited from a parent
shell are removed by `scripts/research_env.sh`; localhost is also forced into
`NO_PROXY`.

The baseline source snapshot, dependency lock, checksums, and validation record
are under `baseline/`. Historical `results/`, the production virtual
environment, Git metadata, caches, and the production `.env` were intentionally
not copied. They are not needed to change or test the agent framework and would
couple the research tree to the running evaluation.

The original framework used a symlink for `bird-interact-full`. The research
tree replaces it with a checksum-verified independent copy (98 files, about
5.4 MiB), so even an accidental research write cannot modify the production
task files.

## First use

```bash
cd /home/user/code/BIRD-Interact/BIRD-Interact-ADK-agent-research
source scripts/research_env.sh
bash scripts/research_preflight.sh
```

Start the independent database only when DB research is required:

```bash
bash scripts/start_research_db.sh
bash scripts/start_research_services.sh
```

Stop only research-owned resources:

```bash
bash scripts/stop_research_services.sh
bash scripts/stop_research_db.sh
```

Do not copy production API keys into this directory. Provider smoke tests must
be explicitly reviewed and configured for a single command.
