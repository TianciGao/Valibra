# Valibra

A research framework for grounded, interactive Text-to-SQL, built on BIRD-Interact-ADK.

[Report — Chinese](https://tiancigao.github.io/Valibra-site/) · [Report — Russian](https://tiancigao.github.io/Valibra-site/ru/) · [Documentation](docs/README.md) · [Getting Started](docs/getting-started.md)

## Overview

Valibra separates evidence gathering from SQL generation. It maintains a structured grounding state covering **tables, join keys, column mappings, and domain knowledge**, with user clarification and budget-constrained evidence retrieval.

Structure, Mapping, and Knowledge build this state. Check assesses whether it supports SQL generation; Gate routes evidence-driven revisions through an isolated draft that is committed only after validation. The SQL-writing agent then executes and submits queries using the checked state. If the primary workflow cannot complete the task, an independent BIRD-Interact agent may attempt recovery within the remaining budget.

See the [architecture guide](docs/architecture.md) for stage responsibilities, state transitions, and recovery conditions.

## Evaluation

Archived comparison on the same **600 BIRD-Interact Full tasks**, using **a-interact** with **GLM-5.2** and a **Claude Haiku 4.5** user simulator.

| Metric | BIRD-Interact agent baseline | Valibra |
| --- | ---: | ---: |
| Phase 1 passed | 144 / 600 (24.00%) | 150 / 600 (25.00%) |
| Both phases passed | 79 / 600 (13.17%) | 75 / 600 (12.50%) |
| Total reward | 124.5 | 127.5 |
| Reported total tokens | 82,272,142 | 103,258,480 |

Reward is `0.7 × Phase 1 passes + 0.3 × both-phase passes`. Token totals include grounding, SQL generation, recovery, and user simulation.

The reward gain is modest: full-task completion declined and reported token use increased by 25.5%. These results do not establish an overall performance or cost advantage. They describe the archived evaluation, not a new benchmark run of subsequent code changes.

[Evaluation protocol and limitations](docs/releases/2026-09-16/README.md) · [Results JSON](docs/releases/2026-09-16/core_results.json) · [Configuration manifest](docs/releases/2026-09-16/candidate_manifest.json)

## Installation

The commands below use Python 3.12 and a Linux / WSL shell.

```bash
git clone --branch research/sql-grounding-v1 https://github.com/TianciGao/Valibra.git
cd Valibra
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest pytest-subtests
```

Run the offline test suite:

```bash
python -m pytest -q -p no:cacheprovider tests
```

Live evaluation additionally requires the benchmark data, PostgreSQL databases, and model credentials. Follow the [setup guide](docs/getting-started.md) before starting services or running paid model calls. See [test documentation](tests/README.md) for optional dependencies and skipped checks.

## Code Organization

- [`valibra_agent/`](valibra_agent/): grounding, state validation, SQL generation, and recovery.
- [`system_agent/`](system_agent/): upstream agent implementation.
- [`db_environment/`](db_environment/), [`user_simulator/`](user_simulator/), [`orchestrator/`](orchestrator/): execution environment, interaction, and evaluation.
- [`tests/`](tests/), [`docs/`](docs/README.md): tests, technical documentation, and release results.

## License and Acknowledgments

Released under the [MIT License](LICENSE). Built on BIRD-Interact-ADK and the [BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) benchmark. Valibra's experimental results are separate from the upstream benchmark results.

See the [upstream attribution and citation](docs/upstream/README.md#citation).
