<h1 align="center">Valibra</h1>

<p align="center">
  <strong>Structured Grounding for Interactive Text-to-SQL</strong><br>
  A research framework built on BIRD-Interact-ADK.
</p>

<p align="center">
  <a href="https://tiancigao.github.io/Valibra-site/">Chinese Report</a> &nbsp;·&nbsp;
  <a href="https://tiancigao.github.io/Valibra-site/ru/">Russian Report</a> &nbsp;·&nbsp;
  <a href="docs/architecture.md">Architecture</a> &nbsp;·&nbsp;
  <a href="docs/getting-started.md">Getting Started</a>
</p>

## Approach

Valibra separates evidence gathering from SQL generation. A shared grounding state records **tables, join keys, column mappings, and domain knowledge**.

1. **Ground.** Structure, Mapping, and Knowledge organize database evidence and user clarifications.
2. **Check and revise.** Check assesses information sufficiency; Gate routes evidence-driven revisions through isolated drafts, committed only after validation.
3. **Generate and recover.** The SQL-writing agent uses the checked state to execute and submit queries. If the primary workflow fails, an independent BIRD-Interact agent may attempt recovery within the remaining budget.

The [architecture guide](docs/architecture.md) details tool permissions, state transitions, and recovery conditions.

## Results

Archived evaluation on **600 BIRD-Interact Full tasks**, comparing Valibra with the BIRD-Interact agent baseline.

**Protocol:** a-interact &nbsp;·&nbsp; **Model:** GLM-5.2 &nbsp;·&nbsp; **User simulator:** Claude Haiku 4.5

| Metric | Baseline | Valibra | Change |
| :--- | ---: | ---: | ---: |
| Phase 1 passed | 144 / 600 | 150 / 600 | +6 |
| Both phases passed | 79 / 600 | 75 / 600 | −4 |
| Total reward | 124.5 | 127.5 | +3.0 |
| Reported tokens | 82,272,142 | 103,258,480 | +25.51% |

Reward = `0.7 × Phase 1 passes + 0.3 × both-phase passes`. Token totals include grounding, SQL generation, recovery, and user simulation.

> Reward improved modestly, while full-task completion declined and token use increased. These results apply to the archived evaluation snapshot and do not establish an overall performance or cost advantage.

[Protocol and limitations](docs/releases/2026-09-16/README.md) · [Results JSON](docs/releases/2026-09-16/core_results.json) · [Configuration manifest](docs/releases/2026-09-16/candidate_manifest.json)

## Getting Started

Use Python 3.12 with a Linux / WSL shell. Live evaluation also requires benchmark data, PostgreSQL databases, and model credentials; see the [setup guide](docs/getting-started.md).

<details>
<summary><strong>Install and run offline tests</strong></summary>

```bash
git clone --branch research/sql-grounding-v1 https://github.com/TianciGao/Valibra.git
cd Valibra

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest pytest-subtests

python -m pytest -q -p no:cacheprovider tests
```

These tests do not require model calls or a running database. Optional dependencies and skipped checks are documented in [tests/README.md](tests/README.md).

</details>

Core implementation: [`valibra_agent/`](valibra_agent/) · Upstream agent: [`system_agent/`](system_agent/) · [Documentation index](docs/README.md)

## License and Acknowledgments

[MIT License](LICENSE). Built on BIRD-Interact-ADK and the [BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) benchmark. Valibra's results are independent research findings, not upstream benchmark results.

[Upstream attribution and citation](docs/upstream/README.md#citation)
