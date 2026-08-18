# SQL Grounding V1 — SG0 Governance and Isolation Freeze

Status: **PASS**

SG0 created `research/sql-grounding-v1` from the immutable Requirement Grounding baseline `22dbf71de443843ec2945fc929022e79f50a3d2e` (`p7.1d-pass`). This stage changed repository governance only; it did not implement SQL Grounding business code.

Frozen migration boundary:

- old Runtime key: `valibra:grounding_runtime`
- new Runtime key: `valibra:sql_grounding_runtime`
- automatic migration: forbidden
- `valibra_agent/requirement_grounding/`: retained read-only
- `valibra_agent/sql_grounding/`: not created in SG0

Offline verification:

- `pip check`: PASS
- `python -m unittest discover -s tests -v`: 376/376 PASS
- `python -m compileall valibra_agent tests/valibra`: PASS
- `git diff --check`: PASS
- Provider / DB / HTTP / User Simulator / benchmark calls: 0

Scope audit found no changes in `system_agent/` or `valibra_agent/requirement_grounding/`, no SQL Grounding implementation, and no movement of `research/main` or `p7.1d-pass`.
