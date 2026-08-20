# SQL Grounding R2 Stage 4A — P2 Lifecycle

Status: PASS

Stage 4A adds one Phase-2 follow-up Grounding call after an Official Phase-1
submit passes with a follow-up. It reuses the task-level schema, complete column
meanings, and complete knowledge definitions already present in the Official
trajectory; it does not run those three bootstrap tools again.

The frozen call budget is phase-scoped: P1 permits one Primary call plus one
first-submit-failure Repair, P2 permits one follow-up call plus one
first-submit-failure Repair, and the task total is capped at four. `execute_sql`
results remain bounded Official evidence only and never trigger Grounding.
Duplicate bootstrap attempts, including attempts after a failed Official tool
result, are rejected before Baseline Bird-Coin charging and Official execution.

No real Provider, database, HTTP service, User Simulator, or benchmark was
called. No fifth Grounding dimension or persistent Evidence Store was added.

Verification:

- Stage 4A focused lifecycle tests: 6/6 PASS
- SQLG/Stage regression: 173/173 PASS
- Full tracked repository suite plus Stage 4A: 470/470 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- External calls: 0

The repository contained pre-existing, unrelated untracked SG7 draft files.
They were excluded from Stage 4A, left unchanged, and are not part of this
evidence or commit.
