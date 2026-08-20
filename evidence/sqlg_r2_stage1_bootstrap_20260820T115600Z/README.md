# SQL Grounding R2 Stage 1 bootstrap evidence

Status: **PASS (local implementation and offline verification)**

Stage 1 implements only the evidence collection/feed foundation:

- Valibra emits `get_schema`, `get_all_column_meanings`, and
  `get_all_knowledge_definitions` as deterministic, ordered ADK function calls.
- Baseline remains the only Bird-Coin deductor and Official trajectory writer.
- Each exact bootstrap result is retained in the Official trajectory and is
  replaced by a bounded completion marker before Main sees tool history.
- No bootstrap result invokes the Grounding updater or changes
  `GroundingRuntime`.
- The five-field Primary input can be rebuilt ephemerally from the original
  query, the three exact Official results, and current four-dimensional State.
- Bulk knowledge definitions are strictly parsed and projected one-by-one as
  validator-supported `business_rule` evidence.

No Provider, database, HTTP service, User Simulator, or benchmark was called.
Stage 2 was not started.

Verification:

- Stage 1 focused tests: 4/4 PASS
- SG3–SG7 focused regression: 85/85 PASS
- All Git-tracked Valibra tests: 428/428 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- Real ADK offline fixture: Grounding calls 0, Bird-Coin cost 3, Runtime revision 0

The complete discovery run executed 467 tests; 466 passed and one unrelated,
pre-existing untracked SG7 protocol test rejected the dirty development HEAD.
That file and its frozen identity assertion are outside Stage 1.

The initial automatic Git delivery was deferred because the task started with
unrelated tracked and untracked changes.  A later explicit user instruction
authorized publishing this verified Stage 1 scope; unrelated SG7 drafts remain
excluded.
