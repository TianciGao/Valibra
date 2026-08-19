# SG6b Active First-Submit Gate

Status: **PASS**

This checkpoint activates the frozen first-submit Attempt Gate and adds the
approved Budget Liveness Bypass.  The pure `evaluate_first_submit_gate()`
decision is unchanged.  The Callback execution policy reads current-focus
tool directions and the Baseline `system_agent.callbacks.TOOL_COSTS`; it does
not copy or replace the Bird-Coin cost table.

## Verified behavior

- Normal CLOSED first submit with an affordable focus direction is blocked
  before Baseline cost handling, paired by exact `function_call_id`, and never
  becomes an Official trajectory event.
- CLOSED first submit in `INITIAL_GROUNDING` bypasses only when no current-focus
  direction is affordable.  It then follows the complete Baseline forced-exit
  path.
- The 0.25 Bird-Coin ADK regression first rejects `get_schema`, then executes
  one real local `submit_sql` stub through Baseline, records the Official
  trajectory, sets budget to `-1`, and transitions the control stage to
  `REPAIR` after the Official failure fact.
- Missing IDs, corrupt Runtime, invalid budgets, cost-map anomalies, and Gate
  evaluation errors fail open without guessed pairing or cost values.
- Active Grounding View and Control Hint behavior and all three frozen SHA
  values remain unchanged.

The only `control.py` extension is an explicit
`allow_initial_forced_exit=True` lifecycle flag used after an approved
liveness-bypass submit produces an Official result.  Default callers retain
the old transition rejection, and the pure Gate function is byte-for-byte
unchanged.

## Offline gates

- SG6b focused: 13/13 PASS
- SG1–SG6b staged regression: 144/144 PASS
- Full unittest: 441/441 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- External Provider/DB/HTTP/User Simulator/benchmark calls: 0

No SQL text, Provider response, credential, database content, or large log is
stored in this evidence directory.
