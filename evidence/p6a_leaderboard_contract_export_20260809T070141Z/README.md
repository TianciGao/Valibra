# P6 leaderboard contract and result-ledger offline wiring

Result: **PASS**

This evidence covers only the offline P6 integration step. No service,
Provider, database, User Simulator, or benchmark was invoked.

Verified outcomes:

- Formal B0 and Research have the same normalized runtime contract for the
  nine tools, tool costs, fixed budget cases, `submit_sql`, four HTTP routes,
  and init/run/cleanup request schemas.
- The formal B0 68-file source/config fingerprint is identical before and
  after this work.
- The orchestrator retains the exact mocked HTTP order, URLs, payloads,
  timeouts, and cleanup call; only the additive `valibra` result key is new.
- Existing result fields, including `token_usage.combined`, SQL, reward, and
  trajectories remain byte-for-byte equivalent as Python values in the
  synthetic regression.
- Main Agent and Grounding model usage are separate. Reasoning tokens are not
  added twice, missing Provider cost remains null, and bird-coin is separate.
- The final bounded Runtime round-trips through JSON. Trajectory counts and
  canonical SHA256 values are stable, with bounded locations for Grounding
  updates and Requirement View model calls.
- The initial `user_query` Grounding summary is attached to the exact Baseline
  model-call record without storing the Grounding prompt or model response.
- Missing/corrupt Runtime and export/audit failures remain fail-open.

Validation:

- `pip check`: PASS
- `python -m unittest discover -s tests -v`: PASS, 249/249
- `python -m compileall valibra_agent orchestrator tests/valibra`: PASS
- `git diff --check`: PASS

See `summary.json` for the bounded machine-readable record.
