# SG5 Control Shadow evidence

- Result: **PASS**
- Branch: `research/sql-grounding-v1`
- Start HEAD / `sqlg-v1-sg4-pass`: `12ae7b54c08cd6a77b2fbbeff9f131888e8a9978`
- Scope: offline Control Shadow only
- External calls: Provider `0`, DB `0`, HTTP `0`, User Simulator `0`, benchmark `0`

## Verified behavior

- `focus_dimension` persists only in `GroundingRuntime`; no second focus snapshot is stored in Pending or audit state.
- `render_control_hint()` is evaluated for bounded SHA/size/tool-direction audit, but its text is never injected into `llm_request`.
- The first-submit Attempt Gate is evaluated before the Baseline budget callback. In Shadow it records `would_block` but always records `blocked=false` and never returns a synthetic tool result.
- Baseline callbacks remain the sole executor and Bird-Coin owner; their return values and the model-visible request are unchanged.
- Official submit classification uses only the exact pending `phase_before` plus Official Session facts: `current_phase`, `phase1_completed`, `phase2_completed`, and `task_done`.
- P1 and P2 failure enter `REPAIR`; accepted Grounding with complete dimensions and `focus=none` returns to `SQL_ATTEMPT` or `P2_INCREMENTAL` according to the Official phase. Terminal Official completion enters `DONE`.
- Closed shadow-gate outcomes are audited without forcing a lifecycle transition that would be impossible once the future Active Gate blocks execution.
- P2 follow-up and `user_answer` remain fail-closed as `skipped_affected_dimensions_unfrozen`; SG5 does not invent an affected-dimension judge.
- Control errors fail open to the last valid Runtime and expose only bounded error types, never exception text.
- Prompt, Form, Config, SQL State, Service authorization, tools, budget, submit state machine, and evaluator are unchanged.

## Verification

- SG5 focused: `15/15` PASS
- SG1–SG5 regression: `119/119` PASS
- Full unittest: `416/416` PASS
- `pip check`: PASS
- `compileall valibra_agent tests/valibra`: PASS
- `git diff --check`: PASS

No raw model data, credentials, provider responses, or large logs are retained in this evidence directory.
