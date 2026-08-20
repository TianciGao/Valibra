# SQL Grounding V1 — SG7-R1 Root-Cause Closure

Status: **PASS**

This evidence closes the SG7-R1 HOLD without running a benchmark or changing
the active first-submit Gate.

## Phase A — offline downstream closure

- Replayed the first successful Smoke A response from its private audit.
- A gitignored counterfactual fixture changed exactly one JSON path:
  `$.sql_grounding_state.column_mapping[0].targets[0]`.
- The only content change was the four ASCII spaces needed for the frozen
  sqlglot 26.16.4 canonical JSON/JSONB expression.
- Form, semantic validation, authorization, StateDiff, Service, and Atomic
  Replace all passed.
- State dimensions changed: tables, join_keys, column_mapping,
  domain_knowledge; counts were 3/2/2/0.
- State SHA changed and grounding revision advanced from 0 to 1.
- This was a private test fixture only. Production response repair remains
  disabled.

## Phase B — one real 180-second Smoke A

- Exactly one Grounding Provider HTTP request was made after `user_query`
  consumed zero requests.
- Provider succeeded in 33951.043 ms with token usage 4981/2531/2320/7512
  (input/output/reasoning/total); Provider cost was unavailable.
- Returned dimensions were 3 tables, 2 join keys, 3 column mappings, and an
  empty domain-knowledge list.
- Strict Form, semantic validation, authorization, and Atomic Replace passed.
- State SHA changed and grounding revision advanced from 0 to 1.
- Every returned relation/field expression was already byte-for-byte canonical;
  no rewrite, repair, retry, alias, or fallback occurred.

## Smoke B and regressions

- Installed Google ADK sibling lifecycle test passed with two Official tools,
  exact audit 2/2, one fake schema Grounding call, zero knowledge-name
  Grounding calls, no race, correct revision, and Baseline Bird-Coin parity.
- SG7-R1/SG1–SG6b focused tests: 154/154 PASS.
- Full repository tests: 463/463 PASS.
- `pip check`, `compileall`, `git diff --check`, scope, secret, and large-file
  checks passed.
- External calls for this continuation: Grounding Provider 1; Main Agent,
  User Simulator, database, HTTP services, and benchmark 0.
- The 48-task experiment was not started.

Private raw request/response and counterfactual fixture remain under
`research-runtime/` and are not included in Git.
