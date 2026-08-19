# SG3 ADK Lifecycle Shadow Evidence

Status: **PASS**

- Branch: `research/sql-grounding-v1`
- Start HEAD: `9695abbfe1d66fcfb343f55b38d031e36564f46f`
- Checkpoint: `sqlg-v1-sg2.1-pass`
- Scope: SQL Grounding V1 lifecycle shadow only; SG4 was not started.

## Frozen behavior

The production callback now has one semantic core: `valibra_agent/sql_grounding/`.
It observes the real ADK lifecycle, calls `process_sql_grounding_observation()`
with a deterministic local passthrough updater, renders the Grounding View for
audit only, and delegates each frozen Baseline callback exactly once. It does
not alter model-visible requests, model responses, tool arguments, Baseline
overrides, Bird-Coin, the official trajectory, or the submit state machine.

Runtime and callback glue are isolated under:

- `valibra:sql_grounding_runtime`
- `valibra:sql_grounding_pending`
- `valibra:sql_grounding_sequence`
- `valibra_sql_grounding_shadow`
- `valibra_sql_grounding_view`
- `valibra_sql_grounding_update`
- `valibra:sql_grounding_error_audits`

The retired `valibra:grounding_runtime` key is not read or written. Pending
records use real `function_call_id` values and remain outside GroundingRuntime.
Control, Attempt Gate, Control Hint, active View injection, and Provider calls
are disabled. Submission/P2 observations that require future Control are
captured and audited as `skipped_control_not_active` without changing stage.

## Validation

- SG3 focused callback/agent/runtime/server tests: 29/29 PASS
- SG2/SG2.1 regression: 34/34 PASS
- SG1 regression: 39/39 PASS
- Full unittest suite: 386/386 PASS, 0 skipped
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- Provider / DB / HTTP / User Simulator / benchmark calls: 0

Only in-process unit/contract tests were executed. The evidence contains no
raw tool result, Provider payload, prompt body, credential, database content,
or benchmark result.

## Frozen hashes

- Prompt: `17455ea076632901a9c2aa3bada96fe4c06baa500e0ce271be854d681ab74962`
- Form: `2d60e788b2a3c1efc581f95945331a124805678fedc857bb2bc39f7462500406`
- Config: `405704b6798c4662df4dbe425ca0d15f284776551827e636bd0297f4f53a13f0`

## Scope review

No production diff exists in `system_agent/`, `valibra_agent/agent.py`,
`valibra_agent/adk_runtime.py`, `valibra_agent/sql_grounding/`,
`valibra_agent/requirement_grounding/`, the evaluator, tools, budget, or submit
logic. Historical Requirement package unit tests remain. Callback integration
tests tied to the retired production semantic core were replaced by SG3 SQL
Grounding lifecycle tests; no compatibility production mode was introduced.
