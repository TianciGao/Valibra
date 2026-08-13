# P7.1b Initial Frame result contract

Result: **PASS**

This checkpoint records the one-shot outcome of the first valid Phase-1
`user_query` without changing the Grounding Prompt, fixed form, Provider
adapter, Requirement semantic projection, Agent-visible Requirement View,
tools, budget, submit state machine, or evaluator formula.

## Frozen behavior

- Runtime records exactly three initialization fields: status, bounded reason,
  and the source Observation ID.
- Status starts as `not_attempted` and can enter one terminal result:
  `ready`, `empty`, or `failed`. Exact terminal replay is idempotent; a
  conflicting overwrite is rejected.
- Only a real Phase-1 `user_query` Observation can be referenced.
- `ready` requires at least one initial Slot. The V1 LLM form cannot yet state
  an explicit empty outcome, so a zero-Slot LLM result is conservatively
  `failed/form_validation_failed`; P7.1c owns the form and Prompt change that
  will make the two `empty` reasons reachable.
- Initialization metadata changes neither `grounding_revision` nor
  `requirement_revision`.
- Old V1 Runtime JSON remains loadable. Its default `not_attempted` and
  `requirement_revision=0` are compatibility sentinels for **unknown history**,
  not evidence that initialization or semantic updates never occurred.
  A Session-level legacy marker prevents later turns from backfilling a false
  first-attempt result.
- `SCHEMA_VERSION` intentionally remains `1.0`; P7.1c must upgrade the unified
  Runtime/Form/Prompt contract to `1.1`.

## Verification

- Focused P7.1b tests: 24/24 passed.
- Complete repository tests: 356/356 passed.
- `pip check`, `compileall`, and `git diff --check`: passed.
- Independent read-only review: no blocker.
- Provider, model, database, User Simulator, HTTP service, and benchmark calls:
  zero.
- `system_agent/`, Prompt/Form/Config SHA, Requirement semantic projection,
  tools, budget, submit behavior, and evaluator formula: unchanged.

## Frozen hashes

- Grounding Prompt SHA-256:
  `5ce6c8061509990d5c42e7e71b7ddfe9c96230eddb00e9c50dff6e591c0d928d`
- Fixed Form Schema SHA-256:
  `441a59c410a99ef0db53b8e974aeeeaea1bcd1735aabdc3cb51f59e0b6e069a2`
- Grounding configuration SHA-256:
  `83ba93c060b110a0e48485f8d5083052d96a67c8a79892ab77024be3c4b5ccd9`

The pre-change rollback point is annotated tag `p7.1a-pass` at
`3812b90e68630a0e768f818b13bab8f6e4a61786`.
