# P7.1a Revision semantic separation

Result: **PASS**

This checkpoint separates the existing technical/CAS revision from the
Requirement-semantic revision without changing the Agent prompt, fixed LLM
form, provider adapter, tools, budget, submit state machine, evaluator, or
schema version.

## Frozen behavior

- `grounding_revision` keeps its V1 behavior and remains the only Patch CAS
  version. A changed Grounding business state (including Evidence) or phase
  advances it.
- `requirement_revision` starts at zero and advances only when the canonical
  Requirement Frame/Ambiguity semantic SHA changes.
- Slot semantic projection includes IDs, kind/role, interpretation, grounding
  status, lifecycle, ambiguity references, and type-specific semantics.
- Mention anchors, Evidence and Evidence references, origin, sequence, phase
  audit fields, raw references, metrics, errors, Pending calls, and Runtime
  phase are excluded from Requirement semantic comparison.
- `SCHEMA_VERSION` intentionally remains `1.0`; the planned V1.1 schema/Prompt/
  fixed-form upgrade belongs to P7.1c, not this substage.

## Verification

- Focused P7.1a tests: 22/22 passed.
- Complete repository tests: 332/332 passed.
- `pip check`, `compileall`, and `git diff --check`: passed.
- Provider, model, database, User Simulator, HTTP service, and benchmark calls:
  zero.
- `system_agent/`, Prompt/Form/Config SHA, tools, budget, and evaluator: unchanged.

## Rollback and compatibility

The annotated tag `p7-v1-baseline` points to pre-change commit
`177ac84828f9e3224a64244eac28bccd6e2767a7`.

New code accepts a V1 Runtime JSON that lacks `requirement_revision` and uses
the default value zero. That default is only a compatibility value: historical
results without the field have an **unknown**, not proven-zero, semantic
revision history. Older code must not reuse a live Session State written by
the new contract because strict extra-field validation would reject it.
