# P4.2d-recovery — parameter object contract clarification

Status: **PASS (offline only)**

The frozen Grounding Prompt now states unambiguously that every
`operation_slots[].parameters` value must itself be a JSON object. It includes
valid object examples, an empty-object example, and invalid string, integer,
and boolean examples. Values inside the object remain restricted to JSON
scalars; nested objects and arrays remain invalid.

The strict Pydantic form, operation enum, mention anchoring, unbound Schema
rule, empty Ambiguity rule, Provider adapter behavior, Callback wiring, and all
other production contracts are unchanged. No scalar-to-object repair, enum
alias, fallback, dependency, or Provider call was added.

Provider capability wording is frozen as follows:

> Provider structured-output request was sent, but schema enforcement is not
> treated as guaranteed. Local strict Pydantic validation remains the
> authoritative boundary.

All 192 unit tests passed. Three private historical invalid responses
(`order_by`, `sort`, and scalar `parameters`) were replayed offline and all
three were rejected atomically with revision `0` and no Frame state. Complete
historical responses remain only in gitignored private audits.

P4.2d remains failed. This recovery only passes the offline Prompt-contract
hardening step; another real lifecycle request has not been authorized.
