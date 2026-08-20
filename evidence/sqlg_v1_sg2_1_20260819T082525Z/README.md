# SG2.1 offline kernel contract closure

Status: **PASS**

SG2.1 closes two narrow offline contract gaps without starting SG3.

## Domain-knowledge authorization

- `schema` and `metadata` may resolve `domain_knowledge` only from `null` to
  `[]`.
- They cannot add or replace non-empty domain knowledge, even when the
  transient `ValidationContext` contains matching canonical knowledge.
- A non-empty canonical domain-knowledge item remains accepted only for a
  `knowledge` Observation.

## Repair lifecycle

Two explicit pure Control events were added without adding a Stage:

- `repair_completed_p1`: `REPAIR -> SQL_ATTEMPT`
- `repair_completed_p2`: `REPAIR -> P2_INCREMENTAL`

Both require a fully evaluated four-dimensional State and `focus_dimension =
none`. They do not change State or `grounding_revision`; a future Callback must
choose the event from the Official BIRD phase and must not infer that phase.

The offline flows cover P1 Repair, a second official submit failure, a
successful retry followed by P2, and P2 Repair returning to another incremental
attempt. Invalid stage, incomplete State, and pending focus are fail-closed.

## Validation

- SG2/SG2.1 focused tests: 34/34 PASS
- SG1/SG1.1/SG1.2 regression tests: 39/39 PASS
- Full unittest suite: 449/449 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- Sensitive-information findings: 0
- Files over 1 MiB in the change scope: 0
- External calls (Provider/DB/HTTP/User Simulator/benchmark): 0

Frozen executable hashes remained unchanged:

- Prompt: `17455ea076632901a9c2aa3bada96fe4c06baa500e0ce271be854d681ab74962`
- Form Schema: `2d60e788b2a3c1efc581f95945331a124805678fedc857bb2bc39f7462500406`
- Configuration: `405704b6798c4662df4dbe425ca0d15f284776551827e636bd0297f4f53a13f0`

## SG3 boundary

`affected_dimensions` remains a transient, fail-closed caller authorization.
SG3 must not invent a new LLM Judge, Planner, or complex rule system to infer
P2 `affected_dimensions`. If no already-frozen safe source is available, Shadow
wiring must remain fail-closed.
