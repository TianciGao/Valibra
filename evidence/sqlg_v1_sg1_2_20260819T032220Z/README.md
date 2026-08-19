# SG1.2 SQL Grounding contract closure

Status: **PASS**

This checkpoint closes two SG1 contract gaps without starting SG2:

- Git governance now distinguishes the immutable legacy rollback tag
  `p7.1d-pass`, the `research/main` integration branch, and the sole active
  development branch `research/sql-grounding-v1`.
- Same-table adjacent-record grounding uses one narrowly whitelisted,
  self-contained PostgreSQL window relation:

  `LAG(orbital_characteristics.orbitalref) OVER (PARTITION BY planets.hostlink ORDER BY orbital_characteristics.period) <> orbital_characteristics.orbitalref`

  The prior same-row expression `stars.rn = stars.rn - 1`, undeclared aliases,
  arbitrary SELECT/JOIN/DML/DDL, offsets, descending or multi-column ordering,
  and unknown identifiers remain rejected.
- During `INITIAL_GROUNDING`, any remaining null dimension requires a
  four-dimensional focus; when all dimensions are evaluated, focus must be
  `none`. `REPAIR` remains unchanged.

The representation is based only on already-saved Full-600 audit categories
and the existing tool trajectories for global tasks 493 and 511. No GT, test
cases, SQL execution, Provider, database, HTTP, User Simulator, or benchmark
was used.

Validation:

- SG1/SG1.1/SG1.2 focused tests: 39/39 PASS
- Full unittest suite: 415/415 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- Sensitive-information findings: 0
- Files over 1 MiB in the change scope: 0
- External calls (Provider/DB/HTTP/User Simulator/benchmark): 0

SG1.1 regressions remain covered: ordinary persisted expressions have no
hidden aliases; expression table references are a subset of State.tables;
correlated scalar relations retain their narrow frozen form; and
StateDiffAuthorization remains a pure contract with no Patch, Reducer, or
Evidence Store.
