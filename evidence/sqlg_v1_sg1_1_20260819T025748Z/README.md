# SG1.1 SQL Grounding V1 reverse-review correction

- Result: **PASS**
- Branch: `research/sql-grounding-v1`
- Start HEAD: `d60b9ddbe8938ee8b2d8df736364a1a6aeb433a4`
- Immutable prior checkpoint: `sqlg-v1-sg1-pass`
- Focused tests: 37/37 passed
- Full unittest suite: 413/413 passed
- `pip check`, `compileall`, and `git diff --check`: passed
- Provider/DB/HTTP/User Simulator/benchmark calls: 0
- Fixed parser: `sqlglot==26.16.4`

## Frozen corrections

Ordinary persisted expressions now use real table identifiers. The transient
`ValidationContext.table_aliases` channel was removed. A local alias is allowed
only inside the narrowly validated scalar correlated subquery that declares it.

The State model parses every field/relation expression and requires every real
referenced table to be present in `SQLGroundingState.tables`. An empty table set
therefore cannot coexist with a table-referencing mapping or relation. Domain
knowledge remains independent of table references.

The only new SELECT-bearing relation shapes are those already identified in
the Full-600 audit: a single correlated lookup column and a single correlated
`MAX(column)`. Both require one explicitly aliased real table and exactly one
equality correlation to an outer real table. Arbitrary SELECT, JOIN, multiple
tables/projections, DISTINCT, LIMIT, extra clauses, unrelated subqueries, and
other aggregate functions remain rejected.

`StateDiffAuthorization(stage, authorized_dimensions)` freezes whole-State
proposal permissions without creating Patch/Reducer/Evidence machinery.
Changed dimensions must be explicitly authorized, evaluated dimensions cannot
regress to null, and SQL_ATTEMPT/DONE cannot change Grounding State. Runtime
revision remains unchanged for a State NoOp and increments exactly once for an
authorized canonical State change.

## SG2 attention

The current SG1 contract validates `domain_knowledge` by exact canonical
`(kind, content)` evidence match. A future SG2 Prompt must copy an approved
canonical knowledge statement verbatim. SG1.1 does not add semantic similarity,
embedding, LLM Judge, or any other fuzzy matching path.

Reverse review found no changes to `system_agent/`, old
`requirement_grounding/`, callbacks, tools, Bird-Coin, submit, evaluator,
dependencies, or historical tags. No SG2 module was created.
