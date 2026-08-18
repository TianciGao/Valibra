# SG1 SQL Grounding V1 data-contract evidence

- Result: **PASS**
- Branch: `research/sql-grounding-v1`
- Start HEAD: `045ea8f3e0b1a470050314c63677cfc529ed32b9`
- Rollback checkpoint: `sqlg-v1-sg0-pass`
- Scope: four-dimensional data contracts, expression validation, transient validation context, tests only
- Focused tests: 22/22 passed
- Full unittest suite: 398/398 passed
- `pip check`: passed
- `compileall`: passed
- `git diff --check`: passed
- External Provider/DB/HTTP/User Simulator/benchmark calls: 0
- Fixed SQL parser: `sqlglot==26.16.4`

The frozen State contains only `tables`, `join_keys`, `column_mapping`, and
`domain_knowledge`. Stage and focus remain Runtime control fields. Revision
changes are validated only against a canonical State SHA. ValidationContext is
a transient projection of the current query, latest legal Observation, and
already-seen Official BIRD trajectory; it is neither persisted nor serialized.

Reverse review found no changes to `system_agent/`, the old
`requirement_grounding/` kernel, callbacks, tools, Bird-Coin, submit, evaluator,
dependencies, historical tags, or `research/main`. No SG2 module was created.
