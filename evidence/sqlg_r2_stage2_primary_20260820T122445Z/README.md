# SQL Grounding R2 Stage 2 evidence

Status: PASS

Stage 2 replaces the four separate initial Grounding updates with one Primary
Grounding request after the three fixed Official BIRD bootstrap tools complete:
`get_schema`, `get_all_column_meanings`, and
`get_all_knowledge_definitions`.

Verified offline boundaries:

- the three complete Official results remain in the existing tool trajectory;
- the Primary request contains exactly `query`, `schema`, `column_meanings`,
  `knowledge_definitions`, and `current_state`;
- one fake Provider response can atomically populate all four SQL Grounding
  dimensions and advance the State revision once;
- identifiers and ordinary columns must be supported by schema evidence;
- JSON/JSONB path keys must be supported by the matching Official
  `fields_meaning` metadata;
- domain knowledge must match an exact Official knowledge definition;
- incomplete or invalid bundles fail closed before State mutation;
- no persistent Evidence store or fifth Grounding dimension was added;
- the existing non-Primary observation path remains in place for Stage 3;
- no real Provider, database, HTTP service, User Simulator, or benchmark was
  invoked.

Checks:

- Stage 2/Stage 1/SG1/SG2 focused tests: 81/81 PASS
- SG3-SG7/R1 cross-stage regression: 81/81 PASS
- all Git-tracked Valibra tests: 432/432 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS

Untracked SG7 evaluation drafts were treated as unrelated user work, left
unchanged, and excluded from this Stage 2 evidence and commit.
