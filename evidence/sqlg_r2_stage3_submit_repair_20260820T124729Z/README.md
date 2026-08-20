# R2 Stage 3 — Submit-driven Repair Grounding

Result: **PASS**

- Start HEAD: `30657af2e7ba035da1409d950bf2cb118e420a29`
- Branch: `research/sql-grounding-v1`
- Stage 3 focused tests: 5/5 PASS
- Cross-stage lifecycle tests: 94/94 PASS
- Full tracked repository tests plus the new Stage 3 suite: 441/441 PASS
- `pip check`, `compileall`, and `git diff --check`: PASS
- Provider, Main Agent, User Simulator, DB, HTTP, and benchmark calls: 0

## Frozen behavior

1. Every `execute_sql` result is retained only as bounded Official trajectory evidence. Success, SQL error, and empty result do not invoke Grounding.
2. The first actual Official `submit_sql` failure is the only event that may invoke Repair Grounding (#2).
3. A successful first submit leaves total Grounding calls at one. A later failed submit cannot invoke a third Grounding call.
4. Repair reuses the existing single `get_schema`, `get_all_column_meanings`, and `get_all_knowledge_definitions` trajectory records. It never executes those tools again.
5. The Repair input contains exactly seven fields and remains subject to the existing 262144-character canonical request bound. Oversized input is rejected before a client call.
6. Stage 4 was not started.

## Frozen contract SHA256

- Prompt: `137f8b917a7e39f6ce4a5b8d885f09dabc0737d3d23dd321ec7ffcd5e73fe373`
- Form: `2d60e788b2a3c1efc581f95945331a124805678fedc857bb2bc39f7462500406`
- Configuration: `d641974ee9a3a08d439b9ad349a33eec889f9b47de9ea38730cee0f1ef42d224`

Only compact, non-sensitive verification metadata is retained here. No prompts, model responses, credentials, database contents, or large logs are included.
