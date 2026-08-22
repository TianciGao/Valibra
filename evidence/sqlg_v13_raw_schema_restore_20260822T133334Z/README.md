# SQL Grounding V1.3 raw-schema restoration

Status: **HOLD**.

The requested minimal correction passed all offline gates. Structure Grounding
now receives the validated, unmodified Official `get_schema` result, including
sample rows and JSONB sample values. The five frozen smoke cases were then run
exactly once (`rerun_count=0`) and all services were cleaned up.

The correction fixed the measured Structure failure: success improved from
1/5 to 5/5, while Structure input grew from roughly 1.1K to 4.4K tokens as
expected. Across all Grounding stages, 32K reasoning exhaustion fell from 5/6
calls to 1/15, and length-with-empty-content fell from 5/6 to 1/15.

The run remains HOLD because no case passed Knowledge Grounding. One response
ended at the 32K limit with empty content. The other four returned nonconforming
Knowledge forms: `domain_knowledge` had the wrong container type, and two also
included forbidden `tables` / `join_keys` fields. Local strict validation
correctly rejected every candidate. This blocker is outside the authorized
raw-schema correction, so no Knowledge Prompt/Form or runtime behavior was
changed and no task was rerun.

Raw prompts, Provider responses, tool payloads, and credentials remain only in
the gitignored private directory
`research-runtime/sqlg_v13_raw_schema_smoke_20260822T131504Z/`.
