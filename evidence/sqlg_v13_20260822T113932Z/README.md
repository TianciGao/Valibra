# SQL Grounding 1.3 implementation and smoke

Status: **HOLD**.

The offline implementation gates passed. The five frozen real smoke tasks were
run exactly once with `rerun_count=0`, then all services were cleaned up. The
private raw request/response material remains under the gitignored
`research-runtime/sqlg_v13_five_task_smoke_20260822T110200Z/` directory.

The lifecycle safety checks passed, including fail-closed behavior, no Repair,
no execute/submit-triggered Grounding, bounded Provider calls, and verified
cleanup. Real viability did not pass: 5 of 6 Grounding calls exhausted 32,768
reasoning tokens with `finish_reason=length` and empty content, so local strict
JSON validation correctly rejected them. No task reached a completed Phase-1
Grounding State or the SQL Writer.

This evidence contains only bounded summaries and hashes. It contains no API
key, Authorization header, full Prompt, full Provider response, or raw tool
payload.
