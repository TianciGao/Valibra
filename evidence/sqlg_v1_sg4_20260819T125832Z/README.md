# SQL Grounding V1 SG4 Provider Shadow

Status: **PASS**

SG4 keeps the SQL Grounding Provider in shadow mode. The Main Agent request,
official tools, Bird-Coin, submit state machine, and official trajectory remain
unchanged. Control, Attempt Gate, and active View injection remain disabled.

## Executable Prompt correction

- The executable Prompt now spells out the exact nested fixed form.
- `column_mapping` items contain exactly `phrase` and plural `targets`.
- `targets` is always a non-empty JSON array.
- No alias, output repair, fallback, or retry was added.
- Provider transport remains `json_object`; local strict Pydantic and semantic
  validation remain authoritative.
- Form schema canonical JSON and SHA are unchanged.

## Real smoke results

Smoke A made one real Grounding request. The raw Provider response used
`targets` as an array, contained no singular `target`, passed strict local Form
and semantic validation, and was atomically accepted at revision 1.

Smoke B made one real Grounding request inside a real ADK `run_turn` with a
local deterministic Main Agent stub. The Provider response and lifecycle
completed, but the first private harness raised after the run because it
compared the post-callback request with a later ADK request containing the
framework-added `config.labels`. A zero-HTTP replay of the saved response
proved that the request is identical immediately before and after the
Grounding Callback; the only later difference is `config.labels`. The replay
also verified strict Runtime validity, shadow-only View, zero tools/control/
gate, unchanged Bird-Coin, and unchanged official trajectory. No second real
request was made.

Real HTTP ledger: two historical SG4 requests plus Smoke A and Smoke B equals
four cumulative requests. Provider retry remained zero.

## Offline gates

- SG1–SG3 focused: 89/89 PASS
- SG4 focused: 15/15 PASS
- Full unittest: 401/401 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- External calls before smoke: 0
- Smoke A real HTTP: 1
- Smoke B real HTTP: 1
- Post-failure forensic replay real HTTP: 0

Private raw request/response bodies remain only under gitignored
`research-runtime/sql-grounding-llm/`.
