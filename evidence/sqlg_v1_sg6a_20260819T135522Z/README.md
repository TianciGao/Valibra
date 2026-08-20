# SG6a Active Grounding View + Control Hint

Status: **PASS**

- Start HEAD: `89150e671bfbf373e01d9d97398f9d17e0b3ad6c`
- Branch: `research/sql-grounding-v1`
- Grounding View: active, bounded by the existing 4096-char / 96-item / 1024-token renderer
- Control Hint: active, additionally bounded to 1024 chars / 256 cl100k tokens
- Injection: one View block followed by one Hint block; both are applied or rolled back atomically
- Attempt Gate: SG5 shadow-only; `would_block` remains observable and `blocked` remains false
- Runtime corruption: suppresses both blocks for that model turn and continues Baseline fail-open
- Valid Runtime plus latest update failure: reuses the last legal Runtime for active visibility
- Prompt/Form/Config SHA: unchanged
- Focused SG6a tests: 12/12 PASS
- SG3–SG6a lifecycle/provider/control tests: 58/58 PASS
- SG1/SG2 regressions: 73/73 PASS
- Full unittest suite: 428/428 PASS
- `pip check`, `compileall`, and `git diff --check`: PASS
- External calls (Grounding/System Provider, BIRD tools, DB, HTTP, User Simulator, benchmark): all 0

The evidence intentionally contains no request bodies, model output, credentials, full system
instruction, raw tool results, or large logs.
