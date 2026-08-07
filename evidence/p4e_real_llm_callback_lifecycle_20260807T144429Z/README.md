# P4.2d real LLM callback lifecycle smoke — FAIL

- UTC run: `2026-08-07T14:48:20Z`
- Branch: `research/main`
- HEAD: `8a0776df8365ae487276bd0304c4b4ace7bc299a`
- Offline gate: `pip check` PASS, `182/182` unit tests PASS, compileall PASS, diff check PASS.
- Rule control: PASS through the real ADK `init_session` / `run_turn` lifecycle with one local deterministic main-model stub call, zero Provider calls, zero tools, and zero bird-coin spent.
- Real Grounding request: exactly one Provider audit was created; retry was zero, tools were absent, and strict JSON Schema was sent.
- Failure: the Provider returned the four required top-level fields and eight proposed slots, including an exact `order` operation, but all four operation `parameters` values had scalar runtime types (`bool`, `int`, `int`, `str`) instead of the contract-required JSON object. Strict Pydantic validation rejected the whole form. The callback correctly failed open, leaving an empty Shadow Frame and business revision `0`.
- No second real Grounding request was made. No System Agent Provider, DB, User Simulator, tool, or benchmark was called.
- Production code was not modified. The current Agent remains Rule Shadow.

The full request and response remain only in the gitignored private audit referenced by `summary.json`. This evidence contains no prompt body, response body, credential, Authorization header, or model output text.
