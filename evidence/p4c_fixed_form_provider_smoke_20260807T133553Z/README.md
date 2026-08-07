# P4.2b fixed-form Grounding Provider smoke

Result: `PASS`.

- Exactly one real Grounding Provider request was made with `retry=0`.
- The request used `openai/glm-5.2`, `max_tokens=32768`, no tools, and the frozen strict JSON Schema.
- The existing strict JSON, Pydantic, verbatim-mention, Patch, and Reducer chain accepted the response atomically.
- Nine provisional slots were produced, including an exact `operation_type="order"`; Grounding revision advanced to `1`.
- System Agent, DB Environment, User Simulator, benchmark, tools, fallback, and enum normalization were not used.
- The current Agent remains Rule Shadow. This smoke completes P4.2b only; P4 overall remains incomplete.
- The complete request and response remain only in the gitignored private audit referenced by `smoke_summary.json`.

This evidence contains no complete Prompt, model response, credential, Authorization header, or credential path.
