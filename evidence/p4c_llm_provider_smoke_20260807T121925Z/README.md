# P4.2b Grounding Provider smoke retry — FAIL

- UTC evidence timestamp: `2026-08-07T12:19:25Z`
- Baseline HEAD: `c2790dedac02d1efc8fb25d355f7a411ce111884`
- Synthetic input only; no Lite/Full task, database, User Simulator, tool, or System Agent call was used.
- Exactly one real Grounding Provider request was made with retry disabled.
- The request sent `model=openai/glm-5.2` and `max_tokens=32768`.
- The response was valid JSON, but it contained the new unsupported `operation_type="sort"`.
- The only authorized boundary alias remains the exact `order_by -> order` mapping. No mapping for `sort` was added and no second real request was attempted.
- The first inspection harness used the wrong post-check attribute name (`ambiguities` instead of `ambiguity_index`) after the service returned. The retained private response was therefore replayed offline through the same strict Updater and Reducer boundary; that replay confirmed the strict rejection of `sort`.
- Full private request/response remains only in the gitignored `research-runtime/grounding-llm/` directory. This evidence contains no prompt, response body, credential, or credential path.

Result: `P4.2b FAIL`. No commit, push, tag, callback switch, or later P4 stage is authorized from this result.
