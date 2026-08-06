# P3 NoOp Shadow wiring evidence

Generated at `2026-08-06T20:59:37Z` from baseline commit
`4f6185a7e77a810e13a898e6738a6f5badc00c2a` on `research/main`.

P3 connects the existing K0 NoOp service to the ADK callback lifecycle. It
does not create semantic Frame/Ambiguity content, inject a Prompt View, add a
tool, call a Provider, or run a Lite/Full evaluation.

Verified properties:

- google-adk 2.5.0 exposes `ToolContext.function_call_id`, and ADK constructs
  each context from the actual `FunctionCall.id`;
- google-adk 2.5.0 exposes the official `on_tool_error_callback` contract;
- Baseline callbacks remain the only budget, trajectory, and model-audit
  writers and are each delegated exactly once;
- Pending calls are paired and removed only by exact `function_call_id`;
- NoOp observations use the original tool response, while the model still
  receives the unmodified Baseline override;
- legal `submit_sql` phase 1 to phase 2 transitions update only the internal
  Grounding runtime lifecycle;
- runtime and Shadow metadata retain only bounded summaries, digests, and an
  existing Baseline trajectory reference;
- all Grounding failures are fail-open and tool infrastructure exceptions are
  not converted into Valibra fallback responses.

The complete test output is intentionally not retained. `test_results.json`
contains the compact result summary and `source_sha256.txt` fingerprints every
source or test file changed in P3.
