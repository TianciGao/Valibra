# P4.2d real LLM Callback Shadow lifecycle retry

Status: **PASS**

This evidence records the final authorized P4.2d retry. The run used the real
ADK `init_session` / `run_turn` lifecycle, one real Grounding Provider request,
and a local deterministic main-model stub. It did not call the System Agent
Provider, tools, DB Environment, User Simulator, or benchmark.

The real response passed strict JSON parsing, local Pydantic validation,
verbatim mention anchoring, Patch construction, and Reducer application. It
created eight hypothesized Slots, including an exact `order` operation; all
four operation `parameters` values were JSON objects containing scalar values.

The Rule and LLM Shadow model-visible request SHA values, contents SHA, tools
SHA, generation-config SHA, and final stub-response SHA were identical. No
Requirement Frame, Grounding State, Evidence, or telemetry was injected into
the main-model request.

The private harness initially exited nonzero after the successful lifecycle
because its final assertion expected Slot origin `llm`. Production code and
the existing contract tests define the LLM origin as `llm_provisional`. An
offline checker applied the authorized acceptance criteria to the already
saved response and passed every check. No second Provider request was made.

The structured-output request was sent with `strict=true`, but Provider-side
schema enforcement is not treated as guaranteed. Local strict Pydantic
validation remains the authoritative boundary.

Full request/response bodies remain only in the gitignored private audit path
identified by `summary.json`.
