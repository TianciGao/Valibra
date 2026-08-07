# P4.2c LLM Frame Callback Shadow offline evidence

Status: PASS

This checkpoint wires the already frozen LLM Frame Updater into the Valibra
ADK callback lifecycle in Shadow mode. All validation used fake clients and
mock callbacks. No real Provider, System Agent model, DB Environment, User
Simulator, benchmark, or tool service was called.

Validated boundaries:

- empty or unset `GROUNDING_UPDATER_MODE` selects Rule Shadow;
- exact `llm` selects LLM Frame Shadow;
- every other mode and invalid LLM configuration fail open without Rule fallback;
- only `user_query` and legal `ask_user` `user_answer` observations call the fake LLM;
- non-user tool observations use NoOp lifecycle recording and consume no LLM budget;
- Baseline callback return values, model request, tool override, budget, and phase logic remain unchanged;
- Pending calls are cleared after LLM success, failure, timeout, and invalid response;
- Shadow State remains bounded and JSON-safe, with no client, credential, or full Provider response;
- Prompt View remains uninjected;
- `system_agent/`, frozen Grounding contracts, Provider adapter, dependencies, and core state/service contracts have zero diff.

Validation completed with 182/182 unit tests passing. The only emitted warning
was the pre-existing Starlette `httpx` deprecation warning.
