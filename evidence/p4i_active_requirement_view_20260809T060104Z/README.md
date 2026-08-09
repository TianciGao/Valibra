# P4.3c Active Requirement View evidence

Status: **PASS**

This evidence records an offline P4.3c verification at starting commit
`393924311295648a937a9266d891a215a46689b3`. The controlled lifecycle smoke
used three fresh ADK sessions, a fake Grounding client, and a deterministic
local main-model stub. It made no Provider, database, User Simulator, tool, or
benchmark calls.

The `off` and `shadow` model-visible requests were identical. The `active`
request differed only by one bounded Requirement View block in the current
system instruction; removing that exact block reproduced the `off` request.
No complete prompt, Session State, model response, or credential is retained
here.

Validation summary:

- `pip check`: PASS
- `python -m unittest discover -s tests -v`: PASS, 237/237
- `python -m compileall valibra_agent tests/valibra`: PASS
- `git diff --check`: PASS
- google-adk: 2.5.0
- `LlmRequest.append_instructions(list[str])`: verified to append only the
  current request's string system instruction and leave contents unchanged
- real Provider calls: 0

See `summary.json` for hashes and bounded structured results.
