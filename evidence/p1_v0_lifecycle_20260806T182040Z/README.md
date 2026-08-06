# P1 Valibra V0 thin shell verification

**Result: P1 PASS (pre-publication verification)**

Valibra V0 is a behavior-preserving a-interact shell around B0. It imports the
Baseline instruction, model builder, nine tools, budget/submit behavior, and
session runtime. Its four callback wrappers each delegate once to the matching
Baseline callback. Grounding is disabled and no P2 state or decision logic is
present.

Offline verification passed `pip check`, all 38 old and new unittest cases,
Python compilation, and Bash syntax checks. The tests compare the B0 and V0
prompt, tool names/order/signatures/functions, callbacks, budget rejection,
tool response override, submit phase transition, HTTP schemas/routes, session
reset/reuse/cleanup, and two function calls in one synthetic turn.

The final service lifecycle ran entirely in persistent PTY session 65376. Research
PostgreSQL became Docker-healthy and host-ready. User Simulator 6101, DB
Environment 6102, and Valibra 6110 passed two independent PID, command, CWD,
listener, and HTTP health checks 30 seconds apart. No `/run_session`, Provider,
model, or evaluation request was made.

The exact Valibra PID stop script terminated only the recorded processes, and
the existing Research database stop script removed the container and network
while preserving the named volume. All Research ports were released. Formal
ports, the formal PostgreSQL identity/start time, and the formal results
metadata fingerprint/count remained unchanged. The three 525-byte service
logs contained only health calls and graceful shutdown and were removed after
their checks were summarized; no raw runtime logs are retained here.

The commit and annotated `p1-pass` tag are intentionally recorded in Notion
after the local commit is created and before publication.
