# P6b real leaderboard integration smoke

Status: **PASS**

This evidence covers the P6 cost-completeness gate and one real Full
`a-interact` integration smoke through Research PostgreSQL, DB Environment,
User Simulator, Valibra Agent, and the orchestrator. The SQL score was not a
P6 pass criterion.

The fixed task was selected before any external call from the 600-row Full
JSONL: require a follow-up and at least one official ambiguity, then choose
the lowest initial bird-coin budget and lexicographically lowest
`instance_id`. This selected `cross_border_15` with an initial budget of 14.0.
No GT, test cases, hidden follow-up, task body, SQL, Prompt, model response, or
credential is stored here.

The final run used one orchestrator invocation, `limit=1`, and
`concurrency=1`. It made 12 Main Agent Provider calls with reported tokens,
one Grounding Provider call, and zero User Simulator Provider calls because
`ask_user` was not used. Grounding cost and total model cost are `null`
because the Provider did not supply a reliable per-call cost; no estimate was
made. Bird-coin remains a separate ledger.

The active Requirement View was injected 13 times, with at most one block per
model request. The final Grounding Runtime had revision 11, three slots,
eleven Evidence objects, no Ambiguity, and no Pending call. The trajectory
manifest was reproduced exactly. P1 and P2 were both false and reward was
zero; that outcome does not fail this integration-only smoke.

Before the final run, a private launcher check exposed that a background shell
function PID was not the uvicorn PID. No orchestrator or Provider call had
occurred. The three actual child commands and CWDs were verified, stopped by
exact PID through the existing stop script, and the final lifecycle was then
run in a fresh persistent PTY using an exec-based private launcher. No
production file was changed for this adjustment.

The final lifecycle passed two health rounds, exact PID/command/CWD checks,
and complete cleanup. Research ports 6101/6102/6110/6433 were released, the
Research container and network were removed, and its named volume was
preserved. Formal ports, formal PostgreSQL identity/health, and the formal
results metadata fingerprint were unchanged.

Private artifacts, including the 15.5 MB result, complete traces, service
logs, task manifest, and raw Grounding audit, remain gitignored under
`research-runtime/` and are referenced only by bounded relative paths in
`summary.json`.

Final offline gates: `pip check` passed, 253/253 unit tests passed,
`compileall` passed, and `git diff --check` passed.
