# P7b frozen paired evaluation

Status: **COMPLETE**

Hard gates: **PASS**

Frozen promotion decision: **NO-GO**

The frozen 30-pair / 60-variant Full evaluation completed in manifest order.
Every score was copied from the original BIRD-Interact task result: all 60
private raw-result hashes, fixed JSON pointers, and copied
`total_reward`/`phase1_passed`/`phase2_passed` values were independently
verified. No SQL rescoring or reward reconstruction was performed.

Valibra-LLM minus B0 mean reward delta was `-0.02333333333333333`; the frozen
10,000-sample paired bootstrap 95% interval was `[-0.07, 0.0]`, and the paired
win/tie/loss count was `0/29/1`. All hard integrity gates passed, but the
preregistered directional rule therefore yields **NO-GO**. This result does
not authorize P8 and must not be overwritten or extended after inspection.

B0 passed P1 on 2/30 tasks and P2 on 0/30. Valibra-LLM passed P1 on 1/30 and
P2 on 0/30. B0 main-agent usage was 3,685,308 total tokens. Valibra main-agent
usage was 3,954,260 total tokens; Grounding made 38 Provider calls, but 11 had
unreliable usage, so Grounding tokens and the Valibra combined model-token
total remain `null` rather than being estimated. Provider cost totals are
also incomplete and remain `null`.

All 60 task/session/database cleanups were verified. Research ports were
released, the Research container and network were removed, and its named
PostgreSQL volume was retained. Formal ports, the formal PostgreSQL container,
the formal results fingerprint, and the frozen 68-file B0 source fingerprint
were unchanged.

Private raw results remain gitignored at
`research-runtime/p7_paired_20260809T100952Z/`. This committed evidence contains
no task text, SQL, Prompt, model response, GT, test case, credential, or
Authorization material.
