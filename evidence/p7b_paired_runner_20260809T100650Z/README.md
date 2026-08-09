# P7b frozen paired runner offline gate

Status: **PASS**

This evidence covers only the offline P7b runner gate.  No B0, Valibra,
Provider, DB Environment, User Simulator, PostgreSQL, or benchmark run was
started while producing it.

The new runner validates the frozen Full input, P7 protocol, manifest, common
configuration, both variant configurations, and the formal B0 source
fingerprint.  It delegates task execution to the existing BIRD-Interact
`orchestrator.ainteract.run_single_task` implementation.  It copies
`total_reward`, `phase1_passed`, and `phase2_passed` from each saved official
raw result and records the raw SHA256 plus fixed JSON pointers in a private
score-provenance ledger.  It contains no SQL evaluator or reward formula.

Offline synthetic coverage verifies exactly 30 adjacent pairs / 60 variant
runs in frozen order, 15 B0-first and 15 Valibra-first pairs, no task
substitution, no post-call rerun, cleanup gating, no interim effect analysis,
strict official score provenance, and zero network calls in validate-only
mode.

Final checks: `pip check` passed, 296/296 unit tests passed, `compileall`
passed, and `git diff --check` passed.  The formal B0 source fingerprint was
68 files and remained
`66fa2d56eb3a4004d186c79ee5b1a160be85594ad20c0c4624b0a469abbe70e0`.

This evidence does not claim that P7b has run or passed.  Real Provider calls
remain gated on committing and pushing this runner first.
