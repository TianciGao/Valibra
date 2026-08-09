# P7a paired experiment protocol and manifest freeze

Status: **PASS**

This evidence covers only the offline P7a preregistration step. No service,
database, Provider, User Simulator, benchmark, B0 run, Valibra run, or result
generation occurred.

The Research repository, remote branch, and annotated `p6-pass` target were
identical at the start. The formal B0 directory was read-only; its frozen
68-file source/config fingerprint remained
`66fa2d56eb3a4004d186c79ee5b1a160be85594ad20c0c4624b0a469abbe70e0`.
The Full input contained 600 unique tasks and matched the frozen SHA256.

Development exclusions were derived only by intersecting real Full task IDs
with structured task-id fields in committed `evidence/`. The only match was
`cross_border_15`, from the P6b development smoke. No historical result
directory, SQL score, reward, GT, test case, task body, or follow-up body was
used for selection.

After that exclusion, the frozen SHA256 selection selected exactly 30 unique
tasks. The byte-reproducible manifest covers 16 databases, 30 follow-up tasks,
29 tasks with official query ambiguity, and 21 with official knowledge
ambiguity. Independent order hashing assigns 15 B0-first and 15
Valibra-first adjacent pairs. Diversity passed without reseeding or manual
substitution.

The protocol freezes common Full/a-interact/Stress configuration, the formal
B0 and Valibra-LLM variant hashes, exact cleanup and restart rules, a strict
60-row paired result ledger, Provider-only latency semantics, separate Token,
cost, and bird-coin ledgers, 10,000 paired bootstrap iterations, and the
pre-registered GO/HOLD/NO-GO rule. The analyzer rejects incomplete,
duplicated, reordered, substituted, or configuration-mismatched inputs before
computing an effect summary.

Offline validation passed `pip check`, 274/274 unit tests, `compileall`,
manifest byte reproduction, and `git diff --check`. Committed summaries and
configs contain no prompt, response, SQL, task text, follow-up text, GT, test
case, credential value, Authorization header, or absolute credential path.

P7 itself is not complete. No paired experiment is authorized by this freeze,
P5 remains an unstarted non-blocking branch, and P8 is not authorized.
