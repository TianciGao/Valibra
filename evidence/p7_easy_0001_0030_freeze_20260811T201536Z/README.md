# P7 Full 1--30 paired diagnostic freeze

Status: PASS

- Frozen selection: Full `records[0:30]`, global indexes 1--30.
- Pair count: 30; variant runs: 60.
- Unique task IDs: 30.
- Variant order balance: 15 B0-first / 15 Valibra-first.
- Database coverage: one database (`solar_panel`), as dictated by original input order.
- Configuration identity: unchanged from `configs/p7/p7_protocol_v1.json`.
- Full input SHA256: `a051f7a78462d6c17e840c048ea15c4be65b9f8eed61aad3d2df1370561b10c0`.
- Protocol semantic SHA256: `790c2e80e5907fe9faec3c60b45feaab76a87c98f98f9850a8e800ab5e4e506c`.
- Manifest SHA256: `b00b51773a8c4236a6d53eca12afd601ffac819c8772d800d6d4480214e9ec9f`.
- Offline validation: `pip check`, 308/308 unittests, `compileall`, manifest byte reproduction, validate-only runner, and `git diff --check` passed.
- External Provider/DB/User Simulator calls during freeze validation: 0.
- Original P7 NO-GO remains unchanged; P8 remains unauthorized.

This evidence contains no task text, SQL, ground truth, test cases, prompts,
model outputs, credentials, or Authorization data.
