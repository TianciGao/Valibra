# P7.1d — Narrow JSON transport normalization

Result: **PASS**

This checkpoint accepts the unchanged strict fixed-form JSON either naked or
wrapped by exactly one complete Markdown fence. The optional fence label is
empty or case-insensitive `json`. Only the outer wrapper is removed; the
payload then follows the existing duplicate-key, non-finite JSON, Pydantic,
mention, Patch, and Reducer boundaries.

Invalid wrappers are classified as `transport_format_invalid`. A legal fence
containing malformed JSON or `NaN` remains `json_invalid`, while duplicate
keys remain `duplicate_json_key`. No fields, enums, parameters, or values are
repaired.

No transport metadata was added to Requirement Runtime or Session telemetry.
The existing private raw Provider audit, response SHA, and private reference
remain available for forensic review. Normalization does not add a revision.

## Frozen P7.1c contract

- Runtime schema: `1.1`
- Prompt SHA256: `ddaaa23fd3c824a704ea1e17a769b4c8b1af5c998949fccfe8d564b18f78f7c4`
- Form Schema SHA256: `f7409a7267b3fddcb40d69574e6187320884951ad4e96980bb590ee0058c38d1`
- Grounding Configuration SHA256: `2ec2accb786a1f1e4d35027affe52c0402957861f59a93832583ac4094066dce`

## Verification

- P7.1d focused tests: 8/8 PASS
- Full repository tests: 376/376 PASS
- `pip check`: PASS
- `compileall`: PASS
- `git diff --check`: PASS
- External calls: 0

This stage did not start Evidence-to-Frame, Ambiguity, Phase-2 optimization,
or P8.
