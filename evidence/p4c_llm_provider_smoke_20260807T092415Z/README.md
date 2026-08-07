# P4.2b Grounding Provider smoke — FAIL

P4.2b stopped after its single authorized real Provider request. The native
async Provider call succeeded, but the returned Frame failed the frozen strict
Pydantic contract because one `operation_type` was `order_by`; the contract
allows only `projection`, `filter`, `aggregation`, `group`, `order`, `limit`,
`distinct`, or `other`.

No retry was made. The current Agent remains Rule Shadow. No benchmark,
database, User Simulator, tool, or System Agent model request was executed.
The last valid business State was preserved at revision 0.

The complete request and response remain only in the gitignored private audit
reference recorded in `smoke_summary.json`. This evidence contains no complete
Prompt or response and no credential material.
