# P4.3b Agent-facing Requirement View Shadow

Status: **PASS**

This offline checkpoint replaces the internal K0 renderer with one stable,
bounded, English Agent-facing view and attaches that view only to the matching
Baseline model-call audit. It does not inject or mutate the main model request.

Synthetic example:

```text
[CURRENT REQUIREMENT]

Values:
- "limit": "5"
- "year": "2024"

Schema concepts:
- "customer_name": "customer names" (schema not yet verified)
- "total": "total" (schema not yet verified)

Operations:
- "limit": "top 5" role="row_limit" params={"limit":5}
- "order": "total descending" role="ordering" params={"direction":"desc","nulls_last":true}

Note: This is a working requirement state, not database ground truth.
Verify unbound schema concepts with the normal BIRD-Interact tools.
```

The example is 486 characters and 128 tokens under the fixed research metric
`tiktoken/cl100k_base`; its SHA256 is
`3b8af60882fae063d43bb7c3820e2b81fb978f2d18c87712bd7d6a916a3a2132`.

All 219 offline tests passed. No Provider, model, database, User Simulator, or
benchmark call was made. The core Grounding state contract and all frozen P4.2
hashes remain unchanged.
