# Historical experiment assertions

These files preserve wording assertions for Check Prompts that are not in the
evaluated Full600 candidate. They are records, not passing tests of current code,
and deliberately do not use the `test_*.py` discovery name.

- `check_authority_first_closure_r2.py`: the experiment was adjudicated
  `HOLD / NOT KEEP`; its authority-first wording is absent from the frozen code.
- `check_semantic_closure_contract_r1.py`: its Prompt was subsequently replaced.
  The evaluated Check instead has the explicit Mapping-omission ownership contract.

Original files are preserved here, not weakened to make obsolete experiments
appear supported. Current behavior remains covered by `test_sql_grounding_v13.py`,
`test_check_semantic_closure.py`, `test_check_mapping_authority_r1.py`, and the
Mapping-omission, Atomic Draft and Final Gate tests. The release contract test
also checks that these retired Prompt instructions have not returned.

The separate historical SG7 evaluation test needs old P7 tooling and private
evaluation fixtures. If those dependencies are absent, collection explicitly
reports a skip; this is not a claim that SG7 has been revalidated.
