"""Protect the evaluated runtime while release-only tests and docs are updated."""

import hashlib
import json
from pathlib import Path
import unittest

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding import updater


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "docs/releases/2026-09-16"


class ReleaseCandidateContractTests(unittest.TestCase):
    def test_release_runtime_differs_only_by_authorized_audit_fix(self):
        manifest = json.loads((RELEASE / "candidate_manifest.json").read_text())
        self.assertEqual(len(manifest["runtime_source_sha256"]), 17)
        overrides = manifest["release_runtime_overrides"]
        self.assertEqual(set(overrides), {"valibra_agent/grounding_callbacks.py"})
        self.assertNotEqual(overrides["valibra_agent/grounding_callbacks.py"],
                            manifest["runtime_source_sha256"]["valibra_agent/grounding_callbacks.py"])
        for filename, expected in manifest["runtime_source_sha256"].items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(),
                    overrides.get(filename, expected),
                )

    def test_evaluated_prompt_form_and_configuration_identities(self):
        manifest = json.loads((RELEASE / "candidate_manifest.json").read_text())
        self.assertEqual(manifest["identities"], {
            "grounding_prompt_sha256": updater.SQL_GROUNDING_PROMPT_SHA256,
            "grounding_form_sha256": updater.SQL_GROUNDING_FORM_SCHEMA_SHA256,
            "grounding_configuration_sha256": updater.SQL_GROUNDING_CONFIGURATION_SHA256,
            "main_prompt_sha256": hashlib.sha256(
                grounding_callbacks._SQL_WRITER_PROMPT.encode("utf-8")
            ).hexdigest(),
        })

    def test_retired_prompt_experiments_are_not_reenabled(self):
        prompt = updater.CHECK_GROUNDING_PROMPT
        self.assertNotIn("Authority-first closure procedure", prompt)
        self.assertNotIn("Required transformation / grain authority", prompt)
        self.assertIn("至少存在两个合理的用户业务解释", prompt)
        self.assertIn("unresolved_mappings 非空时，本轮 Check 不得返回 complete", prompt)

    def test_missing_omission_sidecar_is_still_rejected(self):
        common = {"query": "Show the cost", "current_state": {}}
        for evidence in (
            {"column_meanings": {}},
            {"knowledge_definitions": [], "relevant_column_meanings": {}},
            {"check_context": {"kind": "initial"},
             "previous_official_calls": [], "answered_clarifications": []},
        ):
            with self.subTest(fields=sorted(evidence)):
                with self.assertRaises(ValueError):
                    updater.classify_grounding_input({**common, **evidence}, phase=1)


if __name__ == "__main__":
    unittest.main()
