"""Synthetic tests for offline archive diagnostics; no external calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handoff_audit", ROOT / "scripts/audit_main_handoff.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def request(omitted=0):
    suffix = f"\n... omitted={omitted}" if omitted else ""
    return {"config": {"system_instruction": f"""Writer policy
{audit.BEGIN}
Phase: 1
Original Query: Show selected records.
Follow-up: none
Final Grounding State and answered Clarifications:
{audit.VIEW}
Tables:
- "items"
Relations:
- []
Column mappings:
- {{"phrase":"type","targets":["items.tags -> 'kind'"]}}
Domain knowledge:
- []{suffix}
[MAIN EXECUTION ENVELOPE]
{{}}
Remaining Bird-Coin: 12
{audit.END}"""}, "contents": []}


class HandoffAuditTests(unittest.TestCase):
    def test_writer_detection_does_not_accept_history_or_bootstrap(self):
        self.assertIsNone(audit.writer_input({"contents": [{"parts": [{"text": audit.BEGIN}]}]}))
        self.assertIsNone(audit.writer_input({"config": {"system_instruction": "bootstrap"}}))

    def test_view_and_omissions_are_from_actual_request(self):
        parsed = audit.writer_input(request(4))
        self.assertEqual(parsed["omitted_items"], 4)
        self.assertEqual(parsed["targets"], ["items.tags -> 'kind'"])
        self.assertNotIn("EXECUTION ENVELOPE", parsed["view"])
        self.assertEqual(audit.writer_input(request())["omitted_items"], 0)

    def test_lexical_absence_is_detected_without_truncation(self):
        meanings = {"items.tags": {"fields_meaning": {"kind": "Enum: 'Alpha', 'Beta'."}}}
        parsed = audit.writer_input(request())
        result = audit.lexical_coverage(parsed["targets"], meanings, parsed["visible"])
        self.assertEqual(parsed["omitted_items"], 0)
        self.assertEqual(result["quoted_literals_absent"], 2)
        self.assertEqual(result["enum_targets_with_absent_literals"], 1)

    def test_existing_literal_is_not_counted_absent(self):
        means = {"items.kind": "Enum: 'Alpha', 'Beta'."}
        result = audit.lexical_coverage(["items.kind"], means, "Use Alpha and Beta")
        self.assertEqual(result["quoted_literals_absent"], 0)

    def test_unsupported_expression_is_not_guessed(self):
        result = audit.lexical_coverage(["COALESCE(items.kind, 'X')"], {}, "")
        self.assertEqual(result["targets_without_resolved_description"], 1)
        self.assertNotIn("enum_labelled_targets", result)

    def test_future_and_foreign_evidence_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "provider_audits").mkdir()
            for i, (query, date, text) in enumerate([
                ("Q", "2026-01-01T00:00:01+00:00", "original"),
                ("OTHER", "2026-01-01T00:00:02+00:00", "foreign"),
                ("Q", "2026-01-01T00:00:04+00:00", "future"),
            ]):
                payload = {"query": query, "column_meanings": {"items.kind": text}}
                record = {"completed_at": date, "request": {"messages": [{"content": json.dumps(payload)}]}}
                (directory / "provider_audits" / f"{i}.json").write_text(json.dumps(record))
            means, counts = audit.prior_meanings(directory, "Q", audit.timestamp("2026-01-01T00:00:03+00:00"))
            self.assertEqual(means, {"items.kind": "original"})
            self.assertEqual(counts["foreign_query_audits_excluded"], 1)

    def test_naive_timestamps_fail_closed(self):
        with self.assertRaises(ValueError):
            audit.timestamp("2026-01-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
