"""Offline audit-only regression tests; no Provider, DB or tool execution."""

import copy
import unittest

from valibra_agent import grounding_callbacks as cb


def _stage(kind, size=3000):
    value = {"staged_grounding_kind": kind, "service_status": "accepted", "detail": ""}
    value["detail"] = "x" * (size - len(cb.canonical_json(value).encode("utf-8")))
    return value


def _audit(count=4, size=3000):
    return {
        "function_call_id": "submit-1",
        "service_status": "skipped_control_lifecycle_only",
        "p2_follow_up": {"staged_grounding": [
            _stage(kind, size) for kind in ("structure", "mapping", "knowledge", "check")[:count]
        ]},
    }


class P2AuditStorageTests(unittest.TestCase):
    def test_four_full_sized_stages_are_preserved_exactly(self):
        audit = _audit(size=cb._MAX_AUDIT_BYTES)
        before = copy.deepcopy(audit)
        state = {"budget_remaining": 3.0, "current_phase": 2,
                 cb.GROUNDING_RUNTIME_KEY: {"frozen": "sentinel"},
                 "tool_trajectory": [{"tool": "submit_sql", "result": "passed"}]}
        initial = copy.deepcopy(state)
        control = {"official_outcome": "p1_follow_up"}
        cb._upsert_tool_callback_audit(state, "submit-1", shadow_audit=audit, control_audit=control)
        records = cb._load_tool_callback_audits(state)
        self.assertEqual(records["submit-1"][cb.SHADOW_AUDIT_KEY], before)
        self.assertEqual(records["submit-1"][cb.GROUNDING_CONTROL_AUDIT_KEY], control)
        self.assertEqual(audit, before)
        self.assertLessEqual(len(cb.canonical_json(records["submit-1"]).encode()), cb._MAX_TOOL_AUDIT_RECORD_BYTES)
        self.assertEqual({k: v for k, v in state.items() if k != cb.GROUNDING_TOOL_AUDITS_KEY}, initial)

    def test_research_trajectory_attachment_has_the_same_lossless_bound(self):
        audit = _audit()
        state = {"tool_trajectory": [{"tool": "submit_sql", "result": "passed"}]}
        cb._attach_tool_audit(state, 0, audit)
        self.assertEqual(state["tool_trajectory"][0], {
            "tool": "submit_sql", "result": "passed", cb.SHADOW_AUDIT_KEY: audit,
        })

    def test_partial_p2_prefix_and_small_legacy_audit_remain_valid(self):
        for audit in ({"service_status": "succeeded"}, _audit(count=1), _audit(count=2), _audit(count=3)):
            with self.subTest(audit=repr(audit)[:80]):
                state = {}
                cb._upsert_tool_callback_audit(state, "submit-1", shadow_audit=audit, control_audit=None)
                self.assertEqual(cb._load_tool_callback_audits(state)["submit-1"][cb.SHADOW_AUDIT_KEY], audit)

    def test_invalid_or_oversized_components_fail_without_partial_write(self):
        cases = []
        too_many = _audit()
        too_many["p2_follow_up"]["staged_grounding"].append(_stage("check", 100))
        cases.append(too_many)
        bad_order = _audit()
        bad_order["p2_follow_up"]["staged_grounding"].reverse()
        cases.append(bad_order)
        cases.append(_audit(size=4097))
        cases.append({**_audit(), "padding": "x" * 4096})
        cases.append({"padding": "x" * 4096})
        cases.append({"p2_follow_up": {"staged_grounding": []}})
        cases.append({"p2_follow_up": {"staged_grounding": "not-a-list"}})
        nested = _audit()
        nested["p2_follow_up"]["staged_grounding"][0]["p2_follow_up"] = {}
        cases.append(nested)
        for audit in cases:
            with self.subTest(index=cases.index(audit)):
                state = {"budget_remaining": 3.0}
                with self.assertRaises(ValueError):
                    cb._upsert_tool_callback_audit(state, "submit-1", shadow_audit=audit, control_audit=None)
                self.assertEqual(state, {"budget_remaining": 3.0})

    def test_utf8_bytes_not_character_count_are_bounded(self):
        audit = _audit(count=1, size=100)
        audit["p2_follow_up"]["staged_grounding"][0]["detail"] = "中" * 1400
        self.assertLess(len(cb.canonical_json(audit)), 4096)
        with self.assertRaises(ValueError):
            cb._require_bounded_tool_audit(audit)

    def test_stored_record_tampering_is_rejected_on_read(self):
        state = {}
        cb._upsert_tool_callback_audit(state, "submit-1", shadow_audit=_audit(), control_audit=None)
        record = state[cb.GROUNDING_TOOL_AUDITS_KEY]["submit-1"]
        record[cb.SHADOW_AUDIT_KEY]["p2_follow_up"]["staged_grounding"][0] = _stage("structure", 4097)
        with self.assertRaises(ValueError):
            cb._load_tool_callback_audits(state)

    def test_record_count_and_control_limits_are_unchanged(self):
        state = {}
        for index in range(cb._MAX_TOOL_AUDITS):
            cb._upsert_tool_callback_audit(state, f"call-{index}", shadow_audit={}, control_audit=None)
        before = copy.deepcopy(state)
        with self.assertRaises(ValueError):
            cb._upsert_tool_callback_audit(state, "extra", shadow_audit=_audit(), control_audit=None)
        self.assertEqual(state, before)
        with self.assertRaises(ValueError):
            cb._upsert_tool_callback_audit(state, "call-0", shadow_audit=_audit(), control_audit={"large": "x" * 4096})
        self.assertEqual(state, before)
        cb._upsert_tool_callback_audit(state, "call-0", shadow_audit=_audit(), control_audit=None)
        self.assertEqual(len(cb._load_tool_callback_audits(state)), cb._MAX_TOOL_AUDITS)


if __name__ == "__main__":
    unittest.main()
