"""Synthetic probe tests. No network, credentials, model, or database access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handoff_probe", ROOT / "scripts/probe_main_handoff.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def archived_request():
    return {"model": probe.MODEL, "config": {
        "system_instruction": "Original instruction", "temperature": 0.0,
        "tools": [{"function_declarations": [
            {"name": name, "description": name, "parameters_json_schema": {
                "type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}
            for name in ("execute_sql", "submit_sql")]}]},
        "contents": [{"role": "user", "parts": [{"text": "Original question", "function_call": None}]}]}


def generation():
    return {"temperature": 0.0, "max_tokens": 32768, "reasoning_effort": "high",
            "thinking": {"type": "enabled", "clear_thinking": False}, "tool_choice": "auto"}


def synthetic_plan(output):
    plan = {"max_attempts": 12, "calls": []}
    for task in probe.TASKS:
        a = probe.completion_payload(archived_request(), generation())
        for condition, payload in (("A", a), ("B", probe.evidence_variant(a, [{"target": "t.c", "field_meaning": "Enum: 'X', 'Y'."}]))):
            path = output / f"{task}_{condition}.request.json"
            probe.save(path, payload)
            plan["calls"].append({"task": task, "condition": condition, "request": path.name,
                                  "sha256": probe.sha(path.read_bytes())})
    probe.save(output / "manifest.json", plan)
    return plan


class FakeClient:
    def __init__(self, calls, lock, failure=False):
        self.calls, self.lock, self.failure = calls, lock, failure
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def create(self, **payload):
        with self.lock:
            self.calls.append(payload)
        if self.failure:
            exc = RuntimeError("SECRET MUST NOT APPEAR IN RESULTS")
            exc.status_code = 401
            raise exc
        # Tool calls are never executed, even when SQL would mutate a database.
        response = {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "execute_sql", "arguments": '{"sql":"DELETE FROM fake"}'}}]}}]}
        return SimpleNamespace(model_dump=lambda **kwargs: response)


class ProbeTests(unittest.TestCase):
    def test_conversion_keeps_prompt_user_tools_and_generation(self):
        request = archived_request()
        request["raw_response"] = "FORBIDDEN LATER RESPONSE"
        payload = probe.completion_payload(request, generation())
        self.assertEqual(payload["messages"], [{"role": "system", "content": "Original instruction"},
                                               {"role": "user", "content": "Original question"}])
        self.assertEqual(payload["tools"][0]["function"]["parameters"], request["config"]["tools"][0]["function_declarations"][0]["parameters_json_schema"])
        self.assertEqual(payload["extra_body"]["thinking"], generation()["thinking"])
        self.assertEqual(payload["max_tokens"], 32768)
        self.assertNotIn("FORBIDDEN", json.dumps(payload))

    def test_append_only_preserves_all_enum_options_and_original(self):
        a = probe.completion_payload(archived_request(), generation())
        original = deepcopy(a)
        b = probe.evidence_variant(a, [{"target": "t.c", "field_meaning": "Enum: 'X', 'Y', 'Z'."}])
        self.assertEqual(a, original)
        self.assertIn("Enum: 'X', 'Y', 'Z'.", b["messages"][0]["content"])
        b["messages"][0]["content"] = a["messages"][0]["content"]
        self.assertEqual(a, b)

    def test_history_multimodal_unknown_config_fail_closed(self):
        variants = []
        req = archived_request(); req["contents"].append({"role": "model", "parts": []}); variants.append(req)
        req = archived_request(); req["contents"][0]["parts"][0]["inline_data"] = {"data": "x"}; variants.append(req)
        req = archived_request(); req["config"]["new_parameter"] = True; variants.append(req)
        for req in variants:
            with self.assertRaises(ValueError):
                probe.completion_payload(req, generation())

    def test_exactly_twelve_calls_no_tool_execution_no_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); plan = synthetic_plan(out); calls = []; lock = threading.Lock()
            factory = lambda: FakeClient(calls, lock)
            results = probe.run(plan, out, factory)
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(results), 12)
            sessions = [json.loads(p.read_text())["session_id"] for p in out.glob("*.started.json")]
            self.assertEqual(len(set(sessions)), 12)
            with self.assertRaises(FileExistsError):
                probe.run(plan, out, factory)
            self.assertEqual(len(calls), 12)

    def test_auth_failure_stops_after_initial_pair_without_retry_or_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); plan = synthetic_plan(out); calls = []; lock = threading.Lock()
            results = probe.run(plan, out, lambda: FakeClient(calls, lock, failure=True))
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(results), 2)
            self.assertNotIn("SECRET", json.dumps(results))

    def test_tampering_duplicate_and_call_overrun_rejected_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); plan = synthetic_plan(out)
            variants = [deepcopy(plan) for _ in range(3)]
            variants[0]["calls"].append(plan["calls"][0])
            variants[1]["calls"][0] = plan["calls"][1]
            variants[2]["calls"][0]["sha256"] = "wrong"
            for variant in variants:
                with self.assertRaises(ValueError):
                    probe.validate_plan(variant, out)
            self.assertFalse((out / "RUN_STARTED.json").exists())

    def test_different_tools_in_b_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); plan = synthetic_plan(out)
            call = plan["calls"][1]; path = out / call["request"]
            payload = json.loads(path.read_text()); payload["tools"] = []
            path.write_text(json.dumps(payload)); call["sha256"] = probe.sha(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "beyond evidence"):
                probe.validate_plan(plan, out)


if __name__ == "__main__":
    unittest.main()
