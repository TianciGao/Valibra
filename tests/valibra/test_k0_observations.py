import math
import subprocess
import sys
import textwrap
import unittest

from pydantic import ValidationError

from valibra_agent.requirement_grounding.models import MAX_SUMMARY_CHARS
from valibra_agent.requirement_grounding.observations import (
    MAX_COLLECTION_ITEMS,
    MAX_NESTING_DEPTH,
    ObservationNormalizationError,
    build_observation,
    canonical_json,
)


class K0ObservationTests(unittest.TestCase):
    def test_digest_and_id_are_stable_across_mapping_order(self):
        common = dict(
            task_id="task-1",
            observation_type="user_query",
            phase=1,
            sequence=7,
            source="caller",
        )
        first = build_observation(raw={"b": 2, "a": 1}, **common)
        second = build_observation(raw={"a": 1, "b": 2}, **common)
        self.assertEqual(first.raw_digest, second.raw_digest)
        self.assertEqual(first.observation_id, second.observation_id)
        self.assertEqual(canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}')

    def test_identity_changes_with_task_call_or_raw_content(self):
        base = dict(
            observation_type="schema",
            phase=1,
            sequence=1,
            source="tool",
            function_call_id="call-1",
            tool_name="get_schema",
        )
        first = build_observation(task_id="t1", raw={"x": 1}, **base)
        second = build_observation(task_id="t2", raw={"x": 1}, **base)
        third = build_observation(task_id="t1", raw={"x": 2}, **base)
        self.assertEqual(len({first.observation_id, second.observation_id, third.observation_id}), 3)

    def test_summary_is_bounded_and_raw_value_is_not_stored(self):
        secret_tail = "RAW-LARGE-TEXT-NEVER-STORED"
        observation = build_observation(
            task_id="task",
            observation_type="user_query",
            phase=1,
            sequence=1,
            source="caller",
            raw={"value": "x" * 5000 + secret_tail},
            summary="  " + "summary " * 200 + "  ",
        )
        self.assertEqual(len(observation.summary), MAX_SUMMARY_CHARS)
        self.assertTrue(observation.summary.endswith("…"))
        self.assertNotIn(secret_tail, observation.model_dump_json())
        self.assertNotIn("x" * 100, observation.model_dump_json())

    def test_non_json_inputs_are_rejected_without_coercion(self):
        invalid = [
            (1, 2),
            {"set"},
            object(),
            {1: "non-string-key"},
            math.nan,
            math.inf,
        ]
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ObservationNormalizationError):
                    canonical_json(value)

    def test_collection_and_depth_limits_are_explicit(self):
        with self.assertRaises(ObservationNormalizationError):
            canonical_json(list(range(MAX_COLLECTION_ITEMS + 1)))
        value = "leaf"
        for _ in range(MAX_NESTING_DEPTH + 1):
            value = [value]
        with self.assertRaises(ObservationNormalizationError):
            canonical_json(value)

    def test_total_canonical_byte_limit_is_enforced(self):
        large_but_individually_bounded = ["x" * 230_000 for _ in range(5)]
        with self.assertRaisesRegex(
            ObservationNormalizationError,
            "total byte limit",
        ):
            canonical_json(large_but_individually_bounded)

    def test_tool_observation_requires_call_identity(self):
        with self.assertRaises(ValidationError):
            build_observation(
                task_id="task",
                observation_type="schema",
                phase=1,
                sequence=1,
                source="tool",
                raw={"schema": "bounded"},
            )

    def test_package_imports_when_adk_network_and_model_packages_are_blocked(self):
        script = textwrap.dedent(
            """
            import builtins
            real_import = builtins.__import__
            blocked = ('google.adk', 'google.genai', 'httpx', 'litellm')
            def guarded(name, globals=None, locals=None, fromlist=(), level=0):
                if name.startswith(blocked):
                    raise ImportError('blocked by K0 import test: ' + name)
                return real_import(name, globals, locals, fromlist, level)
            builtins.__import__ = guarded
            import valibra_agent.requirement_grounding as kernel
            print(kernel.RequirementGroundingRuntime().schema_version)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "1.1")


if __name__ == "__main__":
    unittest.main()
