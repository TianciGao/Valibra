import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_official_nonadk_range import (
    build_submission_prompt_flow,
    turns_with_logical_phase,
)
from scripts.validate_nonadk_submission import validate_submission


class NonAdkSubmissionTests(unittest.TestCase):
    def test_successful_phase1_submit_is_not_mislabeled_as_phase2(self) -> None:
        status = {
            "interaction_history": [
                {
                    "turn": 1,
                    "phase": 2,
                    "action": "submit(\"SELECT phase1\")",
                    "reward": 0.7,
                    "observation": "Phase 1 SQL Correct!",
                },
                {
                    "turn": 2,
                    "phase": 2,
                    "action": "submit(\"SELECT phase2\")",
                    "reward": 1.0,
                    "observation": "Phase 2 SQL Correct!",
                },
            ]
        }
        turns = turns_with_logical_phase(status)
        self.assertEqual([turn["logical_phase"] for turn in turns], [1, 2])

    def test_prompt_flow_uses_raw_prompt_and_budget_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            raw = {
                "prompt": "exact prompt",
                "response": "<thought>x</thought><interaction_object>Environment</interaction_object><action>get_schema()</action>",
                "reasoning_content": "hidden",
                "token_usage": {"input_tokens": 10, "output_tokens": 5},
                "response_audit": {
                    "provider_visible_content": "visible",
                    "control_response_source": "content",
                    "finish_reason": "stop",
                    "outcome_classification": "completed_response",
                },
            }
            (task_dir / "results.jsonl.agent_raw_turn_1.jsonl").write_text(
                json.dumps(raw) + "\n", encoding="utf-8"
            )
            status = {
                "total_budget": 12,
                "interaction_history": [{
                    "turn": 1,
                    "phase": 1,
                    "action": "get_schema()",
                    "observation": "schema",
                    "budget_after_action": {"remaining_budget": 11},
                }],
            }
            flow = build_submission_prompt_flow(
                status,
                task_dir,
                system_model="openai/glm-4.7",
                user_model="anthropic/claude-haiku-4-5-20251001",
            )
            self.assertEqual(len(flow), 1)
            self.assertEqual(flow[0]["prompt"], "exact prompt")
            self.assertEqual(flow[0]["action_cost"], 1.0)
            self.assertEqual(flow[0]["model"], "my_model")

    def test_submission_validator_accepts_sanitized_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            submission = root / "submission.jsonl"
            manifest.write_text(
                json.dumps({
                    "global_index": 1,
                    "instance_id": "example_1",
                    "selected_database": "example",
                }) + "\n",
                encoding="utf-8",
            )
            submission.write_text(
                json.dumps({
                    "instance_id": "example_1",
                    "subtask_1_predicted_sql": ["SELECT 1"],
                    "subtask_2_predicted_sql": [],
                    "prompt_flow": [{
                        "model": "my_model",
                        "user_simulator": "claude-haiku-4-5-20251001",
                        "prompt": "prompt",
                        "response": "response",
                        "action": "execute(\"SELECT 1\")",
                        "remaining_budget": 11,
                        "action_input_tokens": 3,
                        "action_output_tokens": 2,
                        "action_cost": 1.0,
                    }],
                }) + "\n",
                encoding="utf-8",
            )
            report = validate_submission(
                submission,
                manifest,
                expected_count=1,
                expected_start=1,
                expected_end=1,
            )
            self.assertTrue(report["valid"])

    def test_submission_validator_rejects_ground_truth_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            submission = root / "submission.jsonl"
            manifest.write_text(
                json.dumps({"global_index": 1, "instance_id": "example_1"}) + "\n",
                encoding="utf-8",
            )
            submission.write_text(
                json.dumps({
                    "instance_id": "example_1",
                    "subtask_1_predicted_sql": [],
                    "subtask_2_predicted_sql": [],
                    "prompt_flow": [{
                        "model": "my_model",
                        "user_simulator": "simulator",
                        "prompt": "prompt",
                        "response": "response",
                        "action": "",
                        "remaining_budget": 12,
                        "action_input_tokens": 0,
                        "action_output_tokens": 0,
                        "action_cost": 0.0,
                        "test_cases": ["private"],
                    }],
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                validate_submission(
                    submission,
                    manifest,
                    expected_count=1,
                    expected_start=1,
                    expected_end=1,
                )

    def test_rejected_post_budget_action_has_zero_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            submission = root / "submission.jsonl"
            manifest.write_text(
                json.dumps({"global_index": 1, "instance_id": "example_1"}) + "\n",
                encoding="utf-8",
            )
            submission.write_text(
                json.dumps({
                    "instance_id": "example_1",
                    "subtask_1_predicted_sql": [],
                    "subtask_2_predicted_sql": [],
                    "prompt_flow": [{
                        "model": "my_model",
                        "user_simulator": "simulator",
                        "prompt": "prompt",
                        "response": "response",
                        "action": "ask(\"more\")",
                        "action_executed": False,
                        "observation": "Budget depleted. Agent failed to submit.",
                        "remaining_budget": 0,
                        "action_input_tokens": 1,
                        "action_output_tokens": 5,
                        "action_cost": 0.0,
                    }],
                }) + "\n",
                encoding="utf-8",
            )
            report = validate_submission(
                submission,
                manifest,
                expected_count=1,
                expected_start=1,
                expected_end=1,
            )
            self.assertTrue(report["valid"])


if __name__ == "__main__":
    unittest.main()
