import random
import unittest

from tests.valibra.k0_fixtures import (
    graph_patch,
    make_ambiguity,
    make_candidate,
    make_observation,
)
from valibra_agent.requirement_grounding.models import RequirementGroundingRuntime
from valibra_agent.requirement_grounding.reducer import apply_patch, validate_runtime
from valibra_agent.requirement_grounding.service import process_observation
from valibra_agent.requirement_grounding.telemetry import increment_metrics


class FixedSeedOperationSequenceTests(unittest.TestCase):
    def test_repeated_operations_preserve_invariants_and_json_round_trip(self):
        rng = random.Random(20260806)
        runtime = RequirementGroundingRuntime()
        initial_observation = make_observation(sequence=1)
        ambiguity = make_ambiguity(
            status="unresolved",
            candidates=(
                make_candidate("cand-a", "A", "effect-a"),
                make_candidate("cand-b", "B", "effect-b"),
            ),
        )
        runtime = apply_patch(
            runtime,
            graph_patch(runtime, initial_observation, ambiguity=ambiguity),
        )
        observations = [initial_observation]

        for sequence in range(2, 122):
            action = rng.choice(("fresh", "replay", "metric"))
            if action == "fresh":
                observation = make_observation(
                    sequence=sequence,
                    raw={"sequence": sequence, "coin": rng.randrange(1000)},
                )
                observations.append(observation)
                runtime = process_observation(runtime, observation).runtime
            elif action == "replay":
                observation = rng.choice(observations)
                before = runtime
                result = process_observation(runtime, observation)
                runtime = result.runtime
                self.assertIs(runtime, before)
            else:
                runtime = increment_metrics(runtime, prompt_view_chars=rng.randrange(4))

            validate_runtime(runtime)
            restored = RequirementGroundingRuntime.model_validate_json(
                runtime.model_dump_json()
            )
            self.assertEqual(restored, runtime)
            self.assertEqual(runtime.grounding_revision, 1)


if __name__ == "__main__":
    unittest.main()
