# GLM-4.7 Full a-Interact Stress submission profile

This profile runs the upstream BIRD-Interact text-ReAct agent with a frozen
GLM-4.7 provider transport. It targets the Full 600-task, a-Interact, Stress
leaderboard setting.

## Frozen evaluation configuration

- Dataset: Full, original 600-row order
- Interaction mode: a-Interact
- Leaderboard mode: Stress
- System model: `openai/glm-4.7`
- Thinking: enabled
- `clear_thinking`: false
- `max_tokens`: 32768
- `temperature`: 0.0
- `top_p`: 0.95
- `tool_choice`: auto
- User Simulator: `anthropic/claude-haiku-4-5-20251001`
- User Simulator mode: encoder/decoder, prompt v2, hidden provider thinking disabled
- User Simulator encoder/decoder output limit: upstream default 6000
- Concurrency: 1
- Maximum turns: 60
- User patience budget component: 6, equivalent to `patience=3`
- Official nine-action prompt, parser, tools, budget costs and evaluator

The source data must contain exactly 600 unique instances and have SHA256:

`a051f7a78462d6c17e840c048ea15c4be65b9f8eed61aad3d2df1370561b10c0`

The upstream source baseline is commit
`451fe2c3518ee1cf908d8139e2913483bd519381` from
`https://github.com/bird-bench/BIRD-Interact`. Exact upstream hashes and the
approved local patch scope are in `configs/official_nonadk_baseline.json`.

## Provider compatibility policy

Provider-visible `content` is authoritative whenever it is non-empty. If it is
empty, `reasoning_content` may be passed to the unchanged BIRD parser only when
it contains exactly one complete tagged action, names one of the official nine
actions, and uses the matching interaction object. Raw visible content,
reasoning, usage and provider response remain in the private audit logs.

An output-limit response without a valid action and a deterministic context
window rejection are model failures with reward 0. Authentication, rate-limit,
network and other provider failures retain the existing retry policy and are
quarantined as infrastructure failures rather than scored.

`clear_thinking=false` is sent exactly. The upstream text-ReAct runner rebuilds
one full user prompt per turn rather than maintaining OpenAI chat-role history,
so Z.AI preserved-thinking history blocks are not active. Enabling them would
change the upstream scaffold and requires a separately named experiment.

## Preflight and formal run

```bash
cd /home/user/code/BIRD-Interact/BIRD-Interact-ADK

MODEL_PRESET=glm47_matched_32768 \
CONCURRENCY=1 \
DRY_RUN=1 \
bash scripts/run_official_nonadk_full_glm47.sh
```

Remove `DRY_RUN=1` to run all 600 tasks. The script defaults to pausing after
the first 100 completed tasks if P1 is below 15%; this gate does not change any
evaluated request or result.

Each task is checkpointed and its database is restored immediately. A run
directory cannot be resumed if its input, configuration, slice, preset or
dependency freeze differs.

## Output separation

- `submission_predictions.jsonl`: minimal SQL-only analysis artifact.
- `submission_official.jsonl`: sanitized BIRD submission containing exact
  system prompts, raw/control responses, reasoning, actions, observations,
  budget, action token counts and costs.
- `result_private.json` and `logs/`: private audit material that may contain GT
  or test-case data and must not be emailed as prediction files.
- `submission_validation.json`: generated only after all 600 tasks pass the
  row/order/schema/private-key/secret checks.

## Manual BIRD validation requirements

Automated preflight cannot grant a leaderboard verified badge. Before sending:

1. Offer temporary access to the proprietary system model/API.
2. Disclose the third-party User Simulator gateway and allow model identity
   validation.
3. Obtain acceptance of the customized User Simulator under BIRD's expert
   rejection-rate requirement (below 15%).
4. Provide this source tree, upstream commit, frozen dependencies and provider
   adapter for BIRD pipeline validation.
5. Send only the sanitized official submission and supporting configuration;
   never send API keys, GT, test cases, or private result files.

BIRD recommends at least three complete runs and reports the best validated
result. This is recommended rather than enforced by the local runner.
