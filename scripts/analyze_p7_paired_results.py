#!/usr/bin/env python3
"""Analyze a complete frozen P7 B0/Valibra paired-result ledger.

The analyzer is deliberately fail-closed on pairing, order, and configuration.
It never reads benchmark inputs, GT, SQL, prompts, or hidden task data.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from scripts.build_p7_manifest import (
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    ManifestContractError,
    canonical_json_bytes,
    load_json,
    protocol_digest,
    sha256_file,
    validate_protocol,
)


EXPECTED_PAIRS = 30
EXPECTED_ROWS = EXPECTED_PAIRS * 2
VARIANTS = ("b0", "valibra_llm")
MODEL_ROLES = ("main_agent", "grounding", "user_simulator")
RESULT_KEYS = {
    "pair_index",
    "task_id",
    "variant",
    "configuration_sha256",
    "reward",
    "p1_passed",
    "p2_passed",
    "models",
    "latency",
    "bird_coin",
    "tools",
    "restart",
    "integrity",
}
MODEL_LEDGER_KEYS = {"calls"}
CALL_KEYS = {
    "kind",
    "usage_reliable",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "started_at",
    "completed_at",
    "cost",
}
INTEGRITY_KEYS = {
    "configuration_match",
    "official_contract_match",
    "credential_leak_absent",
    "ground_truth_leak_absent",
    "trajectory_complete",
    "trajectory_reconstructable",
    "export_complete",
    "pending_count",
    "cleanup_complete",
    "task_substitution_absent",
    "post_provider_rerun_absent",
    "infrastructure_error",
}
RESTART_KEYS = {
    "pair_startup_restart_count",
    "restart_only_before_provider_calls_and_results",
}
LATENCY_KEYS = {"task_wall_ms"}
BIRD_COIN_KEYS = {"initial", "used", "remaining"}
TOOL_KEYS = {"total", "ask_user", "submit_sql"}


class PairedAnalysisError(ValueError):
    """The input cannot be analyzed under the frozen P7 protocol."""


def _require_exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedAnalysisError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise PairedAnalysisError(
            f"{label} fields mismatch: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise PairedAnalysisError(f"{label} must be finite and non-negative")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    number = _finite_number(value, label, nonnegative=True)
    if int(number) != number:
        raise PairedAnalysisError(f"{label} must be an integer")
    return int(number)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PairedAnalysisError(f"{label} must be boolean")
    return value


def _optional_token(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, label)


def _optional_cost(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, label, nonnegative=True)


def _parse_time(value: Any, label: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PairedAnalysisError(f"{label} must be an ISO timestamp or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PairedAnalysisError(f"{label} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise PairedAnalysisError(f"{label} must include a timezone")
    return parsed


def summarize_role_ledger(value: Any, label: str) -> dict[str, Any]:
    ledger = _require_exact_keys(value, MODEL_LEDGER_KEYS, label)
    calls = ledger["calls"]
    if not isinstance(calls, list):
        raise PairedAnalysisError(f"{label}.calls must be a list")

    provider_calls = 0
    local_calls = 0
    token_complete = True
    cost_complete = True
    input_tokens = output_tokens = reasoning_tokens = total_tokens = 0
    costs: list[float] = []
    provider_latency_ms = 0.0
    latency_included_calls = 0
    latency_excluded_calls = 0

    for index, raw_call in enumerate(calls):
        call_label = f"{label}.calls[{index}]"
        call = _require_exact_keys(raw_call, CALL_KEYS, call_label)
        kind = call["kind"]
        if kind not in {"provider", "local"}:
            raise PairedAnalysisError(f"{call_label}.kind must be provider or local")
        usage_reliable = _boolean(
            call["usage_reliable"], f"{call_label}.usage_reliable"
        )
        tokens = {
            key: _optional_token(call[key], f"{call_label}.{key}")
            for key in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        }
        started = _parse_time(call["started_at"], f"{call_label}.started_at")
        completed = _parse_time(
            call["completed_at"], f"{call_label}.completed_at"
        )
        cost = _optional_cost(call["cost"], f"{call_label}.cost")

        if kind == "local":
            local_calls += 1
            if any(value is not None for value in tokens.values()) or cost is not None:
                raise PairedAnalysisError(
                    f"{call_label} local records cannot carry Provider tokens/cost"
                )
            continue

        provider_calls += 1
        if usage_reliable:
            if any(value is None for value in tokens.values()):
                raise PairedAnalysisError(
                    f"{call_label} reliable usage needs all four token fields"
                )
            assert all(value is not None for value in tokens.values())
            if tokens["reasoning_tokens"] > tokens["output_tokens"]:
                raise PairedAnalysisError(
                    f"{call_label} reasoning_tokens must be a subset of output_tokens"
                )
            if tokens["total_tokens"] != (
                tokens["input_tokens"] + tokens["output_tokens"]
            ):
                raise PairedAnalysisError(
                    f"{call_label} total_tokens must equal input + output"
                )
            input_tokens += int(tokens["input_tokens"])
            output_tokens += int(tokens["output_tokens"])
            reasoning_tokens += int(tokens["reasoning_tokens"])
            total_tokens += int(tokens["total_tokens"])
        else:
            token_complete = False

        if cost is None:
            cost_complete = False
        else:
            costs.append(cost)

        if usage_reliable and started is not None and completed is not None:
            if completed < started:
                raise PairedAnalysisError(
                    f"{call_label} completion precedes start"
                )
            provider_latency_ms += (completed - started).total_seconds() * 1000.0
            latency_included_calls += 1
        else:
            latency_excluded_calls += 1

    if provider_calls == 0:
        token_summary: dict[str, int] | None = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        cost_summary: float | None = 0.0
    else:
        token_summary = (
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
            }
            if token_complete
            else None
        )
        cost_summary = sum(costs) if cost_complete else None

    return {
        "provider_calls": provider_calls,
        "local_calls": local_calls,
        "tokens": token_summary,
        "cost": cost_summary,
        "provider_latency_ms": round(provider_latency_ms, 6),
        "provider_latency_included_calls": latency_included_calls,
        "provider_latency_excluded_calls": latency_excluded_calls,
        "provider_latency_complete": latency_excluded_calls == 0,
    }


def _validate_config_hashes(protocol: Mapping[str, Any]) -> dict[str, str]:
    configurations = protocol.get("configurations")
    if not isinstance(configurations, Mapping):
        raise PairedAnalysisError("protocol configurations missing")
    common = configurations.get("common")
    if not isinstance(common, Mapping):
        raise PairedAnalysisError("protocol common configuration missing")
    common_config = common.get("configuration")
    common_sha = hashlib.sha256(canonical_json_bytes(common_config)).hexdigest()
    if common.get("configuration_sha256") != common_sha:
        raise PairedAnalysisError("common configuration SHA mismatch")
    result: dict[str, str] = {}
    for variant in VARIANTS:
        entry = configurations.get(variant)
        if not isinstance(entry, Mapping):
            raise PairedAnalysisError(f"protocol {variant} configuration missing")
        payload = {
            "common_configuration_sha256": common_sha,
            "variant": entry.get("variant"),
        }
        actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if entry.get("configuration_sha256") != actual:
            raise PairedAnalysisError(f"{variant} configuration SHA mismatch")
        result[variant] = actual
    return result


def _load_result_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PairedAnalysisError(
                    f"result row {line_number} must be an object"
                )
            rows.append(value)
    return rows


def _validate_record(
    raw: Any,
    *,
    pair: Mapping[str, Any],
    expected_variant: str,
    configuration_sha256: str,
    row_index: int,
) -> dict[str, Any]:
    label = f"result row {row_index}"
    record = _require_exact_keys(raw, RESULT_KEYS, label)
    if record["pair_index"] != pair["pair_index"]:
        raise PairedAnalysisError(f"{label} pair_index/order mismatch")
    if record["task_id"] != pair["task_id"]:
        raise PairedAnalysisError(f"{label} task_id/order mismatch")
    if record["variant"] != expected_variant:
        raise PairedAnalysisError(f"{label} variant/order mismatch")
    if record["configuration_sha256"] != configuration_sha256:
        raise PairedAnalysisError(f"{label} configuration SHA mismatch")

    reward = _finite_number(record["reward"], f"{label}.reward")
    p1 = _boolean(record["p1_passed"], f"{label}.p1_passed")
    p2 = _boolean(record["p2_passed"], f"{label}.p2_passed")
    models = _require_exact_keys(record["models"], set(MODEL_ROLES), f"{label}.models")
    model_summary = {
        role: summarize_role_ledger(models[role], f"{label}.models.{role}")
        for role in MODEL_ROLES
    }

    latency = _require_exact_keys(record["latency"], LATENCY_KEYS, f"{label}.latency")
    task_wall_ms = _finite_number(
        latency["task_wall_ms"], f"{label}.latency.task_wall_ms", nonnegative=True
    )

    coin = _require_exact_keys(record["bird_coin"], BIRD_COIN_KEYS, f"{label}.bird_coin")
    initial = _finite_number(coin["initial"], f"{label}.bird_coin.initial", nonnegative=True)
    used = _finite_number(coin["used"], f"{label}.bird_coin.used", nonnegative=True)
    remaining = _finite_number(
        coin["remaining"], f"{label}.bird_coin.remaining", nonnegative=True
    )
    if not math.isclose(initial, float(pair["initial_bird_coin"]), abs_tol=1e-9):
        raise PairedAnalysisError(f"{label} initial bird-coin mismatches manifest")
    if not math.isclose(initial, used + remaining, abs_tol=1e-6):
        raise PairedAnalysisError(f"{label} bird-coin ledger does not balance")

    tools = _require_exact_keys(record["tools"], TOOL_KEYS, f"{label}.tools")
    tool_counts = {
        key: _nonnegative_int(tools[key], f"{label}.tools.{key}")
        for key in TOOL_KEYS
    }
    if tool_counts["ask_user"] + tool_counts["submit_sql"] > tool_counts["total"]:
        raise PairedAnalysisError(f"{label} tool subtotals exceed total")

    restart = _require_exact_keys(record["restart"], RESTART_KEYS, f"{label}.restart")
    restart_count = _nonnegative_int(
        restart["pair_startup_restart_count"],
        f"{label}.restart.pair_startup_restart_count",
    )
    restart_clean = _boolean(
        restart["restart_only_before_provider_calls_and_results"],
        f"{label}.restart.restart_only_before_provider_calls_and_results",
    )

    integrity = _require_exact_keys(
        record["integrity"], INTEGRITY_KEYS, f"{label}.integrity"
    )
    flags = {
        key: _boolean(integrity[key], f"{label}.integrity.{key}")
        for key in INTEGRITY_KEYS
        if key not in {"pending_count", "infrastructure_error"}
    }
    pending_count = _nonnegative_int(
        integrity["pending_count"], f"{label}.integrity.pending_count"
    )
    infrastructure_error = _boolean(
        integrity["infrastructure_error"],
        f"{label}.integrity.infrastructure_error",
    )

    hard_failures: list[str] = []
    positive_flags = {
        "configuration_match",
        "official_contract_match",
        "credential_leak_absent",
        "ground_truth_leak_absent",
        "trajectory_complete",
        "trajectory_reconstructable",
        "export_complete",
        "cleanup_complete",
        "task_substitution_absent",
        "post_provider_rerun_absent",
    }
    for key in sorted(positive_flags):
        if not flags[key]:
            hard_failures.append(f"{record['task_id']}:{record['variant']}:{key}")
    if pending_count != 0:
        hard_failures.append(
            f"{record['task_id']}:{record['variant']}:pending_count={pending_count}"
        )
    if infrastructure_error:
        hard_failures.append(
            f"{record['task_id']}:{record['variant']}:infrastructure_error"
        )
    if restart_count > 1 or (restart_count and not restart_clean):
        hard_failures.append(
            f"{record['task_id']}:{record['variant']}:forbidden_pair_restart"
        )
    if expected_variant == "b0" and model_summary["grounding"]["provider_calls"]:
        hard_failures.append(f"{record['task_id']}:b0:grounding_provider_called")

    main_tokens = model_summary["main_agent"]["tokens"]
    grounding_tokens = model_summary["grounding"]["tokens"]
    total_model_tokens = (
        main_tokens["total_tokens"] + grounding_tokens["total_tokens"]
        if main_tokens is not None and grounding_tokens is not None
        else None
    )
    main_cost = model_summary["main_agent"]["cost"]
    grounding_cost = model_summary["grounding"]["cost"]
    total_model_cost = (
        main_cost + grounding_cost
        if main_cost is not None and grounding_cost is not None
        else None
    )
    user_cost = model_summary["user_simulator"]["cost"]
    all_provider_cost = (
        total_model_cost + user_cost
        if total_model_cost is not None and user_cost is not None
        else None
    )

    return {
        "pair_index": pair["pair_index"],
        "task_id": pair["task_id"],
        "variant": expected_variant,
        "reward": reward,
        "p1_passed": p1,
        "p2_passed": p2,
        "models": model_summary,
        "total_model_tokens": total_model_tokens,
        "total_model_cost": total_model_cost,
        "all_provider_cost": all_provider_cost,
        "task_wall_ms": task_wall_ms,
        "bird_coin": {"initial": initial, "used": used, "remaining": remaining},
        "tools": tool_counts,
        "restart_count": restart_count,
        "hard_gate_failures": hard_failures,
    }


def bootstrap_mean_ci(
    deltas: Sequence[float],
    *,
    protocol_sha256: str,
    iterations: int = 10000,
) -> dict[str, Any]:
    if not deltas:
        raise PairedAnalysisError("bootstrap needs paired deltas")
    seed_hex = hashlib.sha256(
        f"valibra-p7-bootstrap-v1{protocol_sha256}".encode("utf-8")
    ).hexdigest()
    generator = random.Random(int(seed_hex, 16))
    count = len(deltas)
    samples = []
    for _ in range(iterations):
        samples.append(
            sum(deltas[generator.randrange(count)] for _ in range(count)) / count
        )
    samples.sort()
    lower_index = math.floor(0.025 * (iterations - 1))
    upper_index = math.ceil(0.975 * (iterations - 1))
    return {
        "iterations": iterations,
        "seed_sha256": seed_hex,
        "method": "paired nonparametric percentile bootstrap",
        "confidence_level": 0.95,
        "lower": samples[lower_index],
        "upper": samples[upper_index],
    }


def preregistered_decision(
    mean_delta: float,
    wins: int,
    losses: int,
    hard_gate_failures: Sequence[str],
) -> str:
    if hard_gate_failures:
        return "NO-GO"
    if mean_delta > 0 and wins > losses:
        return "GO"
    if mean_delta < 0 and losses > wins:
        return "NO-GO"
    return "HOLD"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sum_complete(values: Sequence[int | float | None]) -> int | float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _variant_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(record["reward"]) for record in records]
    result: dict[str, Any] = {
        "tasks": len(records),
        "mean_reward": _mean(rewards),
        "p1_passed": sum(bool(record["p1_passed"]) for record in records),
        "p1_rate": sum(bool(record["p1_passed"]) for record in records) / len(records),
        "p2_passed": sum(bool(record["p2_passed"]) for record in records),
        "p2_rate": sum(bool(record["p2_passed"]) for record in records) / len(records),
        "task_wall_ms": {
            "total": sum(float(record["task_wall_ms"]) for record in records),
            "mean": _mean([float(record["task_wall_ms"]) for record in records]),
        },
        "bird_coin_used": {
            "total": sum(float(record["bird_coin"]["used"]) for record in records),
            "mean": _mean([float(record["bird_coin"]["used"]) for record in records]),
        },
        "tools": {
            key: sum(int(record["tools"][key]) for record in records)
            for key in sorted(TOOL_KEYS)
        },
    }
    for role in MODEL_ROLES:
        role_rows = [record["models"][role] for record in records]
        tokens = [row["tokens"] for row in role_rows]
        result[role] = {
            "provider_calls": sum(row["provider_calls"] for row in role_rows),
            "tokens": {
                token_key: _sum_complete(
                    [value[token_key] if value is not None else None for value in tokens]
                )
                for token_key in (
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                )
            }
            if all(value is not None for value in tokens)
            else None,
            "cost": _sum_complete([row["cost"] for row in role_rows]),
            "provider_latency_ms": sum(
                row["provider_latency_ms"] for row in role_rows
            ),
            "provider_latency_included_calls": sum(
                row["provider_latency_included_calls"] for row in role_rows
            ),
            "provider_latency_excluded_calls": sum(
                row["provider_latency_excluded_calls"] for row in role_rows
            ),
        }
    result["total_model_tokens"] = _sum_complete(
        [record["total_model_tokens"] for record in records]
    )
    result["total_model_cost"] = _sum_complete(
        [record["total_model_cost"] for record in records]
    )
    result["all_provider_cost"] = _sum_complete(
        [record["all_provider_cost"] for record in records]
    )
    return result


def analyze_paired_results(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        protocol_sha = validate_protocol(protocol)
    except ManifestContractError as exc:
        raise PairedAnalysisError(str(exc)) from exc
    configuration_hashes = _validate_config_hashes(protocol)
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIRS:
        raise PairedAnalysisError("analysis requires the frozen complete 30-pair manifest")
    if len(raw_records) != EXPECTED_ROWS:
        raise PairedAnalysisError(
            f"analysis requires exactly {EXPECTED_ROWS} adjacent rows, got {len(raw_records)}"
        )

    expected_sequence: list[tuple[Mapping[str, Any], str]] = []
    for expected_index, pair in enumerate(pairs, 1):
        if not isinstance(pair, Mapping) or pair.get("pair_index") != expected_index:
            raise PairedAnalysisError("manifest pair order is invalid")
        order = pair.get("run_order")
        if order not in (["b0", "valibra_llm"], ["valibra_llm", "b0"]):
            raise PairedAnalysisError("manifest run_order is invalid")
        expected_sequence.extend((pair, variant) for variant in order)

    normalized = [
        _validate_record(
            raw,
            pair=pair,
            expected_variant=variant,
            configuration_sha256=configuration_hashes[variant],
            row_index=index,
        )
        for index, (raw, (pair, variant)) in enumerate(
            zip(raw_records, expected_sequence), 1
        )
    ]

    by_variant = {
        variant: [record for record in normalized if record["variant"] == variant]
        for variant in VARIANTS
    }
    if any(len(records) != EXPECTED_PAIRS for records in by_variant.values()):
        raise PairedAnalysisError("each variant needs exactly 30 records")

    pair_summaries: list[dict[str, Any]] = []
    hard_gate_failures: list[str] = []
    for pair in pairs:
        pair_records = [
            record
            for record in normalized
            if record["pair_index"] == pair["pair_index"]
        ]
        if len(pair_records) != 2:
            raise PairedAnalysisError("incomplete pair after normalization")
        variants = {record["variant"]: record for record in pair_records}
        b0 = variants["b0"]
        valibra = variants["valibra_llm"]
        if b0["restart_count"] != valibra["restart_count"]:
            hard_gate_failures.append(
                f"{pair['task_id']}:inconsistent_pair_restart_count"
            )
        hard_gate_failures.extend(b0["hard_gate_failures"])
        hard_gate_failures.extend(valibra["hard_gate_failures"])
        delta = valibra["reward"] - b0["reward"]
        pair_summaries.append(
            {
                "pair_index": pair["pair_index"],
                "task_id": pair["task_id"],
                "run_order": pair["run_order"],
                "b0_reward": b0["reward"],
                "valibra_llm_reward": valibra["reward"],
                "reward_delta": delta,
                "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
            }
        )

    deltas = [float(pair["reward_delta"]) for pair in pair_summaries]
    wins = sum(pair["outcome"] == "win" for pair in pair_summaries)
    ties = sum(pair["outcome"] == "tie" for pair in pair_summaries)
    losses = sum(pair["outcome"] == "loss" for pair in pair_summaries)
    mean_delta = _mean(deltas)
    hard_gate_failures = sorted(set(hard_gate_failures))
    decision = preregistered_decision(
        mean_delta, wins, losses, hard_gate_failures
    )
    return {
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "complete_pairs": EXPECTED_PAIRS,
        "strict_pairing_valid": True,
        "primary": {
            "metric": "paired_mean_reward_delta_valibra_minus_b0",
            "mean_delta": mean_delta,
            "bootstrap_95_ci": bootstrap_mean_ci(
                deltas, protocol_sha256=protocol_sha
            ),
        },
        "win_tie_loss": {"wins": wins, "ties": ties, "losses": losses},
        "variants": {
            variant: _variant_summary(records)
            for variant, records in by_variant.items()
        },
        "pairs": pair_summaries,
        "hard_gates": {
            "passed": not hard_gate_failures,
            "failures": hard_gate_failures,
        },
        "decision": decision,
        "decision_scope": "Full only; GO does not authorize P8",
        "raw_results_mutated": False,
    }


def _assert_safe_summary(summary: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "prompt",
        "response",
        "sql",
        "sol_sql",
        "test_cases",
        "ground_truth",
        "follow_up",
        "authorization",
        "api_key",
    }

    def walk(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key).lower()
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    leaked = forbidden_keys.intersection(walk(summary))
    if leaked:
        raise PairedAnalysisError(
            f"analysis summary contains forbidden fields: {sorted(leaked)}"
        )
    json.dumps(summary, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.results.resolve() == args.output.resolve():
        raise PairedAnalysisError("output cannot overwrite the raw result ledger")
    protocol = load_json(args.protocol)
    manifest = load_json(args.manifest)
    records = _load_result_jsonl(args.results)
    result = analyze_paired_results(protocol, manifest, records)
    result["raw_results_sha256"] = sha256_file(args.results)
    _assert_safe_summary(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": result["decision"],
                "complete_pairs": result["complete_pairs"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
