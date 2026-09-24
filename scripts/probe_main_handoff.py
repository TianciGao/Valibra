"""Prepare/run a bounded, single-turn evidence-handoff A/B probe.

Preparation is offline. Paid execution requires --run-paid. No Agent, database,
tool execution, historical Runtime restoration, automatic retry, or resume.
Raw requests and responses must stay in the ignored research-runtime directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_main_handoff import prior_meanings, target_description, timestamp, writer_input

TASKS = ("hulushows_11", "hulushows_12", "hulushows_M_1", "hulushows_M_3",
         "hulushows_M_5", "crypto_exchange_M_1")
MAX_ATTEMPTS = 12
MODEL = "openai/glm-5.2"
SUPPLEMENT = "\n\n[PREVIOUSLY ACQUIRED OFFICIAL FIELD DOCUMENTATION]\n"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, data):
    # Exclusive creation prevents silently replacing prior attempts/results.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def completion_payload(request, generation):
    """Lossless conversion of this probe's text-only first-Main requests.

    Fail closed for histories/multimodal input rather than dropping content.
    Only request fields are used: recorded responses/results are never inputs.
    """
    if request["model"] != MODEL:
        raise ValueError("Unexpected archived model")
    if len(request["contents"]) != 1 or request["contents"][0]["role"] != "user":
        raise ValueError("Probe requires one fresh user message")
    parts = request["contents"][0]["parts"]
    if len(parts) != 1 or not isinstance(parts[0].get("text"), str):
        raise ValueError("Probe requires a single text part")
    if any(v is not None for k, v in parts[0].items() if k != "text"):
        raise ValueError("Non-text content must not be discarded")
    config = request["config"]
    if set(config) - {"system_instruction", "temperature", "tools"}:
        raise ValueError("Unsupported request config")
    if config.get("temperature") != generation["temperature"]:
        raise ValueError("Temperature mismatch")
    if set(generation) - {"max_tokens", "temperature", "reasoning_effort", "thinking", "tool_choice", "top_p"}:
        raise ValueError("Unsupported generation setting")
    functions = []
    for group in config["tools"]:
        if set(group) != {"function_declarations"}:
            raise ValueError("Unsupported tool group")
        for declaration in group["function_declarations"]:
            if set(declaration) != {"name", "description", "parameters_json_schema"}:
                raise ValueError("Unsupported tool declaration")
            functions.append({"type": "function", "function": {
                "name": declaration["name"], "description": declaration["description"],
                "parameters": declaration["parameters_json_schema"]}})
    if sorted(f["function"]["name"] for f in functions) != ["execute_sql", "submit_sql"]:
        raise ValueError("Unexpected tool names")
    payload = {"model": MODEL.removeprefix("openai/"), "messages": [
        {"role": "system", "content": config["system_instruction"]},
        {"role": "user", "content": parts[0]["text"]}], "tools": functions,
        "stream": False}
    payload.update({k: v for k, v in generation.items() if k != "thinking"})
    payload["extra_body"] = {"thinking": deepcopy(generation["thinking"])}
    return payload


def evidence_variant(payload, evidence):
    result = deepcopy(payload)
    result["messages"][0]["content"] += SUPPLEMENT + json.dumps(
        {"source": "Official field documentation already acquired before this Main request.",
         "scope": "Descriptions of mapped fields; not a selected answer or an instruction to add predicates.",
         "fields": evidence}, ensure_ascii=False, sort_keys=True)
    return result


def prepare(index, output):
    raw = index.read_bytes()
    rows = json.loads(raw)
    if len({r["task"] for r in rows}) != len(rows):
        raise ValueError("Duplicate archive task")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "singleturn-handoff-ab/v1", "created_at": now(),
                "index_sha256": sha(raw), "model": MODEL, "max_attempts": MAX_ATTEMPTS,
                "sdk_retries": 0, "timeout_seconds": 300, "max_concurrency": 2,
                "design": "Four lexically flagged failures; two historical primary-success controls. One draw per condition.",
                "scope": "First P1 Main request only. No SQL execution, scoring, Agent, or restored Runtime.",
                "intervention": "Append verbatim descriptions for ALL resolvable mapped targets from same-query audits completed before Main.",
                "transport": "Fresh stateless OpenAI-compatible request per attempt; archived text and tool schemas preserved.",
                "adjudication": "Inspect candidate SQL evidence use, unresolved ambiguity, and control regressions; never infer benchmark PASS.",
                "tasks": [], "calls": []}
    for i, name in enumerate(TASKS):
        row = next(r for r in rows if r["task"] == name)
        directory = (ROOT / row["candidate_source"]).resolve()
        if not directory.is_relative_to(ROOT):
            raise ValueError("Archive outside repository")
        source = (directory / "official_result.json").read_bytes()
        record = json.loads(source)
        if (record.get("instance_id") or record.get("task_id")) != name:
            raise ValueError("Task identity mismatch")
        for flow in record["prompt_flow"]:
            writer = writer_input(flow.get("request") or {})
            if writer and writer["phase"] == 1:
                break
        else:
            raise ValueError("No P1 Writer request")
        cutoff = timestamp(flow["timestamp"])
        meanings, counts = prior_meanings(directory, writer["query"], cutoff)
        evidence = [{"target": target, "field_meaning": description}
                    for target in writer["targets"]
                    if (description := target_description(target, meanings)) is not None]
        if not evidence:
            raise ValueError("No prior mapped-field evidence")
        generation = flow["generation_parameters"]
        if not 1 <= generation["max_tokens"] <= 32768:
            raise ValueError("Output cap exceeds authorized probe design")
        a = completion_payload(flow["request"], generation)
        b = evidence_variant(a, evidence)
        # Keep provenance separate from model input; no outcome labels in payloads.
        sources = []
        for path in sorted((directory / "provider_audits").glob("*.json")):
            blob = path.read_bytes()
            audit = json.loads(blob)
            if not audit.get("completed_at") or timestamp(audit["completed_at"]) > cutoff:
                continue
            data = json.loads(audit["request"]["messages"][-1]["content"])
            if data.get("query", data.get("original_query")) == writer["query"] and any(
                isinstance(data.get(k), dict) for k in ("column_meanings", "relevant_column_meanings")):
                sources.append({"file": path.name, "sha256": sha(blob), "completed_at": audit["completed_at"]})
        manifest["tasks"].append({"task": name, "group": "control" if i >= 4 else "flagged",
                                  "archive": str(directory.relative_to(ROOT)), "archive_sha256": sha(source),
                                  "first_main_at": flow["timestamp"], "call_index": flow.get("call_index"),
                                  "generation": generation, "prior_evidence_counts": counts,
                                  "mapped_targets": len(writer["targets"]), "appended_targets": len(evidence),
                                  "supplement_chars": len(b["messages"][0]["content"]) - len(a["messages"][0]["content"]),
                                  "evidence_sources": sources})
        for condition in (("A", "B") if i % 2 == 0 else ("B", "A")):
            path = output / f"{name}_{condition}.request.json"
            save(path, a if condition == "A" else b)
            manifest["calls"].append({"task": name, "condition": condition,
                                      "request": path.name, "sha256": sha(path.read_bytes())})
    save(output / "manifest.json", manifest)
    return manifest


def validate_plan(manifest, output):
    calls = manifest["calls"]
    if len(calls) != MAX_ATTEMPTS or manifest["max_attempts"] != MAX_ATTEMPTS:
        raise ValueError("Exactly 12 attempts required, never more")
    if {(c["task"], c["condition"]) for c in calls} != {(t, a) for t in TASKS for a in ("A", "B")}:
        raise ValueError("Unexpected or duplicate task-condition pair")
    payloads = []
    for call in calls:
        path = output / call["request"]
        if path.resolve().parent != output.resolve() or sha(path.read_bytes()) != call["sha256"]:
            raise ValueError("Request path/hash mismatch")
        payload = json.loads(path.read_text())
        if payload["model"] != MODEL.removeprefix("openai/") or not 1 <= payload["max_tokens"] <= 32768:
            raise ValueError("Unexpected model/token limit")
        payloads.append(payload)
    for name in TASKS:
        pair = {c["condition"]: p for c, p in zip(calls, payloads) if c["task"] == name}
        a, b = pair["A"], deepcopy(pair["B"])
        original = a["messages"][0]["content"]
        if not b["messages"][0]["content"].startswith(original + SUPPLEMENT):
            raise ValueError("B must append evidence to unchanged A")
        b["messages"][0]["content"] = original
        if a != b:
            raise ValueError("A/B differ beyond evidence appendix")
    return payloads


def one_attempt(call, payload, output, client_factory):
    name = f"{call['task']}_{call['condition']}"
    save(output / f"{name}.started.json", {"at": now(), "session_id": str(uuid4()), "request_sha256": call["sha256"]})
    started = time.monotonic()
    try:
        # Client has no Agent tools. Returned tool calls are data only.
        with client_factory() as client:
            response = client.chat.completions.create(**payload)
        result = {"status": "returned", "response": response.model_dump(mode="json")}
    except Exception as exc:
        # No exception strings/headers: provider errors may contain credentials.
        result = {"status": "error", "error_type": type(exc).__name__,
                  "http_status": getattr(exc, "status_code", None)}
    result.update(task=call["task"], condition=call["condition"], completed_at=now(),
                  latency_seconds=round(time.monotonic() - started, 3))
    save(output / f"{name}.result.json", result)
    print(json.dumps({k: result[k] for k in ("task", "condition", "status", "latency_seconds")}), flush=True)
    return result


def run(manifest, output, client_factory):
    payloads = validate_plan(manifest, output)
    # A killed or interrupted run cannot be resumed, avoiding duplicate billing.
    save(output / "RUN_STARTED.json", {"at": now(), "manifest_sha256": sha((output / "manifest.json").read_bytes())})
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i in range(0, MAX_ATTEMPTS, 2):
            jobs = [pool.submit(one_attempt, c, p, output, client_factory)
                    for c, p in zip(manifest["calls"][i:i+2], payloads[i:i+2])]
            batch = [j.result() for j in jobs]
            results.extend(batch)
            if any(r.get("http_status") in (400, 401, 403, 404, 429) for r in batch):
                break  # Configuration/auth/quota issues: do not spend more calls.
    save(output / "RUN_FINISHED.json", {"at": now(), "attempts": len(results),
                                       "returned": sum(r["status"] == "returned" for r in results)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-index", type=Path)
    mode.add_argument("--run-paid", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "research-runtime") or output == ROOT / "research-runtime":
        parser.error("Output must be a fresh subdirectory of ignored research-runtime/")
    if args.prepare_index:
        plan = prepare(args.prepare_index, output)
        validate_plan(plan, output)
        print(json.dumps({"prepared": len(plan["calls"]), "paid_calls": 0, "output": str(output)}))
        return
    sys.path.insert(0, str(ROOT))
    from shared.llm import _connection_kwargs
    from shared.config import settings
    from openai import OpenAI
    if settings.system_agent_model != MODEL:
        raise ValueError("Current provider configuration must match archived model")
    connection = _connection_kwargs(MODEL)
    if not connection.get("api_key") or not connection.get("api_base", "").startswith("https://"):
        raise ValueError("Explicit authenticated HTTPS model endpoint required")
    def client_factory():
        return OpenAI(api_key=connection["api_key"], base_url=connection["api_base"],
                      max_retries=0, timeout=300.0)
    run(json.loads((output / "manifest.json").read_text()), output, client_factory)


if __name__ == "__main__":
    main()
