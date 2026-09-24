"""Offline, read-only audit of recorded SQL Writer inputs.

Reads an explicitly supplied scored-attempt index and local audit archives.
Prints counts and hashes, never prompts, SQL, credentials, or reference answers.
No runtime imports, model calls, database connections, or session restoration.
Literal absence is a lexical diagnostic, NOT evidence of a wrong answer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

BEGIN = "[VALIBRA SQL WRITER CONTEXT BEGIN]"
END = "[VALIBRA SQL WRITER CONTEXT END]"
VIEW = "[VALIBRA DATABASE GROUNDING]"
SECTIONS = {"Tables:", "Relations:", "Column mappings:", "Domain knowledge:",
            "[USER CLARIFICATIONS]"}


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Archive timestamp must include a timezone")
    return parsed


def writer_input(request: dict) -> dict | None:
    instruction = request.get("config", {}).get("system_instruction", "")
    if not isinstance(instruction, str) or BEGIN not in instruction:
        return None
    context = instruction.split(BEGIN, 1)[1].split(END, 1)[0]
    phase = re.search(r"^Phase: ([12])$", context, re.M)
    query = re.search(r"^Original Query: (.*?)\nFollow-up:", context, re.M | re.S)
    if not phase or not query or VIEW not in context or END not in instruction:
        raise ValueError("Incomplete recorded Writer context")
    view = context.split(VIEW, 1)[1]
    view = re.split(r"\n(?:\[MAIN EXECUTION|\[SQL-SAFE|Remaining Bird-Coin:)", view, 1)[0]
    section = None
    targets = set()
    item_counts = Counter()
    for line in view.splitlines():
        if line in SECTIONS:
            section = line
        elif line.startswith("- "):
            payload = json.loads(line[2:])
            item_counts[section] += 1
            if section == "Column mappings:" and isinstance(payload, dict):
                targets.update(payload.get("targets", []))
    omissions = re.search(r"^\.\.\. omitted=(\d+)$", view, re.M)
    # Only text actually sent to Main, not later responses or final state.
    visible = [instruction]
    for content in request.get("contents", []):
        for part in content.get("parts", []):
            if isinstance(part.get("text"), str):
                visible.append(part["text"])
    return {"phase": int(phase[1]), "query": query[1], "targets": sorted(targets),
            "view": VIEW + view.rstrip(), "visible": "\n".join(visible),
            "omitted_items": int(omissions[1]) if omissions else 0,
            "item_counts": dict(item_counts)}


def target_description(target: str, meanings: dict) -> str | None:
    # Deliberately conservative: unsupported SQL expressions are not guessed.
    match = re.fullmatch(r'(\w+)\.(\w+)((?:\s*->>?\s*\'[^\']+\')*)', target)
    if not match:
        return None
    entry = meanings.get(f"{match[1]}.{match[2]}")
    if entry is None:
        return None
    keys = re.findall(r"->>?\s*'([^']+)'", match[3])
    if keys:
        if not isinstance(entry, dict):
            return None
        entry = entry.get("fields_meaning", {})
        for key in keys:
            if not isinstance(entry, dict) or key not in entry:
                return None
            entry = entry[key]
    return entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False, sort_keys=True)


def lexical_coverage(targets: list[str], meanings: dict, visible: str) -> dict:
    counts = Counter(mapped_targets=len(targets))
    for target in targets:
        description = target_description(target, meanings)
        if description is None:
            counts["targets_without_resolved_description"] += 1
            continue
        counts["targets_with_resolved_description"] += 1
        if not re.search(r"\benum(?:eration|erated|erations|s)?\b", description, re.I):
            continue
        literals = set(re.findall(r"'([^'\n]+)'", description))
        if not literals:
            continue
        missing = {literal for literal in literals if literal not in visible}
        counts["enum_labelled_targets"] += 1
        counts["quoted_literals"] += len(literals)
        counts["quoted_literals_absent"] += len(missing)
        counts["enum_targets_with_absent_literals"] += bool(missing)
    return dict(counts)


def prior_meanings(directory: Path, query: str, cutoff: datetime) -> tuple[dict, dict]:
    descriptions = {}
    counts = Counter()
    prior = []
    for path in sorted((directory / "provider_audits").glob("*.json")):
        audit = json.loads(path.read_text())
        if not audit.get("completed_at"):
            counts["audits_without_completion_time"] += 1
            continue
        if timestamp(audit["completed_at"]) > cutoff:
            continue
        payload = json.loads(audit["request"]["messages"][-1]["content"])
        if payload.get("query", payload.get("original_query")) != query:
            counts["foreign_query_audits_excluded"] += 1
            continue
        prior.append((audit["completed_at"], path.name, payload))
    for _, _, payload in sorted(prior, key=lambda item: (timestamp(item[0]), item[1])):
        for key in ("column_meanings", "relevant_column_meanings"):
            if isinstance(payload.get(key), dict):
                descriptions.update(payload[key])
                counts["prior_meaning_payloads"] += 1
    counts["prior_audits"] = len(prior)
    return descriptions, dict(counts)


def audit_task(root: Path, row: dict) -> dict:
    directory = (root / row["candidate_source"]).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Task archive must be within the supplied repository root")
    raw = (directory / "official_result.json").read_bytes()
    result = json.loads(raw)
    if (result.get("instance_id") or result.get("task_id")) != row["task"]:
        raise ValueError("Scored-attempt task identity mismatch")
    first = {}
    calls = Counter()
    for flow in result["prompt_flow"]:
        data = writer_input(flow.get("request") or {})
        if data is None:
            continue
        phase = data["phase"]
        calls[phase] += 1
        if phase not in first:
            first[phase] = (timestamp(flow["timestamp"]), flow.get("call_index"), data)
    phases = []
    for phase, (cutoff, index, data) in sorted(first.items()):
        meanings, evidence_counts = prior_meanings(directory, data["query"], cutoff)
        phases.append({"phase": phase, "first_call_index": index,
                       "writer_calls": calls[phase], "omitted_items": data["omitted_items"],
                       "view_chars": len(data["view"]), "view_sha256": digest(data["view"].encode()),
                       "view_item_counts": data["item_counts"],
                       "evidence_counts": evidence_counts,
                       "lexical_coverage": lexical_coverage(data["targets"], meanings, data["visible"])})
    return {"task": row["task"], "baseline": row["baseline"], "candidate": row["candidate"],
            "primary_p1": row["primary_p1"], "primary_full": row["primary_full"],
            "fallback": row["fallback"], "replaced": row["replaced"],
            "source_sha256": digest(raw), "phases": phases}


def summarize(records: list[dict]) -> dict:
    counts = Counter(tasks=len(records))
    phase_counts = {1: Counter(), 2: Counter()}
    groups = {}
    flagged = []
    for row in records:
        counts["fallback_tasks"] += row["fallback"]
        counts["replacement_tasks"] += row["replaced"]
        counts["tasks_without_writer"] += not row["phases"]
        for p in row["phases"]:
            c = phase_counts[p["phase"]]
            c["tasks_reaching_writer"] += 1
            c["first_views_with_omissions"] += p["omitted_items"] > 0
            c["omitted_items"] += p["omitted_items"]
            c["primary_passed"] += bool(row["primary_p1" if p["phase"] == 1 else "primary_full"])
            c.update(p["lexical_coverage"])
            c["first_views_with_enum_literal_absences"] += p["lexical_coverage"].get("quoted_literals_absent", 0) > 0
            group = f"P{p['phase']}:" + ("truncated" if p["omitted_items"] else "not_truncated")
            groups.setdefault(group, Counter())["tasks"] += 1
            groups[group]["primary_passed"] += bool(row["primary_p1" if p["phase"] == 1 else "primary_full"])
            if p["omitted_items"] or p["lexical_coverage"].get("quoted_literals_absent", 0):
                flagged.append({"task": row["task"], "phase": p["phase"],
                                "baseline": row["baseline"], "candidate": row["candidate"],
                                "omitted_items": p["omitted_items"],
                                "quoted_literals_absent": p["lexical_coverage"].get("quoted_literals_absent", 0)})
    return {"counts": dict(counts), "by_phase": {str(k): dict(v) for k, v in phase_counts.items()},
            "observational_groups_not_causal": {k: dict(v) for k, v in groups.items()},
            "flagged_first_inputs_not_semantic_verdicts": flagged}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    raw = args.index.read_bytes()
    rows = json.loads(raw)
    if len({r["task"] for r in rows}) != len(rows):
        raise ValueError("Duplicate tasks in scored-attempt index")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    records = []
    for i, row in enumerate(rows, 1):
        records.append(audit_task(args.root, row))
        if i % 50 == 0:
            print(f"Audited {i}/{len(rows)} scored attempts", file=sys.stderr, flush=True)
    output = {"schema_version": "main-handoff-offline-audit/v1", "index_sha256": digest(raw),
              "scope": "First actual SQL Writer request per phase of each selected scored attempt.",
              "limitations": ["No new model or database calls; cannot estimate intervention success.",
                              "Literal presence/absence is lexical, not semantic sufficiency or correctness.",
                              "Only enum-labelled, quoted descriptions and conservatively parsed targets are counted.",
                              "Only matching-query audit payloads completed before the Writer request are used.",
                              "Later Writer retries and unavailable descriptions are not evidence of successful handoff.",
                              "Outcome groups are observational and confounded by task difficulty."],
              "summary": summarize(records)}
    if not args.summary_only:
        output["tasks"] = records
    output["records_sha256"] = digest(json.dumps(records, sort_keys=True).encode())
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
