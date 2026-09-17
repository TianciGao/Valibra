"""Offline-only AnswerContract Lite builder/verifier research prototype.

This module deliberately has no provider, database, tool, Main, or candidate-SQL
dependency.  It reads frozen archived snapshots and emits only invariants that a
small deterministic rule can reproduce from provenance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


CONTRACT_VERSION = "answer-contract-lite/v0-prototype"
ALLOWED_SNAPSHOT_KEYS = frozenset(
    {
        "query",
        "follow_up",
        "phase",
        "grounding_revision",
        "state",
        "clarifications",
        "official_definitions",
        "source_path",
        "source_sha256",
    }
)
RATIO_BANNED_MODIFIERS = frozenset(
    {"percent", "percentage", "change", "relative", "normalized", "normalised", "per"}
)
GENERIC_SYMBOL_TOKENS = frozenset(
    {"duration", "time", "hour", "hours", "hrs", "eta", "value", "val", "score", "pct", "percentage"}
)
IDENTIFIER_ALIASES = {"reliab": "reliability", "idx": "index"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_words(value: str) -> tuple[str, ...]:
    words = re.findall(r"[A-Za-z0-9]+", value.replace("\\_", "_"))
    result: list[str] = []
    for word in words:
        lowered = word.casefold()
        lowered = IDENTIFIER_ALIASES.get(lowered, lowered)
        if lowered.endswith("s") and len(lowered) > 4:
            lowered = lowered[:-1]
        result.append(lowered)
    return tuple(result)


def _source_record(
    *,
    source_kind: str,
    source_ref: str,
    source_digest: str,
    source_text: str,
    evidence: str,
    role: str,
) -> dict[str, Any]:
    start = source_text.casefold().find(evidence.casefold())
    if start < 0:
        raise ValueError(f"evidence is absent from source: {evidence!r}")
    return {
        "source_kind": source_kind,
        "source_ref": source_ref,
        "source_digest": source_digest,
        "evidence": source_text[start : start + len(evidence)],
        "byte_start": len(source_text[:start].encode("utf-8")),
        "byte_end": len(source_text[: start + len(evidence)].encode("utf-8")),
        "role": role,
    }


def _snapshot_hash(snapshot: dict[str, Any]) -> str:
    unknown = set(snapshot) - ALLOWED_SNAPSHOT_KEYS
    if unknown:
        raise ValueError(f"snapshot contains forbidden inputs: {sorted(unknown)}")
    return sha256_text(canonical_json(snapshot))


def load_frozen_snapshot(repo_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    source_path = repo_root / fixture["source"]
    source_bytes = source_path.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    if source_sha != fixture["source_sha256"]:
        raise ValueError(f"frozen source SHA changed for {fixture['task']}")
    raw = json.loads(source_bytes)
    runtime = raw.get("valibra:sql_grounding_runtime")
    if not isinstance(runtime, dict):
        raise ValueError("frozen source lacks Grounding runtime")
    state = runtime.get("grounding_state")
    if not isinstance(state, dict):
        raise ValueError("frozen source lacks validated 4D State")
    phase = raw.get("current_phase")
    revision = runtime.get("grounding_revision")
    if not isinstance(phase, int) or not isinstance(revision, int):
        raise ValueError("phase and grounding revision must be integers")

    definitions: list[dict[str, Any]] = []
    for index, event in enumerate(raw.get("tool_trajectory") or []):
        if not isinstance(event, dict) or event.get("phase") != phase:
            continue
        if event.get("tool") not in {
            "get_all_knowledge_definitions",
            "get_knowledge_definition",
        }:
            continue
        result = event.get("result")
        try:
            payload = json.loads(result) if isinstance(result, str) else result
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        event_digest = sha256_text(canonical_json(result))
        for item_index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("definition"), str):
                continue
            if isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int):
                continue
            name = item.get("knowledge")
            if not isinstance(name, str):
                continue
            definitions.append(
                {
                    "id": item["id"],
                    "name": name,
                    "definition": item["definition"],
                    "source_ref": f"{fixture['source']}#tool_trajectory/{index}/item/{item_index}",
                    "source_digest": event_digest,
                }
            )

    return {
        "query": raw.get("user_query"),
        "follow_up": None,
        "phase": phase,
        "grounding_revision": revision,
        "state": copy.deepcopy(state),
        "clarifications": copy.deepcopy(raw.get("valibra:user_clarifications") or []),
        "official_definitions": definitions,
        "source_path": fixture["source"],
        "source_sha256": source_sha,
    }


def _mappings(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    mappings = snapshot["state"].get("column_mapping") or []
    if not isinstance(mappings, list):
        raise ValueError("column_mapping must be a list")
    return mappings


def _mapping_provenance(
    snapshot: dict[str, Any], index: int, mapping: dict[str, Any], *, role: str
) -> dict[str, Any]:
    text = canonical_json(mapping)
    return _source_record(
        source_kind="column_mapping",
        source_ref=f"{snapshot['source_path']}#state/column_mapping/{index}",
        source_digest=sha256_text(text),
        source_text=text,
        evidence=mapping["targets"][0] if len(mapping.get("targets") or []) == 1 else mapping["phrase"],
        role=role,
    )


def _query_provenance(snapshot: dict[str, Any], evidence: str, *, role: str) -> dict[str, Any]:
    query = snapshot["query"]
    return _source_record(
        source_kind="query",
        source_ref=f"{snapshot['source_path']}#user_query",
        source_digest=sha256_text(query),
        source_text=query,
        evidence=evidence,
        role=role,
    )


def _clarification_provenance(
    snapshot: dict[str, Any], index: int, record: dict[str, Any], evidence: str, *, role: str
) -> dict[str, Any]:
    text = record["answer"]
    return _source_record(
        source_kind="user_clarification",
        source_ref=f"{snapshot['source_path']}#clarifications/{index}",
        source_digest=sha256_text(canonical_json(record)),
        source_text=text,
        evidence=evidence,
        role=role,
    )


def _unique_official_for_content(
    snapshot: dict[str, Any], content: str
) -> dict[str, Any] | None:
    matches = [item for item in snapshot["official_definitions"] if item["definition"] == content]
    identities = {(item["id"], item["name"], item["definition"]) for item in matches}
    if len(identities) != 1:
        return None
    return matches[0]


def _official_provenance(item: dict[str, Any], evidence: str, *, role: str) -> dict[str, Any]:
    return _source_record(
        source_kind="official_knowledge",
        source_ref=item["source_ref"],
        source_digest=item["source_digest"],
        source_text=item["definition"],
        evidence=evidence,
        role=role,
    )


def _resolve_symbol_to_target(
    symbol: str, mappings: list[dict[str, Any]]
) -> tuple[int, dict[str, Any], str] | None:
    symbol_tokens = [
        token for token in _normalized_words(symbol) if token not in GENERIC_SYMBOL_TOKENS
    ]
    if not symbol_tokens:
        return None
    matches: list[tuple[int, dict[str, Any], str]] = []
    for index, mapping in enumerate(mappings):
        for target in mapping.get("targets") or []:
            target_tokens = set(_normalized_words(target))
            if all(token in target_tokens for token in symbol_tokens):
                matches.append((index, mapping, target))
    return matches[0] if len(matches) == 1 else None


def _normalized_invariant(
    *,
    kind: str,
    payload: dict[str, Any],
    canonicalizer: str,
    provenance: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    semantic = {
        "kind": kind,
        "payload": payload,
        "authority_class": "B",
        "canonicalizer": canonicalizer,
    }
    return {
        **semantic,
        "invariant_id": f"inv-{sha256_text(canonical_json(semantic))[:16]}",
        "provenance": list(provenance),
        "normalized_sha256": sha256_text(canonical_json(semantic)),
    }


def _ratio_invariants(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    query = snapshot["query"]
    folded = query.casefold()
    candidates: list[dict[str, Any]] = []
    mappings = _mappings(snapshot)
    for left_index, left in enumerate(mappings):
        if len(left.get("targets") or []) != 1:
            continue
        for right_index, right in enumerate(mappings):
            if left_index == right_index or len(right.get("targets") or []) != 1:
                continue
            phrase = f"ratio of {left['phrase']} to {right['phrase']}"
            position = folded.find(phrase.casefold())
            if position < 0:
                continue
            window = folded[max(0, position - 24) : position + len(phrase) + 24]
            if any(re.search(rf"\b{re.escape(token)}\b", window) for token in RATIO_BANNED_MODIFIERS):
                continue
            payload = {
                "expression_ast": {
                    "op": "div",
                    "args": [
                        {"field": left["targets"][0]},
                        {"field": right["targets"][0]},
                    ],
                }
            }
            candidates.append(
                _normalized_invariant(
                    kind="formula",
                    payload=payload,
                    canonicalizer="ratio_of_to/v1",
                    provenance=[
                        _query_provenance(snapshot, query[position : position + len(phrase)], role="operator"),
                        _mapping_provenance(snapshot, left_index, left, role="operand"),
                        _mapping_provenance(snapshot, right_index, right, role="operand"),
                    ],
                )
            )
    return candidates if len(candidates) == 1 else []


_LATEX_FRACTION_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 _-]*)\s*=\s*\\frac\{(?P<num>[^{}]+)\}\{(?P<den>[^{}]+)\}\s*\\times\s*(?P<factor>\d+)\\%"
)


def _official_formula_invariants(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    mappings = _mappings(snapshot)
    for knowledge in snapshot["state"].get("domain_knowledge") or []:
        content = knowledge.get("content")
        if not isinstance(content, str):
            continue
        official = _unique_official_for_content(snapshot, content)
        if official is None:
            continue
        matches = list(_LATEX_FRACTION_RE.finditer(content))
        if len(matches) != 1:
            continue
        match = matches[0]
        numerator = _resolve_symbol_to_target(match.group("num"), mappings)
        denominator = _resolve_symbol_to_target(match.group("den"), mappings)
        if numerator is None or denominator is None or numerator[2] == denominator[2]:
            continue
        payload = {
            "expression_ast": {
                "op": "mul",
                "args": [
                    {
                        "op": "div",
                        "args": [
                            {"field": numerator[2]},
                            {"field": denominator[2]},
                        ],
                    },
                    {"literal": match.group("factor")},
                ],
            }
        }
        result.append(
            _normalized_invariant(
                kind="formula",
                payload=payload,
                canonicalizer="formula_expression/v1",
                provenance=[
                    _official_provenance(official, match.group(0), role="operator"),
                    _mapping_provenance(snapshot, numerator[0], numerator[1], role="operand"),
                    _mapping_provenance(snapshot, denominator[0], denominator[1], role="operand"),
                ],
            )
        )
    return result


_DIFFERENCE_RE = re.compile(
    r"difference between (?P<left>actual (?:delivery )?duration) and "
    r"(?P<right>planned (?:delivery )?(?:duration|time)) exceeds "
    r"(?P<literal>\d+(?:\.\d+)?) (?P<unit>hours?)",
    re.IGNORECASE,
)


def _difference_predicates(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    mappings = _mappings(snapshot)
    for clarification_index, record in enumerate(snapshot["clarifications"]):
        answer = record.get("answer")
        phrase = record.get("phrase")
        if not isinstance(answer, str) or not isinstance(phrase, str):
            continue
        matches = list(_DIFFERENCE_RE.finditer(answer))
        phrase_mappings = [
            (index, mapping)
            for index, mapping in enumerate(mappings)
            if mapping.get("phrase") == phrase and len(mapping.get("targets") or []) == 2
        ]
        if len(matches) != 1 or len(phrase_mappings) != 1:
            continue
        mapping_index, mapping = phrase_mappings[0]
        targets = mapping["targets"]
        actual = [target for target in targets if "actual" in _normalized_words(target)]
        planned = [target for target in targets if "planned" in _normalized_words(target)]
        if len(actual) != 1 or len(planned) != 1:
            continue
        match = matches[0]
        payload = {
            "target_ast": {
                "op": "sub",
                "args": [{"field": actual[0]}, {"field": planned[0]}],
            },
            "operator": ">",
            "literal": {
                "type": "number",
                "value": match.group("literal"),
                "unit": "hour",
            },
            "boundary_semantics": "exclusive",
        }
        result.append(
            _normalized_invariant(
                kind="predicate",
                payload=payload,
                canonicalizer="difference_exceeds/v1",
                provenance=[
                    _clarification_provenance(snapshot, clarification_index, record, match.group(0), role="operator"),
                    _mapping_provenance(snapshot, mapping_index, mapping, role="target"),
                ],
            )
        )
    return result


_COMPARISON_RE = re.compile(
    r"(?P<operator>greater than|more than|above|less than|below)\s+"
    r"(?P<literal>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|hours?)?",
    re.IGNORECASE,
)


def _comparison_operator(wording: str) -> str:
    return ">" if wording.casefold() in {"greater than", "more than", "above"} else "<"


def _following_singleton_mapping(
    snapshot: dict[str, Any], clarification_phrase: str
) -> tuple[int, dict[str, Any]] | None:
    # V0 deliberately does not generalize comparison-bearing phrases.  A
    # phrase such as "mostly" may qualify a ratio/formula rather than the raw
    # categorical field that happens to follow it.  Only a standalone high/low
    # modifier immediately followed by one mapped metric is in this whitelist.
    if clarification_phrase.casefold() not in {"high", "low"}:
        return None
    query = snapshot["query"]
    folded = query.casefold()
    phrase_position = folded.find(clarification_phrase.casefold())
    if phrase_position < 0:
        return None
    after = folded[phrase_position + len(clarification_phrase) :].lstrip()
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, mapping in enumerate(_mappings(snapshot)):
        if len(mapping.get("targets") or []) != 1:
            continue
        mapping_phrase = mapping.get("phrase")
        if isinstance(mapping_phrase, str) and after.startswith(mapping_phrase.casefold()):
            candidates.append((index, mapping))
    return candidates[0] if len(candidates) == 1 else None


def _clarification_comparisons(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for clarification_index, record in enumerate(snapshot["clarifications"]):
        answer = record.get("answer")
        phrase = record.get("phrase")
        if not isinstance(answer, str) or not isinstance(phrase, str):
            continue
        mapping_match = _following_singleton_mapping(snapshot, phrase)
        if mapping_match is None:
            continue
        comparison_matches = list(_COMPARISON_RE.finditer(answer))
        normalized = {
            (_comparison_operator(match.group("operator")), match.group("literal"), match.group("unit"))
            for match in comparison_matches
        }
        if len(normalized) != 1:
            continue
        operator, literal, unit = next(iter(normalized))
        mapping_index, mapping = mapping_match
        evidence_match = comparison_matches[0]
        payload = {
            "target": {"field": mapping["targets"][0]},
            "operator": operator,
            "literal": {"type": "number", "value": literal},
            "boundary_semantics": "exclusive",
        }
        if unit:
            payload["literal"]["unit"] = "%" if unit.casefold() in {"%", "percent"} else "hour"
        result.append(
            _normalized_invariant(
                kind="predicate",
                payload=payload,
                canonicalizer="comparison_phrase/v1",
                provenance=[
                    _clarification_provenance(snapshot, clarification_index, record, evidence_match.group(0), role="operator"),
                    _mapping_provenance(snapshot, mapping_index, mapping, role="target"),
                ],
            )
        )
    return result


_BANDS_RE = re.compile(
    r"The (?P<subject>[A-Za-z ]+?) typically ranges .*?values below (?P<low>\d+(?:\.\d+)?)"
    r".*?values between (?P<middle_low>\d+(?:\.\d+)?)-(?P<middle_high>\d+(?:\.\d+)?)"
    r".*?values above (?P<high>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _subject_target(
    subject: str, mappings: list[dict[str, Any]]
) -> tuple[int, dict[str, Any], str] | None:
    tokens = set(_normalized_words(subject))
    matches: list[tuple[int, dict[str, Any], str]] = []
    for index, mapping in enumerate(mappings):
        if len(mapping.get("targets") or []) != 1:
            continue
        target = mapping["targets"][0]
        if tokens.issubset(set(_normalized_words(target))):
            matches.append((index, mapping, target))
    return matches[0] if len(matches) == 1 else None


def _numeric_bands(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    mappings = _mappings(snapshot)
    for knowledge in snapshot["state"].get("domain_knowledge") or []:
        content = knowledge.get("content")
        if not isinstance(content, str):
            continue
        official = _unique_official_for_content(snapshot, content)
        if official is None:
            continue
        matches = list(_BANDS_RE.finditer(content))
        if len(matches) != 1:
            continue
        match = matches[0]
        target_match = _subject_target(match.group("subject"), mappings)
        if target_match is None:
            continue
        mapping_index, mapping, target = target_match
        if match.group("low") != match.group("middle_low") or match.group("high") != match.group("middle_high"):
            continue
        payload = {
            "target": {"field": target},
            "bands": [
                {"operator": "<", "literal": match.group("low")},
                {
                    "lower": match.group("middle_low"),
                    "upper": match.group("middle_high"),
                    "boundary_semantics": "unspecified",
                },
                {"operator": ">", "literal": match.group("high")},
            ],
        }
        result.append(
            _normalized_invariant(
                kind="predicate_band_set",
                payload=payload,
                canonicalizer="numeric_band/v1",
                provenance=[
                    _official_provenance(official, match.group(0), role="literal"),
                    _mapping_provenance(snapshot, mapping_index, mapping, role="target"),
                ],
            )
        )
    return result


def _ordering_invariants(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    mappings = _mappings(snapshot)
    for clarification_index, record in enumerate(snapshot["clarifications"]):
        answer = record.get("answer")
        if not isinstance(answer, str):
            continue
        for mapping_index, mapping in enumerate(mappings):
            if len(mapping.get("targets") or []) != 1:
                continue
            pattern = re.compile(
                rf"sorted by (?:the )?{re.escape(mapping['phrase'])} in (ascending|descending) order",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(answer))
            if len(matches) != 1:
                continue
            direction = "ASC" if matches[0].group(1).casefold() == "ascending" else "DESC"
            result.append(
                _normalized_invariant(
                    kind="ordering",
                    payload={"target": {"field": mapping["targets"][0]}, "direction": direction},
                    canonicalizer="sort_direction/v1",
                    provenance=[
                        _clarification_provenance(snapshot, clarification_index, record, matches[0].group(0), role="direction"),
                        _mapping_provenance(snapshot, mapping_index, mapping, role="target"),
                    ],
                )
            )
    query = snapshot["query"]
    for mapping_index, mapping in enumerate(mappings):
        phrase = mapping.get("phrase")
        if not isinstance(phrase, str) or len(mapping.get("targets") or []) != 1:
            continue
        first = phrase.casefold().split(maxsplit=1)[0]
        if first not in {"least", "lowest", "highest", "greatest"}:
            continue
        position = query.casefold().find(phrase.casefold())
        if position < 0:
            continue
        direction = "ASC" if first in {"least", "lowest"} else "DESC"
        result.append(
            _normalized_invariant(
                kind="ordering",
                payload={"target": {"field": mapping["targets"][0]}, "direction": direction},
                canonicalizer="extremum_order/v1",
                provenance=[
                    _query_provenance(snapshot, query[position : position + len(phrase)], role="direction"),
                    _mapping_provenance(snapshot, mapping_index, mapping, role="target"),
                ],
            )
        )
    return result


def _explicit_average(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    query = snapshot["query"]
    folded = query.casefold()
    result: list[dict[str, Any]] = []
    for mapping_index, mapping in enumerate(_mappings(snapshot)):
        if len(mapping.get("targets") or []) != 1:
            continue
        phrase = f"average {mapping['phrase']}"
        position = folded.find(phrase.casefold())
        if position < 0:
            continue
        result.append(
            _normalized_invariant(
                kind="aggregation",
                payload={"function": "AVG", "target": {"field": mapping["targets"][0]}, "grain": None},
                canonicalizer="aggregation_function/v1",
                provenance=[
                    _query_provenance(snapshot, query[position : position + len(phrase)], role="function"),
                    _mapping_provenance(snapshot, mapping_index, mapping, role="target"),
                ],
            )
        )
    return result


class LiteBuilder:
    """Build only independently reproducible A/B invariants."""

    def build(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        snapshot_sha = _snapshot_hash(snapshot)
        invariants: list[dict[str, Any]] = []
        for builder in (
            _ratio_invariants,
            _official_formula_invariants,
            _difference_predicates,
            _clarification_comparisons,
            _numeric_bands,
            _ordering_invariants,
            _explicit_average,
        ):
            invariants.extend(builder(snapshot))
        by_semantic_hash = {item["normalized_sha256"]: item for item in invariants}
        verified = sorted(by_semantic_hash.values(), key=lambda item: item["normalized_sha256"])
        envelope: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "phase": snapshot["phase"],
            "grounding_revision": snapshot["grounding_revision"],
            "source_snapshot_sha256": snapshot_sha,
            "verified_invariants": verified,
            "omissions": [] if verified else [{"reason": "NO_VERIFIABLE_A_OR_B_INVARIANT"}],
        }
        envelope["contract_sha256"] = sha256_text(canonical_json(envelope))
        return envelope


class LiteVerifier:
    """Rebuild from provenance-bearing input; never trust contract shape alone."""

    def __init__(self, builder: LiteBuilder | None = None) -> None:
        self.builder = builder or LiteBuilder()

    def verify(self, snapshot: dict[str, Any], contract: dict[str, Any]) -> tuple[bool, str]:
        claimed = copy.deepcopy(contract)
        claimed_hash = claimed.pop("contract_sha256", None)
        if claimed_hash != sha256_text(canonical_json(claimed)):
            return False, "CONTRACT_HASH_MISMATCH"
        try:
            rebuilt = self.builder.build(snapshot)
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"SOURCE_REBUILD_FAILED:{type(exc).__name__}"
        if canonical_json(rebuilt) != canonical_json(contract):
            return False, "PROVENANCE_REPLAY_MISMATCH"
        return True, "VERIFIED"


def semantic_projection(contract: dict[str, Any]) -> list[dict[str, Any]]:
    result = [
        {
            "kind": item["kind"],
            "payload": item["payload"],
            "authority_class": item["authority_class"],
            "canonicalizer": item["canonicalizer"],
        }
        for item in contract["verified_invariants"]
    ]
    return sorted(result, key=canonical_json)
