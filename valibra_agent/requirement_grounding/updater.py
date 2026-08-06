"""NoOp, deterministic Rule, and offline-testable LLM updater contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ConfigDict, Field, JsonValue, model_validator

from shared.model_presets import (
    canonical_json as preset_canonical_json,
    load_model_preset,
)

from valibra_agent.requirement_grounding.linguistic_hints import (
    LinguisticHint,
    extract_linguistic_hints,
)
from valibra_agent.requirement_grounding.models import (
    GroundingEvidence,
    GroundingSlot,
    KernelModel,
    Observation,
    OperationSlot,
    RequirementGroundingPatch,
    RequirementGroundingState,
    SchemaSlot,
    ValueSlot,
)
from valibra_agent.requirement_grounding.telemetry import (
    LLMCallTelemetryRecorder,
)


GROUNDING_LLM_ENV_NAMES = (
    "GROUNDING_UPDATER_MODE",
    "GROUNDING_MODEL_PRESET",
    "GROUNDING_TIMEOUT_SECONDS",
    "GROUNDING_MAX_TOKENS",
    "GROUNDING_MAX_CALLS_PER_TASK",
    "GROUNDING_PROMPT_SHA256",
)

LLM_FRAME_PROMPT = """You extract a provisional requirement frame from bounded JSON input.
Return exactly one JSON object with keys value_slots, schema_slots,
operation_slots, and ambiguities. Do not return Markdown or prose.

Each value slot has exactly: slot_role, mention, interpretation, value_type.
Each schema slot has exactly: slot_role, mention, interpretation. It is only a
natural-language candidate; never emit a table, column, binding, identifier,
database name, confidence, or hidden fact.
Each operation slot has exactly: slot_role, mention, interpretation,
operation_type, parameters. Parameters must contain JSON scalar values only.
ambiguities must be an empty array.

Only use the supplied Observation and current bounded State. All output is
provisional/hypothesized. If evidence is insufficient, return empty arrays.
Never infer from ground truth, test cases, hidden follow-up, schema contents,
prompt history, audit logs, tools, network resources, or outside knowledge."""
LLM_FRAME_PROMPT_SHA256 = hashlib.sha256(
    LLM_FRAME_PROMPT.encode("utf-8")
).hexdigest()

MAX_LLM_FRAME_INPUT_CHARS = 262_144
MAX_LLM_FRAME_RESPONSE_CHARS = 65_536
MAX_LLM_FRAME_SLOTS = 64


class _StrictLLMModel(KernelModel):
    model_config = ConfigDict(**KernelModel.model_config, strict=True)


class GroundingLLMConfig(_StrictLLMModel):
    """Frozen, self-verifying configuration for one LLM Updater experiment."""

    mode: Literal["llm"] = "llm"
    model_preset: str = Field(min_length=1, max_length=128)
    preset_config_json: str = Field(min_length=2, max_length=8192)
    preset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeout_seconds: float = Field(gt=0.0, le=600.0)
    max_tokens: int = Field(ge=1, le=131_072)
    max_calls_per_task: int = Field(ge=1, le=128)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_frozen_hashes(self) -> "GroundingLLMConfig":
        if self.prompt_sha256 != LLM_FRAME_PROMPT_SHA256:
            raise ValueError("GROUNDING_PROMPT_SHA256 does not match frozen prompt")
        try:
            preset_config = json.loads(self.preset_config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("preset_config_json must be valid JSON") from exc
        if not isinstance(preset_config, dict):
            raise ValueError("preset_config_json must encode an object")
        if preset_canonical_json(preset_config) != self.preset_config_json:
            raise ValueError("preset_config_json must be canonical JSON")
        actual_preset_sha = hashlib.sha256(
            self.preset_config_json.encode("utf-8")
        ).hexdigest()
        if actual_preset_sha != self.preset_sha256:
            raise ValueError("normalized model preset SHA256 mismatch")
        preset_max_tokens = preset_config.get("max_tokens")
        if (
            isinstance(preset_max_tokens, bool)
            or not isinstance(preset_max_tokens, int)
            or self.max_tokens > preset_max_tokens
        ):
            raise ValueError(
                "GROUNDING_MAX_TOKENS must not exceed preset max_tokens"
            )
        if self.configuration_sha256 != _grounding_configuration_sha256(
            mode=self.mode,
            model_preset=self.model_preset,
            preset_sha256=self.preset_sha256,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            max_calls_per_task=self.max_calls_per_task,
            prompt_sha256=self.prompt_sha256,
        ):
            raise ValueError("grounding configuration SHA256 mismatch")
        return self

    @property
    def preset_config(self) -> dict[str, JsonValue]:
        return json.loads(self.preset_config_json)


class GroundingLLMRequest(_StrictLLMModel):
    prompt: str = Field(min_length=1, max_length=MAX_LLM_FRAME_INPUT_CHARS)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_preset: str = Field(min_length=1, max_length=128)
    preset_config: dict[str, JsonValue]
    max_tokens: int = Field(ge=1, le=131_072)


class GroundingLLMUsage(_StrictLLMModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_finite_cost(self) -> "GroundingLLMUsage":
        if not math.isfinite(self.cost):
            raise ValueError("LLM cost must be finite")
        return self


class GroundingLLMResponse(_StrictLLMModel):
    content: str = Field(max_length=MAX_LLM_FRAME_RESPONSE_CHARS)
    usage: GroundingLLMUsage = Field(default_factory=GroundingLLMUsage)


class AsyncGroundingLLMClient(Protocol):
    """Injectable async boundary; P4.2a tests provide only fake clients."""

    provider_may_continue_after_cancel: bool
    provider_may_bill_after_cancel: bool

    async def complete(
        self,
        request: GroundingLLMRequest,
    ) -> GroundingLLMResponse | Mapping[str, Any]: ...


class LLMValueSlotProposal(_StrictLLMModel):
    slot_role: str = Field(min_length=1, max_length=64)
    mention: str = Field(max_length=256)
    interpretation: str = Field(min_length=1, max_length=512)
    value_type: str | None = Field(default=None, max_length=64)


class LLMSchemaSlotProposal(_StrictLLMModel):
    slot_role: str = Field(min_length=1, max_length=64)
    mention: str = Field(max_length=256)
    interpretation: str = Field(min_length=1, max_length=512)


class LLMOperationSlotProposal(_StrictLLMModel):
    slot_role: str = Field(min_length=1, max_length=64)
    mention: str = Field(max_length=256)
    interpretation: str = Field(min_length=1, max_length=512)
    operation_type: Literal[
        "projection",
        "filter",
        "aggregation",
        "group",
        "order",
        "limit",
        "distinct",
        "other",
    ]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class LLMFrameProposal(_StrictLLMModel):
    value_slots: list[LLMValueSlotProposal] = Field(max_length=MAX_LLM_FRAME_SLOTS)
    schema_slots: list[LLMSchemaSlotProposal] = Field(max_length=MAX_LLM_FRAME_SLOTS)
    operation_slots: list[LLMOperationSlotProposal] = Field(max_length=MAX_LLM_FRAME_SLOTS)
    ambiguities: list[dict[str, JsonValue]] = Field(max_length=0)

    @model_validator(mode="after")
    def validate_total_slot_limit(self) -> "LLMFrameProposal":
        total = len(self.value_slots) + len(self.schema_slots) + len(self.operation_slots)
        if total > MAX_LLM_FRAME_SLOTS:
            raise ValueError("LLM Frame proposal exceeds total slot limit")
        return self


class LLMUpdater:
    """Async LLM Frame contract; it is not wired into the current Agent."""

    mode = "llm"

    def __init__(
        self,
        client: AsyncGroundingLLMClient,
        config: GroundingLLMConfig,
    ) -> None:
        self.client = client
        self.config = GroundingLLMConfig.model_validate(
            config.model_dump(mode="python")
        )

    async def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
        telemetry: LLMCallTelemetryRecorder,
    ) -> RequirementGroundingPatch:
        detached_observation = Observation.model_validate(
            observation.model_dump(mode="python")
        )
        detached_state = RequirementGroundingState.model_validate(
            state.model_dump(mode="python")
        )
        request = _build_llm_request(
            detached_observation,
            detached_state,
            base_revision=base_revision,
            config=self.config,
        )
        telemetry.mark_provider_attempted()
        raw_response = await self.client.complete(request)
        response = GroundingLLMResponse.model_validate(raw_response)
        telemetry.capture_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            cost=response.usage.cost,
        )
        proposal = _parse_llm_frame_response(response.content)
        return _llm_patch_from_proposal(
            detached_observation,
            detached_state,
            proposal,
            base_revision=base_revision,
        )


def load_grounding_llm_config(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
) -> GroundingLLMConfig:
    """Load but never activate a model preset, then freeze all LLM settings."""

    source = environment if environment is not None else os.environ
    missing = [name for name in GROUNDING_LLM_ENV_NAMES if not source.get(name)]
    if missing:
        raise ValueError(f"missing frozen Grounding configuration: {missing}")
    if source["GROUNDING_UPDATER_MODE"] != "llm":
        raise ValueError("GROUNDING_UPDATER_MODE must be exactly 'llm'")
    if source["GROUNDING_PROMPT_SHA256"] != LLM_FRAME_PROMPT_SHA256:
        raise ValueError("GROUNDING_PROMPT_SHA256 does not match frozen prompt")

    preset = load_model_preset(project_root, source["GROUNDING_MODEL_PRESET"])
    timeout_seconds = _parse_positive_float(
        source["GROUNDING_TIMEOUT_SECONDS"],
        "GROUNDING_TIMEOUT_SECONDS",
    )
    max_tokens = _parse_positive_int(
        source["GROUNDING_MAX_TOKENS"],
        "GROUNDING_MAX_TOKENS",
    )
    max_calls = _parse_positive_int(
        source["GROUNDING_MAX_CALLS_PER_TASK"],
        "GROUNDING_MAX_CALLS_PER_TASK",
    )
    preset_json = preset_canonical_json(preset.normalized_config)
    config_sha = _grounding_configuration_sha256(
        mode="llm",
        model_preset=preset.name,
        preset_sha256=preset.normalized_sha256,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_calls_per_task=max_calls,
        prompt_sha256=LLM_FRAME_PROMPT_SHA256,
    )
    return GroundingLLMConfig(
        mode="llm",
        model_preset=preset.name,
        preset_config_json=preset_json,
        preset_sha256=preset.normalized_sha256,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_calls_per_task=max_calls,
        prompt_sha256=LLM_FRAME_PROMPT_SHA256,
        configuration_sha256=config_sha,
    )


class NoOpUpdater:
    """Produce a deterministic empty Patch without changing State."""

    mode = "noop"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        # Force contract validation of the supplied State without retaining or
        # mutating it.  K0 deliberately derives no semantic facts.
        RequirementGroundingState.model_validate(state.model_dump(mode="python"))
        return RequirementGroundingPatch(
            patch_id=f"noop:{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            diagnostics=("K0 NoOp updater: no business-state operations",),
        )


class RuleUpdater:
    """Build provisional Frame patches from bounded deterministic hints."""

    mode = "rule"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        detached_state = RequirementGroundingState.model_validate(
            state.model_dump(mode="python")
        )
        if observation.observation_type not in {"user_query", "user_answer"}:
            return _rule_noop_patch(observation, base_revision)

        hints = extract_linguistic_hints(_observation_text(observation))
        if not hints:
            return _rule_noop_patch(observation, base_revision)

        evidence_id = f"evidence.{observation.observation_id}"
        evidence = GroundingEvidence(
            evidence_id=evidence_id,
            observation_id=observation.observation_id,
            source_type=f"rule_{observation.observation_type}",
            phase=observation.phase,
            summary=observation.summary,
            raw_digest=observation.raw_digest,
            raw_log_ref=observation.raw_log_ref,
            sequence=observation.sequence,
            timestamp=None,
        )

        existing = _slot_index(detached_state)
        additions = []
        updates = []
        answer_role_counts = Counter(
            (hint.slot_kind, hint.slot_role) for hint in hints
        )
        for hint in hints:
            stable_id = _slot_id(hint)
            current = existing.get(stable_id)
            if current is not None and current.lifecycle != "active":
                continue
            if current is None and observation.observation_type == "user_answer":
                if answer_role_counts[(hint.slot_kind, hint.slot_role)] != 1:
                    continue
                candidates = [
                    slot
                    for slot in existing.values()
                    if slot.lifecycle == "active"
                    and slot.slot_kind == hint.slot_kind
                    and slot.slot_role == hint.slot_role
                ]
                if len(candidates) > 1:
                    continue
                if len(candidates) == 1:
                    current = candidates[0]

            slot = _slot_from_hint(
                hint,
                slot_id=current.slot_id if current is not None else stable_id,
                evidence_id=evidence_id,
                observation=observation,
                current=current,
            )
            if current is None:
                additions.append(slot)
                existing[slot.slot_id] = slot
            else:
                updates.append(slot)
                existing[slot.slot_id] = slot

        if not additions and not updates:
            return _rule_noop_patch(observation, base_revision)
        return RequirementGroundingPatch(
            patch_id=f"rule.{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            slot_additions=tuple(additions),
            slot_updates=tuple(updates),
            ambiguity_additions=(),
            ambiguity_updates=(),
            evidence_additions=(evidence,),
            diagnostics=(
                "P4.1 deterministic provisional Frame; no ambiguity or schema binding",
            ),
        )


def build_noop_patch(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    return NoOpUpdater().propose(
        observation,
        state,
        base_revision=base_revision,
    )


def build_rule_patch(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    return RuleUpdater().propose(
        observation,
        state,
        base_revision=base_revision,
    )


def _rule_noop_patch(
    observation: Observation,
    base_revision: int,
) -> RequirementGroundingPatch:
    return RequirementGroundingPatch(
        patch_id=f"rule-noop.{observation.observation_id}",
        base_revision=base_revision,
        source_observation_ids=(observation.observation_id,),
        diagnostics=("P4.1 RuleUpdater found no safe provisional Frame change",),
    )


def _observation_text(observation: Observation) -> str:
    # Tool string responses were canonicalized as JSON in P3; decode only that
    # exact scalar shape. No history, hidden state, or audit log is consulted.
    text = observation.summary
    if observation.observation_type == "user_answer" and text.startswith('"'):
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            return text
        if isinstance(decoded, str):
            return decoded
    return text


def _slot_index(state: RequirementGroundingState) -> dict[str, GroundingSlot]:
    frame = state.requirement_frame
    return {
        slot.slot_id: slot
        for slot in (
            *frame.value_slots,
            *frame.schema_slots,
            *frame.operation_slots,
        )
    }


def _slot_id(hint: LinguisticHint) -> str:
    return f"slot.{hint.slot_kind}.{hint.hint_id.removeprefix('hint.')}"


def _slot_from_hint(
    hint: LinguisticHint,
    *,
    slot_id: str,
    evidence_id: str,
    observation: Observation,
    current: GroundingSlot | None,
) -> GroundingSlot:
    existing_evidence = tuple(getattr(current, "evidence_refs", ()))
    evidence_refs = tuple(dict.fromkeys((*existing_evidence, evidence_id)))
    introduced_in_phase = getattr(
        current,
        "introduced_in_phase",
        observation.phase,
    )
    common = {
        "slot_id": slot_id,
        "slot_role": hint.slot_role,
        "mention": hint.mention,
        "current_interpretation": hint.interpretation,
        "grounding_status": "hypothesized",
        "evidence_refs": evidence_refs,
        "ambiguity_refs": tuple(getattr(current, "ambiguity_refs", ())),
        "origin": "rule_provisional",
        "lifecycle": "active",
        "introduced_in_phase": introduced_in_phase,
        "last_updated_phase": observation.phase,
        "sequence": observation.sequence,
    }
    if hint.slot_kind == "value":
        return ValueSlot(**common, value_type=hint.value_type)
    if hint.slot_kind == "schema":
        return SchemaSlot(
            **common,
            binding_type="unknown",
            bound_identifier=None,
        )
    if hint.slot_kind == "operation":
        return OperationSlot(
            **common,
            operation_type=hint.operation_type or "other",
            parameters=dict(hint.parameters),
        )
    raise ValueError(f"unsupported deterministic hint kind: {hint.slot_kind}")


def _grounding_configuration_sha256(
    *,
    mode: str,
    model_preset: str,
    preset_sha256: str,
    timeout_seconds: float,
    max_tokens: int,
    max_calls_per_task: int,
    prompt_sha256: str,
) -> str:
    payload = {
        "GROUNDING_UPDATER_MODE": mode,
        "GROUNDING_MODEL_PRESET": model_preset,
        "GROUNDING_MODEL_PRESET_SHA256": preset_sha256,
        "GROUNDING_TIMEOUT_SECONDS": timeout_seconds,
        "GROUNDING_MAX_TOKENS": max_tokens,
        "GROUNDING_MAX_CALLS_PER_TASK": max_calls_per_task,
        "GROUNDING_PROMPT_SHA256": prompt_sha256,
    }
    return hashlib.sha256(
        preset_canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _parse_positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_positive_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _build_llm_request(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
    config: GroundingLLMConfig,
) -> GroundingLLMRequest:
    payload = {
        "base_revision": base_revision,
        "observation": observation.model_dump(mode="json"),
        "state": state.model_dump(mode="json"),
    }
    dynamic_input = preset_canonical_json(payload)
    prompt = f"{LLM_FRAME_PROMPT}\n\nINPUT_JSON\n{dynamic_input}"
    if len(prompt) > MAX_LLM_FRAME_INPUT_CHARS:
        raise ValueError("bounded LLM Frame input exceeds character limit")
    return GroundingLLMRequest(
        prompt=prompt,
        prompt_sha256=config.prompt_sha256,
        configuration_sha256=config.configuration_sha256,
        model_preset=config.model_preset,
        preset_config=config.preset_config,
        max_tokens=config.max_tokens,
    )


def _parse_llm_frame_response(content: str) -> LLMFrameProposal:
    if not isinstance(content, str):
        raise TypeError("LLM Frame response content must be a string")
    if len(content) > MAX_LLM_FRAME_RESPONSE_CHARS:
        raise ValueError("LLM Frame response exceeds character limit")
    try:
        raw = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("LLM Frame response must be strict JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("LLM Frame response root must be an object")
    return LLMFrameProposal.model_validate(raw)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _llm_patch_from_proposal(
    observation: Observation,
    state: RequirementGroundingState,
    proposal: LLMFrameProposal,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    existing = _slot_index(state)
    evidence_id = f"evidence.{observation.observation_id}"
    additions: list[GroundingSlot] = []
    updates: list[GroundingSlot] = []

    typed_proposals: list[tuple[str, KernelModel]] = [
        *(('value', item) for item in proposal.value_slots),
        *(('schema', item) for item in proposal.schema_slots),
        *(('operation', item) for item in proposal.operation_slots),
    ]
    for slot_kind, item in typed_proposals:
        slot_id = _llm_slot_id(slot_kind, item)
        current = existing.get(slot_id)
        if current is not None and current.lifecycle != "active":
            continue
        slot = _llm_slot_from_proposal(
            slot_kind,
            item,
            slot_id=slot_id,
            evidence_id=evidence_id,
            observation=observation,
            current=current,
        )
        if current is None:
            additions.append(slot)
        else:
            updates.append(slot)
        existing[slot.slot_id] = slot

    if not additions and not updates:
        return RequirementGroundingPatch(
            patch_id=f"llm-noop.{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            ambiguity_additions=(),
            ambiguity_updates=(),
            diagnostics=(
                "P4.2a LLM Frame contract returned no provisional changes",
            ),
        )

    evidence = GroundingEvidence(
        evidence_id=evidence_id,
        observation_id=observation.observation_id,
        source_type=f"llm_{observation.observation_type}",
        phase=observation.phase,
        summary=observation.summary,
        raw_digest=observation.raw_digest,
        raw_log_ref=observation.raw_log_ref,
        sequence=observation.sequence,
        timestamp=None,
    )
    return RequirementGroundingPatch(
        patch_id=f"llm.{observation.observation_id}",
        base_revision=base_revision,
        source_observation_ids=(observation.observation_id,),
        slot_additions=tuple(additions),
        slot_updates=tuple(updates),
        ambiguity_additions=(),
        ambiguity_updates=(),
        evidence_additions=(evidence,),
        diagnostics=(
            "P4.2a provisional LLM Frame; no ambiguity or schema binding",
        ),
    )


def _llm_slot_id(slot_kind: str, proposal: KernelModel) -> str:
    identity = {
        "slot_kind": slot_kind,
        "proposal": proposal.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        preset_canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return f"slot.llm.{slot_kind}.{digest[:24]}"


def _llm_slot_from_proposal(
    slot_kind: str,
    proposal: KernelModel,
    *,
    slot_id: str,
    evidence_id: str,
    observation: Observation,
    current: GroundingSlot | None,
) -> GroundingSlot:
    evidence_refs = tuple(
        dict.fromkeys((*tuple(getattr(current, "evidence_refs", ())), evidence_id))
    )
    common = {
        "slot_id": slot_id,
        "slot_role": proposal.slot_role,
        "mention": proposal.mention,
        "current_interpretation": proposal.interpretation,
        "grounding_status": "hypothesized",
        "evidence_refs": evidence_refs,
        "ambiguity_refs": (),
        "origin": "llm_provisional",
        "lifecycle": "active",
        "introduced_in_phase": getattr(
            current,
            "introduced_in_phase",
            observation.phase,
        ),
        "last_updated_phase": observation.phase,
        "sequence": observation.sequence,
    }
    if slot_kind == "value":
        return ValueSlot(**common, value_type=proposal.value_type)
    if slot_kind == "schema":
        return SchemaSlot(
            **common,
            binding_type="unknown",
            bound_identifier=None,
        )
    if slot_kind == "operation":
        return OperationSlot(
            **common,
            operation_type=proposal.operation_type,
            parameters=proposal.parameters,
        )
    raise ValueError(f"unsupported LLM proposal slot kind: {slot_kind}")
