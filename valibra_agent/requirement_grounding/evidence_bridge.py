"""把合法工具 Observation 转成 evidence-only Patch。"""

from __future__ import annotations

from valibra_agent.requirement_grounding.models import (
    GroundingEvidence,
    Observation,
    RequirementGroundingPatch,
    RequirementGroundingState,
)


EVIDENCE_OBSERVATION_TYPES = frozenset(
    {"schema", "metadata", "knowledge", "sql_execution", "submission"}
)


class EvidenceBridgeUpdater:
    """只补一条有界工具证据，不推断 Frame、歧义或 Schema 绑定。"""

    mode = "evidence_bridge"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        del state
        if observation.observation_type not in EVIDENCE_OBSERVATION_TYPES:
            raise ValueError("Evidence Bridge received an ineligible Observation")
        evidence = GroundingEvidence(
            evidence_id=f"evidence.tool.{observation.observation_id}",
            observation_id=observation.observation_id,
            source_type=f"tool_{observation.observation_type}",
            phase=observation.phase,
            summary=observation.summary,
            raw_digest=observation.raw_digest,
            raw_log_ref=observation.raw_log_ref,
            sequence=observation.sequence,
            timestamp=None,
        )
        return RequirementGroundingPatch(
            patch_id=f"evidence-bridge.{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            evidence_additions=(evidence,),
            diagnostics=(
                "P4.3a Evidence Bridge added bounded tool evidence only",
            ),
        )
