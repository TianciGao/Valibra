from __future__ import annotations

import copy
import hashlib
import json

import pytest

from valibra_agent.sql_grounding.models import (
    SQLGroundingState,
    canonical_json,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.resolved_literal_carrier_materialization import (
    ResolvedLiteralCarrierMaterializationError,
    ResolvedLiteralExecutableCarrier,
    active_resolved_literal_carriers,
    begin_resolved_literal_carrier_draft,
    commit_resolved_literal_carrier_draft,
    materialize_resolved_literal_carrier,
    rollback_resolved_literal_carrier_draft,
)
from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
    ResolvedLiteralShadowCarrier,
    ResolvedLiteralShadowValidation,
)


def _payload(
    *,
    query: str,
    phrase: str,
    target: str,
    table: str,
    column: str,
    result: object,
) -> dict:
    return {
        "query": query,
        "current_state": {
            "tables": [table],
            "join_keys": [],
            "column_mapping": [{"phrase": phrase, "targets": [target]}],
            "domain_knowledge": [],
        },
        "latest_tool": {
            "name": "get_column_meaning",
            "arguments": {"table_name": table, "column_name": column},
            "result": result,
        },
    }


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_shadow(
    payload: dict,
    *,
    task_id: str,
    phase: int,
    revision: int,
    phrase: str,
    target: str,
    literal: str,
) -> ResolvedLiteralShadowValidation:
    arguments = payload["latest_tool"]["arguments"]
    raw_result = payload["latest_tool"]["result"]
    result_text = raw_result if isinstance(raw_result, str) else canonical_json(raw_result)
    return ResolvedLiteralShadowValidation(
        verdict="VALID",
        reason="unique_query_lexeme_and_exact_official_enum",
        matching_enum_literals=(literal,),
        carrier=ResolvedLiteralShadowCarrier(
            task_id=task_id,
            phase=phase,
            grounding_revision=revision,
            state_sha256=sql_grounding_state_sha256(
                SQLGroundingState.model_validate(payload["current_state"])
            ),
            phrase=phrase,
            target=target,
            literal=literal,
            source_request_digest=_sha(
                canonical_json(
                    {
                        "arguments": arguments,
                        "name": "get_column_meaning",
                    }
                )
            ),
            source_result_sha256=_sha(result_text),
        ),
    )


def _solar9() -> tuple[dict, ResolvedLiteralShadowValidation]:
    phrase = "electrical grounding fails"
    target = (
        "mechanical_condition.mech_health_snapshot -> "
        "'electrical_integrity' ->> 'grounding_status'"
    )
    payload = _payload(
        query="Show panels where electrical grounding fails.",
        phrase=phrase,
        target=target,
        table="mechanical_condition",
        column="mech_health_snapshot",
        result={
            "fields_meaning": {
                "electrical_integrity": {
                    "grounding_status": (
                        "Electrical grounding status. Possible values: "
                        "Failed, Pending, Running."
                    )
                }
            }
        },
    )
    return payload, _valid_shadow(
        payload,
        task_id="solar_panel_9",
        phase=1,
        revision=7,
        phrase=phrase,
        target=target,
        literal="Failed",
    )


def _cyber12() -> tuple[dict, ResolvedLiteralShadowValidation]:
    phrase = "highest priority level alert"
    target = "alerts.alert_case_management -> 'invest_priority_stat'"
    payload = _payload(
        query="Find the highest priority level alert.",
        phrase=phrase,
        target=target,
        table="alerts",
        column="alert_case_management",
        result={
            "fields_meaning": {
                "invest_priority_stat": (
                    "Investigation priority category. Possible values: "
                    "High, Medium, Low."
                )
            }
        },
    )
    return payload, _valid_shadow(
        payload,
        task_id="cybermarket_pattern_12",
        phase=1,
        revision=4,
        phrase=phrase,
        target=target,
        literal="High",
    )


def _materialize(
    payload: dict,
    validation: ResolvedLiteralShadowValidation,
    *,
    task_id: str,
    revision: int,
):
    before = copy.deepcopy(payload)
    carrier = materialize_resolved_literal_carrier(
        validation,
        grounding_input=payload,
        atomic_draft_id=f"draft-{task_id}",
        task_id=task_id,
        phase=1,
        state_revision=revision,
    )
    assert payload == before
    return carrier


def test_solar9_materializes_existing_scalar_target() -> None:
    payload, validation = _solar9()
    carrier = _materialize(
        payload,
        validation,
        task_id="solar_panel_9",
        revision=7,
    )
    assert carrier is not None
    assert carrier.source_target == carrier.comparison_target
    assert carrier.literal == "Failed"
    assert carrier.operator == "EXACT_EQUALITY"


def test_cyber12_converts_only_terminal_json_operator_to_scalar() -> None:
    payload, validation = _cyber12()
    carrier = _materialize(
        payload,
        validation,
        task_id="cybermarket_pattern_12",
        revision=4,
    )
    assert carrier is not None
    assert carrier.source_target == (
        "alerts.alert_case_management -> 'invest_priority_stat'"
    )
    assert carrier.comparison_target == (
        "alerts.alert_case_management ->> 'invest_priority_stat'"
    )
    assert payload["current_state"]["column_mapping"][0]["targets"] == [
        carrier.source_target
    ]


def test_omitted_solar13_proposal_materializes_no_carrier() -> None:
    payload, _ = _cyber12()
    omitted = ResolvedLiteralShadowValidation(
        verdict="OMITTED",
        reason="no_resolved_literal_proposal",
    )
    assert materialize_resolved_literal_carrier(
        omitted,
        grounding_input=payload,
        atomic_draft_id="draft-solar13",
        task_id="solar_panel_13",
        phase=1,
        state_revision=4,
    ) is None


def test_non_scalar_json_terminal_is_rejected() -> None:
    payload, validation = _cyber12()
    payload["latest_tool"]["result"] = {
        "fields_meaning": {
            "invest_priority_stat": {
                "label": "Nested object, not a scalar terminal leaf"
            }
        }
    }
    raw = validation.carrier.model_dump(mode="json")
    raw["source_result_sha256"] = _sha(
        canonical_json(payload["latest_tool"]["result"])
    )
    forged = ResolvedLiteralShadowValidation(
        verdict="VALID",
        reason="forged_for_materialization_boundary",
        matching_enum_literals=("High",),
        carrier=ResolvedLiteralShadowCarrier.model_validate(raw),
    )
    with pytest.raises(
        ResolvedLiteralCarrierMaterializationError,
        match="bounded scalar enum leaf",
    ):
        _materialize(
            payload,
            forged,
            task_id="cybermarket_pattern_12",
            revision=4,
        )


def test_ambiguous_enum_is_rejected_again_during_materialization() -> None:
    phrase = "active claimed warranty status"
    target = "plants.warrstate"
    payload = _payload(
        query="Show active claimed warranty status.",
        phrase=phrase,
        target=target,
        table="plants",
        column="warrstate",
        result="Warranty status. Possible values: Active, Claimed, Expired.",
    )
    validation = _valid_shadow(
        payload,
        task_id="solar_panel_13",
        phase=1,
        revision=5,
        phrase=phrase,
        target=target,
        literal="Active",
    )
    with pytest.raises(
        ResolvedLiteralCarrierMaterializationError,
        match="no longer unique",
    ):
        _materialize(
            payload,
            validation,
            task_id="solar_panel_13",
            revision=5,
        )


def test_negative_phrase_is_rejected_again_during_materialization() -> None:
    phrase = "not failed"
    target = "jobs.status"
    payload = _payload(
        query="Show jobs that are not failed.",
        phrase=phrase,
        target=target,
        table="jobs",
        column="status",
        result="Job status. Possible values: Failed, Pending, Running.",
    )
    validation = _valid_shadow(
        payload,
        task_id="negative-case",
        phase=1,
        revision=3,
        phrase=phrase,
        target=target,
        literal="Failed",
    )
    with pytest.raises(
        ResolvedLiteralCarrierMaterializationError,
        match="negative or exclusion",
    ):
        _materialize(
            payload,
            validation,
            task_id="negative-case",
            revision=3,
        )


def test_stale_revision_is_rejected() -> None:
    payload, validation = _solar9()
    with pytest.raises(
        ResolvedLiteralCarrierMaterializationError,
        match="revision is stale",
    ):
        _materialize(
            payload,
            validation,
            task_id="solar_panel_9",
            revision=8,
        )


@pytest.mark.parametrize("tamper", ["target", "literal", "source_hash"])
def test_tampered_proposal_or_provenance_is_rejected(tamper: str) -> None:
    payload, validation = _solar9()
    carrier = validation.carrier.model_dump(mode="json")
    if tamper == "target":
        carrier["target"] = "mechanical_condition.other_status"
    elif tamper == "literal":
        carrier["literal"] = "Pending"
    else:
        carrier["source_result_sha256"] = "0" * 64
    forged = ResolvedLiteralShadowValidation(
        verdict="VALID",
        reason="forged_for_tamper_test",
        matching_enum_literals=(carrier["literal"],),
        carrier=ResolvedLiteralShadowCarrier.model_validate(carrier),
    )
    with pytest.raises(ResolvedLiteralCarrierMaterializationError):
        _materialize(
            payload,
            forged,
            task_id="solar_panel_9",
            revision=7,
        )


def test_carrier_model_rejects_non_terminal_comparison_rewrite() -> None:
    payload, validation = _cyber12()
    carrier = _materialize(
        payload,
        validation,
        task_id="cybermarket_pattern_12",
        revision=4,
    )
    assert carrier is not None
    forged = carrier.model_dump(mode="json")
    forged["comparison_target"] = (
        "alerts.alert_case_management ->> 'another_leaf'"
    )
    with pytest.raises(ValueError, match="exact source leaf"):
        ResolvedLiteralExecutableCarrier.model_validate(forged)


def test_draft_rollback_destroys_carrier_and_cannot_commit() -> None:
    payload, validation = _solar9()
    carrier = _materialize(
        payload,
        validation,
        task_id="solar_panel_9",
        revision=7,
    )
    assert carrier is not None
    state = SQLGroundingState.model_validate(payload["current_state"])
    lifecycle = begin_resolved_literal_carrier_draft(
        carrier,
        base_revision=6,
        base_state_sha256="1" * 64,
    )
    assert active_resolved_literal_carriers(
        lifecycle,
        task_id="solar_panel_9",
        phase=1,
        state_revision=7,
        state=state,
    ) == ()
    rolled_back = rollback_resolved_literal_carrier_draft(lifecycle)
    assert rolled_back.status == "ROLLED_BACK"
    assert rolled_back.staged_carriers == ()
    assert rolled_back.active_carriers == ()
    with pytest.raises(ResolvedLiteralCarrierMaterializationError):
        commit_resolved_literal_carrier_draft(
            rolled_back,
            task_id="solar_panel_9",
            phase=1,
            formal_revision_before_commit=6,
            formal_state_before_commit=state,
            committed_revision=7,
            committed_state=state,
        )


def test_draft_commit_activates_carrier_only_for_exact_state_and_phase() -> None:
    payload, validation = _cyber12()
    carrier = _materialize(
        payload,
        validation,
        task_id="cybermarket_pattern_12",
        revision=4,
    )
    assert carrier is not None
    committed_state = SQLGroundingState.model_validate(payload["current_state"])
    base_state = SQLGroundingState(
        tables=("alerts",),
        join_keys=(),
        column_mapping=(),
        domain_knowledge=(),
    )
    lifecycle = begin_resolved_literal_carrier_draft(
        carrier,
        base_revision=3,
        base_state_sha256=sql_grounding_state_sha256(base_state),
    )
    committed = commit_resolved_literal_carrier_draft(
        lifecycle,
        task_id="cybermarket_pattern_12",
        phase=1,
        formal_revision_before_commit=3,
        formal_state_before_commit=base_state,
        committed_revision=4,
        committed_state=committed_state,
    )
    active = active_resolved_literal_carriers(
        committed,
        task_id="cybermarket_pattern_12",
        phase=1,
        state_revision=4,
        state=committed_state,
    )
    assert len(active) == 1
    assert active[0].comparison_target.endswith(
        "->> 'invest_priority_stat'"
    )
    assert active_resolved_literal_carriers(
        committed,
        task_id="cybermarket_pattern_12",
        phase=2,
        state_revision=4,
        state=committed_state,
    ) == ()
    assert active_resolved_literal_carriers(
        committed,
        task_id="another-task",
        phase=1,
        state_revision=4,
        state=committed_state,
    ) == ()
    with pytest.raises(
        ResolvedLiteralCarrierMaterializationError,
        match="stale for current State",
    ):
        active_resolved_literal_carriers(
            committed,
            task_id="cybermarket_pattern_12",
            phase=1,
            state_revision=5,
            state=committed_state,
        )


def test_commit_rejects_state_or_revision_not_created_by_atomic_commit() -> None:
    payload, validation = _solar9()
    carrier = _materialize(
        payload,
        validation,
        task_id="solar_panel_9",
        revision=7,
    )
    assert carrier is not None
    base_state = SQLGroundingState(
        tables=("mechanical_condition",),
        join_keys=(),
        column_mapping=(),
        domain_knowledge=(),
    )
    lifecycle = begin_resolved_literal_carrier_draft(
        carrier,
        base_revision=6,
        base_state_sha256=sql_grounding_state_sha256(base_state),
    )
    with pytest.raises(
        ResolvedLiteralCarrierMaterializationError,
        match="differs from committed State",
    ):
        commit_resolved_literal_carrier_draft(
            lifecycle,
            task_id="solar_panel_9",
            phase=1,
            formal_revision_before_commit=6,
            formal_state_before_commit=base_state,
            committed_revision=7,
            committed_state=base_state,
        )
    committed_state = SQLGroundingState.model_validate(payload["current_state"])
    with pytest.raises(
        ResolvedLiteralCarrierMaterializationError,
        match="revision does not match",
    ):
        commit_resolved_literal_carrier_draft(
            lifecycle,
            task_id="solar_panel_9",
            phase=1,
            formal_revision_before_commit=6,
            formal_state_before_commit=base_state,
            committed_revision=8,
            committed_state=committed_state,
        )


def test_lifecycle_rejects_carrier_identity_tamper() -> None:
    payload, validation = _solar9()
    carrier = _materialize(
        payload,
        validation,
        task_id="solar_panel_9",
        revision=7,
    )
    assert carrier is not None
    lifecycle = begin_resolved_literal_carrier_draft(
        carrier,
        base_revision=6,
        base_state_sha256="1" * 64,
    )
    raw = lifecycle.model_dump(mode="json")
    raw["staged_carriers"][0]["task_id"] = "another-task"
    with pytest.raises(ValueError, match="identity differs"):
        type(lifecycle).model_validate(raw)
