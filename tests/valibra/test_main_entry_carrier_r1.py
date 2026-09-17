from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.main_entry_carrier import (
    MAIN_EXECUTION_ENVELOPES_KEY,
    build_main_execution_envelopes,
    infer_main_action_intent,
    main_execution_envelope_for_phase,
    render_main_execution_envelope,
    render_sql_safe_identifier_overlay,
)
from valibra_agent.sql_grounding.models import ColumnMapping, SQLGroundingState


def _task() -> dict:
    return {
        "amb_user_query": "Show the total amount in descending order.",
        "category": "Query",
        "output_type": "scalar",
        "conditions": {"decimal": 2, "distinct": False, "order": True},
        "sol_sql": ["REFERENCE MUST NOT LEAK"],
        "test_cases": ["CUSTOM TEST MUST NOT LEAK"],
        "follow_up": {
            "query": "Create a view for the same result.",
            "category": "Query",
            "output_type": "table",
            "conditions": {"decimal": -1, "distinct": False, "order": False},
            "sol_sql": ["FOLLOW-UP REFERENCE MUST NOT LEAK"],
        },
    }


def _writer_request() -> SimpleNamespace:
    declarations = [
        SimpleNamespace(name="execute_sql", description="baseline execute"),
        SimpleNamespace(name="submit_sql", description="baseline submit"),
    ]
    config = SimpleNamespace(
        system_instruction="baseline",
        tools=[SimpleNamespace(function_declarations=declarations)],
    )
    part = SimpleNamespace(
        text="User query",
        function_call=None,
        function_response=None,
    )
    return SimpleNamespace(
        config=config,
        tools_dict={"execute_sql": object(), "submit_sql": object()},
        contents=[SimpleNamespace(role="user", parts=[part])],
    )


def test_task_projection_carries_only_public_response_metadata() -> None:
    payload = build_main_execution_envelopes(_task())
    serialized = json.dumps(payload, sort_keys=True)
    assert "REFERENCE MUST NOT LEAK" not in serialized
    assert "CUSTOM TEST MUST NOT LEAK" not in serialized
    assert set(payload["phases"]) == {"1", "2"}

    state = {MAIN_EXECUTION_ENVELOPES_KEY: payload}
    p1 = main_execution_envelope_for_phase(state, 1)
    p2 = main_execution_envelope_for_phase(state, 2)
    assert p1 is not None
    assert p1.response_type == "scalar"
    assert p1.decimal_places == 2
    assert p1.order_sensitive is True
    assert p1.action_intent == "RESULT_QUERY"
    assert p2 is not None
    assert p2.response_type == "table"
    assert p2.decimal_places is None
    assert p2.order_sensitive is False
    assert p2.action_intent == "CREATE_VIEW"


def test_envelope_does_not_invent_direction_aggregation_or_reference_shape() -> None:
    envelope = main_execution_envelope_for_phase(
        {MAIN_EXECUTION_ENVELOPES_KEY: build_main_execution_envelopes(_task())},
        1,
    )
    assert envelope is not None
    rendered = render_main_execution_envelope(envelope)
    assert '"decimal_places":2' in rendered.text
    assert '"action_intent":"RESULT_QUERY"' in rendered.text
    assert '"order_sensitive":true' in rendered.text
    assert '"response_type":"scalar"' in rendered.text
    for forbidden in (
        " ASC",
        " DESC",
        "AVG(",
        "SUM(",
        "compliance_rate",
        "expected rows",
    ):
        assert forbidden not in rendered.text


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Can you generate a table of company totals?", "CREATE_TABLE"),
        ("Please make a materialized view for this summary.", "CREATE_VIEW"),
        ("I need a function that counts launches by year.", "CREATE_FUNCTION"),
        ("VACUUM the monitoring table.", "SQL_UTILITY"),
        (
            "Analyze the monitoring table so statistics are up-to-date.",
            "SQL_UTILITY",
        ),
        ("Make a tool to measure studio contributions.", "RESULT_QUERY"),
        ("Analyze how promotional intensity varies by content.", "RESULT_QUERY"),
        ("List each studio and its title count.", "RESULT_QUERY"),
    ],
)
def test_action_intent_uses_only_explicit_query_action(
    query: str,
    expected: str,
) -> None:
    assert infer_main_action_intent(query) == expected


def test_management_category_alone_does_not_invent_action() -> None:
    task = _task()
    task["category"] = "Management"
    task["output_type"] = None
    task["amb_user_query"] = "Make a tool to measure studio contributions."
    envelope = main_execution_envelope_for_phase(
        {MAIN_EXECUTION_ENVELOPES_KEY: build_main_execution_envelopes(task)},
        1,
    )
    assert envelope is not None
    assert envelope.task_category == "Management"
    assert envelope.action_intent == "RESULT_QUERY"


def test_identifier_overlay_quotes_only_existing_validated_references() -> None:
    state = SQLGroundingState(
        tables=("marketdata", "orders"),
        join_keys=("marketdata.EXCH_SPOT = orders.exchSpot",),
        column_mapping=(
            ColumnMapping(
                phrase="order reference",
                targets=("orders.ORD_STAMP",),
            ),
            ColumnMapping(
                phrase="market depth",
                targets=("marketdata.orderbook_metrics -> 'bid_size'",),
            ),
        ),
        domain_knowledge=(),
    )
    rendered = render_sql_safe_identifier_overlay(state)
    assert rendered is not None
    assert '"sql":"\\"orders\\".\\"ORD_STAMP\\""' in rendered.text
    assert (
        '"sql":"\\"marketdata\\".\\"EXCH_SPOT\\" = '
        '\\"orders\\".\\"exchSpot\\""'
    ) in rendered.text
    assert "latest" not in rendered.text
    assert "AVG" not in rendered.text
    assert "DESC" not in rendered.text


def test_sql_writer_receives_optional_carriers_without_changing_base_prompt() -> None:
    expected_prompt_sha = (
        "86fdcbc7059bd0d5aea9384805a036ce585346b74fe021a645f9175d9b0d8a6d"
    )
    assert hashlib.sha256(
        grounding_callbacks._SQL_WRITER_PROMPT.encode("utf-8")
    ).hexdigest() == expected_prompt_sha

    envelope = main_execution_envelope_for_phase(
        {MAIN_EXECUTION_ENVELOPES_KEY: build_main_execution_envelopes(_task())},
        1,
    )
    assert envelope is not None
    envelope_text = render_main_execution_envelope(envelope).text
    identifiers = render_sql_safe_identifier_overlay(
        SQLGroundingState(
            tables=("DataFlow",),
            join_keys=(),
            column_mapping=(
                ColumnMapping(
                    phrase="flow info",
                    targets=("DataFlow.flow_overview",),
                ),
            ),
            domain_knowledge=(),
        )
    )
    assert identifiers is not None
    request = _writer_request()
    audit = grounding_callbacks._inject_sql_writer_context(
        request,
        phase=1,
        original_query="Show the flow info.",
        follow_up=None,
        view_text="[VALIBRA DATABASE GROUNDING]\nTables:\n- DataFlow",
        budget_remaining=8,
        execution_envelope_text=envelope_text,
        sql_safe_identifiers_text=identifiers.text,
    )
    instruction = request.config.system_instruction
    assert "[MAIN EXECUTION ENVELOPE]" in instruction
    assert "[SQL-SAFE CANONICAL REFERENCES]" in instruction
    assert '\\"DataFlow\\".\\"flow_overview\\"' in instruction
    assert audit["writer_prompt_sha256"] == expected_prompt_sha
    assert audit["main_execution_envelope_sha256"] == hashlib.sha256(
        envelope_text.encode("utf-8")
    ).hexdigest()


def test_malformed_task_metadata_fails_before_entering_agent_state() -> None:
    malformed = _task()
    malformed["conditions"] = {"decimal": 2, "order": "DESC"}
    with pytest.raises(ValueError, match="order condition"):
        build_main_execution_envelopes(malformed)


def test_absent_task_response_metadata_preserves_legacy_no_op() -> None:
    payload = build_main_execution_envelopes(
        {
            "instance_id": "legacy-task",
            "amb_user_query": "List order totals.",
        }
    )
    assert payload["phases"] == {}
    assert main_execution_envelope_for_phase(
        {MAIN_EXECUTION_ENVELOPES_KEY: payload},
        1,
    ) is None
