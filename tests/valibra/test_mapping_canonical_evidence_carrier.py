from __future__ import annotations

import json
import unittest

from valibra_agent import grounding_callbacks
from valibra_agent.sql_grounding.models import (
    ColumnMapping,
    GroundingRuntime,
    SQLGroundingState,
    SQLGroundingValidationError,
    ValidationContext,
    validate_sql_grounding_state,
)


class MappingCanonicalEvidenceCarrierTests(unittest.TestCase):
    CASES = (
        {
            "task": "crypto_exchange_1",
            "query": "Show exchange code and timestamp.",
            "schema": '''CREATE TABLE marketdata (
  "EXCH_SPOT" text,
  "TimeTrack" timestamp,
  orderbook_metrics jsonb
);''',
            "meanings": {
                "crypto_exchange|marketdata|exch_spot": "Exchange code.",
                "crypto_exchange|marketdata|timetrack": "Snapshot timestamp.",
                "crypto_exchange|marketdata|orderbook_metrics": {
                    "column_meaning": "Order-book JSON.",
                    "fields_meaning": {"spread_pct": "Spread percentage."},
                },
            },
            "tables": ("marketdata",),
            "expected": ("marketdata.EXCH_SPOT", "marketdata.TimeTrack"),
            "normalized_wrong": "marketdata.exch_spot",
        },
        {
            "task": "crypto_exchange_10",
            "query": "Show maximum market stats id and order reference.",
            "schema": '''CREATE TABLE marketstats (
  "MKT_STATS_MARK" bigint
);
CREATE TABLE orders (
  "ORD_STAMP" text
);''',
            "meanings": {
                "crypto_exchange|marketstats|mkt_stats_mark": "Market stats id.",
                "crypto_exchange|orders|ord_stamp": "Order reference.",
            },
            "tables": ("marketstats", "orders"),
            "expected": ("marketstats.MKT_STATS_MARK", "orders.ORD_STAMP"),
            "normalized_wrong": "marketstats.mkt_stats_mark",
        },
        {
            "task": "insider_trading_M_10",
            "query": "Show record key.",
            "schema": '''CREATE TABLE trade_records (
  "REC_KEY" text
);''',
            "meanings": {
                "insider_trading|trade_records|rec_key": "Record key."
            },
            "tables": ("trade_records",),
            "expected": ("trade_records.REC_KEY",),
            "normalized_wrong": "trade_records.rec_key",
        },
        {
            "task": "cybermarket_pattern_M_6",
            "query": "Find cross-border sales.",
            "schema": '''CREATE TABLE transactions (
  "CrossBorder" bigint
);''',
            "meanings": {
                "cybermarket_pattern|transactions|crossborder": "Cross-border flag."
            },
            "tables": ("transactions",),
            "expected": ("transactions.CrossBorder",),
            "normalized_wrong": "transactions.crossborder",
        },
    )

    @staticmethod
    def _state(schema: str, meanings: dict[str, object]) -> dict[str, object]:
        return {
            "task_id": "mapping-canonical-carrier",
            "tool_trajectory": [
                {"tool": "get_schema", "result": schema, "phase": 1},
                {
                    "tool": "get_all_column_meanings",
                    "result": json.dumps(meanings, sort_keys=True),
                    "phase": 1,
                },
            ],
        }

    @classmethod
    def _knowledge_state(
        cls,
        schema: str,
        meanings: dict[str, object],
    ) -> dict[str, object]:
        state = cls._state(schema, meanings)
        state["tool_trajectory"].append(
            {
                "tool": "get_all_knowledge_definitions",
                "result": "[]",
                "phase": 1,
            }
        )
        return state

    def test_four_g3_requests_expose_validator_canonical_identifiers(self):
        for case in self.CASES:
            with self.subTest(task=case["task"]):
                state = self._state(case["schema"], case["meanings"])
                raw_meanings = state["tool_trajectory"][1]["result"]
                runtime = GroundingRuntime(
                    grounding_state=SQLGroundingState(
                        tables=case["tables"],
                        join_keys=(),
                    )
                )
                request = grounding_callbacks._build_staged_grounding_request(
                    state,
                    call_kind="mapping",
                    query=case["query"],
                    runtime=runtime,
                    phase=1,
                )
                self.assertTrue(
                    set(case["expected"]).issubset(request["column_meanings"]),
                )
                self.assertFalse(
                    any("|" in target for target in request["column_meanings"])
                )
                self.assertEqual(json.loads(raw_meanings), case["meanings"])
                self.assertEqual(
                    state["tool_trajectory"][1]["result"],
                    raw_meanings,
                )

                tables, columns = grounding_callbacks._parse_schema_projection(
                    case["schema"]
                )
                context = ValidationContext(
                    current_query=case["query"],
                    latest_observation_id=f"{case['task']}-mapping",
                    known_tables=tables,
                    known_columns=columns,
                )
                phrase = case["query"].split(".", 1)[0]
                for canonical in case["expected"]:
                    validate_sql_grounding_state(
                        SQLGroundingState(
                            tables=case["tables"],
                            join_keys=(),
                            column_mapping=(
                                ColumnMapping(phrase=phrase, targets=(canonical,)),
                            ),
                        ),
                        context,
                    )
                with self.assertRaisesRegex(
                    SQLGroundingValidationError,
                    "unknown column",
                ):
                    validate_sql_grounding_state(
                        SQLGroundingState(
                            tables=case["tables"],
                            join_keys=(),
                            column_mapping=(
                                ColumnMapping(
                                    phrase=phrase,
                                    targets=(case["normalized_wrong"],),
                                ),
                            ),
                        ),
                        context,
                    )

    def test_lowercase_schema_preserves_lowercase_target(self):
        schema = """CREATE TABLE metrics (
  value numeric
);"""
        meanings = {"example|metrics|value": "Metric value."}
        request = grounding_callbacks._build_staged_grounding_request(
            self._state(schema, meanings),
            call_kind="mapping",
            query="Show value.",
            runtime=GroundingRuntime(
                grounding_state=SQLGroundingState(
                    tables=("metrics",),
                    join_keys=(),
                )
            ),
            phase=1,
        )
        self.assertEqual(request["column_meanings"], {"metrics.value": "Metric value."})

    def test_json_path_uses_same_canonical_base_as_validator(self):
        schema = '''CREATE TABLE events (
  "Payload" jsonb
);'''
        meanings = {
            "example|events|payload": {
                "column_meaning": "Event payload.",
                "fields_meaning": {"status_key": "Event status."},
            }
        }
        state = self._state(schema, meanings)
        request = grounding_callbacks._build_staged_grounding_request(
            state,
            call_kind="mapping",
            query="Show status.",
            runtime=GroundingRuntime(
                grounding_state=SQLGroundingState(
                    tables=("events",),
                    join_keys=(),
                )
            ),
            phase=1,
        )
        self.assertEqual(set(request["column_meanings"]), {"events.Payload"})

        tables, columns = grounding_callbacks._parse_schema_projection(schema)
        paths = grounding_callbacks._column_meaning_json_paths(
            json.dumps(meanings),
            known_columns=set(columns),
        )
        self.assertEqual(paths, {("events.Payload", ("status_key",))})
        validate_sql_grounding_state(
            SQLGroundingState(
                tables=("events",),
                join_keys=(),
                column_mapping=(
                    ColumnMapping(
                        phrase="status",
                        targets=("events.Payload ->> 'status_key'",),
                    ),
                ),
            ),
            ValidationContext(
                current_query="Show status.",
                latest_observation_id="json-canonical-carrier",
                known_tables=tables,
                known_columns=columns,
                supported_json_paths=paths,
            ),
        )

    def test_ambiguous_casefold_match_is_omitted(self):
        projected = grounding_callbacks._project_mapping_column_meanings(
            {"example|metrics|foo": "Ambiguous."},
            tables=("metrics",),
            canonical_columns={"metrics.Foo", "metrics.foo"},
        )
        self.assertEqual(projected, {})

    def test_knowledge_projection_matches_mapping_canonical_truth(self):
        for case in self.CASES:
            with self.subTest(task=case["task"]):
                state = self._knowledge_state(case["schema"], case["meanings"])
                mappings = tuple(
                    ColumnMapping(
                        phrase=f"concept {index}",
                        targets=(target,),
                    )
                    for index, target in enumerate(case["expected"])
                )
                runtime = GroundingRuntime(
                    grounding_state=SQLGroundingState(
                        tables=case["tables"],
                        join_keys=(),
                        column_mapping=mappings,
                        domain_knowledge=(),
                    ),
                    focus_dimension="none",
                )
                mapping_request = grounding_callbacks._build_staged_grounding_request(
                    state,
                    call_kind="mapping",
                    query="Show concept 0 and concept 1.",
                    runtime=runtime,
                    phase=1,
                )
                knowledge_request = grounding_callbacks._build_staged_grounding_request(
                    state,
                    call_kind="knowledge",
                    query="Show concept 0 and concept 1.",
                    runtime=runtime,
                    phase=1,
                )
                self.assertEqual(
                    knowledge_request["relevant_column_meanings"],
                    mapping_request["column_meanings"],
                )
                self.assertTrue(
                    set(case["expected"]).issubset(
                        knowledge_request["relevant_column_meanings"]
                    )
                )
                self.assertFalse(
                    any(
                        "|" in target
                        for target in knowledge_request["relevant_column_meanings"]
                    )
                )

    def test_knowledge_lowercase_and_json_projection_are_unchanged(self):
        schema = """CREATE TABLE events (
  value numeric,
  payload jsonb
);"""
        meanings = {
            "example|events|value": "Metric value.",
            "example|events|payload": {
                "column_meaning": "Event payload.",
                "fields_meaning": {"status_key": "Event status."},
            },
        }
        request = grounding_callbacks._build_staged_grounding_request(
            self._knowledge_state(schema, meanings),
            call_kind="knowledge",
            query="Show value and status.",
            runtime=GroundingRuntime(
                grounding_state=SQLGroundingState(
                    tables=("events",),
                    join_keys=(),
                    column_mapping=(
                        ColumnMapping(phrase="value", targets=("events.value",)),
                        ColumnMapping(
                            phrase="status",
                            targets=("events.payload ->> 'status_key'",),
                        ),
                    ),
                    domain_knowledge=(),
                ),
                focus_dimension="none",
            ),
            phase=1,
        )
        self.assertEqual(
            request["relevant_column_meanings"],
            {
                "events.payload": meanings["example|events|payload"],
                "events.value": "Metric value.",
            },
        )

    def test_knowledge_ambiguous_casefold_is_fail_closed(self):
        schema = '''CREATE TABLE metrics (
  "Foo" numeric,
  foo numeric
);'''
        request = grounding_callbacks._build_staged_grounding_request(
            self._knowledge_state(
                schema,
                {"example|metrics|foo": "Ambiguous."},
            ),
            call_kind="knowledge",
            query="Show metric.",
            runtime=GroundingRuntime(
                grounding_state=SQLGroundingState(
                    tables=("metrics",),
                    join_keys=(),
                    column_mapping=(),
                    domain_knowledge=(),
                ),
                focus_dimension="none",
            ),
            phase=1,
        )
        self.assertEqual(request["relevant_column_meanings"], {})


if __name__ == "__main__":
    unittest.main()
