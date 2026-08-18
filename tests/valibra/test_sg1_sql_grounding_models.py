from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from valibra_agent.sql_grounding import (
    ALLOWED_CAST_TYPES,
    FIELD_EXPRESSION_AST_WHITELIST,
    FIELD_EXPRESSION_FUNCTION_WHITELIST,
    LEGACY_GROUNDING_RUNTIME_KEY,
    RELATION_EXPRESSION_AST_WHITELIST,
    SQLGLOT_VERSION,
    SQL_GROUNDING_RUNTIME_KEY,
    ColumnMapping,
    DomainKnowledge,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    SQLGroundingValidationError,
    ValidationContext,
    canonical_json,
    canonicalize_field_expression,
    canonicalize_relation_expression,
    sql_grounding_state_sha256,
    validate_grounding_llm_response,
    validate_grounding_runtime_transition,
    validate_sql_grounding_state,
)


def validation_context() -> ValidationContext:
    tables = frozenset(
        {
            "plants",
            "plant_record",
            "operational_metrics",
            "electrical_performance",
            "data_flows",
            "candidates",
            "metrics",
            "robot_details",
            "predictive_models",
            "score_bands",
            "score_rules",
            "incidents",
            "prediction_events",
            "audits",
            "month_targets",
            "stars",
        }
    )
    columns = frozenset(
        {
            "plants.sitekey",
            "plant_record.sitetie",
            "plant_record.snapkey",
            "operational_metrics.snapops",
            "operational_metrics.maintcost",
            "electrical_performance.elec_perf_snapshot",
            "data_flows.flow_overview",
            "candidates.comorbid_detail",
            "metrics.values",
            "metrics.session_telemetry",
            "robot_details.mfgnameval",
            "predictive_models.mfgnameval_lower",
            "score_bands.score",
            "score_rules.min_score",
            "score_rules.max_exclusive",
            "score_rules.include_max",
            "incidents.time_mark",
            "prediction_events.estimated_subscription_date",
            "audits.REMED_DUE",
            "month_targets.audit_month",
            "stars.rn",
        }
    )
    return ValidationContext(
        current_query=(
            "Show maintenance cost, current power, comorbidities, hourly activity, "
            "and the neighboring star for each plant."
        ),
        follow_up_query="Also include the database relation size and live achievements.",
        latest_observation_id="obs-schema-1",
        official_trajectory_observation_ids=("obs-schema-1", "obs-knowledge-1"),
        known_tables=tables,
        known_columns=columns,
        table_aliases=(
            ("p", "plants"),
            ("pr", "plant_record"),
            ("om", "operational_metrics"),
            ("ep", "electrical_performance"),
            ("df", "data_flows"),
            ("c", "candidates"),
            ("m", "metrics"),
            ("s", "metrics"),
            ("rd", "robot_details"),
            ("pm", "predictive_models"),
            ("b", "score_bands"),
            ("r", "score_rules"),
            ("i", "incidents"),
            ("pce", "prediction_events"),
            ("a", "audits"),
            ("mt", "month_targets"),
            ("s1", "stars"),
            ("s2", "stars"),
        ),
        supported_domain_knowledge=frozenset(
            {
                (
                    "business_rule",
                    "downtime score = mttrh / (mtbfh + mttrh)",
                ),
                (
                    "runtime_state",
                    "achievements table exists in the current task runtime",
                ),
                (
                    "database_capability",
                    "pg_relation_size is required to obtain relation size",
                ),
            }
        ),
    )


def complete_state() -> SQLGroundingState:
    return SQLGroundingState(
        tables=("plants", "plant_record", "operational_metrics"),
        join_keys=(
            "plant_record.snapkey = om.snapops",
            "p.sitekey = pr.sitetie",
        ),
        column_mapping=(
            ColumnMapping(
                phrase="maintenance cost",
                targets=("operational_metrics.maintcost",),
            ),
        ),
        domain_knowledge=(),
    )


class FourDimensionShapeTests(unittest.TestCase):
    def test_default_state_is_exactly_four_null_dimensions(self) -> None:
        self.assertEqual(
            SQLGroundingState().model_dump(mode="json"),
            {
                "tables": None,
                "join_keys": None,
                "column_mapping": None,
                "domain_knowledge": None,
            },
        )

    def test_null_empty_and_nonempty_round_trip(self) -> None:
        states = (
            SQLGroundingState(),
            SQLGroundingState(
                tables=(),
                join_keys=(),
                column_mapping=(),
                domain_knowledge=(),
            ),
            complete_state(),
        )
        for state in states:
            with self.subTest(state=state):
                restored = SQLGroundingState.model_validate_json(state.model_dump_json())
                self.assertEqual(restored, state)

    def test_single_table_can_explicitly_require_no_join(self) -> None:
        state = SQLGroundingState(
            tables=("operational_metrics",),
            join_keys=(),
            column_mapping=(),
            domain_knowledge=(),
        )
        self.assertEqual(state.join_keys, ())
        self.assertIs(validate_sql_grounding_state(state, validation_context()), state)

    def test_tables_are_identifiers_not_free_text_and_are_canonical_sets(self) -> None:
        state = SQLGroundingState(tables=("plants", "operational_metrics"))
        reordered = SQLGroundingState(tables=("operational_metrics", "plants"))
        self.assertEqual(state, reordered)
        with self.assertRaises(ValidationError):
            SQLGroundingState(tables=("the table with plant data",))
        with self.assertRaises(ValidationError):
            SQLGroundingState(tables=("plants", "plants"))

    def test_column_mapping_and_domain_knowledge_are_strict_subobjects(self) -> None:
        mapping = ColumnMapping(
            phrase="maintenance cost",
            targets=("operational_metrics.maintcost",),
        )
        knowledge = DomainKnowledge(
            kind="business_rule",
            content="downtime score = mttrh / (mtbfh + mttrh)",
        )
        self.assertEqual(mapping.targets, ("operational_metrics.maintcost",))
        self.assertEqual(knowledge.kind, "business_rule")
        with self.assertRaises(ValidationError):
            DomainKnowledge(kind="other", content="anything")
        with self.assertRaises(ValidationError):
            ColumnMapping(
                phrase="maintenance cost",
                targets=("operational_metrics.maintcost",),
                confidence=0.9,
            )


class ExpressionContractTests(unittest.TestCase):
    def test_plain_json_array_cast_and_full600_expansion_examples(self) -> None:
        examples = (
            "operational_metrics.maintcost",
            "ep.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
            "m.values[1]",
            "UNNEST(STRING_TO_ARRAY(c.comorbid_detail, ','))",
            "JSONB_ARRAY_ELEMENTS_TEXT(s.session_telemetry -> 'activity_pattern' -> 'hourly_observed')",
            "CAST(df.flow_overview AS JSONB) -> 'routing' ->> 'destination_country'",
        )
        for expression in examples:
            with self.subTest(expression=expression):
                self.assertEqual(canonicalize_field_expression(expression), expression)

    def test_equal_normalized_range_temporal_and_ordinal_relations(self) -> None:
        examples = (
            "p.sitekey = pr.sitetie",
            "LOWER(rd.mfgnameval) = pm.mfgnameval_lower",
            "TRIM(rd.mfgnameval) = TRIM(pm.mfgnameval_lower)",
            "b.score BETWEEN r.min_score AND r.max_exclusive",
            "b.score < r.max_exclusive OR (r.include_max AND b.score <= r.max_exclusive)",
            "i.time_mark < pce.estimated_subscription_date",
            "TO_CHAR(a.REMED_DUE, 'YYYY-MM') = mt.audit_month",
            "s1.rn = s2.rn - 1",
        )
        for expression in examples:
            with self.subTest(expression=expression):
                self.assertEqual(canonicalize_relation_expression(expression), expression)

    def test_expression_contract_rejects_sql_and_unapproved_nodes(self) -> None:
        invalid_fields = (
            "SELECT operational_metrics.maintcost FROM operational_metrics",
            "operational_metrics.maintcost; DROP TABLE plants",
            "operational_metrics.maintcost -- comment",
            "REGEXP_REPLACE(operational_metrics.maintcost, 'x', 'y')",
            "operational_metrics.maintcost + 1",
        )
        invalid_relations = (
            "p.sitekey = pr.sitetie; DELETE FROM plants",
            "p.sitekey = pr.sitetie /* comment */",
            "p.sitekey IN (SELECT pr.sitetie FROM plant_record AS pr)",
            "p.sitekey = 1",
            "p.sitekey = pr.sitetie OR 1 = 1",
        )
        for expression in invalid_fields:
            with self.subTest(field=expression), self.assertRaises(
                (SQLGroundingValidationError, ValidationError)
            ):
                canonicalize_field_expression(expression)
        for expression in invalid_relations:
            with self.subTest(relation=expression), self.assertRaises(
                (SQLGroundingValidationError, ValidationError)
            ):
                canonicalize_relation_expression(expression)

    def test_noncanonical_expression_is_rejected_not_silently_rewritten(self) -> None:
        with self.assertRaises(SQLGroundingValidationError):
            canonicalize_relation_expression(
                "lower(rd.mfgnameval) = pm.mfgnameval_lower"
            )
        with self.assertRaises(SQLGroundingValidationError):
            canonicalize_field_expression("unnest(string_to_array(c.comorbid_detail, ','))")

    def test_sqlglot_and_whitelists_are_frozen(self) -> None:
        self.assertEqual(SQLGLOT_VERSION, "26.16.4")
        self.assertEqual(
            FIELD_EXPRESSION_FUNCTION_WHITELIST,
            frozenset(
                {
                    "JSONB_ARRAY_ELEMENTS",
                    "JSONB_ARRAY_ELEMENTS_TEXT",
                    "JSONB_EACH",
                    "JSONB_EACH_TEXT",
                }
            ),
        )
        self.assertIn("JSONExtractScalar", FIELD_EXPRESSION_AST_WHITELIST)
        self.assertIn("Explode", FIELD_EXPRESSION_AST_WHITELIST)
        self.assertNotIn("Select", FIELD_EXPRESSION_AST_WHITELIST)
        self.assertIn("Between", RELATION_EXPRESSION_AST_WHITELIST)
        self.assertIn("TimeToStr", RELATION_EXPRESSION_AST_WHITELIST)
        self.assertNotIn("Select", RELATION_EXPRESSION_AST_WHITELIST)
        self.assertEqual(
            ALLOWED_CAST_TYPES,
            frozenset(
                {
                    "BIGINT",
                    "BOOLEAN",
                    "DATE",
                    "DECIMAL",
                    "DOUBLE",
                    "INT",
                    "JSON",
                    "JSONB",
                    "TEXT",
                    "TIMESTAMP",
                    "TIMESTAMPTZ",
                }
            ),
        )


class ValidationContextTests(unittest.TestCase):
    def test_full600_driven_examples_validate_against_transient_context(self) -> None:
        context = validation_context()
        state = SQLGroundingState(
            tables=("electrical_performance", "data_flows", "metrics"),
            join_keys=("LOWER(rd.mfgnameval) = pm.mfgnameval_lower",),
            column_mapping=(
                ColumnMapping(
                    phrase="current power",
                    targets=(
                        "ep.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                    ),
                ),
                ColumnMapping(
                    phrase="hourly activity",
                    targets=(
                        "JSONB_ARRAY_ELEMENTS_TEXT(s.session_telemetry -> 'activity_pattern' -> 'hourly_observed')",
                    ),
                ),
            ),
            domain_knowledge=(
                DomainKnowledge(
                    kind="runtime_state",
                    content="achievements table exists in the current task runtime",
                ),
                DomainKnowledge(
                    kind="database_capability",
                    content="pg_relation_size is required to obtain relation size",
                ),
            ),
        )
        self.assertIs(validate_sql_grounding_state(state, context), state)

    def test_phrase_must_be_verbatim_in_query_or_follow_up(self) -> None:
        context = validation_context()
        rewritten = SQLGroundingState(
            column_mapping=(
                ColumnMapping(
                    phrase="power currently produced",
                    targets=(
                        "ep.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                    ),
                ),
            )
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "verbatim"):
            validate_sql_grounding_state(rewritten, context)

    def test_unknown_table_and_column_are_rejected(self) -> None:
        context = validation_context()
        with self.assertRaisesRegex(SQLGroundingValidationError, "unknown table"):
            validate_sql_grounding_state(
                SQLGroundingState(tables=("invented_table",)),
                context,
            )
        state = SQLGroundingState(
            column_mapping=(
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("operational_metrics.invented_column",),
                ),
            )
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "unknown column"):
            validate_sql_grounding_state(state, context)

    def test_domain_fact_requires_exact_official_evidence_projection(self) -> None:
        context = validation_context()
        unsupported = SQLGroundingState(
            domain_knowledge=(
                DomainKnowledge(
                    kind="runtime_state",
                    content="an invented runtime table exists",
                ),
            )
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "lacks"):
            validate_sql_grounding_state(unsupported, context)

    def test_context_is_transient_and_cannot_enter_state_or_runtime(self) -> None:
        context = validation_context()
        runtime = GroundingRuntime()
        payload = runtime.model_dump(mode="json")
        self.assertNotIn("validation_context", payload)
        self.assertNotIn("latest_observation_id", json.dumps(payload))
        self.assertFalse(hasattr(context, "gt_sql"))
        self.assertFalse(hasattr(context, "test_cases"))
        self.assertFalse(hasattr(context, "database"))
        self.assertFalse(hasattr(context, "network_client"))

    def test_context_rejects_columns_without_known_table_and_bad_aliases(self) -> None:
        with self.assertRaises(ValueError):
            ValidationContext(
                current_query="query",
                latest_observation_id="obs-1",
                known_tables=frozenset({"plants"}),
                known_columns=frozenset({"invented.id"}),
            )
        with self.assertRaises(ValueError):
            ValidationContext(
                current_query="query",
                latest_observation_id="obs-1",
                known_tables=frozenset({"plants"}),
                table_aliases=(("p", "invented"),),
            )


class ResponseRuntimeAndCanonicalTests(unittest.TestCase):
    def test_initial_response_cannot_focus_none_while_dimension_is_null(self) -> None:
        response = GroundingLLMResponse(
            sql_grounding_state=SQLGroundingState(tables=("plants",)),
            next_focus_dimension="none",
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "dimension is null"):
            validate_grounding_llm_response(
                response,
                stage="INITIAL_GROUNDING",
                context=validation_context(),
            )

    def test_initial_complete_and_repair_focus_contracts(self) -> None:
        context = validation_context()
        initial = GroundingLLMResponse(
            sql_grounding_state=complete_state(),
            next_focus_dimension="none",
        )
        self.assertIs(
            validate_grounding_llm_response(
                initial,
                stage="INITIAL_GROUNDING",
                context=context,
            ),
            initial,
        )
        repair = GroundingLLMResponse(
            sql_grounding_state=complete_state(),
            next_focus_dimension="join_keys",
        )
        self.assertIs(
            validate_grounding_llm_response(
                repair,
                stage="REPAIR",
                context=context,
            ),
            repair,
        )
        repair_none = GroundingLLMResponse(
            sql_grounding_state=complete_state(),
            next_focus_dimension="none",
        )
        self.assertIs(
            validate_grounding_llm_response(
                repair_none,
                stage="REPAIR",
                context=context,
            ),
            repair_none,
        )

    def test_stage_and_focus_are_closed_enums_and_not_state_fields(self) -> None:
        with self.assertRaises(ValidationError):
            GroundingRuntime(stage="PLANNING")
        with self.assertRaises(ValidationError):
            GroundingRuntime(focus_dimension="schema")
        self.assertEqual(
            set(SQLGroundingState.model_fields),
            {"tables", "join_keys", "column_mapping", "domain_knowledge"},
        )
        self.assertEqual(
            set(GroundingRuntime.model_fields),
            {"grounding_revision", "stage", "focus_dimension", "grounding_state"},
        )

    def test_runtime_transition_revision_depends_only_on_canonical_state(self) -> None:
        old = GroundingRuntime()
        control_only = GroundingRuntime(
            grounding_revision=0,
            stage="REPAIR",
            focus_dimension="domain_knowledge",
            grounding_state=old.grounding_state,
        )
        self.assertIs(validate_grounding_runtime_transition(old, control_only), control_only)

        changed = GroundingRuntime(
            grounding_revision=1,
            stage="REPAIR",
            focus_dimension="domain_knowledge",
            grounding_state=SQLGroundingState(tables=("plants",)),
        )
        self.assertIs(validate_grounding_runtime_transition(old, changed), changed)

        with self.assertRaises(SQLGroundingValidationError):
            validate_grounding_runtime_transition(
                old,
                changed.model_copy(update={"grounding_revision": 0}),
            )
        with self.assertRaises(SQLGroundingValidationError):
            validate_grounding_runtime_transition(
                old,
                control_only.model_copy(update={"grounding_revision": 1}),
            )

    def test_canonical_json_and_sha_are_order_independent(self) -> None:
        left = SQLGroundingState(
            tables=("plants", "operational_metrics"),
            column_mapping=(
                ColumnMapping(
                    phrase="current power",
                    targets=(
                        "ep.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                        "operational_metrics.maintcost",
                    ),
                ),
                ColumnMapping(
                    phrase="maintenance cost",
                    targets=("operational_metrics.maintcost",),
                ),
            ),
        )
        right = SQLGroundingState(
            tables=("operational_metrics", "plants"),
            column_mapping=tuple(reversed(left.column_mapping or ())),
        )
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(
            sql_grounding_state_sha256(left),
            sql_grounding_state_sha256(right),
        )

    def test_runtime_json_round_trip_and_old_requirement_runtime_rejection(self) -> None:
        runtime = GroundingRuntime(
            grounding_revision=1,
            stage="SQL_ATTEMPT",
            focus_dimension="none",
            grounding_state=complete_state(),
        )
        self.assertEqual(
            GroundingRuntime.model_validate_json(runtime.model_dump_json()),
            runtime,
        )
        old_requirement_runtime = {
            "schema_version": "1.1",
            "grounding_revision": 12,
            "requirement_revision": 2,
            "phase": 1,
            "grounding_state": {
                "requirement_frame": {
                    "value_slots": [],
                    "schema_slots": [],
                    "operation_slots": [],
                },
                "ambiguity_index": [],
                "evidence": [],
            },
        }
        with self.assertRaises(ValidationError):
            GroundingRuntime.model_validate(old_requirement_runtime)
        self.assertEqual(LEGACY_GROUNDING_RUNTIME_KEY, "valibra:grounding_runtime")
        self.assertEqual(SQL_GROUNDING_RUNTIME_KEY, "valibra:sql_grounding_runtime")


if __name__ == "__main__":
    unittest.main()
