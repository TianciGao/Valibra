from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from valibra_agent.sql_grounding import (
    ALLOWED_CAST_TYPES,
    CORRELATED_RELATION_AST_WHITELIST,
    FIELD_EXPRESSION_AST_WHITELIST,
    FIELD_EXPRESSION_FUNCTION_WHITELIST,
    LEGACY_GROUNDING_RUNTIME_KEY,
    RELATION_EXPRESSION_AST_WHITELIST,
    SAME_TABLE_MULTI_RECORD_AST_WHITELIST,
    SQLGLOT_VERSION,
    SQL_GROUNDING_RUNTIME_KEY,
    ColumnMapping,
    DomainKnowledge,
    GroundingLLMResponse,
    GroundingRuntime,
    SQLGroundingState,
    SQLGroundingValidationError,
    StateDiffAuthorization,
    ValidationContext,
    canonical_json,
    canonicalize_field_expression,
    canonicalize_relation_expression,
    sql_grounding_state_sha256,
    validate_grounding_llm_response,
    validate_grounding_runtime_transition,
    validate_grounding_state_transition,
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
            "encounters",
            "patients",
            "planets",
            "orbital_characteristics",
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
            "orbital_characteristics.orbitalref",
            "orbital_characteristics.period",
            "encounters.time_mark",
            "encounters.pat_ref",
            "patients.pat_key",
            "stars.hostplname",
            "stars.stellarref",
            "planets.hostlink",
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
        supported_json_paths=frozenset(
            {
                (
                    "electrical_performance.elec_perf_snapshot",
                    ("power",),
                ),
                (
                    "electrical_performance.elec_perf_snapshot",
                    ("power", "power_now_w"),
                ),
                ("metrics.session_telemetry", ("activity_pattern",)),
                (
                    "metrics.session_telemetry",
                    ("activity_pattern", "hourly_observed"),
                ),
            }
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
            "plant_record.snapkey = operational_metrics.snapops",
            "plants.sitekey = plant_record.sitetie",
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
            "electrical_performance.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
            "metrics.values[1]",
            "UNNEST(STRING_TO_ARRAY(candidates.comorbid_detail, ','))",
            "JSONB_ARRAY_ELEMENTS_TEXT(metrics.session_telemetry -> 'activity_pattern' -> 'hourly_observed')",
            "CAST(data_flows.flow_overview AS JSONB) -> 'routing' ->> 'destination_country'",
        )
        for expression in examples:
            with self.subTest(expression=expression):
                self.assertEqual(canonicalize_field_expression(expression), expression)

    def test_equal_normalized_range_and_temporal_relations(self) -> None:
        examples = (
            "plants.sitekey = plant_record.sitetie",
            "LOWER(robot_details.mfgnameval) = predictive_models.mfgnameval_lower",
            "TRIM(robot_details.mfgnameval) = TRIM(predictive_models.mfgnameval_lower)",
            "score_bands.score BETWEEN score_rules.min_score AND score_rules.max_exclusive",
            "score_bands.score < score_rules.max_exclusive OR (score_rules.include_max AND score_bands.score <= score_rules.max_exclusive)",
            "incidents.time_mark < prediction_events.estimated_subscription_date",
            "TO_CHAR(audits.REMED_DUE, 'YYYY-MM') = month_targets.audit_month",
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
            "plants.sitekey = plant_record.sitetie; DELETE FROM plants",
            "plants.sitekey = plant_record.sitetie /* comment */",
            "plants.sitekey IN (SELECT inner_record.sitetie FROM plant_record AS inner_record)",
            "plants.sitekey = 1",
            "plants.sitekey = plant_record.sitetie OR 1 = 1",
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
                "lower(robot_details.mfgnameval) = predictive_models.mfgnameval_lower"
            )
        with self.assertRaises(SQLGroundingValidationError):
            canonicalize_field_expression(
                "unnest(string_to_array(candidates.comorbid_detail, ','))"
            )

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
            CORRELATED_RELATION_AST_WHITELIST,
            frozenset(
                {
                    "Column",
                    "EQ",
                    "From",
                    "Identifier",
                    "Max",
                    "Select",
                    "Subquery",
                    "Table",
                    "TableAlias",
                    "Where",
                }
            ),
        )
        self.assertEqual(
            SAME_TABLE_MULTI_RECORD_AST_WHITELIST,
            frozenset(
                {
                    "Column",
                    "Identifier",
                    "Lag",
                    "NEQ",
                    "Order",
                    "Ordered",
                    "Window",
                }
            ),
        )
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

    def test_full600_correlated_scalar_relation_shapes_are_narrowly_supported(
        self,
    ) -> None:
        audited = (
            "encounters.time_mark = (SELECT MAX(inner_encounters.time_mark) FROM encounters AS inner_encounters WHERE inner_encounters.pat_ref = patients.pat_key)",
            "stars.hostplname = (SELECT inner_stars.hostplname FROM stars AS inner_stars WHERE inner_stars.stellarref = planets.hostlink)",
        )
        for expression in audited:
            with self.subTest(expression=expression):
                self.assertEqual(canonicalize_relation_expression(expression), expression)

    def test_arbitrary_or_uncorrelated_selects_remain_rejected(self) -> None:
        rejected = (
            "patients.pat_key = (SELECT inner_encounters.pat_ref FROM encounters AS inner_encounters)",
            "patients.pat_key = (SELECT inner_encounters.pat_ref FROM encounters AS inner_encounters WHERE inner_encounters.pat_ref = inner_encounters.enc_key)",
            "encounters.time_mark = (SELECT MIN(inner_encounters.time_mark) FROM encounters AS inner_encounters WHERE inner_encounters.pat_ref = patients.pat_key)",
            "patients.pat_key = (SELECT inner_encounters.pat_ref FROM encounters AS inner_encounters JOIN patients AS inner_patients ON inner_patients.pat_key = inner_encounters.pat_ref WHERE inner_encounters.pat_ref = patients.pat_key)",
        )
        for expression in rejected:
            with self.subTest(expression=expression), self.assertRaises(
                SQLGroundingValidationError
            ):
                canonicalize_relation_expression(expression)

    def test_audited_same_table_adjacent_record_relation_is_self_contained(
        self,
    ) -> None:
        audited = (
            "LAG(orbital_characteristics.orbitalref) OVER (PARTITION BY planets.hostlink ORDER BY orbital_characteristics.period) <> orbital_characteristics.orbitalref"
        )
        self.assertEqual(canonicalize_relation_expression(audited), audited)
        state = SQLGroundingState(
            tables=("orbital_characteristics", "planets"),
            join_keys=(audited,),
        )
        self.assertIs(validate_sql_grounding_state(state, validation_context()), state)

    def test_same_row_and_unbounded_same_table_forms_are_rejected(self) -> None:
        rejected = (
            "stars.rn = stars.rn - 1",
            "ROW_NUMBER() OVER (PARTITION BY planets.hostlink ORDER BY orbital_characteristics.period) <> orbital_characteristics.orbitalref",
            "LAG(orbital_characteristics.orbitalref, 2) OVER (PARTITION BY planets.hostlink ORDER BY orbital_characteristics.period) <> orbital_characteristics.orbitalref",
            "LAG(orbital_characteristics.orbitalref) OVER (PARTITION BY planets.hostlink ORDER BY orbital_characteristics.period DESC) <> orbital_characteristics.orbitalref",
            "LAG(orbital_characteristics.orbitalref) OVER (PARTITION BY planets.hostlink ORDER BY orbital_characteristics.period, orbital_characteristics.orbitalref) <> orbital_characteristics.orbitalref",
            "EXISTS(SELECT 1 FROM orbital_characteristics AS earlier_record CROSS JOIN orbital_characteristics AS later_record WHERE earlier_record.orbitalref <> later_record.orbitalref)",
        )
        for expression in rejected:
            with self.subTest(expression=expression), self.assertRaises(
                SQLGroundingValidationError
            ):
                canonicalize_relation_expression(expression)

        with self.assertRaisesRegex(ValidationError, "absent from"):
            SQLGroundingState(
                tables=("orbital_characteristics", "planets"),
                join_keys=("p1.rnum = p2.rnum - 1",),
            )

        hidden_alias_state = SQLGroundingState(
            tables=("p1", "p2"),
            join_keys=("p1.rnum = p2.rnum - 1",),
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "unknown table p1"):
            validate_sql_grounding_state(hidden_alias_state, validation_context())

        with self.assertRaisesRegex(ValidationError, "planets"):
            SQLGroundingState(
                tables=("orbital_characteristics",),
                join_keys=(
                    "LAG(orbital_characteristics.orbitalref) OVER "
                    "(PARTITION BY planets.hostlink ORDER BY "
                    "orbital_characteristics.period) <> "
                    "orbital_characteristics.orbitalref",
                ),
            )


class SelfContainedAndCrossDimensionTests(unittest.TestCase):
    def test_normal_expressions_use_real_table_identifiers(self) -> None:
        state = SQLGroundingState(
            tables=("electrical_performance", "plant_record", "plants"),
            join_keys=("plants.sitekey = plant_record.sitetie",),
            column_mapping=(
                ColumnMapping(
                    phrase="current power",
                    targets=(
                        "electrical_performance.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                    ),
                ),
            ),
        )
        self.assertEqual(
            state.column_mapping[0].targets[0],
            "electrical_performance.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
        )

    def test_hidden_shorthand_aliases_cannot_enter_state(self) -> None:
        with self.assertRaisesRegex(ValidationError, "absent from"):
            SQLGroundingState(
                tables=("electrical_performance",),
                column_mapping=(
                    ColumnMapping(
                        phrase="current power",
                        targets=(
                            "ep.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                        ),
                    ),
                ),
            )
        with self.assertRaisesRegex(ValidationError, "absent from"):
            SQLGroundingState(
                tables=("plant_record", "plants"),
                join_keys=("p.sitekey = pr.sitetie",),
            )
        disguised_alias = SQLGroundingState(
            tables=("ep",),
            column_mapping=(
                ColumnMapping(
                    phrase="current power",
                    targets=(
                        "ep.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "unknown table ep"):
            validate_sql_grounding_state(disguised_alias, validation_context())

    def test_correlated_local_alias_is_self_contained_and_context_validated(self) -> None:
        mental = SQLGroundingState(
            tables=("encounters", "patients"),
            join_keys=(
                "encounters.time_mark = (SELECT MAX(inner_encounters.time_mark) FROM encounters AS inner_encounters WHERE inner_encounters.pat_ref = patients.pat_key)",
            ),
        )
        planets = SQLGroundingState(
            tables=("planets", "stars"),
            join_keys=(
                "stars.hostplname = (SELECT inner_stars.hostplname FROM stars AS inner_stars WHERE inner_stars.stellarref = planets.hostlink)",
            ),
        )
        context = validation_context()
        self.assertIs(validate_sql_grounding_state(mental, context), mental)
        self.assertIs(validate_sql_grounding_state(planets, context), planets)

        with self.assertRaises(SQLGroundingValidationError):
            canonicalize_relation_expression(
                "encounters.time_mark = (SELECT MAX(encounters.time_mark) FROM encounters AS encounters WHERE encounters.pat_ref = patients.pat_key)"
            )

    def test_mapping_and_relation_references_must_be_subset_of_tables(self) -> None:
        with self.assertRaisesRegex(ValidationError, "predictive_models"):
            SQLGroundingState(
                tables=("robot_details",),
                join_keys=(
                    "LOWER(robot_details.mfgnameval) = predictive_models.mfgnameval_lower",
                ),
            )
        with self.assertRaisesRegex(ValidationError, "electrical_performance"):
            SQLGroundingState(
                tables=("plants",),
                column_mapping=(
                    ColumnMapping(
                        phrase="current power",
                        targets=(
                            "electrical_performance.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                        ),
                    ),
                ),
            )
        with self.assertRaisesRegex(ValidationError, "plants"):
            SQLGroundingState(
                tables=(),
                join_keys=("plants.sitekey = plant_record.sitetie",),
            )

    def test_single_table_no_join_and_unbound_domain_fact_remain_valid(self) -> None:
        state = SQLGroundingState(
            tables=("operational_metrics",),
            join_keys=(),
            column_mapping=(),
            domain_knowledge=(
                DomainKnowledge(
                    kind="database_capability",
                    content="pg_relation_size is required to obtain relation size",
                ),
            ),
        )
        self.assertEqual(state.join_keys, ())
        self.assertIs(validate_sql_grounding_state(state, validation_context()), state)


class ValidationContextTests(unittest.TestCase):
    def test_full600_driven_examples_validate_against_transient_context(self) -> None:
        context = validation_context()
        state = SQLGroundingState(
            tables=(
                "data_flows",
                "electrical_performance",
                "metrics",
                "predictive_models",
                "robot_details",
            ),
            join_keys=(
                "LOWER(robot_details.mfgnameval) = predictive_models.mfgnameval_lower",
            ),
            column_mapping=(
                ColumnMapping(
                    phrase="current power",
                    targets=(
                        "electrical_performance.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
                    ),
                ),
                ColumnMapping(
                    phrase="hourly activity",
                    targets=(
                        "JSONB_ARRAY_ELEMENTS_TEXT(metrics.session_telemetry -> 'activity_pattern' -> 'hourly_observed')",
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
            tables=("electrical_performance",),
            column_mapping=(
                ColumnMapping(
                    phrase="power currently produced",
                    targets=(
                        "electrical_performance.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
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
            tables=("operational_metrics",),
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

    def test_context_rejects_columns_without_known_table_and_has_no_alias_map(self) -> None:
        with self.assertRaises(ValueError):
            ValidationContext(
                current_query="query",
                latest_observation_id="obs-1",
                known_tables=frozenset({"plants"}),
                known_columns=frozenset({"invented.id"}),
            )
        with self.assertRaises(TypeError):
            ValidationContext(
                current_query="query",
                latest_observation_id="obs-1",
                known_tables=frozenset({"plants"}),
                table_aliases=(("p", "invented"),),
            )
        self.assertNotIn("table_aliases", ValidationContext.__dataclass_fields__)


class StateDiffAuthorizationTests(unittest.TestCase):
    def test_authorization_contract_is_strict_json_safe_and_canonical(self) -> None:
        authorization = StateDiffAuthorization(
            stage="P2_INCREMENTAL",
            authorized_dimensions=("domain_knowledge", "tables"),
        )
        self.assertEqual(
            authorization.authorized_dimensions,
            ("tables", "domain_knowledge"),
        )
        self.assertEqual(
            StateDiffAuthorization.model_validate_json(
                authorization.model_dump_json()
            ),
            authorization,
        )
        with self.assertRaises(ValidationError):
            StateDiffAuthorization(
                stage="P2_INCREMENTAL",
                authorized_dimensions=("tables", "tables"),
            )

    def test_initial_null_to_populated_requires_dimension_authorization(self) -> None:
        previous = SQLGroundingState()
        candidate = SQLGroundingState(tables=("plants",))
        allowed = StateDiffAuthorization(
            stage="INITIAL_GROUNDING",
            authorized_dimensions=("tables",),
        )
        self.assertEqual(
            validate_grounding_state_transition(previous, candidate, allowed),
            ("tables",),
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "unauthorized"):
            validate_grounding_state_transition(
                previous,
                candidate,
                StateDiffAuthorization(stage="INITIAL_GROUNDING"),
            )

    def test_evaluated_dimension_never_regresses_to_null(self) -> None:
        previous = SQLGroundingState(tables=("plants",))
        candidate = SQLGroundingState()
        for stage in ("INITIAL_GROUNDING", "P2_INCREMENTAL"):
            with self.subTest(stage=stage), self.assertRaisesRegex(
                SQLGroundingValidationError,
                "evaluated to null",
            ):
                validate_grounding_state_transition(
                    previous,
                    candidate,
                    StateDiffAuthorization(
                        stage=stage,
                        authorized_dimensions=("tables",),
                    ),
                )

    def test_authorized_initial_correction_can_replace_or_clear_a_dimension(self) -> None:
        previous = SQLGroundingState(tables=("plants",))
        replacement = SQLGroundingState(tables=("operational_metrics",))
        cleared = SQLGroundingState(tables=())
        authorization = StateDiffAuthorization(
            stage="INITIAL_GROUNDING",
            authorized_dimensions=("tables",),
        )
        self.assertEqual(
            validate_grounding_state_transition(previous, replacement, authorization),
            ("tables",),
        )
        self.assertEqual(
            validate_grounding_state_transition(previous, cleared, authorization),
            ("tables",),
        )

    def test_p2_only_changes_follow_up_authorized_dimensions(self) -> None:
        previous = complete_state()
        candidate = previous.model_copy(
            update={
                "domain_knowledge": (
                    DomainKnowledge(
                        kind="business_rule",
                        content="downtime score = mttrh / (mtbfh + mttrh)",
                    ),
                )
            }
        )
        allowed = StateDiffAuthorization(
            stage="P2_INCREMENTAL",
            authorized_dimensions=("domain_knowledge",),
        )
        self.assertEqual(
            validate_grounding_state_transition(previous, candidate, allowed),
            ("domain_knowledge",),
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "unauthorized"):
            validate_grounding_state_transition(
                previous,
                candidate,
                StateDiffAuthorization(
                    stage="P2_INCREMENTAL",
                    authorized_dimensions=("column_mapping",),
                ),
            )

    def test_retired_repair_stage_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            StateDiffAuthorization(stage="REPAIR")
        with self.assertRaises(ValidationError):
            GroundingRuntime(stage="REPAIR")

    def test_attempt_and_done_cannot_change_grounding_state(self) -> None:
        previous = SQLGroundingState()
        candidate = SQLGroundingState(tables=("plants",))
        for stage in ("SQL_ATTEMPT", "DONE"):
            with self.subTest(stage=stage), self.assertRaisesRegex(
                SQLGroundingValidationError,
                "does not authorize",
            ):
                validate_grounding_state_transition(
                    previous,
                    candidate,
                    StateDiffAuthorization(
                        stage=stage,
                        authorized_dimensions=("tables",),
                    ),
                )

    def test_noop_has_no_changed_dimensions(self) -> None:
        state = complete_state()
        self.assertEqual(
            validate_grounding_state_transition(
                state,
                state,
                StateDiffAuthorization(stage="SQL_ATTEMPT"),
            ),
            (),
        )


class ResponseRuntimeAndCanonicalTests(unittest.TestCase):
    def test_initial_response_cannot_focus_none_while_dimension_is_null(self) -> None:
        partial_state = SQLGroundingState(tables=("plants",))
        response = GroundingLLMResponse(
            sql_grounding_state=partial_state,
            user_clarification_requests=(),
            next_focus_dimension="none",
        )
        with self.assertRaisesRegex(SQLGroundingValidationError, "dimension is null"):
            validate_grounding_llm_response(
                response,
                stage="INITIAL_GROUNDING",
                context=validation_context(),
            )
        with self.assertRaisesRegex(ValidationError, "cannot focus none"):
            GroundingRuntime(
                stage="INITIAL_GROUNDING",
                focus_dimension="none",
                grounding_state=partial_state,
            )
        for focus in (
            "tables",
            "join_keys",
            "column_mapping",
            "domain_knowledge",
        ):
            with self.subTest(focus=focus):
                candidate = GroundingLLMResponse(
                    sql_grounding_state=partial_state,
                    user_clarification_requests=(),
                    next_focus_dimension=focus,
                )
                self.assertIs(
                    validate_grounding_llm_response(
                        candidate,
                        stage="INITIAL_GROUNDING",
                        context=validation_context(),
                    ),
                    candidate,
                )
                self.assertEqual(
                    GroundingRuntime(
                        stage="INITIAL_GROUNDING",
                        focus_dimension=focus,
                        grounding_state=partial_state,
                    ).focus_dimension,
                    focus,
                )

    def test_initial_complete_contract_and_retired_repair_stage(self) -> None:
        context = validation_context()
        initial = GroundingLLMResponse(
            sql_grounding_state=complete_state(),
            user_clarification_requests=(),
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
        with self.assertRaisesRegex(
            SQLGroundingValidationError,
            "must return none after all dimensions are evaluated",
        ):
            validate_grounding_llm_response(
                initial.model_copy(update={"next_focus_dimension": "tables"}),
                stage="INITIAL_GROUNDING",
                context=context,
            )
        with self.assertRaisesRegex(
            ValidationError,
            "must focus none after all dimensions are evaluated",
        ):
            GroundingRuntime(
                stage="INITIAL_GROUNDING",
                focus_dimension="tables",
                grounding_state=complete_state(),
            )
        completed_runtime = GroundingRuntime(
            stage="INITIAL_GROUNDING",
            focus_dimension="none",
            grounding_state=complete_state(),
        )
        self.assertEqual(completed_runtime.focus_dimension, "none")
        with self.assertRaises(ValidationError):
            GroundingRuntime(stage="REPAIR")

    def test_stage_and_focus_are_closed_enums_and_not_state_fields(self) -> None:
        with self.assertRaises(ValidationError):
            GroundingRuntime(stage="PLANNING")
        with self.assertRaises(ValidationError):
            GroundingRuntime(stage="REPAIR")
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
            stage="SQL_ATTEMPT",
            focus_dimension="domain_knowledge",
            grounding_state=old.grounding_state,
        )
        self.assertIs(validate_grounding_runtime_transition(old, control_only), control_only)

        changed = GroundingRuntime(
            grounding_revision=1,
            stage="INITIAL_GROUNDING",
            focus_dimension="tables",
            grounding_state=SQLGroundingState(tables=("plants",)),
        )
        initial_tables = StateDiffAuthorization(
            stage="INITIAL_GROUNDING",
            authorized_dimensions=("tables",),
        )
        self.assertIs(
            validate_grounding_runtime_transition(old, changed, initial_tables),
            changed,
        )

        with self.assertRaises(SQLGroundingValidationError):
            validate_grounding_runtime_transition(
                old,
                changed.model_copy(update={"grounding_revision": 0}),
                initial_tables,
            )
        with self.assertRaises(SQLGroundingValidationError):
            validate_grounding_runtime_transition(
                old,
                changed.model_copy(update={"grounding_revision": 2}),
                initial_tables,
            )
        with self.assertRaisesRegex(SQLGroundingValidationError, "stage must match"):
            validate_grounding_runtime_transition(
                old,
                changed,
                StateDiffAuthorization(
                    stage="SQL_ATTEMPT",
                    authorized_dimensions=("tables",),
                ),
            )
        with self.assertRaises(SQLGroundingValidationError):
            validate_grounding_runtime_transition(
                old,
                control_only.model_copy(update={"grounding_revision": 1}),
            )

    def test_canonical_json_and_sha_are_order_independent(self) -> None:
        left = SQLGroundingState(
            tables=(
                "electrical_performance",
                "operational_metrics",
                "plants",
            ),
            column_mapping=(
                ColumnMapping(
                    phrase="current power",
                    targets=(
                        "electrical_performance.elec_perf_snapshot -> 'power' ->> 'power_now_w'",
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
            tables=(
                "plants",
                "operational_metrics",
                "electrical_performance",
            ),
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
