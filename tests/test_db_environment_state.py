import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import httpx

from db_environment import server
from shared import db_utils
from shared.models import InitTaskRequest
from system_agent import tools as agent_tools
from user_simulator import server as user_sim_server


FULL_INPUT = (
    Path(__file__).resolve().parents[2]
    / "Dataset"
    / "bird-interact-full"
    / "bird_interact_data.jsonl"
)


class TaskDatabaseNameTests(unittest.TestCase):
    @unittest.skipUnless(
        FULL_INPUT.is_file(),
        "Full600 dataset is not distributed with the code; install it to run this integrity check.",
    )
    def test_all_full_task_database_names_are_bounded_and_unique(self) -> None:
        records = [
            json.loads(line)
            for line in FULL_INPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 600)

        generated = []
        for record in records:
            names = db_utils.task_database_names(
                record["selected_database"], record["instance_id"]
            )
            self.assertEqual(set(names), {"task", "initial", "phase1"})
            self.assertEqual(len(set(names.values())), 3)
            self.assertTrue(
                all(
                    len(name.encode("utf-8")) <= db_utils.PG_IDENTIFIER_MAX_BYTES
                    for name in names.values()
                )
            )
            generated.extend(names.values())

        self.assertEqual(len(generated), 1800)
        self.assertEqual(len(set(generated)), 1800)

    def test_long_logical_names_no_longer_collide(self) -> None:
        first = db_utils.task_database_names(
            "labor_certification_applications",
            "labor_certification_applications_2",
        )
        second = db_utils.task_database_names(
            "labor_certification_applications",
            "labor_certification_applications_7",
        )
        self.assertTrue(set(first.values()).isdisjoint(second.values()))

    def test_clone_refuses_to_drop_its_own_template(self) -> None:
        with self.assertRaisesRegex(ValueError, "from itself"):
            db_utils._drop_and_create_db("same_database", "same_database")


class EvaluatorCompatibilityTests(unittest.TestCase):
    def test_remove_distinct_preserves_postgresql_distinct_on(self) -> None:
        sql = "SELECT DISTINCT ON (account_id) account_id FROM accounts"
        self.assertEqual(db_utils.remove_distinct([sql]), [sql])
        self.assertEqual(
            db_utils.remove_distinct(["SELECT DISTINCT account_id FROM accounts"]),
            ["SELECT  account_id FROM accounts"],
        )

    def test_split_sql_preserves_dollar_quoted_function_body(self) -> None:
        payload = """
        CREATE FUNCTION example() RETURNS integer AS $$
        BEGIN
            RETURN 1;
        END;
        $$ LANGUAGE plpgsql;
        INSERT INTO audit_log(value) VALUES (1);
        """
        statements = db_utils.split_sql_statements(payload)
        self.assertEqual(len(statements), 2)
        self.assertIn("RETURN 1;", statements[0])
        self.assertTrue(statements[1].startswith("INSERT INTO audit_log"))

    def test_writable_cte_without_returning_is_not_fetched(self) -> None:
        connection = Mock()
        cursor = connection.cursor.return_value
        cursor.description = None

        result, returned_connection, description = db_utils.perform_query(
            "WITH removed AS (DELETE FROM items RETURNING id) "
            "DELETE FROM audit WHERE item_id IN (SELECT id FROM removed)",
            "example",
            conn=connection,
        )

        self.assertIsNone(result)
        self.assertIs(returned_connection, connection)
        self.assertIsNone(description)
        cursor.fetchmany.assert_not_called()
        connection.commit.assert_called_once_with()

    def test_analyze_target_parsing(self) -> None:
        self.assertEqual(
            db_utils._analyze_target_table('ANALYZE "monitoring";'),
            "monitoring",
        )
        self.assertEqual(
            db_utils._analyze_target_table("ANALYZE (VERBOSE) public.events"),
            "events",
        )
        self.assertIsNone(db_utils._analyze_target_table("SELECT 1"))


class KnowledgeCatalogIntegrityTests(unittest.TestCase):
    def test_source_latex_commands_survive_json_decoding(self) -> None:
        raw = (
            r'{"id":1,"knowledge":"formula","description":"",'
            r'"definition":"\text{A} = \frac{B}{C} \times 100"}'
        )

        entry = server._decode_source_knowledge_entry(raw)

        self.assertEqual(
            entry["definition"],
            r"\text{A} = \frac{B}{C} \times 100",
        )
        self.assertNotIn("\t", entry["definition"])
        self.assertNotIn("\f", entry["definition"])

    def test_correct_json_escapes_and_newlines_are_not_rewritten(self) -> None:
        raw = json.dumps(
            {
                "id": 1,
                "knowledge": "formula",
                "description": "",
                "definition": "Line one\n" + r"\text{Line two}",
            },
            separators=(",", ":"),
        )

        entry = server._decode_source_knowledge_entry(raw)

        self.assertEqual(entry["definition"], "Line one\n" + r"\text{Line two}")

    def test_one_bad_record_does_not_poison_catalog_or_change_task_mask(self) -> None:
        rows = [
            json.dumps(
                {
                    "id": 1,
                    "knowledge": "first",
                    "description": "",
                    "definition": "First definition.",
                }
            ),
            '{"id":2,"knowledge":"broken",',
            json.dumps(
                {
                    "id": 3,
                    "knowledge": "third",
                    "description": "",
                    "definition": "Third definition.",
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit_kb.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertLogs(server.logger, level="ERROR"):
                catalog = server._load_knowledge_catalog(str(path))

        self.assertEqual(set(catalog), {"first", "third"})
        with patch.dict(server._external_knowledge_cache, {"unit": catalog}):
            visible = server._filter_knowledge(
                "unit",
                {"knowledge_ambiguity": [{"deleted_knowledge": 3}]},
            )
        self.assertEqual(set(visible), {"first"})

    def test_shipped_malformed_intent_latex_is_preserved(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "bird-interact-full"
            / "mental_health"
            / "mental_health_kb.jsonl"
        )
        if not path.is_file():
            self.skipTest(
                "Full knowledge catalog is not distributed with the code; "
                "install it to run this source-data integrity check."
            )

        catalog = server._load_knowledge_catalog(str(path))
        definition = next(
            item["definition"] for item in catalog.values() if item["id"] == 65
        )

        self.assertTrue(definition.startswith(r"\text{PHRT}"))
        self.assertNotIn("\t", definition)
        self.assertNotIn("\f", definition)


class DatabaseStateFlowTests(unittest.TestCase):
    @patch.object(server, "_load_db_data")
    @patch.object(
        server,
        "_initialise_task_databases",
        side_effect=RuntimeError("synthetic init failure"),
    )
    def test_failed_reinitialization_does_not_keep_stale_state(
        self, initialise, load_data
    ) -> None:
        task_id = "same_task"
        server._task_data[task_id] = {"_task_db": "stale"}
        server._submit_attempts[task_id] = {1: 2, 2: 0}
        server._successful_phase1_sql[task_id] = "SELECT 1"
        request = InitTaskRequest(
            task_id=task_id,
            task_data={"selected_database": "sports_events"},
        )

        with self.assertLogs(server.logger, level="ERROR"):
            with self.assertRaisesRegex(
                Exception, "Task database initialization failed"
            ):
                asyncio.run(server.init_task(request))

        self.assertNotIn(task_id, server._task_data)
        self.assertNotIn(task_id, server._submit_attempts)
        self.assertNotIn(task_id, server._successful_phase1_sql)

    @patch.object(server, "close_pool")
    @patch.object(server, "_execute_required_queries")
    @patch.object(server, "clone_database")
    @patch.object(server, "drop_task_db")
    def test_preprocess_creates_canonical_initial_snapshot(
        self, drop_db, clone_db, execute_required, close_pool
    ) -> None:
        task_id = "sports_events_M_5"
        task_data = {
            "selected_database": "sports_events",
            "preprocess_sql": ["UPDATE example SET value = 1", "SELECT 1"],
        }
        names = db_utils.task_database_names("sports_events", task_id)

        state = server._initialise_task_databases(task_id, task_data)

        self.assertEqual(
            drop_db.call_args_list,
            [call(names["phase1"]), call(names["initial"]), call(names["task"])],
        )
        self.assertEqual(
            clone_db.call_args_list,
            [
                call(names["task"], "sports_events_template"),
                call(names["initial"], names["task"]),
            ],
        )
        execute_required.assert_called_once_with(
            names["task"],
            task_data["preprocess_sql"],
            label="preprocess_sql",
        )
        close_pool.assert_called_once_with(names["task"])
        self.assertEqual(state["_task_db"], names["task"])
        self.assertEqual(state["_initial_snapshot_db"], names["initial"])
        self.assertTrue(state["_initial_snapshot_owned"])
        self.assertEqual(state["_preprocess_statement_count"], 2)

    @patch.object(server, "close_pool")
    @patch.object(server, "_execute_required_queries")
    @patch.object(server, "clone_database")
    @patch.object(server, "drop_task_db")
    def test_empty_preprocess_reuses_official_base_template(
        self, drop_db, clone_db, execute_required, close_pool
    ) -> None:
        task_id = "sports_events_1"
        task_data = {"selected_database": "sports_events", "preprocess_sql": []}
        names = db_utils.task_database_names("sports_events", task_id)

        state = server._initialise_task_databases(task_id, task_data)

        clone_db.assert_called_once_with(names["task"], "sports_events_template")
        execute_required.assert_not_called()
        close_pool.assert_not_called()
        self.assertEqual(state["_initial_snapshot_db"], "sports_events_template")
        self.assertFalse(state["_initial_snapshot_owned"])
        self.assertEqual(state["_preprocess_statement_count"], 0)

    @patch.object(server, "clone_database")
    @patch.object(server, "close_pool")
    @patch.object(server, "_execute_required_queries")
    @patch.object(server, "reset_task_db")
    def test_phase1_snapshot_uses_preprocess_state_then_cleanup(
        self, reset_db, execute_required, close_pool, clone_db
    ) -> None:
        names = db_utils.task_database_names("virtual_idol", "virtual_idol_M_1")
        task_data = {
            "_task_db": names["task"],
            "_initial_snapshot_db": names["initial"],
            "_physical_db_names": names,
            "clean_up_sqls": ["DROP VIEW IF EXISTS generated_view"],
        }
        predicted_sql = ["CREATE VIEW generated_view AS SELECT 1"]

        snapshot = server._create_phase1_snapshot(task_data, predicted_sql)

        reset_db.assert_called_once_with(names["task"], names["initial"])
        self.assertEqual(
            execute_required.call_args_list,
            [
                call(
                    names["task"], predicted_sql, label="accepted Phase 1 SQL"
                ),
                call(
                    names["task"],
                    task_data["clean_up_sqls"],
                    label="clean_up_sqls",
                ),
            ],
        )
        close_pool.assert_called_once_with(names["task"])
        clone_db.assert_called_once_with(names["phase1"], names["task"])
        self.assertEqual(snapshot, names["phase1"])
        self.assertEqual(task_data["_phase1_maintenance_sql"], [])

    def test_phase_maintenance_replays_only_analyze(self) -> None:
        statements = [
            "INSERT INTO monitoring(value) VALUES (1);",
            "ANALYZE monitoring;",
            "  analyze (verbose) public.events;",
        ]
        self.assertEqual(
            server._phase_maintenance_sql(statements),
            ["ANALYZE monitoring;", "  analyze (verbose) public.events;"],
        )


class InfrastructureFailureTests(unittest.TestCase):
    @patch(
        "shared.llm.call_llm_with_details",
        side_effect=RuntimeError("synthetic provider outage"),
    )
    def test_user_simulator_provider_failure_is_not_replaced_by_fallback(
        self, call_llm
    ) -> None:
        state = SimpleNamespace(llm_calls=[], current_phase=1)

        with self.assertRaisesRegex(
            user_sim_server.UserSimulatorProviderError,
            "provider call failed after retries",
        ):
            user_sim_server._call_llm(state, "action_parser", "prompt")

        self.assertEqual(len(state.llm_calls), 1)
        self.assertIn("synthetic provider outage", state.llm_calls[0]["error"])

    def test_user_simulator_protocol_extractor_never_invents_fallback(self) -> None:
        self.assertEqual(
            user_sim_server._extract_tagged_payload("<s>Minimal access</s>"),
            ("Minimal access", ""),
        )
        self.assertEqual(
            user_sim_server._extract_tagged_payload("Minimal access</s>"),
            ("Minimal access", ""),
        )
        payload, reason = user_sim_server._extract_tagged_payload(
            "Minimal access"
        )
        self.assertEqual(payload, "")
        self.assertEqual(reason, "missing_protocol_tags")

    @patch(
        "shared.llm.call_llm_with_details",
        side_effect=[
            {
                "content": "Minimal access",
                "usage": {},
            },
            {
                "content": "<s>Minimal access</s>",
                "usage": {},
            },
        ],
    )
    @patch.object(user_sim_server, "PROTOCOL_MAX_ATTEMPTS", 3)
    def test_user_simulator_protocol_violation_is_retried_and_audited(
        self, call_llm
    ) -> None:
        state = SimpleNamespace(llm_calls=[], current_phase=1)

        result = user_sim_server._call_llm_for_tagged_payload(
            state,
            "response_generator",
            "official prompt",
            max_tokens=1024,
        )

        self.assertEqual(result, "Minimal access")
        self.assertEqual(call_llm.call_count, 2)
        self.assertFalse(state.llm_calls[0]["protocol"]["valid"])
        self.assertTrue(state.llm_calls[1]["protocol"]["valid"])
        retry_messages = call_llm.call_args_list[1].args[0]
        self.assertEqual(retry_messages[0]["content"], "official prompt")
        self.assertEqual(retry_messages[1]["content"], "Minimal access")
        self.assertIn("transport format", retry_messages[2]["content"])

    @patch(
        "shared.llm.call_llm_with_details",
        return_value={"content": "still untagged", "usage": {}},
    )
    @patch.object(user_sim_server, "PROTOCOL_MAX_ATTEMPTS", 3)
    def test_user_simulator_protocol_exhaustion_fails_task_uncheckpointed(
        self, call_llm
    ) -> None:
        state = SimpleNamespace(llm_calls=[], current_phase=1)

        with self.assertRaisesRegex(
            user_sim_server.UserSimulatorProtocolError,
            "malformed output after 3 protocol attempts",
        ):
            user_sim_server._call_llm_for_tagged_payload(
                state,
                "response_generator",
                "official prompt",
                max_tokens=1024,
            )

        self.assertEqual(call_llm.call_count, 3)
        self.assertTrue(
            all(not item["protocol"]["valid"] for item in state.llm_calls)
        )

    @patch.object(agent_tools.httpx, "Client")
    def test_ask_user_503_is_not_returned_as_model_feedback(
        self, client_class
    ) -> None:
        request = httpx.Request("POST", "http://user-sim/ask")
        response = httpx.Response(
            503,
            request=request,
            text='{"detail":"provider call failed"}',
        )
        response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "service unavailable", request=request, response=response
            )
        )
        client = client_class.return_value.__enter__.return_value
        client.post.return_value = response
        tool_context = SimpleNamespace(
            state={"task_id": "example", "dialogue_history": []}
        )

        with self.assertRaisesRegex(RuntimeError, "infrastructure failure"):
            agent_tools.ask_user("Which interpretation?", tool_context)

        self.assertIn("_infrastructure_error", tool_context.state)

    @patch.object(agent_tools.httpx, "Client")
    def test_submit_database_503_is_not_returned_as_model_feedback(
        self, client_class
    ) -> None:
        request = httpx.Request("POST", "http://db-env/submit")
        response = httpx.Response(
            503,
            request=request,
            text='{"detail":"snapshot failed"}',
        )
        response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "service unavailable", request=request, response=response
            )
        )
        client = client_class.return_value.__enter__.return_value
        client.post.return_value = response
        tool_context = SimpleNamespace(state={"task_id": "example", "total_reward": 0})

        with self.assertRaisesRegex(RuntimeError, "infrastructure failure"):
            agent_tools.submit_sql("SELECT 1", tool_context)

        self.assertEqual(tool_context.state["total_reward"], 0)


if __name__ == "__main__":
    unittest.main()
