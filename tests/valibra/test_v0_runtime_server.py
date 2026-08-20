import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from system_agent import server as baseline_server
from system_agent.adk_runtime import AdkRuntime as BaselineAdkRuntime
from valibra_agent import server as valibra_server
from valibra_agent import grounding_callbacks
from valibra_agent.adk_runtime import AdkRuntime
from valibra_agent.agent import build_agent
from valibra_agent.sql_grounding.models import GroundingRuntime


class FakeSessionService:
    def __init__(self):
        self.next_id = 1
        self.created = []
        self.deleted = []

    async def create_session(self, app_name, user_id, state):
        session = SimpleNamespace(
            id=f"session-{self.next_id}",
            state=dict(state),
        )
        self.next_id += 1
        self.created.append((app_name, user_id, session))
        return session

    async def delete_session(self, app_name, user_id, session_id):
        self.deleted.append((app_name, user_id, session_id))


class RuntimeParityTests(unittest.IsolatedAsyncioTestCase):
    def test_runtime_subclasses_baseline_and_swaps_only_builder(self):
        runtime = AdkRuntime()
        self.assertIsInstance(runtime, BaselineAdkRuntime)
        self.assertTrue(runtime.available, runtime.error)
        self.assertIs(runtime._backend["build_agent"], build_agent)

    def test_runtime_reports_unavailable_when_adk_backend_import_fails(self):
        with patch(
            "system_agent.adk_runtime.importlib.import_module",
            side_effect=ImportError("synthetic missing ADK"),
        ):
            runtime = AdkRuntime()
        self.assertFalse(runtime.available)
        self.assertIn("synthetic missing ADK", runtime.error)
        self.assertIsNone(runtime._backend)

    async def test_session_reuse_reset_and_cleanup(self):
        runtime = AdkRuntime()
        service = FakeSessionService()
        runner = SimpleNamespace(session_service=service)
        runtime._get_runner = AsyncMock(
            return_value=(runner, "bird_interact_a_interact")
        )

        first = await runtime.init_session(
            task_id="task-1",
            mode="a-interact",
            state={"budget_remaining": 5.0},
            reset=False,
        )
        reused = await runtime.init_session(
            task_id="task-1", mode="a-interact", state={}, reset=False
        )
        reset = await runtime.init_session(
            task_id="task-1", mode="a-interact", state={}, reset=True
        )
        self.assertEqual(first["session_id"], reused["session_id"])
        self.assertNotEqual(first["session_id"], reset["session_id"])
        self.assertEqual(len(service.created), 2)
        self.assertEqual(
            service.created[0][2].state["budget_remaining"], 5.0
        )
        self.assertEqual(service.created[0][2].state["tool_trajectory"], [])
        self.assertEqual(service.created[0][2].state["adk_events"], [])

        removed = await runtime.cleanup_session("task-1", "a-interact")
        absent = await runtime.cleanup_session("task-1", "a-interact")
        self.assertTrue(removed["session_removed"])
        self.assertFalse(absent["session_removed"])
        self.assertEqual(service.deleted[-1][-1], reset["session_id"])

    async def test_sql_grounding_runtime_reuse_reset_and_cleanup_isolation(self):
        runtime = AdkRuntime()
        service = FakeSessionService()
        runner = SimpleNamespace(session_service=service)
        runtime._get_runner = AsyncMock(
            return_value=(runner, "bird_interact_a_interact")
        )
        persisted = GroundingRuntime().model_dump(mode="json")
        first = await runtime.init_session(
            task_id="task-sqlg",
            mode="a-interact",
            state={grounding_callbacks.GROUNDING_RUNTIME_KEY: persisted},
            reset=False,
        )
        reused = await runtime.init_session(
            task_id="task-sqlg", mode="a-interact", state={}, reset=False
        )
        self.assertEqual(first["session_id"], reused["session_id"])
        self.assertEqual(
            service.created[0][2].state[
                grounding_callbacks.GROUNDING_RUNTIME_KEY
            ],
            persisted,
        )

        reset = await runtime.init_session(
            task_id="task-sqlg", mode="a-interact", state={}, reset=True
        )
        self.assertNotEqual(reset["session_id"], first["session_id"])
        self.assertNotIn(
            grounding_callbacks.GROUNDING_RUNTIME_KEY,
            service.created[1][2].state,
        )
        removed = await runtime.cleanup_session("task-sqlg", "a-interact")
        self.assertTrue(removed["session_removed"])


class HttpContractTests(unittest.TestCase):
    def test_request_models_and_routes_match_baseline(self):
        pairs = [
            (valibra_server.SessionInitRequest, baseline_server.SessionInitRequest),
            (valibra_server.SessionRunRequest, baseline_server.SessionRunRequest),
            (
                valibra_server.SessionCleanupRequest,
                baseline_server.SessionCleanupRequest,
            ),
        ]
        for valibra_model, baseline_model in pairs:
            self.assertEqual(
                valibra_model.model_json_schema()["properties"],
                baseline_model.model_json_schema()["properties"],
            )
            self.assertEqual(
                valibra_model.model_json_schema().get("required"),
                baseline_model.model_json_schema().get("required"),
            )

        def contract(app):
            return {
                (route.path, tuple(sorted(route.methods or [])))
                for route in app.routes
                if route.path in {
                    "/init_session",
                    "/run_session",
                    "/cleanup_session",
                    "/health",
                }
            }

        self.assertEqual(contract(valibra_server.app), contract(baseline_server.app))

    def test_http_passthrough_uses_runtime_without_provider(self):
        fake_runtime = SimpleNamespace(
            available=True,
            error="",
            init_session=AsyncMock(
                return_value={"task_id": "t", "session_id": "s"}
            ),
            run_turn=AsyncMock(return_value={"response": "offline"}),
            cleanup_session=AsyncMock(
                return_value={"status": "ok", "session_removed": True}
            ),
        )
        with patch.object(valibra_server, "runtime", fake_runtime):
            with TestClient(valibra_server.app) as client:
                init_response = client.post(
                    "/init_session",
                    json={
                        "task_id": "t",
                        "mode": "a-interact",
                        "state": {"budget_remaining": 5},
                        "reset": True,
                    },
                )
                run_response = client.post(
                    "/run_session",
                    json={
                        "task_id": "t",
                        "mode": "a-interact",
                        "message": "offline contract test",
                    },
                )
                cleanup_response = client.post(
                    "/cleanup_session",
                    json={"task_id": "t", "mode": "a-interact"},
                )
        self.assertEqual(init_response.status_code, 200)
        self.assertEqual(run_response.status_code, 200)
        self.assertEqual(cleanup_response.status_code, 200)
        fake_runtime.init_session.assert_awaited_once()
        fake_runtime.run_turn.assert_awaited_once()
        fake_runtime.cleanup_session.assert_awaited_once()

    def test_health_identifies_sg6b_active_gate_without_credentials(self):
        with (
            patch.dict(os.environ, {"GROUNDING_UPDATER_MODE": ""}, clear=False),
            TestClient(valibra_server.app) as client,
        ):
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["service"], "valibra_agent")
        self.assertEqual(body["variant"], "SQL-Grounding-V1-SG6b-Active-Gate")
        self.assertTrue(body["configuration_summary"]["grounding_enabled"])
        self.assertEqual(
            body["configuration_summary"]["grounding_mode"],
            "active_view",
        )
        self.assertEqual(
            body["configuration_summary"]["grounding_updater"],
            "passthrough",
        )
        self.assertEqual(
            body["configuration_summary"]["grounding_core"],
            "sql_grounding_v1",
        )
        self.assertFalse(
            body["configuration_summary"]["grounding_provider_enabled"]
        )
        self.assertTrue(body["configuration_summary"]["control_enabled"])
        self.assertTrue(body["configuration_summary"]["attempt_gate_enabled"])
        self.assertEqual(
            body["configuration_summary"]["control_mode"],
            "active_hint",
        )
        self.assertTrue(
            body["configuration_summary"]["control_hint_injection_enabled"]
        )
        self.assertEqual(
            body["configuration_summary"]["attempt_gate_mode"],
            "active_first_submit",
        )
        self.assertTrue(
            body["configuration_summary"]["attempt_gate_blocking_enabled"]
        )
        self.assertTrue(
            body["configuration_summary"]["attempt_gate_budget_liveness_bypass"]
        )
        self.assertTrue(
            body["configuration_summary"]["prompt_view_injected"]
        )
        self.assertTrue(
            body["configuration_summary"]["prompt_view_injection_enabled"]
        )
        self.assertEqual(len(body["configuration_sha256"]), 64)
        self.assertTrue(
            body["git_commit"] == "unknown" or len(body["git_commit"]) == 40
        )
        serialized = response.text.lower()
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("research-offline", serialized)


if __name__ == "__main__":
    unittest.main()
