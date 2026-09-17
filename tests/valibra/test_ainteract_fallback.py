from __future__ import annotations

import asyncio
from types import SimpleNamespace

from system_agent.adk_runtime import AdkRuntime as BaselineAdkRuntime
from valibra_agent.adk_runtime import AdkRuntime
from valibra_agent.fallback.ainteract_fallback import (
    FALLBACK_AUDIT_KEY,
    FALLBACK_FEATURE_FLAG,
    FALLBACK_MODE,
    _fallback_after_tool_callback,
    _fallback_before_tool_callback,
    build_fallback_message,
    fallback_eligibility,
    finish_fallback_state,
    start_fallback_state,
)


class AdkLikeState:
    """Minimum live ADK State surface; intentionally has no ``pop`` method."""

    def __init__(self, value: dict):
        self._value = value

    def get(self, key, default=None):
        return self._value.get(key, default)

    def __getitem__(self, key):
        return self._value[key]

    def __setitem__(self, key, value):
        self._value[key] = value


def base_state(*, budget: float = 3.0) -> dict:
    return {
        "task_id": "task_1",
        "db_name": "demo",
        "user_query": "Show risky sites.",
        "current_phase": 1,
        "initial_budget": 18.0,
        "budget_remaining": budget,
        "total_reward": 0.0,
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
        "tool_trajectory": [],
        "system_agent_llm_calls": [],
        "dialogue_history": [],
        "valibra:user_clarifications": [
            {
                "phase": 1,
                "phrase": "risky sites",
                "question": "What counts as a risky site?",
                "answer": "More than two claims.",
            }
        ],
        "valibra:sql_grounding_runtime": {
            "grounding_state": {
                "tables": ["sites"],
                "join_keys": [],
                "column_mapping": [],
                "domain_knowledge": [],
            }
        },
    }


def test_feature_is_default_off_and_budget_floor_is_three(monkeypatch):
    monkeypatch.delenv(FALLBACK_FEATURE_FLAG, raising=False)
    assert fallback_eligibility(base_state()) == (False, "feature_disabled")

    monkeypatch.setenv(FALLBACK_FEATURE_FLAG, "1")
    assert fallback_eligibility(base_state(budget=2.5)) == (
        False,
        "insufficient_budget",
    )
    assert fallback_eligibility(base_state(budget=3.0)) == (
        True,
        "terminal_fail_with_legal_budget",
    )


def test_completed_or_already_attempted_task_never_falls_back(monkeypatch):
    monkeypatch.setenv(FALLBACK_FEATURE_FLAG, "1")
    state = base_state()
    state["task_done"] = True
    assert fallback_eligibility(state) == (False, "task_already_complete")
    state["task_done"] = False
    state[FALLBACK_AUDIT_KEY] = {"attempted": True}
    assert fallback_eligibility(state) == (False, "already_attempted")


def test_message_carries_clarification_and_marks_4d_state_non_authoritative():
    message = build_fallback_message(base_state())
    assert "More than two claims." in message
    assert "What counts as a risky site?" in message
    assert '"remaining_bird_coin": 3.0' in message
    assert '"tables": ["sites"]' in message
    assert "non-authoritative" in message


def test_p2_message_carries_follow_up_and_accepted_artifact():
    state = base_state(budget=3.5)
    state.update({"current_phase": 2, "phase1_completed": True})
    state["tool_trajectory"] = [
        {
            "tool": "submit_sql",
            "phase": 1,
            "args": {"sql": "CREATE TABLE kept AS SELECT 1 AS x"},
            "result": (
                "Phase 1 correct! (Reward: 0.7). Moving to Phase 2.\n"
                "Follow-up question: Count rows in kept.\n"
                "Budget remaining: 3.5 bird-coins"
            ),
        }
    ]
    message = build_fallback_message(state)
    assert "Count rows in kept." in message
    assert "CREATE TABLE kept AS SELECT 1 AS x" in message
    copied = start_fallback_state(state)
    assert copied["budget_remaining"] == 3.5
    assert copied["current_phase"] == 2
    assert copied[FALLBACK_AUDIT_KEY]["accepted_p1_sql_carried"] is True


def test_exact_duplicate_ask_is_blocked_without_bird_coin_charge():
    state = start_fallback_state(base_state(budget=7.0))
    context = SimpleNamespace(state=state)
    tool = SimpleNamespace(name="ask_user")
    response = asyncio.run(
        _fallback_before_tool_callback(
            tool,
            {"question": "What counts as a risky site?"},
            context,
        )
    )
    assert response["error"].startswith("This clarification question was already answered")
    assert state["budget_remaining"] == 7.0
    assert state.get("tool_trajectory") == []
    assert (
        asyncio.run(_fallback_after_tool_callback(tool, {}, context, response))
        is None
    )
    assert state["budget_remaining"] == 7.0


def test_duplicate_ask_marker_is_consumed_from_adk_state_without_pop():
    raw_state = start_fallback_state(base_state(budget=7.0))
    state = AdkLikeState(raw_state)
    context = SimpleNamespace(state=state)
    tool = SimpleNamespace(name="ask_user")

    response = asyncio.run(
        _fallback_before_tool_callback(
            tool,
            {"question": "What counts as a risky site?"},
            context,
        )
    )

    assert response["error"].startswith(
        "This clarification question was already answered"
    )
    assert (
        asyncio.run(_fallback_after_tool_callback(tool, {}, context, response))
        is None
    )
    assert raw_state["_valibra_fallback_duplicate_ask"] is None
    assert raw_state["budget_remaining"] == 7.0


def test_finish_records_phase_only_and_full_rescue():
    state = start_fallback_state(base_state())
    state["phase1_completed"] = True
    state["current_phase"] = 2
    finish_fallback_state(state, tool_count_before=0)
    assert state[FALLBACK_AUDIT_KEY]["status"] == "phase1_rescued_only"

    state["phase2_completed"] = True
    state["task_done"] = True
    finish_fallback_state(state, tool_count_before=0)
    assert state[FALLBACK_AUDIT_KEY]["status"] == "task_rescued"


def test_runtime_uses_separate_runner_and_inherits_ledger(monkeypatch):
    monkeypatch.setenv(FALLBACK_FEATURE_FLAG, "1")
    runtime = object.__new__(AdkRuntime)
    calls: list[tuple[str, str]] = []
    initialized: dict[str, dict] = {}

    async def fake_init(self, task_id, mode, state=None, reset=False):
        assert mode == FALLBACK_MODE
        assert reset is True
        initialized[mode] = state
        return {"task_id": task_id, "mode": mode}

    async def fake_run(self, task_id, mode, message, **kwargs):
        calls.append((mode, message))
        if mode == "a-interact":
            return {"response": "Valibra terminal", "state": base_state(budget=3.0)}
        state = initialized[mode]
        assert state["budget_remaining"] == 3.0
        assert state["current_phase"] == 1
        state["budget_remaining"] = -1
        state["phase1_completed"] = True
        state["task_done"] = True
        state["tool_trajectory"].append({"tool": "submit_sql"})
        return {"response": "rescued", "state": state}

    async def fake_cleanup(self, task_id, mode):
        assert mode == FALLBACK_MODE
        return {"status": "ok"}

    monkeypatch.setattr(BaselineAdkRuntime, "init_session", fake_init)
    monkeypatch.setattr(BaselineAdkRuntime, "run_turn", fake_run)
    monkeypatch.setattr(BaselineAdkRuntime, "cleanup_session", fake_cleanup)

    result = asyncio.run(
        runtime.run_turn(
            task_id="task_1",
            mode="a-interact",
            message="original",
        )
    )
    assert [mode for mode, _ in calls] == ["a-interact", FALLBACK_MODE]
    assert result["response"] == "rescued"
    assert result["state"][FALLBACK_AUDIT_KEY]["status"] == "task_rescued"
    assert result["state"][FALLBACK_AUDIT_KEY]["budget_before"] == 3.0
    assert result["state"][FALLBACK_AUDIT_KEY]["fallback_tool_calls"] == 1


def test_runtime_flag_off_is_byte_path_equivalent(monkeypatch):
    monkeypatch.delenv(FALLBACK_FEATURE_FLAG, raising=False)
    runtime = object.__new__(AdkRuntime)
    primary = {"response": "terminal", "state": base_state(budget=10.0)}
    calls: list[str] = []

    async def fake_run(self, task_id, mode, message, **kwargs):
        calls.append(mode)
        return primary

    monkeypatch.setattr(BaselineAdkRuntime, "run_turn", fake_run)
    result = asyncio.run(runtime.run_turn("task_1", "a-interact", "original"))
    assert result is primary
    assert calls == ["a-interact"]
    assert FALLBACK_AUDIT_KEY not in result["state"]
