"""User Simulator Service (Port 6001). Two-stage function-driven pipeline."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException

from shared.config import settings
from shared.audit import summarize_usage, utc_now
from shared.models import (
    AskUserRequest,
    AskUserResponse,
    InitTaskRequest,
    PhaseTransitionRequest,
    SchemaRequest,
)
from user_simulator.prompts import USER_SIMULATOR_ACTION_PARSER, USER_SIMULATOR_RESPONSE_GENERATOR
from user_simulator.sql_parser import segment_sql

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact User Simulator", version="1.0.0")

PROMPT_VERSION = settings.prompt_version  # v1=legacy, v2=recommended
PROTOCOL_POLICY = settings.user_sim_protocol_policy
PROTOCOL_MAX_ATTEMPTS = settings.user_sim_protocol_max_attempts
if PROTOCOL_POLICY not in {"official", "strict_retry"}:
    raise RuntimeError(f"Unsupported User Simulator protocol policy: {PROTOCOL_POLICY}")
if PROTOCOL_MAX_ATTEMPTS < 1:
    raise RuntimeError("USER_SIM_PROTOCOL_MAX_ATTEMPTS must be at least 1")


class UserSimulatorProviderError(RuntimeError):
    """A provider failure that must not be converted into a scored answer."""


class UserSimulatorProtocolError(UserSimulatorProviderError):
    """A malformed simulator response that must not become a scored answer."""


class TaskSimState:
    def __init__(self, task_data: Dict[str, Any]):
        self.task_data = task_data
        self.db_name = task_data["selected_database"]
        self.amb_user_query = task_data["amb_user_query"]
        self.clear_query = task_data.get("query", task_data["amb_user_query"])
        self.reference_sql = task_data["sol_sql"]
        self.user_query_ambiguity = task_data.get("user_query_ambiguity", {})
        self.knowledge_ambiguity = task_data.get("knowledge_ambiguity", [])
        self.current_phase = 1
        self.db_schema = ""
        fu = task_data.get("follow_up") or {}
        self.follow_up_sol_sql = fu.get("sol_sql") if fu else None
        self.llm_calls: List[Dict[str, Any]] = []
        self.dialogue_history: List[Dict[str, Any]] = []

    def get_sql_segments(self, sql: str) -> str:
        segs = segment_sql(sql)
        return "\n\n".join(f"{clause}:\n{text}" for clause, text in segs)

    def get_all_sql_segments(self) -> str:
        sql_list = self.reference_sql if isinstance(self.reference_sql, list) else [self.reference_sql]
        return "\n===\n".join(self.get_sql_segments(sql) for sql in sql_list)

    def get_ambiguity_json(self) -> str:
        if self.current_phase == 1:
            return ("user_query_ambiguity: \n"
                    + json.dumps(self.user_query_ambiguity, indent=4)
                    + "\n\nknowledge_ambiguity: \n"
                    + json.dumps(self.knowledge_ambiguity, indent=4))
        return json.dumps({}, indent=4)

    def get_gt_sql_str(self) -> str:
        if isinstance(self.reference_sql, list):
            return "\n".join(self.reference_sql)
        return self.reference_sql

    def transition_to_phase2(self):
        self.current_phase = 2
        if self.follow_up_sol_sql:
            self.reference_sql = self.follow_up_sol_sql


_task_states: Dict[str, TaskSimState] = {}


def _call_llm(
    state: TaskSimState,
    stage: str,
    prompt: str,
    max_tokens: int = 200,
    *,
    protocol_attempt: int = 1,
    previous_content: str = "",
    correction: str = "",
) -> str:
    call_index = len(state.llm_calls) + 1
    started = time.perf_counter()
    messages = [{"role": "user", "content": prompt}]
    if previous_content and correction:
        # Keep the benchmark prompt byte-for-byte unchanged.  A malformed
        # response is repaired in a separate conversational turn containing
        # only a transport-format correction, not additional task guidance.
        messages.extend(
            [
                {"role": "assistant", "content": previous_content},
                {"role": "user", "content": correction},
            ]
        )
    try:
        from shared.llm import call_llm_with_details
        details = call_llm_with_details(
            messages,
            model_name=settings.user_sim_model,
            temperature=0,
            max_tokens=max_tokens,
        )
        details.update({
            "call_index": call_index,
            "stage": stage,
            "phase": state.current_phase,
            "prompt": prompt,
            "request_messages": messages,
            "max_tokens": max_tokens,
            "protocol_attempt": protocol_attempt,
        })
        state.llm_calls.append(details)
        return details.get("content", "")
    except Exception as e:
        logger.exception("User Simulator LLM call failed")
        state.llm_calls.append({
            "call_index": call_index,
            "timestamp": utc_now(),
            "stage": stage,
            "phase": state.current_phase,
            "model": settings.user_sim_model,
            "prompt": prompt,
            "request_messages": messages,
            "max_tokens": max_tokens,
            "protocol_attempt": protocol_attempt,
            "content": "",
            "usage": {},
            "latency_seconds": time.perf_counter() - started,
            "error": f"{type(e).__name__}: {e}",
        })
        raise UserSimulatorProviderError(
            "User Simulator provider call failed after retries "
            f"during {stage} ({type(e).__name__})"
        ) from e


def _extract_tagged_payload(content: str) -> Tuple[str, str]:
    """Extract the official ``<s>...</s>`` payload without inventing text.

    The upstream prompts end with ``<s>``, so the official implementation
    also accepts a response containing only the closing ``</s>`` tag.  A
    response with neither boundary is malformed and must be retried instead
    of being silently replaced by a model-visible fallback sentence.
    """
    text = (content or "").strip()
    if not text:
        return "", "empty_response"

    start = text.find("<s>")
    if start >= 0:
        end = text.find("</s>", start + 3)
        if end < 0:
            return "", "missing_closing_tag"
        payload = text[start + 3:end].strip()
    else:
        end = text.find("</s>")
        if end < 0:
            return "", "missing_protocol_tags"
        payload = text[:end].strip()

    if not payload:
        return "", "empty_protocol_payload"
    return payload, ""


def _protocol_correction(stage: str) -> str:
    noun = "chosen action" if stage == "action_parser" else "user answer"
    return (
        "Your previous response did not follow the required transport format. "
        f"Return the same {noun} enclosed in exactly one <s>...</s> pair. "
        "Do not add analysis, Markdown fences, or any new task information."
    )


def _call_llm_for_tagged_payload(
    state: TaskSimState,
    stage: str,
    prompt: str,
    max_tokens: int,
) -> str:
    """Call the simulator with bounded, audited protocol-only retries."""
    previous_content = ""
    correction = ""
    reasons: List[str] = []

    for attempt in range(1, PROTOCOL_MAX_ATTEMPTS + 1):
        content = _call_llm(
            state,
            stage,
            prompt,
            max_tokens=max_tokens,
            protocol_attempt=attempt,
            previous_content=previous_content,
            correction=correction,
        )
        payload, reason = _extract_tagged_payload(content)
        protocol_audit = {
            "required_format": "<s>...</s>",
            "attempt": attempt,
            "valid": not reason,
            "error": reason,
        }
        if state.llm_calls:
            state.llm_calls[-1]["protocol"] = protocol_audit
        if not reason:
            return payload

        reasons.append(reason)
        logger.warning(
            "User Simulator protocol violation during %s attempt %d/%d: %s",
            stage,
            attempt,
            PROTOCOL_MAX_ATTEMPTS,
            reason,
        )
        previous_content = content
        correction = _protocol_correction(stage)

    raise UserSimulatorProtocolError(
        "User Simulator returned malformed output after "
        f"{PROTOCOL_MAX_ATTEMPTS} protocol attempts during {stage}: "
        + ", ".join(reasons)
    )


def _parse_action(state: TaskSimState, question: str) -> str:
    """Stage 1: Action Parser — maps clarification question to action (AMB/LOC/UNA)."""
    template = USER_SIMULATOR_ACTION_PARSER[PROMPT_VERSION]
    prompt = template.replace("[[clarification_Q]]", question)
    prompt = prompt.replace("[[amb_json]]", state.get_ambiguity_json())
    prompt = prompt.replace("[[SQL_Glot]]", state.get_all_sql_segments())
    prompt = prompt.replace("[[DB_schema]]", state.db_schema)
    # v2 includes <think> reasoning, needs more tokens
    max_tok = 500 if PROMPT_VERSION == "v2" else 200
    if PROTOCOL_POLICY == "strict_retry":
        action = _call_llm_for_tagged_payload(
            state,
            "action_parser",
            prompt,
            max_tokens=max_tok,
        )
    else:
        # Byte-for-byte equivalent to the upstream official parser behavior.
        content = _call_llm(state, "action_parser", prompt, max_tokens=max_tok)
        if "</s>" in content:
            action = content.split("</s>")[0].strip()
        else:
            action = content.split("\n")[0].strip()
        if "<s>" in action:
            action = action[action.find("<s>"):].replace("<s>", "").strip()
    logger.info(f"Parsed action: {action}")
    return action


def _generate_response(state: TaskSimState, question: str, action: str) -> str:
    """Stage 2: Response Generator — produces user response from action + context."""
    template = USER_SIMULATOR_RESPONSE_GENERATOR[PROMPT_VERSION]
    prompt = template.replace("[[clarification_Q]]", question)
    prompt = prompt.replace("[[Action]]", action)
    prompt = prompt.replace("[[clear_query]]", state.clear_query)
    prompt = prompt.replace("[[amb_json]]", state.get_ambiguity_json())
    prompt = prompt.replace("[[GT_SQL]]", state.get_gt_sql_str())
    prompt = prompt.replace("[[SQL_Glot]]", state.get_all_sql_segments())
    prompt = prompt.replace("[[DB_schema]]", state.db_schema)
    if PROTOCOL_POLICY == "strict_retry":
        return _call_llm_for_tagged_payload(
            state,
            "response_generator",
            prompt,
            max_tokens=1024,
        )

    # Byte-for-byte equivalent to the upstream official parser behavior.
    content = _call_llm(state, "response_generator", prompt, max_tokens=1024)
    if "</s>" in content:
        extracted = content.split("</s>")[0].strip()
        if "<s>" in extracted:
            return extracted[extracted.find("<s>"):].replace("<s>", "").strip()
        return extracted
    if "<s>" in content:
        return content.split("<s>")[1].strip()
    return "I'm not sure I understand your question."


@app.post("/init_task")
async def init_task(req: InitTaskRequest):
    state = TaskSimState(req.task_data)
    _task_states[req.task_id] = state
    # Load schema from DB env
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.post(
                f"http://localhost:{settings.db_env_port}/schema",
                json={"task_id": req.task_id})
            state.db_schema = resp.json().get("schema", "")
    except Exception as e:
        logger.warning(f"Could not load schema: {e}")
    return {"status": "ok", "task_id": req.task_id}


def _ask_sync(state: "TaskSimState", question: str) -> str:
    """Two-stage pipeline: parse action, then generate response. Runs in thread pool."""
    action = _parse_action(state, question)
    return _generate_response(state, question, action)


@app.post("/ask", response_model=AskUserResponse)
async def ask_user(req: AskUserRequest):
    state = _task_states.get(req.task_id)
    if not state:
        raise HTTPException(404, f"Task {req.task_id} not initialized")
    try:
        response = await asyncio.to_thread(_ask_sync, state, req.question)
    except UserSimulatorProviderError as exc:
        # A 503 lets the agent/orchestrator distinguish provider downtime from
        # a genuine simulated-user answer.  The failed task is not scored.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    state.dialogue_history.append({
        "phase": state.current_phase,
        "question": req.question,
        "answer": response,
        "timestamp": utc_now(),
    })
    logger.info(f"User response for {req.task_id}: {response[:100]}...")
    return AskUserResponse(answer=response)


@app.post("/phase_transition")
async def phase_transition(req: PhaseTransitionRequest):
    state = _task_states.get(req.task_id)
    if not state:
        raise HTTPException(404, f"Task {req.task_id} not initialized")
    state.transition_to_phase2()
    return {"status": "ok", "phase": 2}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "user_simulator"}


@app.get("/debug_state/{task_id}")
async def debug_state(task_id: str):
    state = _task_states.get(task_id)
    if not state:
        return {"error": "not found"}
    return {
        "db_name": state.db_name,
        "schema_len": len(state.db_schema),
        "schema_first_100": state.db_schema[:100],
        "current_phase": state.current_phase,
        "amb_query": state.amb_user_query[:100],
    }


@app.get("/audit/{task_id}")
async def audit_state(task_id: str):
    """Return the complete user-simulator prompt/response/token trace."""
    state = _task_states.get(task_id)
    if not state:
        raise HTTPException(404, f"Task {task_id} not initialized")
    return {
        "task_id": task_id,
        "model": settings.user_sim_model,
        "prompt_version": PROMPT_VERSION,
        "llm_calls": state.llm_calls,
        "token_usage": summarize_usage(state.llm_calls),
        "dialogue_history": state.dialogue_history,
    }


@app.post("/cleanup_task")
async def cleanup_task(req: SchemaRequest):
    """Release a completed simulator state after its audit has been exported."""
    removed = _task_states.pop(req.task_id, None) is not None
    return {
        "status": "ok",
        "task_id": req.task_id,
        "state_removed": removed,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.user_sim_port)
