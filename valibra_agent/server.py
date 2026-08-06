"""Valibra P3 NoOp Shadow HTTP service on the configured agent port."""

import hashlib
import json
import logging
import subprocess
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import (
    PROJECT_ROOT,
    active_model_preset_report,
    normalized_system_agent_config,
    settings,
)
from valibra_agent.adk_runtime import AdkRuntime

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact Valibra Agent", version="P3-NoOp-Shadow")
runtime = AdkRuntime()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _configuration_summary() -> Dict[str, Any]:
    preset_report = active_model_preset_report()
    normalized = normalized_system_agent_config()
    return {
        "dataset": settings.dataset,
        "prompt_version": settings.prompt_version,
        "patience": settings.patience,
        "model": settings.system_agent_model,
        "generation_parameters": {
            key: value for key, value in normalized.items() if key != "model"
        },
        "user_simulator_model": settings.user_sim_model,
        "model_preset": preset_report["name"] if preset_report else None,
        "normalized_model_sha256": (
            preset_report["normalized_sha256"] if preset_report else None
        ),
        "ports": {
            "valibra_agent": settings.system_agent_port,
            "user_simulator": settings.user_sim_port,
            "db_environment": settings.db_env_port,
            "postgresql": settings.pg_port,
        },
        "grounding_enabled": True,
        "grounding_mode": "shadow",
        "grounding_updater": "noop",
        "prompt_view_injected": False,
    }


def _summary_sha256(summary: Dict[str, Any]) -> str:
    payload = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@app.on_event("startup")
async def log_active_configuration() -> None:
    summary = _configuration_summary()
    logger.info(
        "Valibra variant=P3-NoOp-Shadow git_commit=%s configuration_summary=%s "
        "configuration_sha256=%s",
        _git_commit(),
        summary,
        _summary_sha256(summary),
    )


class SessionInitRequest(BaseModel):
    task_id: str
    mode: str = "a-interact"
    state: Dict[str, Any] = {}
    reset: bool = True


class SessionRunRequest(BaseModel):
    task_id: str
    message: str
    mode: str = "a-interact"


class SessionCleanupRequest(BaseModel):
    task_id: str
    mode: str = "a-interact"


@app.post("/init_session")
async def init_session(req: SessionInitRequest):
    if not runtime.available:
        raise HTTPException(
            status_code=503,
            detail=f"ADK runtime unavailable: {runtime.error}",
        )
    return await runtime.init_session(
        task_id=req.task_id,
        mode=req.mode,
        state=req.state,
        reset=req.reset,
    )


@app.post("/run_session")
async def run_session(req: SessionRunRequest):
    if not runtime.available:
        raise HTTPException(
            status_code=503,
            detail=f"ADK runtime unavailable: {runtime.error}",
        )
    return await runtime.run_turn(
        task_id=req.task_id,
        mode=req.mode,
        message=req.message,
    )


@app.post("/cleanup_session")
async def cleanup_session(req: SessionCleanupRequest):
    if not runtime.available:
        raise HTTPException(
            status_code=503,
            detail=f"ADK runtime unavailable: {runtime.error}",
        )
    return await runtime.cleanup_session(
        task_id=req.task_id,
        mode=req.mode,
    )


@app.get("/health")
async def health():
    normalized = normalized_system_agent_config()
    preset_report = active_model_preset_report()
    summary = _configuration_summary()
    return {
        "status": "healthy",
        "service": "valibra_agent",
        "variant": "P3-NoOp-Shadow",
        "git_commit": _git_commit(),
        "configuration_summary": summary,
        "configuration_sha256": _summary_sha256(summary),
        "model": settings.system_agent_model,
        "generation_parameters": {
            key: value for key, value in normalized.items() if key != "model"
        },
        "model_preset": (
            {
                "name": preset_report["name"],
                "normalized_sha256": preset_report["normalized_sha256"],
                "file_sha256": preset_report["file_sha256"],
            }
            if preset_report is not None
            else None
        ),
        "adk_available": runtime.available,
        "adk_error": runtime.error,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.system_agent_port)
