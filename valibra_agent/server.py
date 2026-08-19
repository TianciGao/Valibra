"""Valibra 的 HTTP 服务入口，端口由 SYSTEM_AGENT_PORT 决定。"""

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
from valibra_agent.sql_grounding.updater import sql_grounding_provider_health_report

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact Valibra Agent", version="SQLG-V1-SG6b-Active")
runtime = AdkRuntime()


def _git_commit() -> str:
    """读取当前提交号；失败时返回 unknown，不影响服务启动。"""

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
    """生成不含密钥的运行配置摘要，便于复现实验。"""

    preset_report = active_model_preset_report()
    normalized = normalized_system_agent_config()
    grounding_provider = sql_grounding_provider_health_report(PROJECT_ROOT)
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
        "grounding_core": "sql_grounding_v1",
        "grounding_mode": "active_view",
        "grounding_updater": grounding_provider["effective_updater_mode"],
        "grounding_requested_updater_mode": grounding_provider[
            "requested_updater_mode"
        ],
        "grounding_effective_updater_mode": grounding_provider[
            "effective_updater_mode"
        ],
        "grounding_provider_configuration_valid": grounding_provider[
            "provider_configuration_valid"
        ],
        "grounding_provider_enabled": grounding_provider["provider_enabled"],
        "grounding_provider_configuration_error_type": grounding_provider[
            "configuration_error_type"
        ],
        "grounding_prompt_view_effective_mode": "active",
        "prompt_view_injection_enabled": True,
        # These booleans report configured active injection.  Per-call truth
        # (including fail-open suppression) lives in the bounded Callback audit.
        "prompt_view_injected": True,
        "control_mode": "active_hint",
        "control_hint_injection_enabled": True,
        "attempt_gate_mode": "active_first_submit",
        "attempt_gate_blocking_enabled": True,
        "attempt_gate_budget_liveness_bypass": True,
        "control_enabled": True,
        "attempt_gate_enabled": True,
    }


def _variant(summary: Dict[str, Any]) -> str:
    """Return the SG6b Active Gate service identity."""

    del summary
    return "SQL-Grounding-V1-SG6b-Active-Gate"


def _summary_sha256(summary: Dict[str, Any]) -> str:
    """给配置摘要生成稳定指纹。"""

    payload = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@app.on_event("startup")
async def log_active_configuration() -> None:
    """启动时记录版本和配置，但不记录凭据。"""

    summary = _configuration_summary()
    logger.info(
        "Valibra variant=%s git_commit=%s configuration_summary=%s "
        "configuration_sha256=%s",
        _variant(summary),
        _git_commit(),
        summary,
        _summary_sha256(summary),
    )


class SessionInitRequest(BaseModel):
    """初始化或重置一个任务会话。"""

    task_id: str
    mode: str = "a-interact"
    state: Dict[str, Any] = {}
    reset: bool = True


class SessionRunRequest(BaseModel):
    """向已有会话发送一轮用户消息。"""

    task_id: str
    message: str
    mode: str = "a-interact"


class SessionCleanupRequest(BaseModel):
    """清理指定任务的会话状态。"""

    task_id: str
    mode: str = "a-interact"


@app.post("/init_session")
async def init_session(req: SessionInitRequest):
    """创建会话；ADK 不可用时明确返回 503。"""

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
    """执行一轮 Agent 交互。"""

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
    """释放指定任务的运行时资源。"""

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
    """返回健康状态和可公开的配置指纹。"""

    normalized = normalized_system_agent_config()
    preset_report = active_model_preset_report()
    summary = _configuration_summary()
    return {
        "status": "healthy",
        "service": "valibra_agent",
        "variant": _variant(summary),
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
