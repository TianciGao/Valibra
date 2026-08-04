"""
通过 FastAPI 接口封装 LLM 智能体，使编排器能够通过 HTTP 端口与全部三个组件通信：

系统智能体： 端口 6000（本服务）
用户模拟器： 端口 6001
数据库环境： 端口 6002

支持两种模式：

/chat：简单的提示输入、文本输出模式（由 c-interact 使用）
/init_session + /run_session：基于 ADK 的会话运行时（由 a-interact 使用）
/chat_with_tools：面向旧版编排器客户端的传统非 ADK 回退模式
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import (
    active_model_preset_report,
    normalized_system_agent_config,
    settings,
)
from system_agent.adk_runtime import AdkRuntime

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact System Agent", version="1.0.0")
runtime = AdkRuntime()


@app.on_event("startup")
async def log_active_model_preset():
    report = active_model_preset_report()
    if report is not None:
        logger.info(
            "MODEL_PRESET=%s normalized_config=%s normalized_sha256=%s "
            "preset_file_sha256=%s",
            report["name"],
            report["normalized_config"],
            report["normalized_sha256"],
            report["file_sha256"],
        )


# ── Request / Response models ──────────────────────────────────────────────

class SessionInitRequest(BaseModel): # 表示请求内容必须符合以下字段：
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


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/init_session")
async def init_session(req: SessionInitRequest): # 建立一项任务的 Agent Runner 和会话记忆
    """Initialize an ADK runner session for a task."""
    if not runtime.available:
        raise HTTPException(status_code=503, detail=f"ADK runtime unavailable: {runtime.error}")
    return await runtime.init_session(
        task_id=req.task_id,
        mode=req.mode,
        state=req.state,
        reset=req.reset,
    )


@app.post("/run_session")
async def run_session(req: SessionRunRequest): # 把一条消息送进已有会话，让 Agent 真正运行一轮
    """Run one ADK turn on an existing task session."""
    if not runtime.available:
        raise HTTPException(status_code=503, detail=f"ADK runtime unavailable: {runtime.error}")
    return await runtime.run_turn(
        task_id=req.task_id,
        mode=req.mode,
        message=req.message,
    )


@app.post("/cleanup_session")
async def cleanup_session(req: SessionCleanupRequest):
    """Release a completed ADK session after its full state has been exported."""
    if not runtime.available:
        raise HTTPException(status_code=503, detail=f"ADK runtime unavailable: {runtime.error}")
    return await runtime.cleanup_session(task_id=req.task_id, mode=req.mode)


@app.get("/health") # 查看 6000 服务和 ADK 加载状态
async def health():
    normalized = normalized_system_agent_config()
    preset_report = active_model_preset_report()
    return {
        "status": "healthy",
        "service": "system_agent",
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


if __name__ == "__main__": # 支持直接使用 python -m system_agent.server 启动服务
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.system_agent_port)
