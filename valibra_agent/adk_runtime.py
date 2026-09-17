"""Valibra 运行时：只改 Agent 构造和单轮入口，其余沿用 Baseline。"""

import logging

from system_agent.adk_runtime import AdkRuntime as BaselineAdkRuntime
from valibra_agent.agent import build_agent
from valibra_agent.fallback.ainteract_fallback import (
    FALLBACK_AUDIT_KEY,
    FALLBACK_MODE,
    build_fallback_agent,
    build_fallback_message,
    fallback_eligibility,
    finish_fallback_state,
    start_fallback_state,
)
from valibra_agent.grounding_callbacks import (
    _bind_turn_message,
    _reset_turn_message,
)


logger = logging.getLogger(__name__)


class AdkRuntime(BaselineAdkRuntime):
    """在 Baseline 运行时上挂入 Valibra Agent。"""

    def _load_backend(self) -> None:
        """先加载原后端，再把 Agent 工厂替换成 Valibra 版本。"""

        super()._load_backend()
        if self._backend is not None:
            self._backend["build_agent"] = build_agent

    async def _run_native_fallback(
        self,
        *,
        task_id: str,
        primary_result: dict,
    ) -> dict:
        primary_state = primary_result.get("state")
        if not isinstance(primary_state, dict):
            return primary_result
        fallback_state = start_fallback_state(primary_state)
        message = build_fallback_message(fallback_state)
        trajectory = fallback_state.get("tool_trajectory")
        tool_count_before = len(trajectory) if isinstance(trajectory, list) else 0
        fallback_runtime = None
        initialized = False
        try:
            fallback_runtime = BaselineAdkRuntime()
            if not fallback_runtime.available or fallback_runtime._backend is None:
                raise RuntimeError(
                    fallback_runtime.error or "native fallback runtime unavailable"
                )
            fallback_runtime._backend["build_agent"] = (
                lambda _mode: build_fallback_agent()
            )
            await BaselineAdkRuntime.init_session(
                fallback_runtime,
                task_id=task_id,
                mode=FALLBACK_MODE,
                state=fallback_state,
                reset=True,
            )
            initialized = True
            fallback_result = await BaselineAdkRuntime.run_turn(
                fallback_runtime,
                task_id=task_id,
                mode=FALLBACK_MODE,
                message=message,
            )
            final_state = fallback_result.get("state")
            if not isinstance(final_state, dict):
                raise RuntimeError("native fallback returned no session state")
            finish_fallback_state(
                final_state,
                tool_count_before=tool_count_before,
            )
            return {
                **fallback_result,
                "mode": "a-interact",
                "valibra_primary_response": primary_result.get("response", ""),
            }
        except Exception as exc:
            logger.warning(
                "Native A-Interact fallback failed for %s: %s",
                task_id,
                type(exc).__name__,
            )
            failed_state = finish_fallback_state(
                fallback_state,
                tool_count_before=tool_count_before,
                error_type=type(exc).__name__[:128],
            )
            primary_result["state"] = failed_state
            return primary_result
        finally:
            if initialized and fallback_runtime is not None:
                try:
                    await BaselineAdkRuntime.cleanup_session(
                        fallback_runtime,
                        task_id=task_id,
                        mode=FALLBACK_MODE,
                    )
                except Exception as exc:
                    logger.warning(
                        "Native fallback session cleanup failed for %s: %s",
                        task_id,
                        type(exc).__name__,
                    )

    async def run_turn(
        self,
        task_id: str,
        mode: str,
        message: str,
        **kwargs,
    ):
        """临时绑定本轮用户消息，再完整执行 Baseline 流程。"""

        if mode != "a-interact":
            # Valibra 只处理 a-interact，其他模式不做任何增强。
            return await super().run_turn(
                task_id=task_id,
                mode=mode,
                message=message,
                **kwargs,
            )
        token = _bind_turn_message(task_id, mode, message)
        try:
            primary_result = await super().run_turn(
                task_id=task_id,
                mode=mode,
                message=message,
                **kwargs,
            )
            state = primary_result.get("state")
            if not isinstance(state, dict):
                return primary_result
            eligible, reason = fallback_eligibility(state)
            if not eligible:
                return primary_result
            state[FALLBACK_AUDIT_KEY] = {
                "attempted": False,
                "status": "eligible",
                "eligibility_reason": reason,
            }
            return await self._run_native_fallback(
                task_id=task_id,
                primary_result=primary_result,
            )
        finally:
            # ContextVar 必须在本轮结束后恢复，避免并发任务串消息。
            _reset_turn_message(token)
