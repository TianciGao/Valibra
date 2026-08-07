"""Valibra 运行时：只改 Agent 构造和单轮入口，其余沿用 Baseline。"""

from system_agent.adk_runtime import AdkRuntime as BaselineAdkRuntime
from valibra_agent.agent import build_agent
from valibra_agent.grounding_callbacks import (
    _bind_turn_message,
    _reset_turn_message,
)


class AdkRuntime(BaselineAdkRuntime):
    """在 Baseline 运行时上挂入 Valibra Agent。"""

    def _load_backend(self) -> None:
        """先加载原后端，再把 Agent 工厂替换成 Valibra 版本。"""

        super()._load_backend()
        if self._backend is not None:
            self._backend["build_agent"] = build_agent

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
            return await super().run_turn(
                task_id=task_id,
                mode=mode,
                message=message,
                **kwargs,
            )
        finally:
            # ContextVar 必须在本轮结束后恢复，避免并发任务串消息。
            _reset_turn_message(token)
