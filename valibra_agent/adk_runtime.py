"""Valibra Rule Shadow runtime extending only the Baseline turn entry."""

from system_agent.adk_runtime import AdkRuntime as BaselineAdkRuntime
from valibra_agent.agent import build_agent
from valibra_agent.grounding_callbacks import (
    _bind_turn_message,
    _reset_turn_message,
)


class AdkRuntime(BaselineAdkRuntime):
    """Replace only the Agent construction entry used by the Baseline runtime."""

    def _load_backend(self) -> None:
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
        """Bind the exact user message, then run the complete Baseline turn."""

        if mode != "a-interact":
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
            _reset_turn_message(token)
