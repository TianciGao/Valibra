"""Valibra Shadow runtime using the complete Baseline session implementation."""

from system_agent.adk_runtime import AdkRuntime as BaselineAdkRuntime
from valibra_agent.agent import build_agent


class AdkRuntime(BaselineAdkRuntime):
    """Replace only the Agent construction entry used by the Baseline runtime."""

    def _load_backend(self) -> None:
        super()._load_backend()
        if self._backend is not None:
            self._backend["build_agent"] = build_agent
