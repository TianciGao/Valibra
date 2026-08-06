"""Valibra Shadow Agent: Baseline behavior plus NoOp grounding lifecycle."""

from shared.config import settings
from system_agent.agent import (
    ADK_AVAILABLE,
    ADK_IMPORT_ERROR,
    AINTERACT_INSTRUCTION,
    Agent,
    _build_model,
    build_agent as build_baseline_agent,
    types,
)
from valibra_agent.grounding_callbacks import (
    after_model_callback,
    after_tool_callback,
    before_model_callback,
    before_tool_callback,
    on_tool_error_callback,
)


def build_agent(mode: str = "a-interact") -> Agent:
    """Build Shadow while keeping the Baseline prompt, tools, and behavior.

    Shadow is an a-interact shell. Other modes delegate to the Baseline builder
    so the shared HTTP mode field remains backward compatible.
    """
    if mode != "a-interact":
        return build_baseline_agent(mode)
    if not ADK_AVAILABLE:
        raise RuntimeError(
            f"google-adk runtime unavailable: {ADK_IMPORT_ERROR}"
        )

    from system_agent.tools import get_ainteract_tools

    return Agent(
        model=_build_model(settings.system_agent_model),
        name="bird_interact_agent",
        description="Text-to-SQL agent for BIRD-Interact a-interact benchmark.",
        instruction=AINTERACT_INSTRUCTION,
        tools=get_ainteract_tools(),
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        before_tool_callback=before_tool_callback,
        after_tool_callback=after_tool_callback,
        on_tool_error_callback=on_tool_error_callback,
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.system_agent_temperature
        ),
    )
