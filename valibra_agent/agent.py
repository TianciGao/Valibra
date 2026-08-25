"""Valibra SQL Grounding Agent built around the frozen Official lifecycle."""

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
    """Build the SQL Grounding agent while reusing Official tools and model."""

    if mode != "a-interact":
        # 非 a-interact 模式直接使用原 Agent，保证接口兼容。
        return build_baseline_agent(mode)
    if not ADK_AVAILABLE:
        raise RuntimeError(
            f"google-adk runtime unavailable: {ADK_IMPORT_ERROR}"
        )

    from system_agent.tools import get_ainteract_tools

    # Valibra 只替换回调；模型、Prompt 和九个官方工具都复用 Baseline。
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
