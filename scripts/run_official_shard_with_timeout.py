"""Launch the official shard runner with an authorized HTTP wait override.

The immutable benchmark implementation remains unchanged. This wrapper only
replaces the orchestrator-to-system-agent wait used by ``run_agent_session``
for the current process.
"""

from __future__ import annotations

import os

from orchestrator import ainteract


def _authorized_timeout() -> float:
    value = float(os.environ.get("SYSTEM_AGENT_RUN_TIMEOUT_SECONDS", "1800"))
    if value <= 0:
        raise ValueError("SYSTEM_AGENT_RUN_TIMEOUT_SECONDS must be positive")
    return value


async def _run_agent_session(task_id: str, message: str):
    return await ainteract._post(
        f"{ainteract.SYSTEM_AGENT_URL}/run_session",
        {"task_id": task_id, "mode": "a-interact", "message": message},
        timeout=_authorized_timeout(),
    )


ainteract.run_agent_session = _run_agent_session

from orchestrator.official_shard_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
