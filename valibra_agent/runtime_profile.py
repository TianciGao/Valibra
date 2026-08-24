"""Explicit Valibra execution profile with a research-safe default."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal, cast

# Importing shared.config also applies the project's established environment
# priority, including its project-local .env loading.  The profile itself stays
# outside the frozen BIRD settings and model-preset contracts.
from shared import config as _shared_config  # noqa: F401


VALIBRA_EXECUTION_PROFILE_ENV = "VALIBRA_EXECUTION_PROFILE"
ValibraExecutionProfile = Literal["research", "leaderboard"]


def valibra_execution_profile(
    environment: Mapping[str, str] | None = None,
) -> ValibraExecutionProfile:
    """Return the strict runtime profile; omitted means current research mode."""

    source = os.environ if environment is None else environment
    value = str(source.get(VALIBRA_EXECUTION_PROFILE_ENV, "research")).strip()
    if value not in {"research", "leaderboard"}:
        raise ValueError(
            f"{VALIBRA_EXECUTION_PROFILE_ENV} must be 'research' or 'leaderboard'"
        )
    return cast(ValibraExecutionProfile, value)


def is_leaderboard_profile() -> bool:
    """Whether emitted Official actions require unconditional pass-through."""

    return valibra_execution_profile() == "leaderboard"
