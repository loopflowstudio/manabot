"""Deprecated alias; import native policies from managym.bots instead."""

import warnings

from managym.bots import (
    DEFAULT_ROLLOUTS_PER_WORLD,
    DEFAULT_SEARCH_SIMS,
    FlatSearchPolicy,
)

__all__ = ["DEFAULT_SEARCH_SIMS", "DEFAULT_ROLLOUTS_PER_WORLD", "FlatSearchPolicy"]

warnings.warn(
    "manabot.sim.search.flat is deprecated; use managym.bots. Removed in 0.4.0.",
    DeprecationWarning,
    stacklevel=2,
)
