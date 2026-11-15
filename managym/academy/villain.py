"""Build a native or optional training opponent from the game's configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from managym.bots import (
    DEFAULT_SEARCH_SIMS,
    BotPolicy,
    FlatSearchPolicy,
    RandomPolicy,
    passive_policy,
)

from .opponent import policy_registry

if TYPE_CHECKING:
    from .trace import GameConfig

NATIVE_WATCH_PLAYERS = ("random", "search")
MAX_SEARCH_SIMS = 4096


def build_villain_policy(config: "GameConfig") -> BotPolicy:
    """Build a bot policy from a parsed GameConfig."""
    seed = config.seed if config.seed is not None else 0
    villain_type = config.villain_type
    search = config.search[1] if config.search else None
    if search is not None and villain_type != "policy":
        raise ValueError("Mirror search requires a registered search-capable policy.")

    if villain_type == "passive":
        return passive_policy
    if villain_type == "random":
        return RandomPolicy(seed=seed)
    if villain_type == "search":
        sims = config.villain_sims or DEFAULT_SEARCH_SIMS
        return FlatSearchPolicy(sims=sims, seed=seed)
    if villain_type == "policy":
        if config.opponent is None:
            raise ValueError("Policy opponent requires a registered identity.")
        return policy_registry().load(
            config.opponent.get("base_sha256", config.opponent["sha256"]),
            config.to_rust(),
            seat=1,
            seed=seed,
            search=search,
        )
    raise ValueError(f"Unsupported villain_type: {villain_type}")
