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

from .extensions import bot_extension

if TYPE_CHECKING:
    from .trace import GameConfig

NATIVE_WATCH_PLAYERS = ("random", "search")
MAX_SEARCH_SIMS = 4096


def build_villain_policy(config: "GameConfig") -> BotPolicy:
    """Build a bot policy from a parsed GameConfig."""
    seed = config.seed if config.seed is not None else 0
    villain_type = config.villain_type

    if villain_type == "passive":
        return passive_policy
    if villain_type == "random":
        return RandomPolicy(seed=seed)
    if villain_type == "search":
        sims = config.villain_sims or DEFAULT_SEARCH_SIMS
        return FlatSearchPolicy(sims=sims, seed=seed)
    if villain_type == "checkpoint":
        if not config.villain_checkpoint:
            raise ValueError("checkpoint villain requires villain_checkpoint path.")
        extension = bot_extension()
        if extension is None:
            raise ValueError("Trained opponents need manabot installed on this server.")
        return extension.checkpoint_policy(
            config.villain_checkpoint,
            deterministic=config.villain_deterministic,
            player_configs=config.to_rust(),
        )
    raise ValueError(f"Unsupported villain_type: {villain_type}")
