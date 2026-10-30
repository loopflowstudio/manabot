"""A freshly dealt interactive-deck game for belief tests that need a live engine."""

from __future__ import annotations

from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.verify.games import INTERACTIVE_DECK


def fresh_env(seed: int) -> Env:
    """Return an `Env` reset to `seed` with both seats on the interactive deck."""
    match = Match(
        MatchConfig(
            hero="teacher-a",
            villain="teacher-b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(
        match,
        ObservationSpace(),
        Reward(RewardConfig()),
        seed=seed,
        auto_reset=False,
    )
    env.reset(seed=seed)
    return env
