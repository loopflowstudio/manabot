"""
match.py
Defines configuration for a Magic: The Gathering match between two players.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Dict

# Local imports
from manabot.infra.hypers import MatchHypers, RewardHypers
import managym


@dataclass
class Match:
    """Python-side player configuration that we pass into manabot Env.

    This represents a match between two players (hero and villain) with their
    respective decks. The default deck is a simple two-color aggressive deck.
    """

    hero: str
    villain: str
    hero_deck: Dict[str, int]
    villain_deck: Dict[str, int]
    hypers: MatchHypers

    def __init__(self, hypers: MatchHypers = MatchHypers()):
        self.hypers = hypers
        self.hero = hypers.hero
        self.villain = hypers.villain
        self.hero_deck = deepcopy(hypers.hero_deck)
        self.villain_deck = deepcopy(hypers.villain_deck)

    def to_rust_hero(self) -> "managym.PlayerConfig":
        return managym.PlayerConfig(self.hero, self.hero_deck)

    def to_rust_villain(self) -> "managym.PlayerConfig":
        return managym.PlayerConfig(self.villain, self.villain_deck)

    def to_rust(self) -> "list[managym.PlayerConfig]":
        return [self.to_rust_hero(), self.to_rust_villain()]

    def __str__(self) -> str:
        """Return a human-readable string representation of the match."""
        return f"Match({self.hero} vs {self.villain})"


def parse_deck(deck_str: str) -> Dict[str, int]:
    """Parse a deck string into a dictionary of card counts.

    Accepts either:
    1. JSON string: '{"Mountain": 12, "Forest": 12}'
    2. Simple format: 'Mountain:12,Forest:12'
    """
    try:
        # First try JSON format
        return json.loads(deck_str)
    except json.JSONDecodeError:
        # Fall back to simple format
        deck = {}
        for pair in deck_str.split(","):
            if ":" not in pair:
                raise ValueError(f"Invalid deck format: {deck_str}")
            card, count = pair.split(":")
            deck[card.strip()] = int(count)
        return deck


class Reward:
    """
    Reward policy for an environment.

    Hypers:
        managym: If true, directly use the reward from the managym engine.
        trivial: If True, return 1.0 for all rewards.
        win_reward: Reward for winning the game.
        lose_reward: Reward for losing the game.
        land_play_reward: Reward when a new friendly land enters the battlefield.
        creature_play_reward: Reward when a new friendly creature enters the battlefield.
        opponent_life_loss_reward: Reward when opponent life drops across a step.
    """

    def __init__(self, hypers: RewardHypers):
        self.hypers = hypers

    def compute(
        self,
        raw_reward: float,
        _last_obs: managym.Observation,
        new_obs: managym.Observation,
    ) -> float:
        if self.hypers.managym:
            return raw_reward

        if self.hypers.trivial:
            return 1.0

        reward = float(raw_reward)
        reward += self._progress_shaping(_last_obs, new_obs)

        if not new_obs.game_over:
            return reward

        return reward + (
            self.hypers.win_reward if new_obs.won else self.hypers.lose_reward
        )

    def _progress_shaping(
        self,
        last_obs: managym.Observation,
        new_obs: managym.Observation,
    ) -> float:
        reward = 0.0

        if self.hypers.land_play_reward != 0.0:
            reward += self.hypers.land_play_reward * max(
                0,
                self._count_battlefield_lands(new_obs.agent_cards)
                - self._count_battlefield_lands(last_obs.agent_cards),
            )

        if self.hypers.creature_play_reward != 0.0:
            reward += self.hypers.creature_play_reward * max(
                0,
                self._count_battlefield_creatures(new_obs.agent_cards)
                - self._count_battlefield_creatures(last_obs.agent_cards),
            )

        if (
            self.hypers.opponent_life_loss_reward != 0.0
        ):
            reward += self.hypers.opponent_life_loss_reward * max(
                0,
                last_obs.opponent.life - new_obs.opponent.life,
            )

        return reward

    @staticmethod
    def _count_battlefield_lands(cards: list[managym.Card]) -> int:
        return sum(
            1
            for card in cards
            if int(card.zone) == int(managym.ZoneEnum.BATTLEFIELD)
            and bool(card.card_types.is_land)
        )

    @staticmethod
    def _count_battlefield_creatures(cards: list[managym.Card]) -> int:
        return sum(
            1
            for card in cards
            if int(card.zone) == int(managym.ZoneEnum.BATTLEFIELD)
            and bool(card.card_types.is_creature)
        )
