"""
match.py
Defines configuration for a Magic: The Gathering match between two players.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Dict

# Local imports
<<<<<<< HEAD
from manabot.infra.hypers import MatchHypers, RewardHypers
=======
from manabot.infra.config import MatchConfig, RewardConfig
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
    hero_sideboard: Dict[str, int]
    villain_sideboard: Dict[str, int]
<<<<<<< HEAD
    hypers: MatchHypers

    def __init__(self, hypers: MatchHypers = MatchHypers()):
        self.hypers = hypers
        self.hero = hypers.hero
        self.villain = hypers.villain
        self.hero_deck = deepcopy(hypers.hero_deck)
        self.villain_deck = deepcopy(hypers.villain_deck)
        self.hero_sideboard = deepcopy(hypers.hero_sideboard)
        self.villain_sideboard = deepcopy(hypers.villain_sideboard)

    def to_rust_hero(self) -> "managym.PlayerConfig":
        return managym.PlayerConfig(
            self.hero, self.hero_deck, self.hero_sideboard, self.hypers.content_pack
=======
    config: MatchConfig

    def __init__(self, config: MatchConfig = MatchConfig()):
        self.config = config
        self.hero = config.hero
        self.villain = config.villain
        self.hero_deck = deepcopy(config.hero_deck)
        self.villain_deck = deepcopy(config.villain_deck)
        self.hero_sideboard = deepcopy(config.hero_sideboard)
        self.villain_sideboard = deepcopy(config.villain_sideboard)

    def to_rust_hero(self) -> "managym.PlayerConfig":
        return managym.PlayerConfig(
            self.hero, self.hero_deck, self.hero_sideboard, self.config.content_pack
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        )

    def to_rust_villain(self) -> "managym.PlayerConfig":
        return managym.PlayerConfig(
            self.villain,
            self.villain_deck,
            self.villain_sideboard,
<<<<<<< HEAD
            self.hypers.content_pack,
=======
            self.config.content_pack,
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        )

    def to_rust(self) -> "list[managym.PlayerConfig]":
        return [self.to_rust_hero(), self.to_rust_villain()]

    def swapped(self) -> "Match":
        """Return a copy with hero and villain seats exchanged.

        The managym engine always gives player 0 the first turn
        (managym/src/flow/setup.rs: TurnState::new(PlayerId(0))), so passing
        a swapped Match to Env.reset puts the hero on the draw. Used for
        seat-balanced evaluation.
        """
        return Match(
<<<<<<< HEAD
            MatchHypers(
                content_pack=self.hypers.content_pack,
=======
            MatchConfig(
                content_pack=self.config.content_pack,
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
                hero=self.villain,
                villain=self.hero,
                hero_deck=self.villain_deck,
                villain_deck=self.hero_deck,
                hero_sideboard=self.villain_sideboard,
                villain_sideboard=self.hero_sideboard,
            )
        )

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

<<<<<<< HEAD
    Hypers:
=======
    Configuration:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        managym: If true, directly use the reward from the managym engine.
        trivial: If True, return 1.0 for all rewards.
        win_reward: Reward for winning the game.
        lose_reward: Reward for losing the game.
        land_play_reward: Reward when a new friendly land enters the battlefield.
        creature_play_reward: Reward when a new friendly creature enters the battlefield.
        opponent_life_loss_reward: Reward when opponent life drops across a step.
    """
<<<<<<< HEAD

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
=======

    def __init__(self, config: RewardConfig):
        self.config = config
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

    def compute(
        self,
        raw_reward: float,
        _last_obs: managym.Observation,
        new_obs: managym.Observation,
    ) -> float:
        if self.config.managym:
            return raw_reward

        if self.config.trivial:
            return 1.0

        reward = float(raw_reward)
        reward += self._progress_shaping(_last_obs, new_obs)
        reward += self._potential_shaping(_last_obs, new_obs)

        if not new_obs.game_over:
            return reward

        return reward + (
<<<<<<< HEAD
            self.hypers.win_reward if new_obs.won else self.hypers.lose_reward
=======
            self.config.win_reward if new_obs.won else self.config.lose_reward
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        )

    def _progress_shaping(
        self,
        last_obs: managym.Observation,
        new_obs: managym.Observation,
    ) -> float:
        reward = 0.0

<<<<<<< HEAD
        if self.hypers.land_play_reward != 0.0:
            reward += self.hypers.land_play_reward * max(
=======
        if self.config.land_play_reward != 0.0:
            reward += self.config.land_play_reward * max(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
                0,
                self._count_battlefield_lands(new_obs.agent_cards)
                - self._count_battlefield_lands(last_obs.agent_cards),
            )

<<<<<<< HEAD
        if self.hypers.creature_play_reward != 0.0:
            reward += self.hypers.creature_play_reward * max(
=======
        if self.config.creature_play_reward != 0.0:
            reward += self.config.creature_play_reward * max(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
                0,
                self._count_battlefield_creatures(new_obs.agent_cards)
                - self._count_battlefield_creatures(last_obs.agent_cards),
            )

<<<<<<< HEAD
        if self.hypers.opponent_life_loss_reward != 0.0:
            reward += self.hypers.opponent_life_loss_reward * max(
=======
        if self.config.opponent_life_loss_reward != 0.0:
            reward += self.config.opponent_life_loss_reward * max(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
                0,
                last_obs.opponent.life - new_obs.opponent.life,
            )

        return reward

    def _potential_shaping(
        self,
        last_obs: managym.Observation,
        new_obs: managym.Observation,
    ) -> float:
        """Potential-based shaping term gamma * Phi(s') - Phi(s).

        Phi(terminal) is treated as 0 so the shaping telescopes out over an
        episode and preserves policy invariance (Ng, Harada & Russell 1999).
        """
<<<<<<< HEAD
        if not self.hypers.potential_enabled:
=======
        if not self.config.potential_enabled:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            return 0.0

        prev_phi = self._potential(last_obs)
        next_phi = 0.0 if new_obs.game_over else self._potential(new_obs)
<<<<<<< HEAD
        return self.hypers.potential_gamma * next_phi - prev_phi
=======
        return self.config.potential_gamma * next_phi - prev_phi
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

    def _potential(self, obs: managym.Observation) -> float:
        """Hero-perspective board-state potential Phi(s)."""

        phi = 0.0
<<<<<<< HEAD
        if self.hypers.potential_land_weight != 0.0:
            phi += self.hypers.potential_land_weight * (
                self._count_battlefield_lands(obs.agent_cards)
                - self._count_battlefield_lands(obs.opponent_cards)
            )
        if self.hypers.potential_creature_weight != 0.0:
            phi += self.hypers.potential_creature_weight * (
                self._count_battlefield_creatures(obs.agent_cards)
                - self._count_battlefield_creatures(obs.opponent_cards)
            )
        if self.hypers.potential_life_weight != 0.0:
            phi += (
                self.hypers.potential_life_weight
=======
        if self.config.potential_land_weight != 0.0:
            phi += self.config.potential_land_weight * (
                self._count_battlefield_lands(obs.agent_cards)
                - self._count_battlefield_lands(obs.opponent_cards)
            )
        if self.config.potential_creature_weight != 0.0:
            phi += self.config.potential_creature_weight * (
                self._count_battlefield_creatures(obs.agent_cards)
                - self._count_battlefield_creatures(obs.opponent_cards)
            )
        if self.config.potential_life_weight != 0.0:
            phi += (
                self.config.potential_life_weight
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
                * (float(obs.agent.life) - float(obs.opponent.life))
                / 20.0
            )
        return phi

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
