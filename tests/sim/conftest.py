"""Shared setup for checkpoints trained on the interactive mirror matchup."""

import pytest

from manabot.env import Match
from manabot.infra.config import MatchConfig
from manabot.verify.util import INTERACTIVE_DECK


@pytest.fixture
def interactive_player_configs():
    return Match(
        MatchConfig(
            hero_deck=dict(INTERACTIVE_DECK), villain_deck=dict(INTERACTIVE_DECK)
        )
    ).to_rust()
