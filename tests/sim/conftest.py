"""Shared setup for checkpoints trained on the interactive mirror matchup."""

import pytest

from manabot.env import Match
<<<<<<< HEAD
from manabot.infra.hypers import MatchHypers
=======
from manabot.infra.config import MatchConfig
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from manabot.verify.util import INTERACTIVE_DECK


@pytest.fixture
def interactive_player_configs():
    return Match(
<<<<<<< HEAD
        MatchHypers(
=======
        MatchConfig(
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
            hero_deck=dict(INTERACTIVE_DECK), villain_deck=dict(INTERACTIVE_DECK)
        )
    ).to_rust()
