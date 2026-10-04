"""
__init__.py
Behavioral study of the Allies versus Lessons matchup

Holds what is specific to this matchup: its setup and seatings, the players
worth studying, Learn choices, and the report command. General questions belong
in `manabot.study.measures`.
"""

from .matchup import DECKS, PLAYERS, learn_choices, schedule
from .report import build_report

__all__ = ["DECKS", "PLAYERS", "build_report", "learn_choices", "schedule"]
