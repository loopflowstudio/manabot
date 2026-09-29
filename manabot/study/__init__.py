"""
__init__.py
Behavioral study of recorded games: record, measure, report

Matchup-independent pieces live here. A matchup gets a subpackage (see
`allies_lessons`) holding its setup, schedule, extra measures and command.
"""

from .measures import (
    Rate,
    attacks,
    blocks,
    casting,
    choices,
    contested_attacks,
    game_lengths,
    held_back,
    land_drops,
    land_drops_by_lands,
    open_attacks,
    outcomes,
    win_rates,
)
from .record import (
    Decision,
    GameRecord,
    GameSpec,
    Permanent,
    read_games,
    record_game,
    record_games,
    write_games,
)

__all__ = [
    "Decision",
    "GameRecord",
    "GameSpec",
    "Permanent",
    "Rate",
    "attacks",
    "blocks",
    "casting",
    "choices",
    "contested_attacks",
    "game_lengths",
    "held_back",
    "land_drops",
    "land_drops_by_lands",
    "open_attacks",
    "outcomes",
    "read_games",
    "record_game",
    "record_games",
    "win_rates",
    "write_games",
]
