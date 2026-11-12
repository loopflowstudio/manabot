"""Public completed-game results, joinable to the pools and builds that played.

One row per completed native game. Seats are native game seats, so seat 0 was
on the play. A seat names its deck's pool and build: `build` is None for the
human build, and `pool` is None for a deck with no pool. Rows carry no replay
inputs; failed games belong to the private game-failure records. This module
imports nothing from the engine or from training.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field

from manabot.infra.identity import StrictModel


class GameSeat(StrictModel):
    pool: str | None = Field(min_length=1)
    build: str | None = Field(min_length=1)


class CompletedGame(StrictModel):
    schema_version: Literal[1] = 1
    stream: int = Field(ge=0)
    game: int = Field(ge=1)
    seats: tuple[GameSeat, GameSeat]
    winner: Literal[0, 1] | None


def read_game_results(path: Path) -> list[CompletedGame]:
    """Load public outcomes; malformed rows are errors, never missing games."""
    with path.open() as stream:
        return [CompletedGame.model_validate_json(line) for line in stream]
