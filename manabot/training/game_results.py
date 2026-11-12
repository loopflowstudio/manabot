"""Public completed-game attribution, independent of private crash recovery.

Seats are native game seats: seat zero is on the play. Population entry IDs
retain event identity even for identical decks. Journals are durable while a
stage runs; sealing publishes immutable digest-named rows beside its artifacts.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from manabot.infra.identity import StrictModel, file_sha256


class GameSeat(StrictModel):
    entry_id: str = Field(min_length=1)
    build_episode: str | None = None


class CompletedGame(StrictModel):
    schema_version: Literal[1] = 1
    stream: int = Field(ge=0)
    game: int = Field(ge=1)
    seats: tuple[GameSeat, GameSeat]
    on_play: Literal[0, 1] = 0
    winner: Literal[0, 1] | None


class GameResultWriter:
    """Retain outcomes before the collector resets, including failed updates."""

    def __init__(self, directory: Path, stage: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{stage}-game-results.jsonl.partial"
        self.path.touch(exist_ok=False)

    def record(self, result: CompletedGame) -> None:
        with self.path.open("a") as stream:
            stream.write(result.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def seal(self) -> Path:
        # A journal can span an entire stage; hash it without retaining its rows.
        digest = file_sha256(self.path)
        target = self.path.parent / f"{digest}.jsonl"
        if target.exists() and file_sha256(target) != digest:
            raise ValueError("game result artifact digest mismatch")
        self.path.replace(target)
        return target


def read_game_results(path: Path) -> list[CompletedGame]:
    """Load public outcomes; malformed rows are errors, never missing games."""
    with path.open() as stream:
        return [CompletedGame.model_validate_json(line) for line in stream]
