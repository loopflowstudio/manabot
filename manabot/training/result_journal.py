"""Durable journal of a stage's completed games, sealed as a public artifact.

Rows (`manabot.game_results`) are fsynced as games finish, before the collector
resets, so they survive a later optimizer failure. Sealing publishes immutable
digest-named rows beside the run's artifacts.
"""

import os
from pathlib import Path

from manabot.game_results import CompletedGame
from manabot.infra.identity import file_sha256


class GameResultWriter:
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
