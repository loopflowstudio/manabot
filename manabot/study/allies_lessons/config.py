"""
config.py
Every parameter of an Allies versus Lessons study, in one saved file

A run writes its resolved `study.json` beside its games and report. To re-run
with a change, load that file and override the fields that differ. Anything
that alters which games are played belongs here, not in a command flag alone.
"""

from __future__ import annotations

# Standard library
from pathlib import Path

# Third-party imports
from pydantic import BaseModel, ConfigDict, Field

# Local imports
from ..record import GameSpec
from .balance import edited, parse_edits
from .matchup import Lists, authored_lists, schedule


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field("search", description="player whose behavior is studied")
    baseline: str = Field("random", description="player shown beside it")
    mirror_deals: int = Field(60, description="deal seeds for subject against itself")
    versus_deals: int = Field(15, description="deal seeds for subject against baseline")
    baseline_deals: int = Field(
        200, description="deal seeds for baseline against itself"
    )
    seed: int = Field(
        91_000, description="first deal seed; pairings are offset from it"
    )
    ur_lessons: str = Field("", description="edits to UR Lessons, like '+2 Tiger-Seal'")
    gw_allies: str = Field("", description="edits to GW Allies")
    keep_decisions: bool = Field(True, description="False keeps outcomes only")
    max_commands: int = 10_000
    workers: int = 8

    @classmethod
    def load(cls, path: Path | str) -> "StudyConfig":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    def changed(self, **fields) -> "StudyConfig":
        """A validated copy with some fields replaced."""
        return StudyConfig.model_validate({**self.model_dump(), **fields})

    def lists(self) -> Lists:
        lists = edited(authored_lists(), "ur_lessons", parse_edits(self.ur_lessons))
        return edited(lists, "gw_allies", parse_edits(self.gw_allies))

    def games(self) -> list[GameSpec]:
        """The full schedule this configuration describes."""
        lists = self.lists() if self.ur_lessons or self.gw_allies else None
        pairings = [
            (self.subject, self.subject, self.mirror_deals, 0),
            (self.subject, self.baseline, self.versus_deals, 1_000),
            (self.baseline, self.baseline, self.baseline_deals, 2_000),
        ]
        seen = set()
        specs = []
        for first, second, deals, offset in pairings:
            key = tuple(sorted((first, second)))
            if deals <= 0 or key in seen:
                continue
            seen.add(key)
            specs += schedule(
                first,
                second,
                deals=deals,
                seed=self.seed + offset,
                max_commands=self.max_commands,
                lists=lists,
                keep_decisions=self.keep_decisions,
            )
        return specs
