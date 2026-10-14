"""Retained debugging attempts shared by the ETU-118 executor and offline reader.

A completed cohort records execution, not sustained-baseline acceptance. Missing
rows remain missing; neither parsing nor reporting imputes failed games.
"""

from typing import Literal

from pydantic import Field

from manabot.training.models import Strict


class Score(Strict):
    deal: int
    win: float = Field(ge=0, le=1)
    replay: bool
    seat: int | None = None
    leg: int | None = None
    action: int | None = None
    winner: int | None = None
    probabilities: list[float] | None = None
    root: str | None = None
    terminal: str | None = None


class Attempt(Strict):
    seed: int
    frozen: bool
    status: Literal["running", "completed", "failed"] = "running"
    initial: list[Score] = Field(default_factory=list)
    rows: list[Score] = Field(default_factory=list)
    seconds: float = Field(default=0, ge=0)
    error: str | None = None
    run_id: str | None = None
    checkpoint_sha256: str | None = None
    initial_sha256: str | None = None


class Result(Strict):
    status: Literal["running", "completed", "failed"] = "running"
    attempts: list[Attempt] = Field(default_factory=list)
    seconds: float = Field(default=0, ge=0)
    prior_seconds: float = Field(default=0, ge=0)
    prior_sha256: str | None = None
    full_game: bool = False
    error: str | None = None
