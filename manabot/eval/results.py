"""Shared replay-checked arena rows and block-bootstrap mean intervals."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from manabot.eval.models import EvaluationKey


class EvaluationRow(BaseModel):
    """Narrow the arena JSON boundary while retaining every additional field."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    arena_key: EvaluationKey
    deal_seed: int
    leg: int = Field(ge=0, le=7)
    player_a: str
    player_b: str
    player_a_registration_sha256: str
    player_b_registration_sha256: str
    score_a: float | None = Field(strict=True)
    failure: str | None
    terminated: bool
    truncated: bool
    replay_passed: bool
    trace_path: str
    game_seconds: float = Field(ge=0)
    integrity: dict[str, int]

    @field_validator("score_a")
    @classmethod
    def valid_score(cls, value: float | None) -> float | None:
        if value not in (None, 0.0, 0.5, 1.0):
            raise ValueError("arena score must be loss, draw, win or unavailable")
        return value

    @property
    def valid(self) -> bool:
        return (
            self.failure is None
            and self.terminated
            and not self.truncated
            and self.replay_passed
            and self.score_a is not None
            and not any(self.integrity.values())
        )


class RateInterval(BaseModel):
    """A mean rate or rate difference with its uncertainty bounds."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mean: float
    lower: float
    upper: float


def bootstrap_mean_interval(
    block_means: Sequence[float] | npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
) -> RateInterval:
    """Estimate a mean's 95% interval from caller-defined independent blocks.

    Callers aggregate correlated legs and supply the resample indices. Reusing
    those indices across checkpoints preserves paired comparisons.
    """
    blocks = np.asarray(block_means, dtype=np.float64)
    lower, upper = np.quantile(blocks[indices].mean(axis=1), [0.025, 0.975])
    return RateInterval(
        mean=float(blocks.mean()), lower=float(lower), upper=float(upper)
    )
