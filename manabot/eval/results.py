"""Shared replay-checked arena rows and paired-deal uncertainty records."""

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
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mean: float
    lower: float
    upper: float
