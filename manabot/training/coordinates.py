"""Training progress coordinates shared by checkpoint and evaluation records."""

from pydantic import Field

from manabot.infra.schema import Strict


class TrainingCoordinates(Strict):
    # Segment-local completed iterations; updates includes inherited absolute offsets.
    segment_updates: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    optimizer_updates: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    stage_id: str
    updates: int = Field(ge=0)
    training_seconds: float = Field(ge=0)
    active_training_seconds: float | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    environment_decisions: int = Field(default=0, ge=0)
    learner_transitions: int = Field(default=0, ge=0)
    optimizer_exposures: int = Field(default=0, ge=0)
    games: int = Field(default=0, ge=0)
