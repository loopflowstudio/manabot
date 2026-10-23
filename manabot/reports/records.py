"""Read retained recipes as evidence, without granting execution admission.

Recorded stage parameters stay JSON, including unknown historical learning tags.
No recipe is translated, repaired or passed to an executor. Measured run fields
share RunRecord validation with live training; original exports remain unchanged.
"""

from pydantic import BaseModel, ConfigDict, JsonValue

from manabot.infra.config import ModelConfig
from manabot.training.records import RunRecord, TrainingRun


class RecordedStage(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue]
    id: str
    operation: str


class RecordedRegime(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue]
    id: str
    world: str
    agent: ModelConfig
    stages: list[RecordedStage]


class RetainedRun(RunRecord[RecordedRegime]):
    """Reporting projection with opaque recipe parameters, never an executable run."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue]

    def updates_through(self, stage_id: str | None = None) -> int:
        """Count recorded update stages without admitting their learning rules."""
        learning = {
            stage.id
            for stage in self.regime.stages
            if stage.operation
            in {"train_self_play", "train_supervised", "train_compound"}
        }
        return self._updates_through(learning, stage_id)


type ReportRun = TrainingRun | RetainedRun
