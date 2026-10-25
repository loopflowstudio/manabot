"""Read retained recipes as evidence, without granting execution admission.

Recorded stage parameters stay JSON, including unknown historical learning tags.
No recipe is translated, repaired or passed to an executor. Measured run fields
share RunRecord validation with live training; original exports remain unchanged.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def _count_retained_updates(cls, data: Any) -> Any:
        """Exports written before StageRecord.updates kept one row per update."""
        if isinstance(data, dict) and isinstance(data.get("stages"), list):
            data = dict(data)
            data["stages"] = [
                {"updates": len(stage["diagnostics"]), **stage}
                if isinstance(stage, dict)
                and "updates" not in stage
                and isinstance(stage.get("diagnostics"), list)
                else stage
                for stage in data["stages"]
            ]
        return data

    def updates_through(self, stage_id: str | None = None) -> int:
        """Count recorded update stages without admitting their learning rules."""
        learning = {
            stage.id
            for stage in self.regime.stages
            if stage.operation in {"train_self_play", "train_compound"}
        }
        return self._updates_through(learning, stage_id)


type ReportRun = TrainingRun | RetainedRun
