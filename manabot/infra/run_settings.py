"""Self-play instrumentation shared by recipe configs and game capture.

Recipe configs serialize these options as flat keys. Etude compares the config
on resume, while the learner digest excludes instrumentation.
"""

from pydantic import BaseModel, ConfigDict, Field

from managym.diagnostics import Level


class Instrumentation(BaseModel):
    record_games: float = Field(default=0.0, ge=0, le=1)
    diagnostics: Level = "standard"
    stall_seconds: float = Field(default=600, gt=0)
    statistics: bool = False

    @property
    def run_settings(self) -> "RunSettings":
        return RunSettings.model_validate(
            self.model_dump(include=set(RunSettings.model_fields))
        )


class RunSettings(Instrumentation):
    model_config = ConfigDict(extra="forbid", frozen=True)
