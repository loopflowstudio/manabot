"""Self-play instrumentation: game recording, statistics, diagnostics and stalls.

`RunSettings` is the value SelfPlay and game capture receive. Recipe configs
subclass it, so the same options are flat keys of every self-play recipe
config; `run_settings` takes the plain value back out. Etude compares the whole
config on resume, while the learner digest excludes instrumentation.
"""

from pydantic import BaseModel, ConfigDict, Field

from managym.diagnostics import Level


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_games: float = Field(default=0.0, ge=0, le=1)
    diagnostics: Level = "standard"
    stall_seconds: float = Field(default=600, gt=0)
    statistics: bool = False

    @property
    def run_settings(self) -> "RunSettings":
        """These options alone, so a recipe config's other fields stay out."""
        return RunSettings(**self.model_dump(include=set(RunSettings.model_fields)))
