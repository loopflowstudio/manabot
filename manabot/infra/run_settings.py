"""Runtime instrumentation, separate from learning identity and resume admission.

RunSettings travels unchanged from an entry point to captures. TrainingJob keeps
its flat serialized fields; deprecated switches resolve only at that boundary or
the CLI, before execution receives this value.
"""

from pydantic import BaseModel, ConfigDict, Field

from managym.diagnostics import Level


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_games: float = Field(default=0.0, ge=0, le=1)
    diagnostics: Level = "standard"
    stall_seconds: float = Field(default=600, gt=0)
    statistics: bool = False
