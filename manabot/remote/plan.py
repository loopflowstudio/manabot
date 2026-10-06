"""Pure deployment planning. Rental declarations, never host CPU counts, own placement.

DeploymentPlan retains the input bytes and a fully resolved TrainingRegime. Source
identity is supplied by the CLI so compilation itself needs no provider or GPU.
"""

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.training.models import TrainingRegime, TrainSelfPlay


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class HardwareMix(Frozen):
    schema_version: Literal[1] = 1
    provider: Literal["runpod"] = "runpod"
    role: Literal["training"] = "training"
    gpu_count: Literal[1] = 1
    gpu_types: tuple[str, ...] = Field(min_length=1)
    workers: Literal[1] = 1
    thread_limit: int = Field(default=4, ge=1, le=4)
    vcpus: int = Field(ge=1)
    memory_gb: int = Field(ge=1)
    container_gb: int = Field(default=30, ge=10)
    volume_gb: int = Field(default=20, ge=1)
    image: str = Field(pattern=r"^[a-zA-Z0-9./_-]+@sha256:[a-f0-9]{64}$")
    hourly_ceiling: float = Field(gt=0)
    storage_hourly_allowance: float = Field(default=0.02, ge=0)
    dollar_cap: float = Field(gt=0, lt=5)
    wall_seconds: int = Field(ge=60)
    setup_seconds: int = Field(default=300, ge=1)
    transfer_seconds: int = Field(default=300, ge=1)
    cleanup_seconds: int = Field(default=120, ge=30)

    @property
    def projected_dollars(self) -> float:
        return (
            (self.hourly_ceiling + self.storage_hourly_allowance)
            * self.wall_seconds
            / 3600
        )

    @model_validator(mode="after")
    def budget(self) -> "HardwareMix":
        if self.projected_dollars > self.dollar_cap:
            raise ValueError("rental allowance exceeds dollar cap")
        if (
            self.setup_seconds + self.transfer_seconds + self.cleanup_seconds
            >= self.wall_seconds
        ):
            raise ValueError("reserves consume rental allowance")
        return self


class Source(Frozen):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DeploymentPlan(Frozen):
    schema_version: Literal[1] = 1
    input_json: str
    input_sha256: str
    regime: TrainingRegime
    mix: HardwareMix
    source: Source
    seed: int = Field(ge=0)
    projected_dollars: float

    @model_validator(mode="after")
    def consistent(self) -> "DeploymentPlan":
        if digest(self.input_json.encode()) != self.input_sha256:
            raise ValueError("input regime digest differs")
        expected = resolve(self.input_json, self.mix)
        if (
            expected != self.regime
            or self.projected_dollars != self.mix.projected_dollars
        ):
            raise ValueError("resolved deployment differs from input and mix")
        return self


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve(input_json: str, mix: HardwareMix) -> TrainingRegime:
    regime = TrainingRegime.model_validate_json(input_json)
    if regime.recovery_max_microsteps is not None:
        raise ValueError("remote recovery is unsupported")
    if regime.agent.compound_decisions or regime.agent.belief_count_buckets:
        raise ValueError("remote deployment requires ordinary self-play")
    for stage in regime.stages:
        if not isinstance(stage, TrainSelfPlay) or stage.opponent is not None:
            raise ValueError("remote deployment requires self-contained self-play")
        if stage.execution.memory_bytes > mix.memory_gb * 1024**3:
            raise ValueError("declared rental memory is below the stage requirement")
        stage.execution.device = "cuda"
        stage.execution.threads = min(
            stage.execution.threads, mix.thread_limit, mix.vcpus, 4
        )
    reserves = mix.setup_seconds + mix.transfer_seconds + mix.cleanup_seconds
    if regime.wall_seconds + reserves > mix.wall_seconds:
        raise ValueError("run watchdog and reserves exceed rental allowance")
    if (
        sum(stage.execution.wall_seconds for stage in regime.stages)
        > regime.wall_seconds
    ):
        raise ValueError("stage watchdogs exceed run allowance")
    return TrainingRegime.model_validate(regime.model_dump())


def compile_plan(
    input_json: str, mix: HardwareMix, source: Source, seed: int
) -> DeploymentPlan:
    return DeploymentPlan(
        input_json=input_json,
        input_sha256=digest(input_json.encode()),
        regime=resolve(input_json, mix),
        mix=mix,
        source=source,
        seed=seed,
        projected_dollars=mix.projected_dollars,
    )
