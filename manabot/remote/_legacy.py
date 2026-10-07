"""Private schema-1 persistence readers. Never author new work with these shapes.

Retained plans serialize through their original schema so job IDs, claims and
source receipts remain valid. The runtime consumes the normalized JobSpec.
"""

from typing import Literal

from pydantic import Field, model_validator

from manabot.training.models import TrainingRegime, TrainSelfPlay

from .plan import DeploymentPlan, Frozen, JobSpec, Machine, Source, digest


class _LegacyMix(Machine):
    """Frozen v1 deployment input; read only at persistence boundaries."""

    dollar_cap: float = Field(gt=0)
    wall_seconds: int | float = Field(ge=60)
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
    def budget(self) -> "_LegacyMix":
        if self.projected_dollars > self.dollar_cap:
            raise ValueError("rental allowance exceeds dollar cap")
        if (
            self.setup_seconds + self.transfer_seconds + self.cleanup_seconds
            >= self.wall_seconds
        ):
            raise ValueError("reserves consume rental allowance")
        return self


class _LegacyPlan(Frozen):
    schema_version: Literal[1] = 1
    input_json: str
    input_sha256: str
    regime: TrainingRegime
    mix: _LegacyMix
    source: Source
    seed: int = Field(ge=0)
    projected_dollars: float

    @model_validator(mode="after")
    def consistent(self) -> "_LegacyPlan":
        if digest(self.input_json.encode()) != self.input_sha256:
            raise ValueError("input regime digest differs")
        if self.schema_version == 1 and self.mix.dollar_cap >= 5:
            raise ValueError("frozen v1 deployment requires a dollar cap below five")
        if (
            _resolve(self.input_json, self.mix) != self.regime
            or self.projected_dollars != self.mix.projected_dollars
        ):
            raise ValueError("resolved deployment differs from input and mix")
        return self

    def specification(self) -> JobSpec:
        # v1 reserves had no checkpoint split. This projection is for hardware
        # access only; v1 execution keeps its original watchdog/deadline rules.
        return JobSpec.model_construct(
            machine=Machine.model_validate(
                {name: getattr(self.mix, name) for name in Machine.model_fields}
            ),
            lifetime_hours=self.mix.wall_seconds / 3600,
            spending_limit=self.mix.dollar_cap,
            setup_seconds=self.mix.setup_seconds,
            checkpoint_seconds=0,
            upload_seconds=self.mix.transfer_seconds,
            cleanup_seconds=self.mix.cleanup_seconds,
        )


def read_plan(value: object) -> DeploymentPlan:
    legacy = _LegacyPlan.model_validate(value)
    return DeploymentPlan.model_construct(
        schema_version=legacy.schema_version,
        spec=legacy.specification(),
        input_json=legacy.input_json,
        input_sha256=legacy.input_sha256,
        regime=legacy.regime.model_copy(deep=True),
        source=legacy.source,
        seed=legacy.seed,
        projected_dollars=legacy.projected_dollars,
        _legacy=legacy,
    )


def _resolve(input_json: str, mix: _LegacyMix) -> TrainingRegime:
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
