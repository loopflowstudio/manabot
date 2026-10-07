"""Pure deployment planning. Rental declarations, never host CPU counts, own placement.

DeploymentPlan retains the input bytes and a fully resolved TrainingRegime. Source
identity is supplied by the CLI so compilation itself needs no provider or GPU.
"""

import hashlib
import math
from typing import TYPE_CHECKING, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from manabot.infra.artifacts import split_s3_uri
from manabot.training.models import TrainingRegime, TrainSelfPlay

if TYPE_CHECKING:
    from ._legacy import _LegacyPlan

MAX_JOB_SECONDS = 12 * 3600


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Machine(Frozen):
    """Provider shape and price ceilings, without an authority duration."""

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


class AccessScope(Frozen):
    """Artifact namespace only. Issuer and provider credentials stay on the launcher."""

    destination: str = "s3://etudefantasia/manabot/jobs"

    @model_validator(mode="after")
    def literal_prefix(self) -> "AccessScope":
        _, prefix = split_s3_uri(self.destination)
        if not prefix or any(c in prefix for c in "*?[]") or ".." in prefix.split("/"):
            raise ValueError("access scope requires a literal private S3 prefix")
        return self


class JobSpec(Frozen):
    """One machine allocation; hours include setup, checkpoint, upload and cleanup.

    Construction is planning, not execution admission. Long allocations cannot
    execute on the current RunPod adapter without renewal and portable recovery.
    No throughput estimate or learning target belongs here.
    """

    machine: Machine
    lifetime_hours: float = Field(gt=0)
    spending_limit: float = Field(gt=0)
    access: AccessScope = AccessScope()
    setup_seconds: int = Field(default=300, ge=1)
    checkpoint_seconds: int = Field(default=120, ge=1)
    upload_seconds: int = Field(default=300, ge=1)
    cleanup_seconds: int = Field(default=120, ge=30)

    @property
    def lifetime_seconds(self) -> float:
        return self.lifetime_hours * 3600

    @property
    def projected_dollars(self) -> float:
        return (
            (self.machine.hourly_ceiling + self.machine.storage_hourly_allowance)
            * self.lifetime_seconds
            / 3600
        )

    @model_validator(mode="after")
    def budget(self) -> "JobSpec":
        if self.projected_dollars > self.spending_limit:
            raise ValueError("rental allowance exceeds spending limit")
        if (
            self.setup_seconds
            + self.checkpoint_seconds
            + self.upload_seconds
            + self.cleanup_seconds
            >= self.lifetime_seconds
        ):
            raise ValueError("reserves consume rental allowance")
        return self

    def admit(self, now: float) -> "Allocation":
        if self.lifetime_seconds > MAX_JOB_SECONDS - 60:
            raise ValueError(
                "RunPod allocation unavailable: renewable worker credentials and "
                "durable complete-state CUDA recovery across replacement workers "
                "are not implemented (single STS session limit: twelve hours)"
            )
        return Allocation(spec=self, admitted_at=now)


class Allocation(Frozen):
    """Absolute authority receipt. Every cutoff is derived from one admission."""

    spec: JobSpec
    admitted_at: float = Field(ge=0)

    @property
    def deadline(self) -> float:
        # The guardian uses Unix seconds; round down once for both enforcement owners.
        return math.floor(self.admitted_at + self.spec.lifetime_seconds)

    @property
    def pause_at(self) -> float:
        return (
            self.deadline
            - self.spec.checkpoint_seconds
            - self.spec.upload_seconds
            - self.spec.cleanup_seconds
        )

    @property
    def checkpoint_deadline(self) -> float:
        return self.deadline - self.spec.upload_seconds - self.spec.cleanup_seconds

    def renewal_cutoff(self, now: float) -> float:
        if now >= self.deadline:
            raise ValueError("allocation has expired; renewal forbidden")
        # This bound is useful to adapters but does not implement a renewer.
        return self.deadline

    def extend(self, lifetime_hours: float) -> "Allocation":
        raise ValueError(
            "RunPod extension unavailable: persisted intent, guardian and issued "
            "session policies cannot be updated atomically; original deadline remains"
        )


class Source(Frozen):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DeploymentPlan(Frozen):
    """Resolved learning and one machine request; legacy bytes stay at the boundary."""

    schema_version: Literal[1, 2] = 2
    spec: JobSpec
    input_json: str
    input_sha256: str
    regime: TrainingRegime
    source: Source
    seed: int = Field(ge=0)
    projected_dollars: float
    _legacy: "_LegacyPlan | None" = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def read_saved(
        cls, value: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        if isinstance(value, dict) and value.get("schema_version", 2) == 1:
            from ._legacy import read_plan

            # The private reader validates the old shape before normalizing it.
            return cast(Self, read_plan(value))
        return handler(value)

    @model_serializer(mode="wrap")
    def write_saved(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, object]:
        if self._legacy is not None:
            # Never silently serialize an edited projection as frozen evidence.
            original = self._legacy
            if (
                self.spec != original.specification()
                or self.regime != original.regime
                or self.source != original.source
                or self.seed != original.seed
                or self.input_json != original.input_json
                or self.input_sha256 != original.input_sha256
                or self.projected_dollars != original.projected_dollars
                or self.schema_version != original.schema_version
            ):
                raise ValueError(
                    "historical deployment is immutable; compile a new plan"
                )
            return original.model_dump(
                mode=info.mode, exclude=info.exclude, include=info.include
            )
        return handler(self)

    @model_validator(mode="after")
    def consistent(self) -> "DeploymentPlan":
        if self._legacy is not None:
            return self
        if self.schema_version != 2:
            raise ValueError("new deployments require schema 2")
        if digest(self.input_json.encode()) != self.input_sha256:
            raise ValueError("input regime digest differs")
        if (
            resolve(self.input_json, self.spec) != self.regime
            or self.projected_dollars != self.spec.projected_dollars
        ):
            raise ValueError("resolved deployment differs from input and spec")
        return self

    def allocation_at(self, now: float) -> Allocation | None:
        if self._legacy is not None and self._legacy.schema_version == 1:
            return None
        return self.spec.admit(now)

    def deadline_at(self, now: float) -> float:
        allocation = self.allocation_at(now)
        if allocation is not None:
            return allocation.deadline
        assert self._legacy is not None
        return now + self._legacy.mix.wall_seconds


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve(input_json: str, spec: JobSpec) -> TrainingRegime:
    regime = TrainingRegime.model_validate_json(input_json)
    if regime.recovery_max_microsteps is not None:
        raise ValueError("remote recovery is unsupported")
    if regime.agent.compound_decisions or regime.agent.belief_count_buckets:
        raise ValueError("remote deployment requires ordinary self-play")
    if len(regime.stages) != 1 or regime.schedule_clock != "iteration_fraction":
        raise ValueError(
            "JobSpec requires one step-target stage with iteration_fraction schedules"
        )
    machine = spec.machine
    for stage in regime.stages:
        if not isinstance(stage, TrainSelfPlay) or stage.opponent is not None:
            raise ValueError("remote deployment requires self-contained self-play")
        if stage.active_seconds is not None:
            raise ValueError(
                "JobSpec requires update targets; active-time recipes are frozen v1 records"
            )
        if stage.execution.memory_bytes > machine.memory_gb * 1024**3:
            raise ValueError("declared rental memory is below the stage requirement")
        stage.execution.device = "cuda"
        stage.execution.threads = min(
            stage.execution.threads, machine.thread_limit, machine.vcpus, 4
        )
    return TrainingRegime.model_validate(regime.model_dump())


def compile_plan(
    input_json: str, spec: JobSpec, source: Source, seed: int
) -> DeploymentPlan:
    # Re-admit copied declarations; historical projections cannot author new work.
    spec = JobSpec.model_validate(spec.model_dump())
    return DeploymentPlan(
        spec=spec,
        input_json=input_json,
        input_sha256=digest(input_json.encode()),
        regime=resolve(input_json, spec),
        source=source,
        seed=seed,
        projected_dollars=spec.projected_dollars,
    )
