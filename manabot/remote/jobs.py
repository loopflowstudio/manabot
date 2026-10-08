"""Durable remote job contracts, independent of a submitting process.

S3 intent and create claims own rental identity. TrainingRun and VerifyStore still
own learning; JobRecord reports supervisor liveness and evidence publication.
No state here authorizes restarting an interrupted learner.
"""

import json
import time
from typing import Literal

from pydantic import Field, model_validator

from manabot.infra.artifacts import StoredArtifact, split_s3_uri
from manabot.training.checkpoint_queue import MonitoringBudget

from .plan import MAX_JOB_SECONDS, Allocation, DeploymentPlan, Frozen, digest
from .provider import Pod

DEFAULT_JOBS = "s3://etudefantasia/manabot/jobs"
JobPhase = Literal[
    "accepted",
    "running",
    "finalizing",
    "completed",
    "failed",
    "cancelled",
    "deadline",
    "paused",
]


class Job(Frozen):
    """Admitted execution intent: exact plan, identity and absolute deadline."""

    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    plan: DeploymentPlan
    created_at: float
    deadline: float
    destination: str = DEFAULT_JOBS
    monitoring: MonitoringBudget | None = None
    checkpoint_seconds: float = Field(default=60, gt=0)
    publish_seconds: float = Field(default=30, ge=5)
    # Optional original Experiment authoring receipt; not a second recipe owner.
    experiment_json: str | None = None
    # Omit the absent gate so historical job identities remain byte-for-byte stable.
    validate_numerics: bool = Field(default=False, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def valid(self) -> "Job":
        bucket, prefix = split_s3_uri(self.destination)
        if not prefix or any(c in prefix for c in "*?[]") or ".." in prefix.split("/"):
            raise ValueError("job destination requires a literal private S3 prefix")
        expected_deadline = self.plan.deadline_at(self.created_at)
        if self.deadline != expected_deadline:
            raise ValueError("job deadline must bind the original rental allowance")
        if self.plan.spec.lifetime_seconds > MAX_JOB_SECONDS:
            raise ValueError(
                "remote job credential lifetime is bounded to twelve hours"
            )
        if self.allocation is not None:
            if self.destination != self.plan.spec.access.destination:
                raise ValueError("job destination differs from JobSpec access scope")
        if self.monitoring is not None:
            if self.monitoring.require_initial_admission and (
                len(self.plan.regime.stages) != 1
                or self.plan.regime.stages[0].operation != "train_self_play"
            ):
                raise ValueError("initial admission requires one self-play stage")
            if self.plan.spec.machine.vcpus < 2:
                raise ValueError("monitoring requires a separate allocated CPU")
            if any(
                s.execution.threads >= self.plan.spec.machine.vcpus
                for s in self.plan.regime.stages
            ):
                raise ValueError("reserve one allocated CPU for the evaluator")
            if self.monitoring.attempt_seconds > self.monitoring.seconds:
                raise ValueError("evaluation attempt exceeds cumulative allocation")
        if self.experiment_json is not None:
            receipt = json.loads(self.experiment_json)
            if receipt.get("regime") != json.loads(self.plan.input_json):
                raise ValueError("Experiment receipt differs from deployment input")
        return self

    @property
    def allocation(self) -> Allocation | None:
        return self.plan.allocation_at(self.created_at)

    @property
    def identity(self) -> str:
        return digest(self.model_dump_json().encode())

    @property
    def prefix(self) -> str:
        return f"{self.destination.rstrip('/')}/{self.job_id}"

    @property
    def work_deadline(self) -> float:
        if self.allocation is not None:
            return self.allocation.checkpoint_deadline
        return (
            self.deadline
            - self.plan.spec.upload_seconds
            - self.plan.spec.cleanup_seconds
        )


class Cancellation(Frozen):
    requested_at: float | None = None


class CreateClaim(Frozen):
    spec_sha256: str
    name: str
    purpose: Literal["guardian", "training"]
    intent_time: float
    deadline: float


class Resource(Frozen):
    claim: CreateClaim
    pod: Pod

    def validate_for(self, spec: Job) -> None:
        """Require the claimed job identity, price and hardware before execution."""
        pod, machine = self.pod, spec.plan.spec.machine
        if self.claim.spec_sha256 != spec.identity or pod.name != self.claim.name:
            raise ValueError("provider resource does not belong to this job")
        if (
            pod.gpu_count != machine.gpu_count
            or not 0 < pod.rate <= machine.hourly_ceiling
            or pod.vcpus < machine.vcpus
            or pod.memory_gb < machine.memory_gb
        ):
            raise ValueError(
                "assigned rental fails declared price/resource admission; cancel job"
            )


class JobRecord(Frozen):
    spec_sha256: str
    pod_id: str
    phase: JobPhase
    accepted_at: float
    heartbeat_at: float
    generation: int = 0
    manifest: StoredArtifact | None = None
    artifacts_complete: bool = False
    run_id: str | None = None
    updates: int = 0
    evaluations_completed: int = 0
    evaluator_seconds: float = 0
    cancel_acknowledged_at: float | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.phase in {"completed", "failed", "cancelled", "deadline", "paused"}


class Deletion(Frozen):
    confirmed_at: float
    estimated_dollars: float


class JobStatus(Frozen):
    spec: Job
    record: JobRecord | None
    provider_state: Literal["present", "absent", "unknown", "not-created", "ambiguous"]
    cleanup: Deletion | None = None
    cancel_requested_at: float | None = None
    observed_at: float = Field(default_factory=time.time)

    @property
    def stale(self) -> bool:
        return self.record is None or (
            not self.record.terminal
            and self.observed_at - self.record.heartbeat_at
            > max(90, 3 * self.spec.publish_seconds)
        )
