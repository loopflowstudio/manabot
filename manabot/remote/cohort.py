"""Serial execution of frozen Experiment jobs through the existing deploy lifecycle.

S3 owns intent, attempts and cancellation. A managed service holds a local flock
and an immutable host binding; restart observes the same jobs, never replacement
learners. No artifact download, notebook or telemetry runs in the scheduling loop.
"""

from pathlib import Path
import re
import time
from typing import Callable, Literal

from pydantic import Field, model_validator

from manabot.training.checkpoint_queue import MonitoringBudget

from .job_client import cancel_job, job_manifest, persist_job, reconcile_job, submit_job
from .job_store import JobStore, S3JobStore, cancellation_requested
from .jobs import Cancellation, Job, JobStatus
from .plan import AccessScope, DeploymentPlan, Frozen, digest

DEFAULT_COHORTS = "s3://etudefantasia/manabot/cohorts"


class CohortEntry(Frozen):
    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    plan: DeploymentPlan
    monitoring: MonitoringBudget | None = None
    checkpoint_seconds: float = Field(default=3600, gt=0)
    experiment_json: str | None = None

    def job(self, now: float) -> Job:
        return Job(
            job_id=self.job_id,
            plan=self.plan,
            created_at=now,
            deadline=self.plan.deadline_at(now),
            destination=self.plan.spec.access.destination,
            monitoring=self.monitoring,
            checkpoint_seconds=self.checkpoint_seconds,
            experiment_json=self.experiment_json,
        )


class Cohort(Frozen):
    """Immutable order and inclusive allocation; no failed-job replacement policy."""

    schema_version: Literal[1] = 1
    cohort_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    entries: tuple[CohortEntry, ...] = Field(min_length=1)
    destination: str = DEFAULT_COHORTS
    deadline: float = Field(gt=0)
    spending_limit: float = Field(gt=0)
    prior_dollars: float = Field(ge=0)
    controller_dollars: float = Field(ge=0)

    @model_validator(mode="after")
    def valid(self) -> "Cohort":
        AccessScope(destination=self.destination)
        if len({e.job_id for e in self.entries}) != len(self.entries):
            raise ValueError("cohort job IDs must be unique")
        if len({e.plan.source for e in self.entries}) != 1:
            raise ValueError("one cohort requires one frozen source checkout")
        if self.reserved_dollars > self.spending_limit:
            raise ValueError("cohort reservation exceeds inclusive spending limit")
        # Validate job/monitoring admission without persisting or spending.
        for entry in self.entries:
            entry.job(0)
        return self

    @property
    def reserved_dollars(self) -> float:
        return (
            self.prior_dollars
            + self.controller_dollars
            + sum(e.plan.spec.spending_limit for e in self.entries)
        )

    @property
    def prefix(self) -> str:
        return f"{self.destination.rstrip('/')}/{self.cohort_id}"

    @property
    def identity(self) -> str:
        return digest(self.model_dump_json().encode())


Phase = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "paused",
    "cancelled",
    "uncertain",
    "missing",
    "budget-exhausted",
    "deadline",
]


class CohortAttempt(Frozen):
    spec: Job
    observation: JobStatus | None = None
    manifest_verified: str | None = None


class CohortState(Frozen):
    cohort_sha256: str
    owner: str
    heartbeat_at: float = 0
    phase: Phase = "pending"
    attempts: tuple[CohortAttempt, ...] = ()
    error: str | None = None

    @property
    def settled(self) -> bool:
        """No further autonomous observation can admit or clean up a job."""
        if self.phase == "completed":
            return True
        if self.phase not in {
            "failed",
            "paused",
            "cancelled",
            "deadline",
            "missing",
            "budget-exhausted",
        }:
            return False
        if not self.attempts:
            return True
        observed = self.attempts[-1].observation
        return observed is not None and (
            observed.cleanup is not None or observed.provider_state == "not-created"
        )

    def charged_dollars(self, cohort: Cohort) -> float:
        """Unsettled jobs retain their entire reservation; settled charges count once."""
        return (
            cohort.prior_dollars
            + cohort.controller_dollars
            + sum(
                a.observation.cleanup.estimated_dollars
                if a.observation is not None and a.observation.cleanup is not None
                else a.spec.plan.spec.spending_limit
                for a in self.attempts
            )
        )


def prepare_cohort(cohort: Cohort, store: JobStore | None = None) -> None:
    store = store or S3JobStore(cohort.prefix)
    data = cohort.model_dump_json().encode()
    if not store.create("spec.json", data):
        previous = store.read("spec.json")
        if previous is None or previous.data != data:
            raise ValueError("cohort ID already binds different immutable content")
    store.create("cancel.json", Cancellation().model_dump_json().encode())


def load_cohort(cohort_id: str, destination: str = DEFAULT_COHORTS) -> Cohort:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", cohort_id) is None:
        raise ValueError("invalid cohort ID")
    AccessScope(destination=destination)
    raw = S3JobStore(f"{destination.rstrip('/')}/{cohort_id}").read("spec.json")
    if raw is None:
        raise ValueError("cohort intent missing")
    cohort = Cohort.model_validate_json(raw.data)
    if cohort.cohort_id != cohort_id or cohort.destination != destination:
        raise ValueError("cohort location differs")
    return cohort


def cancel_cohort(
    cohort: Cohort,
    store: JobStore | None = None,
    *,
    cancel: Callable[[Job], None] = cancel_job,
) -> None:
    """Stick cancellation before forwarding it, even when the owner is offline."""
    store = store or S3JobStore(cohort.prefix)
    data = Cancellation(requested_at=time.time()).model_dump_json().encode()
    while True:
        value = store.read("cancel.json")
        if value is None:
            if store.create("cancel.json", data):
                break
        elif Cancellation.model_validate_json(value.data).requested_at is not None:
            break
        elif store.replace("cancel.json", data, value.etag):
            break
    state_raw = store.read("state.json")
    if state_raw is not None:
        state = CohortState.model_validate_json(state_raw.data)
        if state.cohort_sha256 != cohort.identity:
            raise ValueError("cohort state belongs to different intent")
        if state.attempts:
            # The cohort marker fences any later admission. Repeating this exact
            # job cancellation is idempotent if forwarding loses its response.
            cancel(state.attempts[-1].spec)


def verify_manifest(status: JobStatus, cache: Path) -> str:
    """Check only the final commit manifest. Bulk checkpoint retrieval is separate."""
    record = status.record
    if record is None or record.manifest is None or not record.artifacts_complete:
        raise ValueError("final artifact commit missing")
    manifest = job_manifest(status.spec, record, cache)
    if (
        manifest.spec_sha256 != status.spec.identity
        or manifest.generation != record.generation
        or not manifest.complete
    ):
        raise ValueError("final manifest identity differs")
    return record.manifest.sha256


class CohortSupervisor:
    """One tick reconciles one durable attempt; every irreversible effect is fenced.

    The caller must hold the service's host lock throughout this object's lifetime.
    Network failures only retry observation/submission of the same admitted ID.
    Failed training, ambiguous creates and absent evidence never spawn replacements.
    """

    def __init__(
        self,
        cohort: Cohort,
        owner: str,
        cache: Path,
        *,
        store: JobStore | None = None,
        observe: Callable[[Job], JobStatus] = reconcile_job,
        submit: Callable[[Job], JobStatus] = submit_job,
        persist: Callable[[Job], Job] = persist_job,
        cancel: Callable[[Job], None] = cancel_job,
        verify: Callable[[JobStatus, Path], str] = verify_manifest,
    ) -> None:
        self.cohort, self.cache = cohort, cache
        self.store = store or S3JobStore(cohort.prefix)
        self.observe, self.submit, self.persist = observe, submit, persist
        self.cancel, self.verify = cancel, verify
        prepare_cohort(cohort, self.store)
        state = CohortState(cohort_sha256=cohort.identity, owner=owner)
        self.store.create("state.json", state.model_dump_json().encode())
        self.previous = self.store.read("state.json")
        if self.previous is None:
            raise RuntimeError("cohort state acknowledgement unavailable")
        self.state = CohortState.model_validate_json(self.previous.data)
        if self.state.cohort_sha256 != cohort.identity or self.state.owner != owner:
            raise ValueError(
                "cohort bound to another owner or intent; no automatic takeover"
            )

    def _save(self, **updates: object) -> CohortState:
        state = self.state.model_copy(update={"heartbeat_at": time.time(), **updates})
        data = state.model_dump_json().encode()
        if not self.store.replace("state.json", data, self.previous.etag):
            raise RuntimeError("cohort ownership changed")
        raw = self.store.read("state.json")
        if raw is None or raw.data != data:
            raise RuntimeError("cohort state write uncertain; restart and reconcile")
        self.previous, self.state = raw, state
        return state

    def observation_failed(self, error: Exception) -> CohortState:
        """Retain last observations and expose a redacted control-plane failure."""
        return self._save(phase="uncertain", error=type(error).__name__)

    def tick(self) -> CohortState:
        cohort = self.cohort
        cancelled = cancellation_requested(self.store) is not None
        expired = time.time() >= cohort.deadline
        attempts = self.state.attempts
        if attempts:
            attempt = attempts[-1]
            # Reconcile even terminal failures so cancellation, final publication and
            # late deletion/cost evidence can still become visible after restart.
            self.persist(attempt.spec)
            if cancelled or expired:
                # Cancellation must not depend on provider inventory availability.
                self.cancel(attempt.spec)
            status = self.observe(attempt.spec)
            attempt = attempt.model_copy(update={"observation": status})
            attempts = (*attempts[:-1], attempt)
            self._save(attempts=attempts, error=None)
            if cancelled or expired:
                return self._save(phase="cancelled" if cancelled else "deadline")
            if status.provider_state in {"ambiguous", "unknown"}:
                return self._save(phase="uncertain")
            record = status.record
            if record is not None and record.terminal:
                if record.phase != "completed":
                    phase: Phase = (
                        "deadline" if record.phase == "deadline" else record.phase
                    )
                    return self._save(phase=phase)
                if not record.artifacts_complete or record.manifest is None:
                    return self._save(
                        phase="missing", error="final artifacts incomplete"
                    )
                if status.cleanup is None:
                    return self._save(phase="running")
                if attempt.manifest_verified != record.manifest.sha256:
                    verified = self.verify(status, self.cache)
                    attempt = attempt.model_copy(update={"manifest_verified": verified})
                    attempts = (*attempts[:-1], attempt)
                    self._save(attempts=attempts)
            elif status.cleanup is not None or (
                time.time() >= attempt.spec.deadline
                and status.provider_state == "not-created"
            ):
                return self._save(
                    phase="missing", error="job ended without terminal evidence"
                )
            else:
                if record is None:
                    # Reuse existing job/create claims. Lost responses must not
                    # create a new ID, reset deadlines or consume another allowance.
                    self.submit(attempt.spec)
                return self._save(phase="uncertain" if status.stale else "running")
        if cancelled or expired:
            return self._save(phase="cancelled" if cancelled else "deadline")
        remaining = cohort.entries[len(attempts) :]
        reserved = self.state.charged_dollars(cohort) + sum(
            e.plan.spec.spending_limit for e in remaining
        )
        if reserved > cohort.spending_limit:
            return self._save(phase="budget-exhausted")
        if not remaining:
            return self._save(phase="completed")
        entry = remaining[0]
        spec = entry.job(time.time())
        if spec.deadline > cohort.deadline:
            return self._save(
                phase="deadline", error="remaining cohort time cannot fit allocation"
            )
        # Atomic write before any job intent/provider effect. Reuse exact admission
        # time on restart, including a crash immediately after this write.
        return self._save(
            attempts=(*attempts, CohortAttempt(spec=spec)), phase="pending", error=None
        )
