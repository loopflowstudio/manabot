"""Private sampler evidence lifecycle, separate from capture and optimization.

Etude owns jobs, worker closure and dollar receipts. EvidenceRoot consumes fresh
launcher observations through an injected reader; it neither polls processes nor
infers settlement from output files. Intent is retained before submission, so an
unknown launch response blocks continuation instead of creating a second job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
from typing import Annotated, Concatenate, Literal, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.belief.sampling_collection import _write_private, read_capture_dataset
from manabot.belief.sampling_data import SamplerDataset, read_dataset, save_dataset
from manabot.belief.sampling_types import _digest


class CaptureWork(BaseModel):
    """An exact subset of one immutable schedule; admission is a separate phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["capture"] = "capture"
    schedule_identity: str = Field(min_length=1)
    slots: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_slots(self) -> CaptureWork:
        if tuple(sorted(set(self.slots))) != self.slots or self.slots[0] < 0:
            raise ValueError("slots must be distinct nonnegative ordinals in order")
        return self


class FitWork(BaseModel):
    """The seed is consumed by dispatch, even if no started receipt is delivered."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["fit"] = "fit"
    dataset_identity: str = Field(min_length=1)
    configuration_identity: str = Field(min_length=1)
    seed: Literal[357, 358, 359]


class JobAllocation(BaseModel):
    """A pre-submit launcher lookup key, exact work and its whole-window cap.

    The adapter must obtain a durable lookup key from its compatible launcher;
    this owner does not manufacture run IDs or assume idempotent submission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    run: str = Field(min_length=1)
    phase: Literal["admission", "collection", "fit"]
    max_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    work: Annotated[CaptureWork | FitWork, Field(discriminator="kind")]

    @model_validator(mode="after")
    def matching_phase(self) -> JobAllocation:
        if (self.phase == "fit") != isinstance(self.work, FitWork):
            raise ValueError("phase does not match scheduled work")
        return self

    @property
    def outputs(self) -> tuple[str, ...]:
        if isinstance(self.work, CaptureWork):
            phase = "admission" if self.phase == "admission" else "main"
            return tuple(
                f"{phase}/game-{slot:06d}.{suffix}.json"
                for slot in self.work.slots
                for suffix in ("started", "result")
            )
        return (
            f"fits/{self.work.seed}/started.json",
            f"fits/{self.work.seed}/complete.json",
            f"fits/{self.work.seed}/sampler.pt",
        )


class JobSettlement(BaseModel):
    """A launcher observation, not an experiment-authored cost estimate.

    writers_closed includes queued/retry workers: none can start or write later.
    usd is the launcher's cumulative amount for this run, counted once. Unknown
    cost or unconfirmed teardown uses None/False and blocks new dispatch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str = Field(min_length=1)
    writers_closed: bool
    usd: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    receipt_identity: str = Field(min_length=1)
    execution: Literal["unknown", "queued", "running", "ended"]
    # Delivered relative paths and byte identities, read from launcher delivery.
    outputs: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None


LauncherSubmit = Callable[[JobAllocation], None]
LauncherRead = Callable[[tuple[str, ...]], tuple[JobSettlement, ...]]


@dataclass(frozen=True, slots=True)
class WorkProgress:
    allocation: JobAllocation
    observation: JobSettlement
    delivery: Literal["unknown", "running", "unavailable", "delivered"]


@dataclass(frozen=True, slots=True)
class Continuation:
    jobs: tuple[WorkProgress, ...]
    capture_slots: dict[str, tuple[int, ...]]
    fit_seeds: tuple[int, ...]
    blocked: tuple[str, ...]


def _bytes_identity(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


P = ParamSpec("P")
R = TypeVar("R")


def _serial(
    method: Callable[Concatenate[EvidenceRoot, P], R],
) -> Callable[Concatenate[EvidenceRoot, P], R]:
    """Serialize admission/closure mutations across invocations of one root."""

    @wraps(method)
    def locked(self: EvidenceRoot, *args: P.args, **kwargs: P.kwargs) -> R:
        self._require_controller()
        descriptor = os.open(
            self.directory / ".controller.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        with os.fdopen(descriptor, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._require_controller()
            return method(self, *args, **kwargs)

    return locked


class EvidenceRoot:
    """One immutable allocation ledger for admission and the main cohort.

    Use only from the serial experiment controller. The launcher reader must
    reconcile shared capacity as well as the named jobs before returning. A
    failed read is unknown, never an empty inventory or a zero-dollar receipt.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        authority = directory / "controller.json"
        if not authority.exists() and not (directory / "worker.json").exists():
            try:
                _write_private(authority, self._controller_identity())
            except FileExistsError:
                pass  # Another invocation established this same directory.
        self.jobs = directory / "jobs"

    def _controller_identity(self) -> dict[str, str | int]:
        stat = self.directory.stat()
        return {
            "host": socket.gethostname(),
            "path": str(self.directory.resolve()),
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }

    def _require_controller(self) -> None:
        authority = self.directory / "controller.json"
        if (
            (self.directory / "worker.json").exists()
            or not authority.exists()
            or json.loads(authority.read_text()) != self._controller_identity()
        ):
            raise ValueError(
                "copied root or worker snapshot has no controller authority"
            )

    def allocations(self) -> tuple[JobAllocation, ...]:
        return tuple(
            JobAllocation.model_validate_json(path.read_bytes())
            for path in sorted(self.jobs.glob("*.json"))
        )

    @_serial
    def observe(self, read: LauncherRead) -> tuple[JobSettlement, ...]:
        """Retain fresh inventory even when it blocks execution or delivery."""
        return self._observe(read)

    def _observe(self, read: LauncherRead) -> tuple[JobSettlement, ...]:
        declared = self.allocations()
        refs = tuple(job.run for job in declared)
        settlements = read(refs)  # Even an empty root requires shared inventory.
        if len({s.run for s in settlements}) != len(settlements) or {
            s.run for s in settlements
        } != set(refs):
            raise ValueError(
                "launcher did not reconcile every declared job exactly once"
            )
        observations = self.directory / "observations"
        observations.mkdir(exist_ok=True, mode=0o700)
        payload = [s.model_dump(mode="json") for s in settlements]
        path = (
            observations
            / f"{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}.json"
        )
        if not path.exists():
            _write_private(path, {"jobs": payload})
        return settlements

    def reconcile(self, read: LauncherRead) -> tuple[JobSettlement, ...]:
        return self._require_closed(self.observe(read))

    @staticmethod
    def _require_closed(
        settlements: tuple[JobSettlement, ...],
    ) -> tuple[JobSettlement, ...]:
        for result in settlements:
            if (
                result.execution != "ended"
                or not result.writers_closed
                or result.usd is None
            ):
                raise ValueError(
                    "launcher job has live/unknown writers or unknown cost"
                )
        return settlements

    @_serial
    def dispatch(
        self, job: JobAllocation, read: LauncherRead, submit: LauncherSubmit
    ) -> None:
        """Reserve through submission under one lock; never resubmit an intent.

        A thrown/lost submission response leaves the durable intent consumed.
        Reconcile its lookup key, including queued retries, before proceeding.
        """
        self._reserve(job, read)
        submit(job)

    def _reserve(self, job: JobAllocation, read: LauncherRead) -> None:
        """Retain one intent before dispatch. Reusing an intent never submits again.

        The caller supplies Etude's stable run identity before invoking launch.
        If submission acknowledgement is lost, reconcile that identity first.
        """
        path = self.jobs / f"{_digest(job.run)}.json"
        if path.exists():
            if JobAllocation.model_validate_json(path.read_bytes()) != job:
                raise ValueError("job allocation changed")
            raise ValueError(
                "job already allocated; reconcile it instead of dispatching"
            )
        settlements = self._require_closed(self._observe(read))
        self._verify_admitted(settlements)
        declared = {j.run: j for j in self.allocations()}
        # All prior jobs are settled, so only their launcher amounts count.
        # The new reservation covers its whole startup/run/finalization window.
        committed = sum(
            (s.usd for s in settlements if s.usd is not None),
            Decimal(0),
        )
        admission = sum(
            (
                s.usd
                for s in settlements
                if declared[s.run].phase == "admission" and s.usd is not None
            ),
            Decimal(0),
        )
        if committed + job.max_usd > Decimal(25):
            raise ValueError("cumulative USD25 allocation exceeded")
        if admission + (
            job.max_usd if job.phase == "admission" else Decimal(0)
        ) > Decimal(2):
            raise ValueError("USD2 admission allocation exceeded")
        if job.phase in ("admission", "collection"):
            phase = "admission" if job.phase == "admission" else "main"
            if (self.directory / phase / "closure.json").exists():
                raise ValueError("collection is closed")
        for previous in declared.values():
            if previous.phase != job.phase:
                continue
            if isinstance(job.work, CaptureWork) and isinstance(
                previous.work, CaptureWork
            ):
                if job.work.schedule_identity != previous.work.schedule_identity or set(
                    job.work.slots
                ) & set(previous.work.slots):
                    raise ValueError(
                        "scheduled work already allocated or schedule changed"
                    )
            elif isinstance(job.work, FitWork) and isinstance(previous.work, FitWork):
                if job.work.seed == previous.work.seed:
                    raise ValueError("fit seed already allocated")
                if (
                    job.work.dataset_identity != previous.work.dataset_identity
                    or job.work.configuration_identity
                    != previous.work.configuration_identity
                ):
                    raise ValueError("declared fit dataset/configuration changed")
        self._validate_work(job)
        self.jobs.mkdir(exist_ok=True, mode=0o700)
        _write_private(path, job.model_dump(mode="json"))

    def stage_worker(self, job: JobAllocation, destination: Path) -> WorkerSnapshot:
        """Export one retained intent and immutable inputs, never controller state.

        Destination must be new. The launcher transports this private snapshot;
        it supplies no dispatch, cost, closure or reconciliation authority.
        """
        self._require_controller()
        if job not in self.allocations():
            raise ValueError("worker has no retained dispatch intent")
        self._validate_work(job)
        destination.mkdir(mode=0o700)
        _write_private(destination / "worker.json", job.model_dump(mode="json"))
        if isinstance(job.work, CaptureWork):
            phase = "admission" if job.phase == "admission" else "main"
            target = destination / phase
            target.mkdir(mode=0o700)
            shutil.copyfile(
                self.directory / phase / "schedule.json", target / "schedule.json"
            )
            (target / "schedule.json").chmod(0o600)
        else:
            self.dataset()
            for name in ("dataset.json", "dataset-binding.json"):
                shutil.copyfile(self.directory / name, destination / name)
                (destination / name).chmod(0o600)
        return WorkerSnapshot(destination)

    @_serial
    def admit_outputs(
        self, job: JobAllocation, returned: Path, read: LauncherRead
    ) -> Literal["unavailable", "delivered"]:
        """Admit launcher-inventoried worker bytes after every writer is closed.

        Partial/fatal output is retained without inventing completion. All input
        files are checked before writing; retries can only add identical bytes.
        A copied ledger, closure, binding or another job's output is rejected.
        """
        if job not in self.allocations():
            raise ValueError("delivery has no retained dispatch intent")
        settlements = self._require_closed(self._observe(read))
        observation = next(s for s in settlements if s.run == job.run)
        allowed = self._output_names(job)
        if set(observation.outputs) - allowed:
            raise ValueError("launcher delivery contains unassigned output")
        if isinstance(job.work, CaptureWork):
            phase = "admission" if job.phase == "admission" else "main"
            if (self.directory / phase / "closure.json").exists() and any(
                not (self.directory / name).exists() for name in observation.outputs
            ):
                raise ValueError("collection is closed")
        # A private staging directory lets validation finish before any bytes
        # enter the authoritative evidence tree, including on corrupt delivery.
        with tempfile.TemporaryDirectory(dir=self.directory) as temporary:
            staged = Path(temporary)
            for name, identity in observation.outputs.items():
                source = returned / name
                if source.is_symlink() or not source.is_file():
                    raise ValueError("launcher output is missing or is a symlink")
                if not source.resolve().is_relative_to(returned.resolve()):
                    raise ValueError("launcher output escapes returned directory")
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(source, target)
                target.chmod(0o600)
                if _bytes_identity(target) != identity:
                    raise ValueError("launcher delivery byte identity mismatch")
                existing = self.directory / name
                if existing.exists() and _bytes_identity(existing) != identity:
                    raise ValueError("returned output conflicts with retained evidence")
            self._validate_returned(job, staged)
            for name in observation.outputs:
                target = self.directory / name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not target.exists():
                    os.link(staged / name, target)
        admitted = self.directory / "deliveries"
        admitted.mkdir(exist_ok=True, mode=0o700)
        binding = {"run": job.run, "outputs": observation.outputs}
        receipt = admitted / f"{_digest(binding)}.json"
        if not receipt.exists():
            _write_private(receipt, binding)
        return self.delivery(job, observation)

    def _verify_admitted(self, settlements: tuple[JobSettlement, ...]) -> None:
        observations = {s.run: s for s in settlements}
        for job in self.allocations():
            observation = observations[job.run]
            present = {
                name
                for name in self._output_names(job)
                if (self.directory / name).exists()
            }
            if not present and not observation.outputs:
                continue
            binding = {"run": job.run, "outputs": observation.outputs}
            receipt = self.directory / "deliveries" / f"{_digest(binding)}.json"
            if not receipt.exists() or json.loads(receipt.read_text()) != binding:
                raise ValueError("worker output has not been admitted")
            for name in present:
                if (
                    name not in observation.outputs
                    or _bytes_identity(self.directory / name)
                    != observation.outputs[name]
                ):
                    raise ValueError("admitted worker output changed")
            if set(observation.outputs) != present:
                raise ValueError("admitted worker output is missing")

    @staticmethod
    def _output_names(job: JobAllocation) -> set[str]:
        if isinstance(job.work, FitWork):
            return set(job.outputs) | {f"fits/{job.work.seed}/stop.json"}
        phase = "admission" if job.phase == "admission" else "main"
        return {
            f"{phase}/game-{slot:06d}.{suffix}.json"
            for slot in job.work.slots
            for suffix in ("started", "result", "game", "stop")
        }

    def _validate_returned(self, job: JobAllocation, directory: Path) -> None:
        if isinstance(job.work, FitWork):
            started = directory / job.outputs[0]
            complete = directory / job.outputs[1]
            if not started.exists() and any(
                (directory / name).exists()
                for name in self._output_names(job) - {job.outputs[0]}
            ):
                raise ValueError("fit delivery has no started receipt")
            if started.exists():
                binding = json.loads(started.read_text())
                if (
                    binding["dataset_identity"] != job.work.dataset_identity
                    or binding["seed"] != job.work.seed
                    or _digest(binding["config"]) != job.work.configuration_identity
                ):
                    raise ValueError("fit delivery binding mismatch")
            if complete.exists():
                receipt = json.loads(complete.read_text())
                checkpoint = directory / job.outputs[2]
                if (
                    not started.exists()
                    or receipt["binding"] != binding
                    or not checkpoint.exists()
                    or receipt["checkpoint_identity"] != _bytes_identity(checkpoint)
                ):
                    raise ValueError("fit delivery completion mismatch")
            return
        phase = "admission" if job.phase == "admission" else "main"
        schedule = json.loads((self.directory / phase / "schedule.json").read_text())
        for slot in job.work.slots:
            prefix = directory / phase / f"game-{slot:06d}"
            started = prefix.with_suffix(".started.json")
            result = prefix.with_suffix(".result.json")
            shard = prefix.with_suffix(".game.json")
            stop = prefix.with_suffix(".stop.json")
            if any(p.exists() for p in (result, shard, stop)) and not started.exists():
                raise ValueError("capture delivery has no started receipt")
            if started.exists() and json.loads(started.read_text()) != {
                "schedule_identity": job.work.schedule_identity,
                "slot": schedule["capture"]["slots"][slot],
            }:
                raise ValueError("capture delivery binding mismatch")
            if shard.exists():
                game = json.loads(shard.read_text())["game"]
                slot_binding = schedule["capture"]["slots"][slot]
                if game is not None and any(
                    game[key] != slot_binding[key]
                    for key in ("game_id", "seed", "assignment", "split")
                ):
                    raise ValueError(
                        "capture delivery game differs from scheduled slot"
                    )
            for path in (result, stop):
                if (
                    path.exists()
                    and json.loads(path.read_text())["schedule_identity"]
                    != job.work.schedule_identity
                ):
                    raise ValueError("capture delivery binding mismatch")
            if result.exists():
                receipt = json.loads(result.read_text())
                status = receipt["status"]
                if status not in ("completed", "failed", "interrupted"):
                    raise ValueError("capture delivery status mismatch")
                if status != "interrupted":
                    if not shard.exists():
                        raise ValueError("capture delivery missing terminal shard")
                    payload = json.loads(shard.read_text())
                    if _digest(payload) != receipt["shard_identity"] or (
                        status == "failed"
                    ) != (payload["failure"] is not None):
                        raise ValueError("capture delivery shard mismatch")

    @_serial
    def continuation(self, read: LauncherRead) -> Continuation:
        """Inspect fresh launcher state and derive untouched work, never dispatch it.

        The following dispatch rechecks under the handoff lock. Output presence
        cannot release a reservation or establish that a worker has ended.
        """
        observations = {result.run: result for result in self._observe(read)}
        allocations = self.allocations()
        jobs = tuple(
            WorkProgress(
                job, observations[job.run], self.delivery(job, observations[job.run])
            )
            for job in allocations
        )
        blocked = [
            f"{result.run}: live/unknown writers or unknown cost"
            for result in observations.values()
            if result.execution != "ended"
            or not result.writers_closed
            or result.usd is None
        ]
        slots: dict[str, tuple[int, ...]] = {}
        for phase, directory_name in (
            ("admission", "admission"),
            ("collection", "main"),
        ):
            directory = self.directory / directory_name
            schedule_path = directory / "schedule.json"
            if not schedule_path.exists() or (directory / "closure.json").exists():
                continue
            schedule = json.loads(schedule_path.read_text())
            if _digest(schedule["capture"]) != schedule["identity"]:
                raise ValueError("capture schedule digest mismatch")
            unresolved = [
                stop
                for stop in directory.glob("game-*.stop.json")
                if not stop.with_suffix(".restored.json").exists()
                or json.loads(stop.with_suffix(".restored.json").read_text()).get(
                    "stop_identity"
                )
                != _digest(json.loads(stop.read_text()))
            ]
            if unresolved:
                blocked.append(
                    f"{phase}: fatal capture prerequisite has not been restored"
                )
            claimed = {
                slot
                for job in allocations
                if job.phase == phase and isinstance(job.work, CaptureWork)
                for slot in job.work.slots
            }
            attempted = {
                path.name.split(".")[0] for path in directory.glob("game-*.json")
            }
            slots[phase] = tuple(
                ordinal
                for ordinal in range(len(schedule["capture"]["slots"]))
                if ordinal not in claimed and f"game-{ordinal:06d}" not in attempted
            )
        seeds = ()
        if (self.directory / "dataset-binding.json").exists():
            self.dataset()
            claimed_seeds = {
                job.work.seed for job in allocations if isinstance(job.work, FitWork)
            }
            seeds = tuple(
                seed
                for seed in (357, 358, 359)
                if seed not in claimed_seeds
                and not (self.directory / f"fits/{seed}").exists()
            )
        return Continuation(
            jobs, {} if blocked else slots, () if blocked else seeds, tuple(blocked)
        )

    def _validate_work(self, job: JobAllocation) -> None:
        work = job.work
        if isinstance(work, CaptureWork):
            phase = "admission" if job.phase == "admission" else "main"
            directory = self.directory / phase
            schedule = json.loads((directory / "schedule.json").read_text())
            if (
                _digest(schedule["capture"]) != work.schedule_identity
                or schedule["identity"] != work.schedule_identity
            ):
                raise ValueError("dispatch schedule identity mismatch")
            attempted = {
                path.name.split(".")[0] for path in directory.glob("game-*.json")
            }
            for slot in work.slots:
                if slot >= len(schedule["capture"]["slots"]):
                    raise ValueError("dispatch slot outside schedule")
                if f"game-{slot:06d}" in attempted:
                    raise ValueError("dispatch slot already attempted")
            for stop in directory.glob("game-*.stop.json"):
                restored = stop.with_suffix(".restored.json")
                if not restored.exists() or json.loads(restored.read_text()).get(
                    "stop_identity"
                ) != _digest(json.loads(stop.read_text())):
                    raise ValueError("fatal capture prerequisite has not been restored")
        else:
            if self.dataset().identity != work.dataset_identity:
                raise ValueError("dispatch dataset identity mismatch")
            if (self.directory / f"fits/{work.seed}").exists():
                raise ValueError("fit seed already attempted")

    def delivery(
        self, job: JobAllocation, observation: JobSettlement
    ) -> Literal["unknown", "running", "unavailable", "delivered"]:
        """Delivery is separate from scientific completion and loader admission."""
        if observation.run != job.run:
            raise ValueError("delivery lookup identity mismatch")
        if observation.execution == "unknown":
            return "unknown"
        if observation.execution != "ended" or not observation.writers_closed:
            return "running"
        if not all(name in observation.outputs for name in job.outputs):
            return "unavailable"
        for name in job.outputs:
            path = self.directory / name
            if not path.is_file():
                return "unavailable"
            if _bytes_identity(path) != observation.outputs[name]:
                raise ValueError("launcher delivery byte identity mismatch")
        if isinstance(job.work, FitWork):
            binding = json.loads((self.directory / job.outputs[0]).read_text())
            receipt = json.loads((self.directory / job.outputs[1]).read_text())
            if (
                binding["dataset_identity"] != job.work.dataset_identity
                or binding["seed"] != job.work.seed
                or _digest(binding["config"]) != job.work.configuration_identity
                or receipt["binding"] != binding
                or receipt["checkpoint_identity"] != observation.outputs[job.outputs[2]]
            ):
                raise ValueError("fit delivery binding mismatch")
        else:
            phase = "admission" if job.phase == "admission" else "main"
            schedule = json.loads(
                (self.directory / phase / "schedule.json").read_text()
            )
            for slot in job.work.slots:
                prefix = f"{phase}/game-{slot:06d}"
                started = json.loads(
                    (self.directory / f"{prefix}.started.json").read_text()
                )
                receipt = json.loads(
                    (self.directory / f"{prefix}.result.json").read_text()
                )
                if (
                    started
                    != {
                        "schedule_identity": job.work.schedule_identity,
                        "slot": schedule["capture"]["slots"][slot],
                    }
                    or receipt["schedule_identity"] != job.work.schedule_identity
                    or receipt["status"] not in ("completed", "failed", "interrupted")
                ):
                    raise ValueError("capture delivery binding mismatch")
                if receipt["status"] != "interrupted":
                    name = f"{prefix}.game.json"
                    path = self.directory / name
                    if name not in observation.outputs or not path.exists():
                        return "unavailable"
                    shard = json.loads(path.read_text())
                    if (
                        _bytes_identity(path) != observation.outputs[name]
                        or _digest(shard) != receipt["shard_identity"]
                        or (receipt["status"] == "failed")
                        != (shard["failure"] is not None)
                    ):
                        raise ValueError("capture delivery shard mismatch")
        return "delivered"

    @_serial
    def restore_capture(
        self, phase: Literal["admission", "main"], *, reason: str, read: LauncherRead
    ) -> None:
        """Record a repaired prerequisite only after the old jobs cannot write."""
        if not reason.strip():
            raise ValueError("restoration requires the repaired prerequisite")
        settlements = self._require_closed(self._observe(read))
        self._verify_admitted(settlements)
        directory = self.directory / phase
        if (directory / "closure.json").exists():
            raise ValueError("collection is closed")
        for stop in sorted(directory.glob("game-*.stop.json")):
            restored = stop.with_suffix(".restored.json")
            if not restored.exists():
                _write_private(
                    restored,
                    {
                        "stop_identity": _digest(json.loads(stop.read_text())),
                        "reason": reason,
                        "jobs": [s.model_dump(mode="json") for s in settlements],
                    },
                )

    @_serial
    def close_collection(
        self, phase: Literal["admission", "main"], *, reason: str, read: LauncherRead
    ) -> None:
        """Close every scheduled slot; retain missing work and all private sources.

        The closure is irreversible. It does not require nonempty splits and
        never relabels an orphan shard as a completed game.
        """
        if not reason.strip():
            raise ValueError("closure requires a stop reason")
        settlements = self._require_closed(self._observe(read))
        self._verify_admitted(settlements)
        directory = self.directory / phase
        closure = directory / "closure.json"
        if closure.exists():
            self._verify_sources(phase)
            return
        schedule = json.loads((directory / "schedule.json").read_text())
        if _digest(schedule["capture"]) != schedule["identity"]:
            raise ValueError("capture schedule digest mismatch")
        statuses: list[str] = []
        for ordinal, _ in enumerate(schedule["capture"]["slots"]):
            prefix = directory / f"game-{ordinal:06d}"
            result = prefix.with_suffix(".result.json")
            started = prefix.with_suffix(".started.json")
            if result.exists():
                receipt = json.loads(result.read_text())
                if receipt["schedule_identity"] != schedule["identity"] or receipt[
                    "status"
                ] not in ("completed", "failed", "interrupted"):
                    raise ValueError("invalid terminal game receipt")
                status = receipt["status"]
                if not started.exists():
                    raise ValueError("terminal game has no started receipt")
                if status != "interrupted":
                    shard = json.loads(prefix.with_suffix(".game.json").read_text())
                    if _digest(shard) != receipt["shard_identity"]:
                        raise ValueError("terminal game shard digest mismatch")
                    if (status == "failed") != (shard["failure"] is not None):
                        raise ValueError("terminal status disagrees with game failure")
            else:
                status = (
                    "interrupted"
                    if started.exists() or prefix.with_suffix(".game.json").exists()
                    else "unattempted"
                )
            if started.exists() and json.loads(started.read_text()) != {
                "schedule_identity": schedule["identity"],
                "slot": schedule["capture"]["slots"][ordinal],
            }:
                raise ValueError("started game differs from schedule")
            statuses.append(status)
        sources = {
            path.name: _bytes_identity(path)
            for path in sorted(directory.glob("*.json"))
        }
        _write_private(
            closure,
            {
                "schedule_identity": schedule["identity"],
                "reason": reason,
                "statuses": statuses,
                "sources": sources,
                "jobs": [s.model_dump(mode="json") for s in settlements],
            },
        )

    def _verify_sources(self, phase: Literal["admission", "main"]) -> str:
        directory = self.directory / phase
        closure = directory / "closure.json"
        payload = json.loads(closure.read_text())
        for name, identity in payload["sources"].items():
            if Path(name).name != name or _bytes_identity(directory / name) != identity:
                raise ValueError("closed collection evidence changed")
        current = {p.name for p in directory.glob("game-*.json")} | {"schedule.json"}
        if current != set(payload["sources"]):
            raise ValueError("closed collection gained or lost source evidence")
        return _bytes_identity(closure)

    @_serial
    def freeze_dataset(self) -> SamplerDataset:
        """Freeze one main dataset after closure; admission can never enter it."""
        closure_identity = self._verify_sources("main")
        dataset = read_capture_dataset(self.directory / "main")
        path = self.directory / "dataset.json"
        if path.exists():
            if read_dataset(path) != dataset:
                raise ValueError("frozen dataset differs from closed collection")
        else:
            save_dataset(dataset, path)
        binding = {
            "closure_identity": closure_identity,
            "dataset_identity": dataset.identity,
            "dataset_sha256": _bytes_identity(path),
        }
        frozen = self.directory / "dataset-binding.json"
        if frozen.exists():
            if json.loads(frozen.read_text()) != binding:
                raise ValueError("frozen dataset binding changed")
        else:
            _write_private(frozen, binding)
        return dataset

    def dataset(self) -> SamplerDataset:
        """Admit frozen bytes and their closed sources before fitting or reporting."""
        binding = json.loads((self.directory / "dataset-binding.json").read_text())
        if (
            self._verify_sources("main") != binding["closure_identity"]
            or _bytes_identity(self.directory / "dataset.json")
            != binding["dataset_sha256"]
        ):
            raise ValueError("frozen dataset binding changed")
        dataset = read_dataset(self.directory / "dataset.json")
        if dataset.identity != binding["dataset_identity"]:
            raise ValueError("frozen dataset identity changed")
        return dataset


class WorkerSnapshot:
    """Private execution inputs for one job, without controller capabilities."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        if (directory / "controller.json").exists():
            raise ValueError("controller root is not a worker snapshot")
        self.allocation = JobAllocation.model_validate_json(
            (directory / "worker.json").read_bytes()
        )

    def dataset(self) -> SamplerDataset:
        if not isinstance(self.allocation.work, FitWork):
            raise ValueError("capture snapshot has no fit dataset")
        binding = json.loads((self.directory / "dataset-binding.json").read_text())
        path = self.directory / "dataset.json"
        if _bytes_identity(path) != binding["dataset_sha256"]:
            raise ValueError("worker dataset bytes changed")
        dataset = read_dataset(path)
        if (
            dataset.identity != binding["dataset_identity"]
            or dataset.identity != self.allocation.work.dataset_identity
        ):
            raise ValueError("worker dataset identity mismatch")
        return dataset
