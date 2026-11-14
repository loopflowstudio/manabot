"""Private sampler evidence lifecycle, separate from capture and optimization.

Etude owns jobs, worker closure and dollar receipts. EvidenceRoot consumes fresh
launcher observations through an injected reader; it neither polls processes nor
infers settlement from output files. Intent is retained before submission, so an
unknown launch response blocks continuation instead of creating a second job.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
from typing import Concatenate, Literal, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from manabot.belief.sampling_collection import _write_private, read_capture_dataset
from manabot.belief.sampling_data import SamplerDataset, read_dataset, save_dataset
from manabot.belief.sampling_types import _digest

Phase = Literal["admission", "collection", "fit", "evaluation"]


class JobAllocation(BaseModel):
    """Charge startup/setup to the phase it prepares, including admission's USD2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str = Field(min_length=1)
    phase: Phase
    max_usd: Decimal = Field(ge=0, allow_inf_nan=False)


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


LauncherRead = Callable[[tuple[str, ...]], tuple[JobSettlement, ...]]


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
        descriptor = os.open(
            self.directory / ".controller.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        with os.fdopen(descriptor, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
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
        self.jobs = directory / "jobs"
        self.jobs.mkdir(exist_ok=True, mode=0o700)

    def allocations(self) -> tuple[JobAllocation, ...]:
        return tuple(
            JobAllocation.model_validate_json(path.read_bytes())
            for path in sorted(self.jobs.glob("*.json"))
        )

    def reconcile(self, read: LauncherRead) -> tuple[JobSettlement, ...]:
        declared = self.allocations()
        refs = tuple(job.run for job in declared)
        settlements = read(refs)  # Even an empty root requires shared inventory.
        if len({s.run for s in settlements}) != len(settlements) or {
            s.run for s in settlements
        } != set(refs):
            raise ValueError(
                "launcher did not reconcile every declared job exactly once"
            )
        for result in settlements:
            if not result.writers_closed or result.usd is None:
                raise ValueError(
                    "launcher job has live/unknown writers or unknown cost"
                )
        return settlements

    @_serial
    def reserve(self, job: JobAllocation, read: LauncherRead) -> None:
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
        settlements = self.reconcile(read)
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
        if job.phase == "admission" and admission + job.max_usd > Decimal(2):
            raise ValueError("USD2 admission allocation exceeded")
        if job.phase in ("admission", "collection"):
            phase = "admission" if job.phase == "admission" else "main"
            if (self.directory / phase / "closure.json").exists():
                raise ValueError("collection is closed")
        _write_private(path, job.model_dump(mode="json"))

    @_serial
    def restore_capture(
        self, phase: Literal["admission", "main"], *, reason: str, read: LauncherRead
    ) -> None:
        """Record a repaired prerequisite only after the old jobs cannot write."""
        if not reason.strip():
            raise ValueError("restoration requires the repaired prerequisite")
        settlements = self.reconcile(read)
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
        settlements = self.reconcile(read)
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
