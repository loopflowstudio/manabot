"""Bounded checkpoint evaluation scheduling over the existing arena and dashboards.

CheckpointQueue is an internal worker queue polled by its execution owner while
learning runs in another process. Immutable jobs bind TrainingRun snapshots, coordinates and protocol;
MonitorResult owns arena evidence. A host/account lease spans all worktrees and
is inherited by the evaluator so a dead supervisor cannot admit a second worker.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Literal

from pydantic import Field, model_validator
import torch

from manabot.arena.models import canonical_sha256
from manabot.training.execution import atomic_json
from manabot.training.models import (
    ArtifactReference,
    Strict,
    TrainingCoordinates,
    TrainingRun,
)
from manabot.training.monitor_evaluation import (
    Checkpoint,
    MonitorProtocol,
    MonitorResult,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.training.monitoring import evaluation_dashboard, training_dashboard


class MonitoringBudget(Strict):
    """Execution allocation, frozen before training; never a scientific protocol."""

    seconds: float = Field(gt=0)
    attempt_seconds: float = Field(default=600, gt=0)
    protocol: MonitorProtocol = Field(default_factory=MonitorProtocol)
    include_initial: bool = Field(default=False, exclude_if=lambda value: not value)
    require_initial_admission: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    terminal_protocols: tuple[MonitorProtocol, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def separate_terminal_deals(self) -> "MonitoringBudget":
        if self.require_initial_admission and not self.include_initial:
            raise ValueError("initial admission requires initialization evaluation")
        used = set(self.protocol.deal_seeds)
        for protocol in self.terminal_protocols:
            if used.intersection(protocol.deal_seeds):
                raise ValueError(
                    "terminal and monitoring deal cohorts must be disjoint"
                )
            used.update(protocol.deal_seeds)
        return self

    def protocols_for(
        self, run: TrainingRun, checkpoint: Checkpoint
    ) -> list[MonitorProtocol]:
        """Select final cohorts only for the last completed stage's raw artifact."""
        if self.terminal_protocols and run.stages:
            final = stage_checkpoint(run, run.regime.stages[-1].id)
            if final is not None and final.artifact == checkpoint.artifact:
                return list(self.terminal_protocols)
        return [self.protocol]


def checkpoints(run: TrainingRun, *, include_initial: bool = False) -> list[Checkpoint]:
    """Admitted raw exports with their original cumulative training coordinates."""
    found = [
        Checkpoint(
            artifact=c.artifact,
            coordinates=TrainingCoordinates.model_validate(
                c.model_dump(exclude={"ordinal", "artifact", "error"})
            ),
        )
        for c in run.monitoring_checkpoints
        if c.artifact is not None and c.error is None
    ]
    for stage in run.stages:
        if include_initial and "initial_raw" in stage.artifacts:
            found.append(
                Checkpoint(
                    artifact=stage.artifacts["initial_raw"],
                    coordinates=TrainingCoordinates(
                        stage_id=stage.id, updates=0, training_seconds=0
                    ),
                )
            )
        checkpoint = stage_checkpoint(run, stage.id)
        if checkpoint is not None:
            found.append(checkpoint)
    return sorted(found, key=lambda c: c.coordinates.training_seconds or 0)


class EvaluationJob(Strict):
    run: TrainingRun
    checkpoint: Checkpoint
    protocol: MonitorProtocol
    allowance_seconds: float
    deadline_unix: float


class Attempt(Strict):
    ordinal: int
    identity: str
    job_sha256: str
    status: Literal["running", "completed", "failed", "interrupted"] = "running"
    reserved_seconds: float
    started_unix: float | None = None
    finished_unix: float | None = None
    charged_seconds: float = 0
    error: str | None = None


class CheckpointQueue:
    """One independent evaluator, with conservative crash charges and no retries.

    Call tick at bounded intervals and close in the supervisor's finally block.
    Restart may discover new checkpoints, but retains every previous attempt.
    """

    def __init__(
        self,
        out: Path,
        config: MonitoringBudget,
        *,
        clock: Callable[[], float] = time.monotonic,
        lease: Path | None = None,
        resolve_artifact: Callable[[ArtifactReference], ArtifactReference]
        | None = None,
        protocols_for: Callable[[TrainingRun, Checkpoint], list[MonitorProtocol]]
        | None = None,
    ) -> None:
        self.out, self.config, self.clock = out, config, clock
        self.resolve_artifact = resolve_artifact
        self.protocols_for = protocols_for
        self.process: subprocess.Popen[bytes] | None = None
        self.active: Attempt | None = None
        self.started = 0.0
        self.attempts: list[Attempt] = []
        self.pending = 0
        self.out.mkdir(parents=True, exist_ok=True)
        lease = lease or Path.home() / ".cache/manabot/checkpoint-evaluator.lock"
        lease.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lease.open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            binding = self.out / "config.json"
            if binding.exists():
                if MonitoringBudget.model_validate_json(binding.read_text()) != config:
                    raise ValueError("monitoring allocation/protocol changed")
            else:
                atomic_json(binding, config.model_dump(mode="json"))
            for path in sorted(self.out.glob("attempt-*/attempt.json")):
                attempt = Attempt.model_validate_json(path.read_text())
                job_path = path.parent / "job.json"
                job = EvaluationJob.model_validate_json(job_path.read_text())
                if canonical_sha256(job.model_dump(mode="json")) != attempt.job_sha256:
                    raise ValueError("monitoring job changed")
                if attempt.status == "running":
                    # The lease proves the previous worker is gone. Charge the
                    # entire reserved allowance, including unobserved startup.
                    attempt.status = "interrupted"
                    attempt.error = "supervisor stopped; reserved allowance charged"
                    attempt.charged_seconds = attempt.reserved_seconds
                    atomic_json(path, attempt.model_dump(mode="json"))
                self.attempts.append(attempt)
            self.attempts.sort(key=lambda a: a.ordinal)
            self._publish()
        except BaseException:
            self.lock.close()
            raise

    @property
    def charged_seconds(self) -> float:
        return sum(a.charged_seconds for a in self.attempts)

    def _directory(self, attempt: Attempt) -> Path:
        return self.out / f"attempt-{attempt.identity}"

    def _finish(self, error: str | None = None) -> None:
        assert self.process is not None and self.active is not None
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        code = self.process.wait()
        self.active.finished_unix = time.time()
        self.active.charged_seconds = self.clock() - self.started
        if error is None and code == 0:
            result_path = self._directory(self.active) / "evaluation/monitor.json"
            if (
                not result_path.exists()
                or MonitorResult.model_validate_json(result_path.read_text()).status
                != "completed"
            ):
                error = "evaluator exited without a completed MonitorResult"
        self.active.status = "failed" if error or code else "completed"
        self.active.error = error or (f"evaluator exited {code}" if code else None)
        atomic_json(
            self._directory(self.active) / "attempt.json",
            self.active.model_dump(mode="json"),
        )
        self.process, self.active = None, None

    def tick(self, sources: list[Path], *, launch: bool = True) -> None:
        if self.process is not None:
            assert self.active is not None
            if self.process.poll() is not None:
                self._finish()
            elif self.clock() - self.started >= self.active.reserved_seconds:
                self._finish("monitoring allowance exhausted")
        seen = {a.identity for a in self.attempts}
        queue: list[tuple[str, TrainingRun, Checkpoint, MonitorProtocol]] = []
        for source in sorted(sources):
            run = TrainingRun.model_validate_json(source.read_text())
            run_dir = self.out / f"run-{canonical_sha256(run.id)[:24]}"
            run_dir.mkdir(exist_ok=True)
            binding = {
                "id": run.id,
                "regime_digest": run.regime_digest,
                "seed": run.seed,
                "regime": run.regime.model_dump(mode="json"),
                "identities": run.identities,
            }
            binding_path = run_dir / "binding.json"
            if binding_path.exists():
                if json.loads(binding_path.read_text()) != binding:
                    raise ValueError("monitoring TrainingRun binding changed")
            else:
                atomic_json(binding_path, binding)
            atomic_json(
                run_dir / "training-dashboard.json",
                training_dashboard(run).model_dump(mode="json"),
            )
            for checkpoint in checkpoints(
                run, include_initial=self.config.include_initial
            ):
                artifact_binding = (
                    run_dir
                    / f"artifact-{canonical_sha256(checkpoint.artifact['path'])}.json"
                )
                if artifact_binding.exists():
                    previous = Checkpoint.model_validate_json(
                        artifact_binding.read_text()
                    )
                    if previous != checkpoint:
                        raise ValueError("checkpoint artifact or coordinates changed")
                else:
                    atomic_json(artifact_binding, checkpoint.model_dump(mode="json"))
                protocols = (
                    self.protocols_for(run, checkpoint)
                    if self.protocols_for is not None
                    else self.config.protocols_for(run, checkpoint)
                )
                for protocol in protocols:
                    identity = canonical_sha256(
                        {
                            "run": run.id,
                            "checkpoint": checkpoint.model_dump(mode="json"),
                            "protocol": protocol.model_dump(mode="json"),
                        }
                    )
                    if identity not in seen:
                        queue.append((identity, run, checkpoint, protocol))
                        seen.add(identity)
        self.pending = len(queue)
        remaining = self.config.seconds - self.charged_seconds
        # Do not start a cohort whose full declared allowance cannot fit.
        if (
            launch
            and self.process is None
            and queue
            and remaining >= self.config.attempt_seconds
        ):
            identity, run, checkpoint, protocol = queue[0]
            local_checkpoint = checkpoint
            if self.resolve_artifact is not None:
                local = self.resolve_artifact(checkpoint.artifact)
                if any(
                    local[key] != checkpoint.artifact[key]
                    for key in ("sha256", "bytes")
                ):
                    raise ValueError("relocated checkpoint identity differs")
                local_checkpoint = checkpoint.model_copy(update={"artifact": local})
            job = EvaluationJob(
                run=run,
                checkpoint=local_checkpoint,
                protocol=protocol,
                allowance_seconds=self.config.attempt_seconds,
                deadline_unix=time.time() + self.config.attempt_seconds,
            )
            attempt = Attempt(
                ordinal=len(self.attempts),
                started_unix=time.time(),
                identity=identity,
                job_sha256=canonical_sha256(job.model_dump(mode="json")),
                reserved_seconds=self.config.attempt_seconds,
            )
            directory = self._directory(attempt)
            directory.mkdir()
            atomic_json(directory / "job.json", job.model_dump(mode="json"))
            atomic_json(directory / "attempt.json", attempt.model_dump(mode="json"))
            self.attempts.append(attempt)
            self.started = self.clock()
            try:
                with (directory / "worker.log").open("ab") as log:
                    self.process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "manabot.training.checkpoint_queue",
                            "--job",
                            str(directory / "job.json"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(self.lock.fileno(),),
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        },
                    )
                self.active = attempt
            except Exception as error:
                attempt.status, attempt.error = (
                    "failed",
                    f"{type(error).__name__}: {error}",
                )
                attempt.charged_seconds = self.clock() - self.started
                atomic_json(directory / "attempt.json", attempt.model_dump(mode="json"))
            self.pending -= 1
        self._publish()

    def _publish(self) -> None:
        grouped: dict[tuple[str, str], list[MonitorResult]] = {}
        for attempt in self.attempts:
            path = self._directory(attempt) / "evaluation/monitor.json"
            if path.exists():
                result = MonitorResult.model_validate_json(path.read_text())
                # A killed cohort cannot publish its partial rates as complete.
                if attempt.status in {"failed", "interrupted"}:
                    result.status = "incomplete"
                    result.score = result.win = result.draw = None
                    result.error = attempt.error
                grouped.setdefault(
                    (
                        result.run_id,
                        canonical_sha256(result.protocol.model_dump(mode="json")),
                    ),
                    [],
                ).append(result)
        curves: list[str] = []
        for (run_id, protocol_id), results in grouped.items():
            directory = self.out / f"run-{canonical_sha256(run_id)[:24]}"
            directory.mkdir(exist_ok=True)
            path = directory / (
                "dashboard.json"
                if self.protocols_for is None
                else f"dashboard-{protocol_id}.json"
            )
            atomic_json(path, evaluation_dashboard(results).model_dump(mode="json"))
            curves.append(str(path))
        atomic_json(
            self.out / "dashboard.json",
            {
                "purpose": "checkpoint-evaluation"
                if self.protocols_for is not None
                else "monitoring-not-scientific-evaluation",
                "allocation_seconds": self.config.seconds,
                "charged_evaluator_process_seconds": self.charged_seconds,
                "active_reserved_seconds": self.active.reserved_seconds
                if self.active
                else 0,
                "pending_checkpoints": self.pending,
                "curves": curves,
                "attempts": [a.model_dump(mode="json") for a in self.attempts],
                "contention": "one CPU evaluator overlaps learner; no throughput correction",
            },
        )

    def close(self) -> None:
        try:
            if self.process is not None:
                self._finish("evaluation owner stopped monitoring")
            self._publish()
        finally:
            self.lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    if os.getpid() != os.getpgrp():
        raise ValueError("internal worker requires an isolated process session")
    job = EvaluationJob.model_validate_json(args.job.read_text())

    # Even if the supervisor dies, stop this whole evaluator session at its cap.
    # The supervisor retains the immutable attempt and conservatively charges it.
    def expire(signum: int, frame: object) -> None:
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGALRM, expire)
    remaining = job.deadline_unix - time.time()
    if remaining <= 0:
        expire(signal.SIGALRM, None)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    torch.set_num_threads(1)
    result = evaluate_checkpoint(
        job.run,
        job.checkpoint.artifact,
        job.checkpoint.coordinates,
        args.job.parent / "evaluation",
        protocol=job.protocol,
        concurrent_activity="campaign learner may overlap; one evaluator CPU thread; no contention correction",
    )
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
