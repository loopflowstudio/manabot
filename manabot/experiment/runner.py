"""Shared explicit execution for declarative Experiment comparisons.

run_experiment admits configured local hardware, schedules regime/seed processes,
feeds immutable checkpoints to the existing evaluator and exports one editable
notebook. TrainingRun/VerifyStore own learning and recovery; failed attempts are
never retried without explicit selection. No research conclusions are authored.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

from pydantic import JsonValue
import torch

from manabot.eval.monitor import MonitorProtocol
from manabot.experiment.checkpoint_evaluation import Checkpoint
from manabot.experiment.config import Experiment
from manabot.experiment.evaluation_queue import CheckpointQueue, stop_process_group
from manabot.experiment.execution import ExperimentRun, HardwareInventory, RegimeAttempt
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import source_bundle_sha256
from manabot.training.clock import arm_worker_deadline, boot_identity
from manabot.training.execution import execute_regime
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime
from manabot.training.state.recovery import attempt_lock
from manabot.verify.store import VerifyStore
import managym


def _runtime() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "world": managym.WORLD_VERSION,
        "source": source_bundle_sha256(sorted((root / "manabot").rglob("*.py"))),
        "native": source_bundle_sha256(sorted((root / "managym").glob("*.so"))),
    }


def _stop(process: subprocess.Popen[bytes]) -> None:
    stop_process_group(process)


def run_experiment(
    experiment: Experiment,
    hardware: HardwareInventory,
    out: Path,
    *,
    resume: bool = False,
    recover: tuple[int, ...] = (),
) -> ExperimentRun:
    """Run an explicit allocation; resume pending work, never silently retry failures.

    recover selects stopped training attempt ordinals for the existing same-host
    recovery path. Unsupported snapshots fail with retained evidence. No new budget
    is granted by resume. No tracker is required and report refresh preserves edits.
    """
    entered, cpu_entered = time.monotonic(), time.process_time()
    if experiment.jobs:
        raise ValueError("JobSpec runs use Experiment.prepare_job and deploy submit")
    schedule = experiment.schedule
    if schedule is None:
        raise ValueError("Experiment execution requires an explicit schedule")
    selected = hardware.select(schedule.hardware)
    resolved = experiment.resolve()
    regimes = resolved.regimes
    if schedule.active_runtime and (
        any(r.recovery is None for r in regimes.values())
        or not schedule.monitoring.active_runtime
    ):
        raise ValueError("active experiment requires active recovery and monitoring")
    for regime in regimes.values():
        schedule.admit_regime(regime)
    if any(
        getattr(stage, "execution", None) is not None
        and (
            stage.execution.device != "cpu"
            or stage.execution.threads > selected.cpu_threads - 1
        )
        for regime in regimes.values()
        for stage in regime.stages
    ):
        raise ValueError(
            "regime exceeds configured local CPU capacity; reserve one evaluator thread"
        )
    order = schedule.order or tuple(tuple(range(len(regimes))) for _ in schedule.seeds)
    if len(order) != len(schedule.seeds) or any(
        sorted(row) != list(range(len(regimes))) for row in order
    ):
        raise ValueError("each seed must schedule every case exactly once")
    intent: dict[str, JsonValue] = {
        "experiment": resolved.receipt(),
        "schedule": schedule.model_dump(mode="json"),
    }
    identity = canonical_sha256(intent)
    out = out.resolve()
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError("resume requires an existing experiment")
    if recover and not resume:
        raise ValueError("recovery selection requires resume")
    lease_root = Path.home() / ".cache/manabot"
    lease_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # Placement currently supports one experiment per configured local host.
        # Keep the same inode across executions; never unlink a lease.
        host_lease = (lease_root / "experiment-host.lock").open("a+b")
        stack.callback(host_lease.close)

        fcntl.flock(host_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stack.enter_context(attempt_lock(out / "execution.lock"))
        store = stack.enter_context(VerifyStore(out / "experiment.sqlite"))
        if resume:
            exported = ExperimentRun.model_validate_json(
                (out / "experiment.json").read_text()
            )
            record = store.experiment_run(exported.id)
            if (
                record.intent_sha256 != identity
                or record.hardware != selected
                or record.runtime != _runtime()
            ):
                raise ValueError("experiment intent, hardware or runtime changed")
            gap = max(0, time.time() - (record.last_seen_unix or time.time()))
            unknown = gap
            if (
                schedule.active_runtime
                and record.boot_identity == boot_identity()
                and record.last_seen_active is not None
            ):
                unknown = max(0, time.monotonic() - record.last_seen_active)
            if record.status == "running":
                record.elapsed_seconds += unknown
                record.uncertain_seconds += unknown
                record.downtime_seconds += max(0, gap - unknown)
            else:
                record.downtime_seconds += gap
            record.calendar_seconds += gap
            for attempt in record.attempts:
                if attempt.status == "running":
                    attempt.status = "interrupted"
                    attempt.error = (
                        "owner stopped; full process allowance conservatively charged"
                    )
                    attempt.process_seconds = (
                        attempt.process_seconds + unknown
                        if schedule.active_runtime
                        else attempt.allowance_seconds
                    )
            for ordinal in recover:
                parent = next(
                    (a for a in record.attempts if a.ordinal == ordinal), None
                )
                if parent is None or parent.status not in {"failed", "interrupted"}:
                    raise ValueError("recovery requires a stopped training attempt")
                path = Path(parent.path) / "run.json"
                run = TrainingRun.model_validate_json(path.read_text())
                if any(a.recovery_parent == run.id for a in record.attempts):
                    raise ValueError("attempt already has a recovery child")
                record.attempts.append(
                    RegimeAttempt(
                        ordinal=len(record.attempts),
                        case=parent.case,
                        seed=parent.seed,
                        path=str(
                            out / "training" / f"attempt-{len(record.attempts):04d}"
                        ),
                        allowance_seconds=max(
                            0, parent.allowance_seconds - parent.process_seconds
                        )
                        if schedule.active_runtime
                        else parent.allowance_seconds,
                        recovery_parent=run.id,
                    )
                )
        else:
            record = ExperimentRun(
                id=f"experiment-{uuid.uuid4().hex}",
                intent=intent,
                intent_sha256=identity,
                hardware=selected,
                runtime=_runtime(),
                notebook=str(out / "comparison.ipynb"),
            )
            for seed, indexes in zip(schedule.seeds, order, strict=True):
                for index in indexes:
                    case = resolved.cases[index]
                    record.attempts.append(
                        RegimeAttempt(
                            ordinal=len(record.attempts),
                            case=case.name,
                            seed=seed,
                            path=str(
                                out / "training" / f"attempt-{len(record.attempts):04d}"
                            ),
                            allowance_seconds=case.regime.wall_seconds,
                        )
                    )
        started, prior_elapsed = entered, record.elapsed_seconds
        calendar_started, prior_calendar = time.time(), record.calendar_seconds
        prior_downtime = record.downtime_seconds
        cpu_started, prior_cpu = cpu_entered, record.coordinator_cpu_seconds
        last_monitor_poll = float("-inf")
        queue: CheckpointQueue | None = None
        record.status, record.error, record.paused = "running", None, False
        pause_path = out / "pause.request"
        if resume:
            pause_path.unlink(missing_ok=True)

        def save() -> None:
            record.coordinator_cpu_seconds = (
                prior_cpu + time.process_time() - cpu_started
            )
            record.host_load = os.getloadavg()
            record.last_seen_unix = time.time()
            record.last_seen_active = time.monotonic()
            record.boot_identity = boot_identity() if schedule.active_runtime else None
            record.elapsed_seconds = prior_elapsed + time.monotonic() - started
            record.calendar_seconds = prior_calendar + time.time() - calendar_started
            record.downtime_seconds = prior_downtime + max(
                0, time.time() - calendar_started - (time.monotonic() - started)
            )
            record.evaluator_seconds = (
                queue.charged_seconds if queue else record.evaluator_seconds
            )
            record.process_seconds = (
                sum(a.process_seconds for a in record.attempts)
                + record.evaluator_seconds
            )
            record.host_dollars = (
                record.elapsed_seconds * selected.dollars_per_hour / 3600
                if selected.dollars_per_hour is not None
                else None
            )
            store.save_experiment_run(record)
            atomic_json(out / "experiment.json", record.model_dump(mode="json"))

        def sources() -> list[Path]:
            return [
                p for a in record.attempts if (p := Path(a.path) / "run.json").exists()
            ]

        def check() -> None:
            if prior_elapsed + time.monotonic() - started >= schedule.wall_seconds:
                raise TimeoutError("experiment wall allowance exhausted")
            if shutil.disk_usage(out).free < schedule.disk_reserve_bytes:
                raise RuntimeError("experiment evidence disk reserve exhausted")

        save()
        try:
            record.report(out)

            def protocols_for(
                run: TrainingRun, checkpoint: Checkpoint
            ) -> list[MonitorProtocol]:
                protocols = [schedule.monitoring.protocol]
                if (
                    schedule.diagnostic_protocol is not None
                    and checkpoint.coordinates.updates in schedule.diagnostic_updates
                ):
                    protocols.append(schedule.diagnostic_protocol)
                return protocols

            queue = CheckpointQueue(
                out / "monitoring",
                schedule.monitoring,
                protocols_for=protocols_for
                if schedule.diagnostic_protocol is not None
                else None,
            )

            def execution_order(attempt: RegimeAttempt) -> tuple[int, int]:
                ancestor = attempt
                while ancestor.recovery_parent is not None:
                    ancestor = next(
                        a
                        for a in record.attempts
                        if a.run_id == ancestor.recovery_parent
                    )
                return ancestor.ordinal, attempt.ordinal

            for attempt in sorted(record.attempts, key=execution_order):
                if attempt.status != "pending":
                    continue
                check()
                if pause_path.exists():
                    record.paused = True
                    break
                if attempt.allowance_seconds <= 0:
                    raise TimeoutError("recovered attempt has no remaining allocation")
                # Reserve monitoring once. Its overlapping process time cannot
                # disappear inside learner wall time or be spent twice.
                learning_spent = sum(a.process_seconds for a in record.attempts)
                if (
                    learning_spent
                    + attempt.allowance_seconds
                    + schedule.monitoring.seconds
                    > schedule.process_seconds
                ):
                    raise TimeoutError(
                        "next training attempt cannot fit with monitoring reserve"
                    )
                directory = Path(attempt.path)
                job_path = out / f"training-job-{attempt.ordinal:04d}.json"
                atomic_json(
                    job_path,
                    {
                        "regime": regimes[attempt.case].model_dump(mode="json"),
                        "seed": attempt.seed,
                        "out": str(directory),
                        "store": str(store.path.resolve()),
                        "resume_from": attempt.recovery_parent,
                        "checkpoint_seconds": schedule.checkpoint_seconds,
                        "checkpoint_updates": schedule.checkpoint_updates,
                        "allowance_seconds": attempt.allowance_seconds,
                        "deadline_unix": time.time() + attempt.allowance_seconds,
                        "active_runtime": schedule.active_runtime,
                        "pause_path": str(pause_path)
                        if schedule.active_runtime
                        else None,
                    },
                )
                attempt.status = "running"
                attempt.started_unix = time.time()
                save()
                child_started = time.monotonic()
                process: subprocess.Popen[bytes] | None = None
                try:
                    with (out / f"training-{attempt.ordinal:04d}.log").open(
                        "ab"
                    ) as log:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "manabot.experiment.runner",
                                "--job",
                                str(job_path),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            pass_fds=(host_lease.fileno(), queue.lock.fileno()),
                            env={
                                **os.environ,
                                "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                            },
                        )
                    while process.poll() is None:
                        check()
                        if (
                            time.monotonic() - child_started
                            >= attempt.allowance_seconds
                        ):
                            raise TimeoutError("training process allowance exhausted")
                        if time.monotonic() - last_monitor_poll >= 5:
                            queue.tick(sources(), launch=not pause_path.exists())
                            attempt.process_seconds = time.monotonic() - child_started
                            save()
                            last_monitor_poll = time.monotonic()
                        time.sleep(0.2)
                    path = directory / "run.json"
                    run = (
                        TrainingRun.model_validate_json(path.read_text())
                        if path.exists()
                        else None
                    )
                    attempt.run_id = run.id if run else None
                    if (
                        run is not None
                        and run.error
                        and run.error.startswith("TrainingPaused:")
                    ):
                        attempt.status, attempt.error = "interrupted", run.error
                        record.paused = True
                        continue
                    if process.returncode or run is None or run.status != "completed":
                        raise RuntimeError(
                            f"learner exited {process.returncode}: {run.error if run else 'missing TrainingRun'}"
                        )
                    attempt.status = "completed"
                except Exception as error:
                    attempt.status, attempt.error = (
                        "failed",
                        f"{type(error).__name__}: {error}",
                    )
                except BaseException as error:
                    attempt.status, attempt.error = (
                        "interrupted",
                        f"{type(error).__name__}: {error}",
                    )
                    raise
                finally:
                    if process is not None:
                        _stop(process)
                    attempt.process_seconds = time.monotonic() - child_started
                    attempt.finished_unix = time.time()
                    save()
                if record.paused:
                    break
                if schedule.active_runtime and attempt.status != "completed":
                    # Operational failures need explicit recovery selection. A
                    # silent next-seed launch can consume the entire allocation.
                    break
                queue.tick(sources())
            while True:
                check()
                queue.tick(sources(), launch=not record.paused)
                save()
                if queue.process is None:
                    break
                time.sleep(0.2)
            latest = {(a.case, a.seed): a for a in record.attempts}
            record.status = (
                "completed"
                if all(a.status == "completed" for a in latest.values())
                else "incomplete"
            )
            if queue.pending or any(a.status != "completed" for a in queue.attempts):
                record.status = "incomplete"
        except BaseException as error:
            record.status, record.error = (
                "incomplete",
                f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            if queue is not None:
                queue.close()
            save()
            record.report(out)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal Experiment training worker")
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    if os.getpid() != os.getpgrp():
        raise ValueError("internal worker requires an isolated process session")
    payload = json.loads(args.job.read_text())

    arm_worker_deadline(
        active_runtime=payload.get("active_runtime", False),
        allowance_seconds=payload["allowance_seconds"],
        deadline_unix=payload["deadline_unix"],
    )
    with VerifyStore(payload["store"]) as store:
        run = execute_regime(
            TrainingRegime.model_validate(payload["regime"]),
            payload["seed"],
            payload["out"],
            store,
            resume_from=payload["resume_from"],
            pause_path=Path(payload["pause_path"])
            if payload.get("pause_path")
            else None,
            checkpoint_seconds=payload["checkpoint_seconds"],
            checkpoint_updates=tuple(payload["checkpoint_updates"])
            if isinstance(payload.get("checkpoint_updates"), list)
            else payload.get("checkpoint_updates"),
        )
    if run.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
