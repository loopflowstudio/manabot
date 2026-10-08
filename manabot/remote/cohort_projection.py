"""Bounded report/W&B delivery, independently supervised from cohort scheduling.

Only existing Dashboard exports are downloaded. Local reports precede telemetry;
failed publication retains evidence and retries against W&B's acknowledged cursor.
A child process bounds network/render time, and reservations survive owner restart.
"""

import fcntl
from html import escape
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from threading import Timer
import time
from typing import Protocol

from pydantic import Field, model_validator

from manabot.infra.artifacts import verify_file
from manabot.training.execution import atomic_json
from manabot.training.monitoring import Dashboard, publish_dashboard

from .cohort import Cohort, CohortState
from .cohort_service import owner_lock
from .job_client import fetch_job_file, job_manifest
from .job_store import S3JobStore
from .jobs import JobRecord
from .plan import Frozen
from .supervisor import stop_process


class DashboardPublisher(Protocol):
    def __call__(
        self,
        dashboard: Dashboard,
        out: Path,
        *,
        project: str,
        entity: str | None = None,
    ) -> str | None: ...


class ProjectionConfig(Frozen):
    """Reporting allowance is inside the cohort's declared controller allocation."""

    project: str | None = None
    entity: str | None = None
    interval_seconds: float = Field(default=300, ge=30)
    attempt_seconds: float = Field(default=120, ge=15, le=600)
    total_seconds: float = Field(default=3600, gt=0)

    @model_validator(mode="after")
    def valid(self) -> "ProjectionConfig":
        if (self.project is None) != (self.entity is None):
            raise ValueError("W&B projection requires both project and entity")
        if self.attempt_seconds > self.total_seconds:
            raise ValueError("projection attempt exceeds total allowance")
        return self


class ProjectionAttempt(Frozen):
    started_at: float
    reserved_seconds: float
    elapsed_seconds: float | None = None
    exit_code: int | None = None

    @property
    def charged_seconds(self) -> float:
        return (
            self.reserved_seconds
            if self.elapsed_seconds is None
            else self.elapsed_seconds
        )


class ProjectionLedger(Frozen):
    cohort_sha256: str
    config: ProjectionConfig
    attempts: tuple[ProjectionAttempt, ...] = ()

    @property
    def charged_seconds(self) -> float:
        return sum(a.charged_seconds for a in self.attempts)


class ProjectionRow(Frozen):
    job_id: str
    phase: str
    generation: int = 0
    heartbeat_at: float | None = None
    updates: int = 0
    evaluations: int = 0
    dashboards: tuple[str, ...] = ()
    error: str | None = None


def _claim(cohort: Cohort, directory: Path, config: ProjectionConfig) -> None:
    # Report publication needs the same exclusive-host policy as scheduling.
    # This short lock obtains the identity; publish.lock below owns the whole pass.
    with owner_lock(directory / "identity") as owner:
        data = json.dumps(
            {
                "owner": owner,
                "config": config.model_dump(),
                "cohort_sha256": cohort.identity,
            },
            sort_keys=True,
        ).encode()
        store = S3JobStore(cohort.prefix)
        if not store.create("projection-owner.json", data):
            value = store.read("projection-owner.json")
            if value is None or value.data != data:
                raise ValueError("projection has another owner or configuration")


def _write_report(
    cohort: Cohort, state: CohortState, rows: list[ProjectionRow], directory: Path
) -> None:
    atomic_json(
        directory / "report.json",
        {
            "cohort_sha256": cohort.identity,
            "observed_at": time.time(),
            "cohort_phase": state.phase,
            "intended_jobs": len(cohort.entries),
            "admitted_jobs": len(state.attempts),
            "supervisor_heartbeat_at": state.heartbeat_at,
            "charged_or_reserved_dollars": state.charged_dollars(cohort),
            "jobs": [row.model_dump(mode="json") for row in rows],
        },
    )
    body = [
        f"<h1>{escape(cohort.cohort_id)}</h1>",
        f"<p>Admitted {len(state.attempts)} of {len(cohort.entries)} intended jobs.</p>",
        f"<p>Saved at {time.time():.0f} Unix seconds. Cohort: {escape(state.phase)}. "
        f"Supervisor heartbeat: {state.heartbeat_at:.0f}. Charged/reserved: ${state.charged_dollars(cohort):.4f}.</p>",
        "<p>Reload to see refreshed saved observations. Development monitoring is not a completed scientific comparison.</p>",
        "<table><tr><th>Job</th><th>Phase</th><th>Updates</th><th>Evaluations</th><th>Generation / heartbeat (Unix seconds)</th><th>Saved dashboards / error</th></tr>",
    ]
    for row in rows:
        links = " ".join(
            f'<a href="{escape(path, quote=True)}">dashboard</a>'
            for path in row.dashboards
        )
        body.append(
            f"<tr><td>{escape(row.job_id)}</td><td>{escape(row.phase)}</td><td>{row.updates}</td><td>{row.evaluations}</td><td>{row.generation} / {row.heartbeat_at}</td><td>{links} {escape(row.error or '')}</td></tr>"
        )
    body.append(
        "</table><p>Full notebook/HTML analysis remains available through deploy fetch/report. W&B uses these same retained Dashboard rows.</p>"
    )
    temporary = directory / "report.html.tmp"
    temporary.write_text(
        '<!doctype html><meta charset="utf-8"><title>manabot cohort</title><style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem}table{border-collapse:collapse}td,th{padding:.6rem;border-bottom:1px solid #bbb;text-align:left}</style>'
        + "".join(body)
    )
    temporary.replace(directory / "report.html")


def project_once(
    cohort: Cohort,
    directory: Path,
    config: ProjectionConfig,
    *,
    publish: DashboardPublisher = publish_dashboard,
) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "publish.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _claim(cohort, directory, config)
        raw = S3JobStore(cohort.prefix).read("state.json")
        if raw is None:
            raise ValueError("cohort supervisor state missing")
        state = CohortState.model_validate_json(raw.data)
        if state.cohort_sha256 != cohort.identity:
            raise ValueError("cohort state identity differs")
        rows: list[ProjectionRow] = []
        pending: list[tuple[Dashboard, Path]] = []
        for attempt in state.attempts:
            spec = attempt.spec
            row = ProjectionRow(job_id=spec.job_id, phase="unavailable")
            try:
                value = S3JobStore(spec.prefix).read("runtime/record.json")
                if value is None:
                    raise ValueError("job record missing")
                record = JobRecord.model_validate_json(value.data)
                manifest = job_manifest(spec, record, directory / "cache")
                saved: list[str] = []
                for entry in manifest.bundle.files:
                    if entry.relative_path != "training-dashboard.json" and not (
                        entry.relative_path.startswith("monitoring/run-")
                        and Path(entry.relative_path).name.startswith("dashboard")
                        and entry.relative_path.endswith(".json")
                    ):
                        continue
                    # monitoring/dashboard.json is the queue index, not a
                    # Dashboard. Per-run files also include protocol suffixes.
                    source = fetch_job_file(
                        spec, record, entry.relative_path, directory / "cache"
                    )
                    assert source is not None
                    dashboard = Dashboard.model_validate_json(source.read_bytes())
                    target = directory / spec.job_id / f"{entry.sha256}.json"
                    target.parent.mkdir(exist_ok=True)
                    if target.exists():
                        verify_file(target, entry.sha256, entry.size)
                    else:
                        with tempfile.TemporaryDirectory(
                            dir=target.parent
                        ) as temporary:
                            staged = Path(temporary) / "dashboard.json"
                            shutil.copyfile(source, staged)
                            staged.replace(target)
                    saved.append(target.relative_to(directory).as_posix())
                    pending.append((dashboard, target))
                row = ProjectionRow(
                    job_id=spec.job_id,
                    phase=record.phase,
                    generation=record.generation,
                    heartbeat_at=record.heartbeat_at,
                    updates=record.updates,
                    evaluations=record.evaluations_completed,
                    dashboards=tuple(saved),
                )
            except Exception as error:
                row = row.model_copy(update={"error": type(error).__name__})
            rows.append(row)
        # Always persist local evidence/report before optional network telemetry.
        _write_report(cohort, state, rows, directory)
        failures: list[str] = []
        if config.project is not None:
            for dashboard, path in pending:
                if path.with_suffix(".publication.json").exists():
                    continue
                try:
                    url = publish(
                        dashboard,
                        directory,
                        project=config.project,
                        entity=config.entity,
                    )
                    atomic_json(
                        path.with_suffix(".publication.json"),
                        {"url": url, "at": time.time()},
                    )
                except Exception as error:
                    failures.append(type(error).__name__)
        history = directory / "telemetry-history"
        history.mkdir(exist_ok=True)
        atomic_json(
            history / f"{time.time_ns()}.json",
            {
                "at": time.time(),
                "enabled": config.project is not None,
                "errors": failures,
            },
        )
        atomic_json(
            directory / "telemetry.json",
            {
                "at": time.time(),
                "enabled": config.project is not None,
                "errors": failures,
            },
        )
        if failures or any(row.error for row in rows):
            raise RuntimeError("projection incomplete; local evidence retained")


def project_before(
    cohort: Cohort,
    directory: Path,
    config: ProjectionConfig,
    deadline: float,
) -> None:
    """Own a private session and deadline, even if the companion parent dies."""
    if os.getsid(0) != os.getpid():
        os.setsid()  # Fail if this caller cannot isolate itself; never kill a shell group.
    group = os.getpgrp()
    timer = Timer(
        max(0.01, deadline - time.time()), lambda: os.killpg(group, signal.SIGKILL)
    )
    timer.daemon = True
    timer.start()
    try:
        project_once(cohort, directory, config)
    finally:
        timer.cancel()


def follow_projection(
    cohort: Cohort, directory: Path, config: ProjectionConfig, command: list[str]
) -> None:
    with owner_lock(directory / "service"):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / "ledger.json"
        ledger = ProjectionLedger(cohort_sha256=cohort.identity, config=config)
        if path.exists():
            ledger = ProjectionLedger.model_validate_json(path.read_bytes())
            if ledger.cohort_sha256 != cohort.identity or ledger.config != config:
                raise ValueError(
                    "projection allocation already binds different content"
                )
        while time.time() < cohort.deadline + 1800:
            remaining = config.total_seconds - ledger.charged_seconds
            if remaining < config.attempt_seconds:
                return
            attempt = ProjectionAttempt(
                started_at=time.time(), reserved_seconds=config.attempt_seconds
            )
            ledger = ledger.model_copy(update={"attempts": (*ledger.attempts, attempt)})
            atomic_json(path, ledger.model_dump(mode="json"))
            started = time.monotonic()
            code = -1
            with (directory / "projection.log").open("ab") as log:
                child = subprocess.Popen(
                    [
                        *command,
                        "--deadline",
                        str(
                            min(
                                attempt.started_at + config.attempt_seconds,
                                cohort.deadline + 1800,
                            )
                            - 10
                        ),
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                try:
                    code = child.wait(
                        timeout=min(
                            config.attempt_seconds,
                            max(1, cohort.deadline + 1800 - time.time()),
                        )
                    )
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    stop_process(child)
            attempt = attempt.model_copy(
                update={
                    "elapsed_seconds": time.monotonic() - started,
                    "exit_code": code,
                }
            )
            ledger = ledger.model_copy(
                update={"attempts": (*ledger.attempts[:-1], attempt)}
            )
            atomic_json(path, ledger.model_dump(mode="json"))
            if code == 0:
                raw = S3JobStore(cohort.prefix).read("state.json")
                if (
                    raw is not None
                    and CohortState.model_validate_json(raw.data).settled
                ):
                    return
            time.sleep(config.interval_seconds)
