"""Follow exported cohort checkpoints with serial Etude CPU evaluations.

This observer cannot rent or mutate training. Verified manifests bind inputs;
per-digest intent is durable before starting a bounded child. Interrupted and
failed attempts remain visible and are never silently replayed. Launchd owns
this observer on the controller; Etude owns the evaluation records and tables.
"""

from __future__ import annotations

import fcntl
from html import escape
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from etude.eval.query import connect
from etude.eval.report import render
from etude.eval.tournament import Curve, curves
from pydantic import BaseModel

from manabot.arena.benchmark import BENCHMARK
from manabot.arena.monitor import (
    Checkpoint,
    MonitorProtocol,
    checkpoint_manifest,
    stage_checkpoint,
)
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json
from manabot.remote.cohort import Cohort, CohortState, load_cohort
from manabot.remote.job_client import fetch_job_file, job_manifest
from manabot.remote.job_store import S3JobStore
from manabot.remote.jobs import Job, JobRecord
from manabot.remote.supervisor import stop_process
from manabot.training.records import TrainingRun


class Attempt(BaseModel):
    digest: str
    projection: Literal["native", "retained-arena"] = "native"
    job: str
    updates: int
    learner_decisions: int
    training_seconds: float
    phase: Literal["running", "completed", "failed", "interrupted"]
    started_at: float
    finished_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    error: str | None = None


class Ledger(BaseModel):
    cohort_sha256: str
    attempts: dict[str, Attempt] = {}
    error: str | None = None
    observed_at: float = 0
    cohort_phase: str = "unknown"
    training_updates: dict[str, int] = {}
    training_heartbeats: dict[str, float] = {}
    settled_jobs: dict[str, str] = {}


def report(directory: Path, ledger: Ledger) -> None:
    """Atomically refresh the live page from Etude's retained match tables."""
    public = directory / "public"
    public.mkdir(exist_ok=True)
    summaries: list[Curve] = []
    if (directory / "track").exists():
        db = connect(directory / "track")
        try:
            count = db.execute("select count(*) from runs").fetchone()
            if count and count[0]:
                page = render(db)
                temporary = public / "eval.html.tmp"
                temporary.write_text(
                    page.replace(
                        "<head>", '<head><meta http-equiv="refresh" content="30">', 1
                    )
                )
                temporary.replace(public / "eval.html")
                tables = {row[0] for row in db.execute("show tables").fetchall()}
                if "plays" in tables:
                    summaries = curves(db, BENCHMARK, table="plays", value="score")
        finally:
            db.close()
    atomic_json(public / "status.json", ledger.model_dump(mode="json"))
    atomic_json(
        public / "curves.json", [curve.model_dump(mode="json") for curve in summaries]
    )
    score_rows: list[str] = []
    for curve in summaries:
        for point in curve.points:
            attempt = ledger.attempts.get(point.checkpoint)
            interval = point.interval
            bounds = (
                f"[{interval[0]:.1%}, {interval[1]:.1%}]" if interval else "unavailable"
            )
            score_rows.append(
                f"<tr><td>{attempt.updates if attempt else escape(point.name)}</td><td>{point.value:.1%}</td><td>{bounds}</td><td>{point.matches}/25 deals</td></tr>"
            )
    score_table = (
        "<h2>Heuristic score</h2><table><tr><th>Updates</th><th>Score</th><th>95% deal interval</th><th>Coverage</th></tr>"
        + "".join(score_rows)
        + "</table>"
    )
    rows = "".join(
        f"<tr><td>{a.updates}</td><td>{escape(a.phase)}</td><td>{escape(a.projection)}</td><td>{a.learner_decisions:,}</td><td>{a.training_seconds / 3600:.2f}</td><td>{a.digest[:12]}</td><td>{escape(a.error or '')}</td></tr>"
        for a in sorted(ledger.attempts.values(), key=lambda a: a.updates)
    )
    training = "; ".join(
        f"{escape(job)}: {updates:,} updates"
        + (
            f" (snapshot heartbeat {max(0, time.time() - ledger.training_heartbeats[job]) / 60:.1f} minutes ago)"
            if job in ledger.training_heartbeats
            else " (snapshot time unavailable)"
        )
        for job, updates in ledger.training_updates.items()
    )
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        "<title>Two-seat ladder — live Etude evaluations</title>"
        "<style>body{font:16px system-ui;margin:2rem;background:#fbf8f3;color:#2b2520}td,th{padding:.5rem;text-align:left}iframe{width:100%;height:1000px;border:0}</style>"
        "<h1>Two-seat ladder: live development report</h1>"
        f"<p>Updated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(ledger.observed_at))}. "
        "Etude evaluates each exported 250-update checkpoint and final paused/completed raw export: 25 fixed deals × eight deck/seat legs, heuristic, one CPU worker on the Mac mini. No evaluation rentals.</p>"
        "<p>Repeated development monitoring, one training seed, unmatched one-seat training cost. Intervals resample whole deals; partial checkpoints are labeled running. Training updates, learner decisions and wall time are different clocks.</p>"
        f"<p>Cohort: {escape(ledger.cohort_phase)}. Latest training snapshot: {training}.</p>{score_table}<p>{escape(ledger.error or '')}</p><table><tr><th>Updates</th><th>Evaluation</th><th>Evidence</th><th>Learner decisions</th><th>Training hours</th><th>Checkpoint SHA</th><th>Error</th></tr>{rows}</table>"
        '<p><a href="eval.html">Etude evaluation report</a> · <a href="curves.json">Curves JSON</a> · <a href="status.json">Status JSON</a></p><iframe src="eval.html"></iframe>'
    )
    temporary = public / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(public / "index.html")


def _refresh(directory: Path, ledger: Ledger) -> None:
    # Reporting failure cannot terminate a valid evaluator or its child games.
    try:
        report(directory, ledger)
    except Exception as error:
        ledger.error = f"report unavailable: {type(error).__name__}: {error}"
        atomic_json(directory / "ledger.json", ledger.model_dump(mode="json"))
        public = directory / "public"
        public.mkdir(exist_ok=True)
        atomic_json(public / "status.json", ledger.model_dump(mode="json"))


def _checkpoints(run: TrainingRun) -> list[Checkpoint]:
    """Milestones and actual budget endpoints; identical artifacts are scored once."""
    found = {
        item.artifact["sha256"]: Checkpoint(artifact=item.artifact, coordinates=item)
        for item in run.monitoring_checkpoints
        if item.artifact is not None and item.updates > 0 and item.updates % 250 == 0
    }
    trained = {
        stage.id for stage in run.regime.stages if stage.operation == "train_self_play"
    }
    for stage in run.stages:
        if stage.id in trained:
            final = stage_checkpoint(run, stage.id)
            if final is not None and final.coordinates.updates > 0:
                found.setdefault(final.artifact["sha256"], final)
    return sorted(found.values(), key=lambda item: item.coordinates.updates)


def _observe(spec: Job, ledger: Ledger) -> JobRecord | None:
    """Refresh small supervisor metadata without changing a pinned evaluation input."""
    raw = S3JobStore(spec.prefix).read("runtime/record.json")
    if raw is None:
        return None
    record = JobRecord.model_validate_json(raw.data)
    if record.spec_sha256 != spec.identity:
        raise ValueError("training observation differs from cohort job")
    ledger.training_updates[spec.job_id] = record.updates
    ledger.training_heartbeats[spec.job_id] = record.heartbeat_at
    return record


def evaluate_exports(
    cohort: Cohort, state: CohortState, directory: Path, ledger: Ledger
) -> None:
    """Discover raw checkpoints from a single pinned generation per job."""
    for attempt in state.attempts:
        spec = attempt.spec
        record = _observe(spec, ledger)
        if record is None:
            continue
        if record.manifest is None:
            continue
        cache = directory / "cache"
        manifest = job_manifest(spec, record, cache)
        entries = {entry.relative_path: entry for entry in manifest.bundle.files}
        run_entry = entries.get("run/run.json")
        unseen_final = (
            record.terminal
            and record.artifacts_complete
            and run_entry is not None
            and ledger.settled_jobs.get(spec.job_id) != run_entry.sha256
        )
        if not unseen_final and not any(
            Path(name).name.startswith("monitor-")
            and name.endswith(".pt")
            and entry.sha256 not in ledger.attempts
            for name, entry in entries.items()
        ):
            continue
        run_path = fetch_job_file(spec, record, "run/run.json", cache)
        if run_path is None:
            continue
        run = TrainingRun.model_validate_json(run_path.read_bytes())
        for exported in _checkpoints(run):
            checkpoint = exported.coordinates
            if time.time() >= cohort.deadline + 3600:
                return
            artifact = exported.artifact
            if artifact["sha256"] in ledger.attempts:
                continue
            name = "run/" + Path(artifact["path"]).name
            entry = entries.get(name)
            if (
                entry is None
                or entry.sha256 != artifact["sha256"]
                or entry.size != artifact["bytes"]
            ):
                raise ValueError("checkpoint receipt differs from committed snapshot")
            path = fetch_job_file(spec, record, name, cache)
            if path is None:
                raise ValueError("committed checkpoint unavailable")
            verify_file(path, artifact["sha256"], artifact["bytes"])
            admission = checkpoint_manifest(
                run,
                {**artifact, "path": str(path.resolve())},
                checkpoint,
                MonitorProtocol(
                    deal_seeds=tuple(range(1_590_100_000, 1_590_100_025)),
                    include_mirrors=True,
                    opponent="heuristic",
                    game_seconds=600,
                    bootstrap_seed=17,
                    bootstrap_replicates=4000,
                ),
            )
            ordinal = 1 + sum(
                item.digest == artifact["sha256"] for item in ledger.attempts.values()
            )
            target = (
                directory / "evaluations" / f"{artifact['sha256']}-attempt-{ordinal}"
            )
            target.mkdir(parents=True, exist_ok=True)
            atomic_json(target / "admission.json", admission.model_dump(mode="json"))
            item = Attempt(
                digest=artifact["sha256"],
                job=spec.job_id,
                updates=checkpoint.updates,
                learner_decisions=checkpoint.learner_transitions,
                training_seconds=checkpoint.training_seconds,
                phase="running",
                started_at=time.time(),
            )
            ledger.attempts[item.digest] = item
            atomic_json(directory / "ledger.json", ledger.model_dump(mode="json"))
            with (target / "console.log").open("ab") as console:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "manabot.arena.benchmark",
                        "--manifest",
                        str(target / "admission.json"),
                    ],
                    stdout=console,
                    stderr=console,
                    start_new_session=True,
                )
                item.pid = child.pid
                atomic_json(directory / "ledger.json", ledger.model_dump(mode="json"))
                try:
                    deadline = time.monotonic() + min(
                        3600, max(0, cohort.deadline + 3600 - time.time())
                    )
                    while child.poll() is None and time.monotonic() < deadline:
                        try:
                            _observe(spec, ledger)
                        except Exception as error:
                            ledger.error = f"training status unavailable: {type(error).__name__}: {error}"
                        ledger.observed_at = time.time()
                        _refresh(directory, ledger)
                        try:
                            child.wait(
                                timeout=min(30, max(0, deadline - time.monotonic()))
                            )
                        except subprocess.TimeoutExpired:
                            pass
                    if child.poll() is None:
                        stop_process(child)
                        item.error = "one-hour evaluation deadline reached"
                    item.exit_code = child.wait()
                    item.phase = (
                        "completed"
                        if item.exit_code == 0 and (target / "result.json").is_file()
                        else "failed"
                    )
                    item.finished_at = time.time()
                finally:
                    if child.poll() is None:
                        stop_process(child)
                    atomic_json(
                        directory / "ledger.json", ledger.model_dump(mode="json")
                    )
            _refresh(directory, ledger)
        if unseen_final and run_entry is not None:
            ledger.settled_jobs[spec.job_id] = run_entry.sha256


def follow(
    cohort_id: str, directory: Path, *, retry_checkpoints: tuple[str, ...] = ()
) -> None:
    """Follow until cohort terminal, at most one hour beyond its absolute deadline."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(cohort_id)
    with (directory / "owner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "ledger.json"
        ledger = (
            Ledger.model_validate_json(path.read_bytes())
            if path.exists()
            else Ledger(cohort_sha256=cohort.identity)
        )
        if ledger.cohort_sha256 != cohort.identity:
            raise ValueError("evaluation owner bound to another cohort")
        for attempt in ledger.attempts.values():
            if attempt.phase == "running":
                if attempt.pid is not None:
                    try:
                        os.kill(attempt.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError(
                            "prior evaluator PID still exists; reconcile before restarting"
                        )
                attempt.phase = "interrupted"
                attempt.error = "observer restarted; retained attempt not replayed"
        for digest in retry_checkpoints:
            previous = ledger.attempts.get(digest)
            if previous is None or previous.phase not in {"failed", "interrupted"}:
                raise ValueError(
                    "explicit retry requires a retained failed or interrupted checkpoint"
                )
            ledger.attempts[f"{digest}-retired-{previous.started_at:.6f}"] = (
                ledger.attempts.pop(digest)
            )
            # A final raw export is discovered through the settled run cursor,
            # rather than a monitor filename. Explicit retry must revisit it.
            ledger.settled_jobs.pop(previous.job, None)
        atomic_json(path, ledger.model_dump(mode="json"))
        observer_deadline = cohort.deadline + 3600
        while time.time() < observer_deadline:
            terminal = False
            try:
                raw = S3JobStore(cohort.prefix).read("state.json")
                if raw is None:
                    raise ValueError("cohort state unavailable")
                state = CohortState.model_validate_json(raw.data)
                if state.cohort_sha256 != cohort.identity:
                    raise ValueError("cohort state identity differs")
                ledger.error = None
                ledger.cohort_phase = state.phase
                evaluate_exports(cohort, state, directory, ledger)
                terminal = state.phase not in {"running", "pending"}
            except Exception as error:
                ledger.error = f"{type(error).__name__}: {error}"
            ledger.observed_at = time.time()
            atomic_json(path, ledger.model_dump(mode="json"))
            _refresh(directory, ledger)
            if terminal:
                break
            time.sleep(60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--retry-checkpoint", action="append", default=[])
    args = parser.parse_args()
    follow(args.cohort, args.directory, retry_checkpoints=tuple(args.retry_checkpoint))
