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
from etude.eval.tournament import curves
from pydantic import BaseModel

from manabot.infra.artifacts import verify_file
from manabot.remote.cohort import Cohort, CohortState, load_cohort
from manabot.remote.job_client import fetch_job_file, job_manifest
from manabot.remote.job_store import S3JobStore
from manabot.remote.jobs import JobRecord
from manabot.remote.supervisor import stop_process
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorProtocol, checkpoint_manifest


class Attempt(BaseModel):
    digest: str
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


def report(directory: Path, ledger: Ledger) -> None:
    """Atomically refresh the live page from Etude's retained match tables."""
    public = directory / "public"
    public.mkdir(exist_ok=True)
    summaries: list[dict[str, object]] = []
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
                summaries = [
                    c.model_dump(mode="json")
                    for c in curves(
                        db, "two-seat-ladder-heuristic", table="plays", value="score"
                    )
                ]
        finally:
            db.close()
    atomic_json(public / "status.json", ledger.model_dump(mode="json"))
    atomic_json(public / "curves.json", summaries)
    rows = "".join(
        f"<tr><td>{a.updates}</td><td>{escape(a.phase)}</td><td>{a.learner_decisions:,}</td><td>{a.training_seconds / 3600:.2f}</td><td>{a.digest[:12]}</td><td>{escape(a.error or '')}</td></tr>"
        for a in sorted(ledger.attempts.values(), key=lambda a: a.updates)
    )
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        "<title>Two-seat ladder — live Etude evaluations</title>"
        "<style>body{font:16px system-ui;margin:2rem;background:#fbf8f3;color:#2b2520}td,th{padding:.5rem;text-align:left}iframe{width:100%;height:1000px;border:0}</style>"
        "<h1>Two-seat ladder: live development report</h1>"
        f"<p>Updated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(ledger.observed_at))}. "
        "Etude evaluates each exported 250-update checkpoint: 25 fixed deals × eight deck/seat legs, heuristic, one CPU worker on the Mac mini. No evaluation rentals.</p>"
        "<p>Repeated development monitoring, one training seed, unmatched one-seat training cost. Intervals resample whole deals; partial checkpoints are labeled running. Training updates, learner decisions and wall time are different clocks.</p>"
        f"<p>{escape(ledger.error or '')}</p><table><tr><th>Updates</th><th>Evaluation</th><th>Learner decisions</th><th>Training hours</th><th>Checkpoint SHA</th><th>Error</th></tr>{rows}</table>"
        '<p><a href="eval.html">Etude evaluation report</a> · <a href="curves.json">Curves JSON</a> · <a href="status.json">Status JSON</a></p><iframe src="eval.html"></iframe>'
    )
    temporary = public / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(public / "index.html")


def evaluate_exports(
    cohort: Cohort, state: CohortState, directory: Path, ledger: Ledger
) -> None:
    """Discover raw checkpoints from a single pinned generation per job."""
    for attempt in state.attempts:
        spec = attempt.spec
        raw = S3JobStore(spec.prefix).read("runtime/record.json")
        if raw is None:
            continue
        record = JobRecord.model_validate_json(raw.data)
        if record.manifest is None:
            continue
        cache = directory / "cache"
        manifest = job_manifest(spec, record, cache)
        entries = {entry.relative_path: entry for entry in manifest.bundle.files}
        if not any(
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
        for checkpoint in run.monitoring_checkpoints:
            artifact = checkpoint.artifact
            if (
                checkpoint.updates <= 0
                or checkpoint.updates % 250
                or artifact is None
                or artifact["sha256"] in ledger.attempts
            ):
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
            target = directory / "evaluations" / artifact["sha256"]
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
                        "manabot.training.etude_evaluation",
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
                    deadline = time.monotonic() + 3600
                    while child.poll() is None and time.monotonic() < deadline:
                        ledger.observed_at = time.time()
                        report(directory, ledger)
                        try:
                            child.wait(timeout=30)
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
            report(directory, ledger)


def follow(cohort_id: str, directory: Path) -> None:
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
                evaluate_exports(cohort, state, directory, ledger)
                terminal = state.phase not in {"running", "pending"}
            except Exception as error:
                ledger.error = f"{type(error).__name__}: {error}"
            ledger.observed_at = time.time()
            atomic_json(path, ledger.model_dump(mode="json"))
            report(directory, ledger)
            if terminal:
                break
            time.sleep(60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    follow(args.cohort, args.directory)
