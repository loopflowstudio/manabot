"""Evaluate MTG-181's exports on the 10×10 deck grid while the seeds train.

For each seed's job, the follower reads the committed generation through
`manabot.verify.learnability.elemental_remote.observe` (verified manifests and
digest-checked fetches), admits the initialization, every export at a multiple
of the frozen cadence, and the actual endpoint once the job is terminal with
confirmed cleanup. It plays each admitted checkpoint's grid once, oldest update
first, and never rents, changes or cancels training.

Grid results are the durable authority. After each evaluation the report under
`<directory>/report` is regenerated: Etude metric runs (one per seed, clocks
`updates` and `learner_transitions`) are rebuilt from those results beside the
training runs' own Etude records, and `index.html` adds each seed's grid tables,
seed spread, training counters and inclusive cost.

    uv run --extra artifacts python -m experiments.elemental_two_seat_follow \
        --directory .runs/mtg181/follow --until <unix-seconds>
"""

from __future__ import annotations

import argparse
import fcntl
from html import escape
from pathlib import Path
import shutil
import statistics
import time

from etude.run.records import Clock
from etude.train.report import pages, write
from etude.train.run import track
from pydantic import BaseModel

from experiments.elemental_grid import (
    DECKS,
    SCHEDULE_SHA256,
    Summary,
    evaluate,
    grid_table,
    schedule_sha256,
    summarize,
)
from experiments.elemental_two_seat import CADENCE, CAP_DOLLARS, JOBS
from manabot.infra.files import atomic_json
from manabot.remote.job_client import fetch_job_file, job_manifest, load_job
from manabot.remote.job_store import S3JobStore
from manabot.remote.jobs import JobRecord
from manabot.verify.learnability.elemental_remote import observe

EXPERIMENT = "mtg181-elemental-grid"
TERMINAL = ("completed", "paused", "failed", "cancelled", "deadline")


class Point(BaseModel):
    seed: int
    job: str
    training_run: str
    updates: int
    label: str
    checkpoint: str
    sha256: str
    coordinates: dict[str, object]
    phase: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    restarts: int = 0

    @property
    def key(self) -> str:
        return f"{self.seed}-{self.updates:06d}-{self.sha256[:12]}"


class JobView(BaseModel):
    phase: str = "not-accepted"
    updates: int = 0
    terminal: bool = False
    cleanup_confirmed: bool = False
    estimated_dollars: float = 0.0
    observed_at: str | None = None
    run: dict[str, object] | None = None


class Ledger(BaseModel):
    schedule_sha256: str
    points: dict[str, Point] = {}
    jobs: dict[str, JobView] = {}
    error: str | None = None
    observed_at: float = 0.0
    settled: bool = False


def _run_view(run: dict | None) -> dict[str, object] | None:
    """Counters the report needs; the full TrainingRun stays in the fetch cache."""
    if run is None:
        return None
    errors = run.get("game_errors") or {}
    return {
        "id": run["id"],
        "seed": run.get("seed"),
        "status": run.get("status"),
        "stages": [
            {k: v for k, v in stage.items() if k not in ("artifacts", "diagnostics")}
            for stage in run.get("stages", [])
        ],
        "game_errors": {
            k: v for k, v in errors.items() if k not in ("by_cell", "window")
        },
    }


def _training_etude(job: str, directory: Path) -> None:
    """Copy the pod's own Etude training run out of the committed generation."""
    spec = load_job(job)
    raw = S3JobStore(spec.prefix).read("runtime/record.json")
    if raw is None:
        return
    record = JobRecord.model_validate_json(raw.data)
    if record.manifest is None:
        return
    cache = directory / "cache" / job
    manifest = job_manifest(spec, record, cache)
    for entry in manifest.bundle.files:
        parts = Path(entry.relative_path).parts
        if "etude" not in parts:
            continue
        relative = Path(*parts[parts.index("etude") + 1 :])
        target = directory / "training-etude" / relative
        source = fetch_job_file(spec, record, entry.relative_path, cache)
        if source is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def _admit(seed: int, job: str, directory: Path, ledger: Ledger) -> None:
    observation = observe(job, directory / "cache" / job)
    status = observation["status"]
    record = status.get("record")
    view = JobView(
        phase=record["phase"] if record else "not-accepted",
        updates=record["updates"] if record else 0,
        cleanup_confirmed=status.get("cleanup") is not None,
        estimated_dollars=status["cleanup"]["estimated_dollars"]
        if status.get("cleanup")
        else observation["estimated_cumulative_dollars"],
        observed_at=observation["observed_at"],
        run=_run_view(observation.get("run")),
    )
    view.terminal = view.phase in TERMINAL
    ledger.jobs[job] = view
    found = observation["checkpoints"]
    if not found:
        return
    training_run = observation["run"]["id"]
    endpoint = max(c["coordinates"]["updates"] for c in found)
    for checkpoint in found:
        updates = checkpoint["coordinates"]["updates"]
        is_endpoint = view.terminal and view.cleanup_confirmed and updates == endpoint
        if updates % CADENCE and not is_endpoint:
            continue
        point = Point(
            seed=seed,
            job=job,
            training_run=training_run,
            updates=updates,
            label="initial"
            if updates == 0
            else "endpoint"
            if is_endpoint
            else "cadence",
            checkpoint=checkpoint["path"],
            sha256=checkpoint["sha256"],
            coordinates=checkpoint["coordinates"],
        )
        if point.key in ledger.points:
            if is_endpoint:
                ledger.points[point.key].label = "endpoint"
            continue
        ledger.points[point.key] = point


def _evaluate(point: Point, directory: Path, ledger: Ledger, path: Path) -> None:
    target = directory / "evaluations" / point.key
    if point.phase == "running":
        point.restarts += 1  # completed groups are kept; failed groups never replay
    point.phase = "running"
    point.started_at = point.started_at or time.time()
    atomic_json(path, ledger.model_dump(mode="json"))
    rows = evaluate(Path(point.checkpoint), point.sha256, point.seed, target / "groups")
    summary = summarize(rows)
    atomic_json(target / "summary.json", summary.model_dump(mode="json"))
    point.phase = "done"
    point.finished_at = time.time()
    atomic_json(path, ledger.model_dump(mode="json"))


def _summary(directory: Path, point: Point) -> Summary | None:
    path = directory / "evaluations" / point.key / "summary.json"
    if not path.exists():
        return None
    return Summary.model_validate_json(path.read_bytes())


# ---------------------------------------------------------------------------
# Report


def _etude(directory: Path, ledger: Ledger, report: Path) -> list[str]:
    """Rebuild grid metric runs from durable results, beside training runs."""
    staging = directory / "track.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for seed in sorted({p.seed for p in ledger.points.values()}):
        done = sorted(
            (p for p in ledger.points.values() if p.seed == seed and p.phase == "done"),
            key=lambda p: p.updates,
        )
        if not done:
            continue
        run = track(
            experiment=EXPERIMENT,
            run=f"grid-{seed}",
            seed=seed,
            kind="evaluation",
            clocks=[
                Clock(name="updates", kind="step", meaning="absolute training updates"),
                Clock(
                    name="learner_transitions",
                    kind="step",
                    meaning="collected learner decisions",
                ),
            ],
            config={
                "job": done[0].job,
                "training_run": done[0].training_run,
                "schedule_sha256": SCHEDULE_SHA256,
                "opponent": "heuristic",
            },
            directory=staging,
        )
        for point in done:
            summary = _summary(directory, point)
            if summary is None:
                continue
            values: dict[str, float] = {
                "score/overall": summary.overall.score or 0.0,
                "score/play": summary.model_seats["play"].score or 0.0,
                "score/draw": summary.model_seats["draw"].score or 0.0,
                "games/invalid": float(summary.overall.invalid),
            }
            for deck in DECKS:
                values[f"model_deck/{deck}"] = summary.model_decks[deck].score or 0.0
                values[f"heuristic_deck/{deck}"] = (
                    summary.heuristic_decks[deck].score or 0.0
                )
            run.log(
                values,
                updates=point.updates,
                learner_transitions=float(
                    point.coordinates.get("learner_transitions", 0)  # type: ignore[arg-type]
                ),
            )
        run.finish()
    # The pods' own training metric runs, retained in each committed generation.
    for training in (directory / "training-etude").glob("*/run.json"):
        shutil.copytree(training.parent, staging / training.parent.name)
    track_dir = directory / "track"
    shutil.rmtree(track_dir, ignore_errors=True)
    staging.replace(track_dir)
    names = []
    for name, page in pages(track_dir).items():
        write(page, report, refresh=300)
        names.append(name)
    return names


def _spread(directory: Path, ledger: Ledger) -> str:
    by_update: dict[int, dict[int, float]] = {}
    for point in ledger.points.values():
        summary = _summary(directory, point) if point.phase == "done" else None
        if summary is not None and summary.overall.score is not None:
            by_update.setdefault(point.updates, {})[point.seed] = summary.overall.score
    rows = "".join(
        f"<tr><td>{u:,}</td>"
        + "".join(
            f"<td>{scores[s]:.1%}</td>" if s in scores else "<td>—</td>"
            for s in sorted(JOBS)
        )
        + (
            f"<td>{statistics.mean(scores.values()):.1%}</td>"
            f"<td>{max(scores.values()) - min(scores.values()):.1%}</td>"
            if len(scores) == len(JOBS)
            else "<td>—</td><td>—</td>"
        )
        + "</tr>"
        for u, scores in sorted(by_update.items())
    )
    head = "".join(f"<th>Seed {s}</th>" for s in sorted(JOBS))
    return (
        "<h2>Overall score against the heuristic by update</h2>"
        f"<table><tr><th>Updates</th>{head}<th>Mean</th><th>Range</th></tr>{rows}</table>"
        "<p>Endpoints differ by seed; only shared cadence updates have a mean.</p>"
    )


def _training(ledger: Ledger) -> str:
    rows = []
    for seed, job in sorted(JOBS.items()):
        view = ledger.jobs.get(job, JobView())
        run = view.run or {}
        stage = next(iter(run.get("stages", [])), {})  # type: ignore[call-overload]
        errors = run.get("game_errors") or {}
        rows.append(
            "<tr>"
            f"<td>{seed}</td><td>{escape(view.phase)}</td><td>{view.updates:,}</td>"
            f"<td>{stage.get('learner_transitions', 0):,}</td>"
            f"<td>{stage.get('environment_decisions', 0):,}</td>"
            f"<td>{stage.get('games', 0):,}</td>"
            f"<td>{stage.get('collection_seconds', 0) / 3600:.2f}</td>"
            f"<td>{stage.get('learning_seconds', 0) / 3600:.2f}</td>"
            f"<td>{errors.get('crashed', 0)} / {errors.get('attempted', 0)}</td>"
            f"<td>${view.estimated_dollars:.3f}{'' if view.cleanup_confirmed else ' (running)'}</td>"
            "</tr>"
        )
    total = sum(v.estimated_dollars for v in ledger.jobs.values())
    return (
        "<h2>Training</h2><table><tr><th>Seed</th><th>Phase</th><th>Updates</th>"
        "<th>Learner decisions</th><th>Environment decisions</th><th>Games</th>"
        "<th>Collection h</th><th>Learning h</th><th>Quarantined / attempted games</th>"
        "<th>Estimated cost</th></tr>"
        + "".join(rows)
        + f"</table><p>Estimated inclusive cost ${total:.3f} of the ${CAP_DOLLARS:.0f} cap "
        "(intent through observed deletion; not an invoice). Counters come from the "
        "latest committed TrainingRun and lag the live job.</p>"
    )


def report(directory: Path, ledger: Ledger) -> None:
    target = directory / "report"
    target.mkdir(parents=True, exist_ok=True)
    try:
        etude = _etude(directory, ledger, target)
    except Exception as error:  # the grid report must survive Etude failures
        etude = []
        ledger.error = f"etude report: {type(error).__name__}: {error}"
    sections = []
    points = sorted(ledger.points.values(), key=lambda p: (p.seed, p.updates))
    for seed in sorted(JOBS):
        mine = [p for p in points if p.seed == seed]
        rows = []
        for point in mine:
            summary = _summary(directory, point)
            name = f"grid-{point.key}.html"
            if summary is not None:
                caption = (
                    f"Seed {seed}, {point.label} at update {point.updates:,}: "
                    "model score against the heuristic, 50 games per cell, 95% Wilson intervals"
                )
                (target / name).write_text(_page(caption, grid_table(summary, caption)))
            overall = summary.overall if summary else None
            rows.append(
                f"<tr><td>{point.updates:,}</td><td>{point.label}</td><td>{point.phase}</td>"
                f"<td>{point.coordinates.get('learner_transitions', 0):,}</td>"
                + (
                    f"<td>{overall.score:.1%} [{overall.low:.1%}, {overall.high:.1%}]</td>"
                    f"<td>{summary.model_seats['play'].score:.1%} / {summary.model_seats['draw'].score:.1%}</td>"
                    f'<td>{overall.invalid}</td><td><a href="{name}">grid</a></td>'
                    if summary and overall and overall.score is not None
                    else "<td>—</td><td>—</td><td>—</td><td></td>"
                )
                + "</tr>"
            )
        latest = next(
            (p for p in reversed(mine) if p.phase == "done"),
            None,
        )
        latest_grid = ""
        if latest is not None and (summary := _summary(directory, latest)):
            latest_grid = grid_table(
                summary,
                f"Seed {seed}, latest evaluated checkpoint ({latest.label}, update {latest.updates:,})",
            )
        sections.append(
            f"<h2>Seed {seed}</h2><table><tr><th>Update</th><th>Point</th><th>Phase</th>"
            "<th>Learner decisions</th><th>Overall [95%]</th><th>Play / draw</th>"
            "<th>Invalid</th><th></th></tr>" + "".join(rows) + "</table>" + latest_grid
        )
    links = " · ".join(
        f'<a href="{escape(n)}.html">Etude: {escape(n)}</a>' for n in etude
    )
    body = (
        "<h1>MTG-181: two-seat self-play on ten Elemental decks</h1>"
        f"<p>Updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(ledger.observed_at))}. "
        "Each evaluated checkpoint plays the frozen 10×10 grid against the heuristic: rows are the "
        "model's deck, columns the heuristic's, 50 games per cell with the seat drawn per game "
        f"from seed 18190001 (schedule {SCHEDULE_SHA256[:12]}). Win = 1, draw = ½; only valid, "
        "replay-checked games count. Intervals cover game noise for one checkpoint, not "
        "variation across training seeds.</p>"
        f"<p>{links}</p><p>{escape(ledger.error or '')}</p>"
        + _training(ledger)
        + _spread(directory, ledger)
        + "".join(sections)
    )
    partial = target / "index.html.partial"
    partial.write_text(_page("MTG-181 Elemental two-seat", body, refresh=300))
    partial.replace(target / "index.html")


def _page(title: str, body: str, *, refresh: int | None = None) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return (
        f'<!doctype html><meta charset="utf-8">{meta}<title>{escape(title)}</title>'
        "<style>body{font:15px system-ui;margin:2rem;background:#fbf8f3;color:#2b2520}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{padding:.3rem .5rem;"
        "border:1px solid #e3dcd2;text-align:center}table.grid td{min-width:4.2rem}"
        "small{display:block;color:#6b625a;font-size:.72rem}caption{text-align:left;"
        "font-weight:600;padding:.4rem 0}</style>" + body
    )


# ---------------------------------------------------------------------------
# Loop


def follow(directory: Path, until: float) -> None:
    if schedule_sha256() != SCHEDULE_SHA256:
        raise ValueError("grid schedule differs from the frozen digest")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ledger.json"
    with (directory / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = (
            Ledger.model_validate_json(path.read_bytes())
            if path.exists()
            else Ledger(schedule_sha256=SCHEDULE_SHA256)
        )
        polled = 0.0
        while time.time() < until and not ledger.settled:
            if time.time() - polled > 300:
                polled = time.time()
                ledger.error = None
                for seed, job in sorted(JOBS.items()):
                    try:
                        _admit(seed, job, directory, ledger)
                        _training_etude(job, directory)
                    except Exception as error:
                        ledger.error = f"{job}: {type(error).__name__}: {error}"
                ledger.observed_at = time.time()
                atomic_json(path, ledger.model_dump(mode="json"))
                report(directory, ledger)
            pending = sorted(
                (p for p in ledger.points.values() if p.phase != "done"),
                key=lambda p: (p.updates, p.seed),
            )
            if pending:
                _evaluate(pending[0], directory, ledger, path)
                ledger.observed_at = time.time()
                report(directory, ledger)
                continue
            # Terminal jobs with confirmed cleanup have admitted their endpoints.
            ledger.settled = all(
                ledger.jobs.get(job, JobView()).terminal
                and ledger.jobs[job].cleanup_confirmed
                for job in JOBS.values()
            )
            if not ledger.settled:
                time.sleep(60)
        atomic_json(path, ledger.model_dump(mode="json"))
        report(directory, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--until", type=float, required=True)
    args = parser.parse_args()
    follow(args.directory.resolve(), args.until)


if __name__ == "__main__":
    main()
