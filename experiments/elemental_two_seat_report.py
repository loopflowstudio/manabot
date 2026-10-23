"""Summarize MTG-181 from the follower's durable evidence into one JSON record.

Reads the follower directory (ledger, grid summaries, fetched TrainingRuns) and
each job's settled status, and writes `summary.json`: per seed, learner and
environment decisions, games, collection/learning time, throughput, game-error
counts, inclusive cost and every evaluated grid point; across seeds, the score
spread at shared cadence updates and at each endpoint. It plays no games and
rents nothing.

    uv run --extra artifacts python -m experiments.elemental_two_seat_report \
        --directory .runs/mtg181/follow --out .runs/mtg181/summary.json
"""

import argparse
from pathlib import Path
import statistics

from pydantic import BaseModel

from experiments.elemental_grid import Score, Summary
from experiments.elemental_two_seat import CADENCE, FAILED_JOBS, JOBS
from experiments.elemental_two_seat_follow import FAILED_PROBE_DOLLARS, Ledger, Point
from manabot.remote.job_client import job_status, load_job


class Grid(BaseModel):
    """One checkpoint's 10x10 heuristic grid, reduced to its margins."""

    overall: Score
    play: Score
    draw: Score
    model_decks: dict[str, Score]
    heuristic_decks: dict[str, Score]


class PointRecord(BaseModel):
    updates: int
    label: str
    phase: str
    sha256: str
    coordinates: dict[str, object]
    evaluation_seconds: float | None
    # None until the follower has finished this point's grid.
    grid: Grid | None


class SeedRecord(BaseModel):
    seed: int
    job: str
    training_run: str | None
    phase: str | None
    updates: int | None
    # Run counters come from the follower's last TrainingRun fetch, which can
    # lag the live job status above.
    counters_observed_at: str | None
    counters_updates: int | None
    cleanup_confirmed: bool
    estimated_dollars: float | None
    learner_decisions: int
    environment_decisions: int
    games: int
    optimizer_exposures: object
    optimizer_updates: object
    # Wall time, settled only when the stage ends.
    training_seconds: float | None
    collection_seconds: float | None
    learning_seconds: float | None
    # Collection + learning, so throughput is defined while the stage runs.
    active_seconds: float | None
    learner_decisions_per_active_hour: float | None
    game_errors: object
    error: object
    points: list[PointRecord]

    def score(self, label: str) -> float | None:
        point = next((p for p in self.points if p.label == label), None)
        return point.grid.overall.score if point and point.grid else None


class Spread(BaseModel):
    n: int
    mean: float
    min: float
    max: float
    sd: float


class Report(BaseModel):
    seeds: list[SeedRecord]
    failed_jobs: dict[int, str]
    failed_probe_dollars: float
    estimated_inclusive_dollars: float
    # Cadence updates every seed has evaluated; endpoints differ by seed.
    shared_updates: dict[int, Spread]
    endpoint_score: Spread | None
    endpoint_gain: Spread | None


def _grid(directory: Path, point: Point) -> Grid | None:
    path = directory / "evaluations" / point.key / "summary.json"
    if not path.exists():
        return None
    summary = Summary.model_validate_json(path.read_bytes())
    return Grid(
        overall=summary.overall,
        play=summary.model_seats["play"],
        draw=summary.model_seats["draw"],
        model_decks=summary.model_decks,
        heuristic_decks=summary.heuristic_decks,
    )


def _point(directory: Path, point: Point) -> PointRecord:
    started, finished = point.started_at, point.finished_at
    return PointRecord(
        updates=point.updates,
        label=point.label,
        phase=point.phase,
        sha256=point.sha256,
        coordinates=point.coordinates,
        evaluation_seconds=finished - started
        if started is not None and finished is not None
        else None,
        grid=_grid(directory, point),
    )


def _seed(directory: Path, ledger: Ledger, seed: int, job: str) -> SeedRecord:
    status = job_status(load_job(job))
    record = status.record
    view = ledger.jobs.get(job)
    run = (view.run if view else None) or {}
    stage = next(iter(run.get("stages", [])), {})  # type: ignore[call-overload]
    learner = int(stage.get("learner_transitions") or 0)
    active = float(stage.get("collection_seconds") or 0) + float(
        stage.get("learning_seconds") or 0
    )
    points = sorted(
        (p for p in ledger.points.values() if p.seed == seed), key=lambda p: p.updates
    )
    return SeedRecord(
        seed=seed,
        job=job,
        training_run=run.get("id"),  # type: ignore[arg-type]
        phase=record.phase if record else None,
        updates=record.updates if record else None,
        counters_observed_at=view.observed_at if view else None,
        counters_updates=view.updates if view else None,
        cleanup_confirmed=status.cleanup is not None,
        estimated_dollars=status.cleanup.estimated_dollars
        if status.cleanup is not None
        else (view.estimated_dollars if view else None),
        learner_decisions=learner,
        environment_decisions=int(stage.get("environment_decisions") or 0),
        games=int(stage.get("games") or 0),
        optimizer_exposures=stage.get("optimizer_exposures"),
        optimizer_updates=stage.get("optimizer_updates"),
        training_seconds=stage.get("seconds")
        if stage.get("status") != "running"
        else None,
        collection_seconds=stage.get("collection_seconds"),
        learning_seconds=stage.get("learning_seconds"),
        active_seconds=active or None,
        learner_decisions_per_active_hour=learner / active * 3600 if active else None,
        game_errors=run.get("game_errors"),
        error=stage.get("error"),
        points=[_point(directory, p) for p in points],
    )


def _spread(values: list[float]) -> Spread | None:
    if not values:
        return None
    return Spread(
        n=len(values),
        mean=statistics.mean(values),
        min=min(values),
        max=max(values),
        sd=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def summarize(directory: Path) -> Report:
    ledger = Ledger.model_validate_json((directory / "ledger.json").read_bytes())
    seeds = [_seed(directory, ledger, s, j) for s, j in sorted(JOBS.items())]
    by_update: dict[int, list[float]] = {}
    for seed in seeds:
        for p in seed.points:
            if p.grid and p.grid.overall.score is not None and p.updates % CADENCE == 0:
                by_update.setdefault(p.updates, []).append(p.grid.overall.score)
    endpoints = [s for s in (seed.score("endpoint") for seed in seeds) if s is not None]
    gains = [
        end - start
        for seed in seeds
        if (start := seed.score("initial")) is not None
        and (end := seed.score("endpoint")) is not None
    ]
    return Report(
        seeds=seeds,
        failed_jobs=FAILED_JOBS,
        failed_probe_dollars=FAILED_PROBE_DOLLARS,
        estimated_inclusive_dollars=sum(s.estimated_dollars or 0 for s in seeds)
        + FAILED_PROBE_DOLLARS,
        shared_updates={
            u: spread
            for u, v in sorted(by_update.items())
            if len(v) == len(JOBS) and (spread := _spread(v))
        },
        endpoint_score=_spread(endpoints),
        endpoint_gain=_spread(gains),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(summarize(args.directory).model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
