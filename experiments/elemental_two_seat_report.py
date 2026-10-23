"""Summarize MTG-181 from the follower's durable evidence into one JSON record.

Reads the follower directory (ledger, grid summaries, fetched TrainingRuns) and
each job's settled status, and writes `summary.json`: per seed, learner and
environment decisions, games, collection/learning time, throughput, game-error
counts, inclusive cost and every evaluated grid point; across seeds, the score
spread at shared updates and at each endpoint. It plays no games and rents
nothing.

    uv run --extra artifacts python -m experiments.elemental_two_seat_report \
        --directory .runs/mtg181/follow --out .runs/mtg181/summary.json
"""

import argparse
import json
from pathlib import Path
import statistics

from experiments.elemental_grid import DECKS, Summary
from experiments.elemental_two_seat import FAILED_JOBS, JOBS
from experiments.elemental_two_seat_follow import FAILED_PROBE_DOLLARS, Ledger
from manabot.remote.job_client import job_status, load_job


def _seed(directory: Path, ledger: Ledger, seed: int, job: str) -> dict[str, object]:
    status = job_status(load_job(job))
    record = status.record
    view = ledger.jobs.get(job)
    run = (view.run if view else None) or {}
    stage = next(iter(run.get("stages", [])), {})  # type: ignore[call-overload]
    learner = int(stage.get("learner_transitions") or 0)
    # The stage's wall `seconds` is settled only when the stage ends; throughput
    # uses active collection + learning time so it is defined while running.
    active = float(stage.get("collection_seconds") or 0) + float(
        stage.get("learning_seconds") or 0
    )
    points = []
    for point in sorted(
        (p for p in ledger.points.values() if p.seed == seed), key=lambda p: p.updates
    ):
        path = directory / "evaluations" / point.key / "summary.json"
        summary = (
            Summary.model_validate_json(path.read_bytes()) if path.exists() else None
        )
        points.append(
            {
                "updates": point.updates,
                "label": point.label,
                "phase": point.phase,
                "sha256": point.sha256,
                "coordinates": point.coordinates,
                "evaluation_seconds": point.finished_at - point.started_at
                if point.finished_at is not None and point.started_at is not None
                else None,
                "overall": summary.overall.model_dump() if summary else None,
                "play": summary.model_seats["play"].model_dump() if summary else None,
                "draw": summary.model_seats["draw"].model_dump() if summary else None,
                "model_decks": {d: summary.model_decks[d].model_dump() for d in DECKS}
                if summary
                else None,
                "heuristic_decks": {
                    d: summary.heuristic_decks[d].model_dump() for d in DECKS
                }
                if summary
                else None,
                "invalid": summary.overall.invalid if summary else None,
            }
        )
    cost = (
        status.cleanup.estimated_dollars
        if status.cleanup is not None
        else (view.estimated_dollars if view else None)
    )
    return {
        "seed": seed,
        "job": job,
        "training_run": run.get("id"),
        "phase": record.phase if record else None,
        "updates": record.updates if record else None,
        # Run counters come from the follower's last fetch, which can lag the
        # live job status above.
        "counters_observed_at": view.observed_at if view else None,
        "counters_updates": view.updates if view else None,
        "cleanup_confirmed": status.cleanup is not None,
        "estimated_dollars": cost,
        "learner_decisions": learner,
        "environment_decisions": int(stage.get("environment_decisions") or 0),
        "games": int(stage.get("games") or 0),
        "optimizer_exposures": stage.get("optimizer_exposures"),
        "optimizer_updates": stage.get("optimizer_updates"),
        "training_seconds": stage.get("seconds")
        if stage.get("status") != "running"
        else None,
        "collection_seconds": stage.get("collection_seconds"),
        "learning_seconds": stage.get("learning_seconds"),
        "active_seconds": active or None,
        "learner_decisions_per_active_hour": learner / active * 3600
        if active
        else None,
        "game_errors": run.get("game_errors"),
        "error": stage.get("error"),
        "points": points,
    }


def summarize(directory: Path) -> dict[str, object]:
    ledger = Ledger.model_validate_json((directory / "ledger.json").read_bytes())
    seeds = [_seed(directory, ledger, s, j) for s, j in sorted(JOBS.items())]
    by_update: dict[int, list[float]] = {}
    endpoints, gains = [], []
    for seed in seeds:
        scored = [
            p
            for p in seed["points"]  # type: ignore[union-attr]
            if p["overall"] and p["overall"]["score"] is not None
        ]
        for p in scored:
            if p["label"] != "endpoint" or p["updates"] % 200 == 0:
                by_update.setdefault(p["updates"], []).append(p["overall"]["score"])
        initial = next((p for p in scored if p["updates"] == 0), None)
        endpoint = next((p for p in scored if p["label"] == "endpoint"), None)
        if endpoint:
            endpoints.append(endpoint["overall"]["score"])
        if initial and endpoint:
            gains.append(endpoint["overall"]["score"] - initial["overall"]["score"])

    def spread(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "n": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    dollars = sum(s["estimated_dollars"] or 0 for s in seeds) + FAILED_PROBE_DOLLARS  # type: ignore[operator]
    return {
        "seeds": seeds,
        "failed_jobs": {str(k): v for k, v in FAILED_JOBS.items()},
        "failed_probe_dollars": FAILED_PROBE_DOLLARS,
        "estimated_inclusive_dollars": dollars,
        "shared_updates": {
            str(u): spread(v)
            for u, v in sorted(by_update.items())
            if len(v) == len(JOBS)
        },
        "endpoint_score": spread(endpoints),
        "endpoint_gain": spread(gains),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
