"""Task c5fba4 calibration: per-update cost, memory and a rehearsed evaluation pass.

Each size runs its protocol regime, at its four producers, for a few updates on
the GPU chosen for the frontier runs, so a size that cannot fit, or would reach
too few updates, changes in a new protocol version before any scoring. Each
reading's `expected_updates` is the damping horizon the protocol takes from it
(`fitted_horizon`). A third pod runs w64 at one producer, so the readings show
what the four producers buy. Calibration is not a first attempt
at the frozen runs: its regime ids and this file's experiment keep it out of
`experiments.tla_frontier.size_run`, and its games touch no frontier slot. The
protocol's Launch sequence owns the commands and both launch gates.

Update timing starts from a fresh policy, so its games need not last as long as
a trained policy's. The reading is a median that sets aside the first update's
warm-up, not a forecast of late-run cost. Host memory follows rows per update,
and an update is about one game per stream, so longer late-run games raise it:
`peak_rss_bytes` and `median_rows_per_update` show how far the regime's 40 GiB
stop is from a fresh policy's reading. Neither is a gate.

A stored run is named by its launcher id, so each reading shows its job name. A
gate reading depends on its host, so it shows its pod's step speed beside it,
and its settled games by label (`engine_error`, `resource_limit`) once its
receipt exists. Neither adds a gate.

A failed game is never replayed, so the first rented evaluation pass must not be
a scored one. Once both calibration runs have ended, or the launcher has closed
them, `eval` plays the frontier's own pass job over calibration's checkpoints, and
`read_rehearsal` (`--rehearsal`) reads it against the second launch gate. The
launcher holds a pass's cost, so that read takes the control client. These
checkpoints are nearly fresh policies, so the reading says what the pass does on a
rental, not how long a trained policy's games run.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from statistics import median
from typing import Literal

from etude.eval.query import statsdb
from etude.eval.records import Matchup
from etude.experiment import ExperimentState, TrainingRun
from etude.launch import client
from etude.launch.records import Job, Run
from etude.run.listing import as_storage
from etude.train.read import RunData, read_all

from experiments.tla_frontier import (
    GAMES_PER_PAIR,
    LARGE,
    SMALL,
    PodReading,
    Protocol,
    SizeLabel,
    capped_runs,
    checkpoint_inference,
    environment,
    evaluation_jobs,
    frozen,
    last_update,
    paired_matchup,
    pod_reading,
    shortened,
    tagged_runs,
    with_launcher_ends,
)
from experiments.tla_pilot_analysis import stored_game_columns
from manabot.infra.identity import StrictModel
from manabot.recipes.builders import fitted_horizon
from manabot.training.regime import TrainingRegime

EXPERIMENT = Path(__file__).stem
UPDATES = 5
# 2,400 s stay usable under the 600 s stop reserve: one update well past the
# protocol's 1,500 s launch gate, and all five at MTG-213's 150 s.
WALL_SECONDS = 3600.0
# Etude's 15-minute setup allowance and these hours fit each pod's ceiling at
# the USD0.67/hour the training budget admits.
MAX_HOURS = 1.2
# The w64 pod at one producer, beside the frontier's four.
ONE_PRODUCER = "w64-p1"
type Pod = Literal["w64", "w160l8", "w64-p1"]
# USD2 in total, inside the protocol's reserve.
MAX_USD: dict[Pod, Decimal] = {
    SMALL: Decimal("0.45"),
    ONE_PRODUCER: Decimal("0.45"),
    LARGE: Decimal("1.1"),
}
# The second launch gate, a rough line at half the protocol's 600 s game limit.
REHEARSAL_GAME_SECONDS = 300.0
REHEARSAL_COMPARISONS = 2


class Calibration(StrictModel):
    """One run's reading; None means it logged no such row."""

    run: str
    # Stored runs carry the launcher's id; this says which size it was.
    job: str | None
    state: str
    updates: int
    seconds_per_update: float | None
    # The damping horizon S at that cost: the frontier wall's updates, capped
    # at the reference.
    expected_updates: int | None
    peak_allocated_cuda_bytes: int | None
    peak_reserved_cuda_bytes: int | None
    # Sampled once per update, so a peak inside an update can exceed it.
    peak_rss_bytes: int | None
    median_rows_per_update: float | None
    # Calibration's five updates reach only part of the host window.
    pod: PodReading


class Rehearsal(StrictModel):
    """Every evaluation pass under this experiment, against the second launch gate.

    A pass that delivered nothing has no stored run, so `etude status` holds it
    and its cost; `runs` and `pass_usd` cover the passes that reached the store.
    """

    runs: tuple[str, ...]
    matches: int
    failed_matches: int
    games: int
    # Killed, truncated or never finished: nothing a scored pass could replay.
    invalid_games: int
    median_game_seconds: float | None
    slowest_game_seconds: float | None
    pass_usd: Decimal
    passed: bool


def _regime_id(pod: Pod) -> str:
    return f"tla-frontier-c5fba4-calibration-{pod}"


def calibration_regime(regime: TrainingRegime, pod: Pod) -> TrainingRegime:
    """The protocol regime, changed only in its identity, update target and wall.

    The one-producer pod also sets `execution.workers` to 1.
    """
    return shortened(
        regime,
        id=_regime_id(pod),
        updates=UPDATES,
        wall_seconds=WALL_SECONDS,
        workers=1 if pod == ONE_PRODUCER else None,
    )


def runs(revision: str) -> list[Run]:
    """One capped allocation per pod; a pod that fails is itself the reading."""
    regimes, protocol = frozen()
    pods: list[tuple[Pod, TrainingRegime]] = [
        (SMALL, regimes[SMALL]),
        (LARGE, regimes[LARGE]),
        (ONE_PRODUCER, regimes[SMALL]),
    ]
    return [
        run
        for pod, regime in pods
        for run in capped_runs(
            calibration_regime(regime, pod),
            experiment=EXPERIMENT,
            revision=revision,
            seed=protocol.training_seed,
            max_hours=MAX_HOURS,
            max_usd=MAX_USD[pod],
        )
    ]


# -- Evaluation rehearsal -----------------------------------------------------


def _furthest_run(state: ExperimentState, label: SizeLabel) -> TrainingRun:
    """The size's run that got furthest, known by its tag since stored names are ids.

    A failed calibration can be launched again, and any of its runs may have
    stored checkpoints; the rehearsal wants the heaviest evidence of the size.
    """
    found = tagged_runs(state, _regime_id(label))
    if not found:
        raise ValueError(f"no {label} calibration checkpoint: {label} fails its gate")
    return max(found, key=lambda run: last_update(run) or 0)


def rehearsal_matchups(
    state: ExperimentState, protocol: Protocol
) -> tuple[Matchup, Matchup]:
    """W160l8 against its own start, then against w64; both last checkpoints.

    W160l8 in both seats is the slowest game the frontier schedules. A w160l8 that
    stored only update 0 has no distinct pair, and no training reading for the
    first gate either, so that is refused rather than rehearsed.
    """
    large = _furthest_run(state, LARGE)
    small = _furthest_run(state, SMALL)
    first = large.checkpoints.at("updates", 0)
    last = large.checkpoints.last
    if not first or last is None or last_update(large) == 0:
        raise ValueError(f"{LARGE} stored no update past 0: it fails its gate")
    opponent = small.checkpoints.last
    assert opponent is not None
    inference = checkpoint_inference()
    candidate = last.model(inference)
    return (
        paired_matchup(
            f"{LARGE} last vs {LARGE} u0",
            (candidate, first[0].model(inference)),
            "rehearsal-initial",
            protocol,
        ),
        paired_matchup(
            f"{LARGE} last vs {SMALL} last",
            (candidate, opponent.model(inference)),
            "rehearsal-head-to-head",
            protocol,
        ),
    )


def eval(state: ExperimentState) -> list[Job]:
    """The rehearsal, once no calibration run can still save a later checkpoint."""
    if state.experiment != EXPERIMENT:
        raise ValueError("rehearsal requires calibration's runs")
    closed = with_launcher_ends(state)
    if not closed.runs or not all(run.ended for run in closed.runs):
        return []
    _, protocol = frozen()
    return evaluation_jobs(
        "tla-frontier-c5fba4-rehearsal-v1",
        environment(
            protocol,
            "tla-frontier-c5fba4-calibration-v1",
            "evaluation-rehearsal-not-scored",
        ),
        rehearsal_matchups(closed, protocol),
    )


# -- Readings -----------------------------------------------------------------


def _peak(data: RunData, metric: str) -> int | None:
    rows = data.points(metric, "updates")
    return int(rows.y.max()) if len(rows) else None


def _calibration(data: RunData, pod: PodReading, wall_seconds: float) -> Calibration:
    collection = data.points("time/collection", "updates")
    learning = data.points("time/learning", "updates")
    seconds = [float(a + b) for a, b in zip(collection.y, learning.y, strict=True)]
    cost = median(seconds[1:] or seconds) if seconds else None
    rows = data.points("rows", "updates")
    return Calibration(
        run=data.name,
        job=data.file.declaration.job_name if data.file else None,
        state=data.state.state if data.state else "running",
        updates=int(rows.x[-1]) if len(rows) else 0,
        seconds_per_update=cost,
        expected_updates=fitted_horizon(wall_seconds, cost) if cost else None,
        peak_allocated_cuda_bytes=_peak(data, "memory/peak_allocated_cuda_bytes"),
        peak_reserved_cuda_bytes=_peak(data, "memory/peak_reserved_cuda_bytes"),
        peak_rss_bytes=_peak(data, "coordinates/rss_bytes"),
        median_rows_per_update=float(median(rows.y)) if len(rows) else None,
        pod=pod,
    )


def read_calibration(source: Path | str) -> list[Calibration]:
    """Every calibration run under `source`, retries included.

    The rehearsal's passes are runs here too, so they are set aside.
    """
    storage = as_storage(source)
    wall_seconds = frozen()[0][SMALL].wall_seconds
    return [
        _calibration(data, pod_reading(data, storage), wall_seconds)
        for data in read_all(storage)
        if data.experiment == EXPERIMENT
        and (data.file is None or data.file.declaration.kind != "evaluation")
    ]


def read_rehearsal(source: Path | str) -> Rehearsal:
    """The rehearsal's games and cost, and whether they pass the second launch gate.

    It passes only with every scheduled game played, valid and under the line.
    Each game repeats on both seats, so seat 0 reads it.
    """
    with statsdb(EXPERIMENT, source) as db:
        receipts = db.execute("SELECT run, state FROM matches").fetchall()
        columns = stored_game_columns(db)
        # Etude omits an all-null column, as when every game was killed.
        seconds_column = "seconds" if "seconds" in columns else "NULL"
        games = (
            db.execute(
                f"SELECT valid, {seconds_column} FROM games WHERE seat = 0"
            ).fetchall()
            if columns
            else []
        )
    runs = tuple(sorted({run for run, _ in receipts}))
    failed = sum(state != "done" for _, state in receipts)
    invalid = sum(not valid for valid, _ in games)
    seconds = [float(each) for _, each in games if each is not None]
    slowest = max(seconds, default=None)
    scheduled = REHEARSAL_COMPARISONS * len(frozen()[1].pairs) * GAMES_PER_PAIR
    return Rehearsal(
        runs=runs,
        matches=len(receipts),
        failed_matches=failed,
        games=len(games),
        invalid_games=invalid,
        median_game_seconds=median(seconds) if seconds else None,
        slowest_game_seconds=slowest,
        pass_usd=sum((client.status(run).usd for run in runs), Decimal(0)),
        passed=(
            len(games) == scheduled
            and not failed
            and not invalid
            and slowest is not None
            and slowest < REHEARSAL_GAME_SECONDS
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Etude training store")
    parser.add_argument(
        "--rehearsal", action="store_true", help="read the evaluation rehearsal"
    )
    args = parser.parse_args()
    if args.rehearsal:
        print(read_rehearsal(args.source).model_dump_json(indent=2))
        return
    readings = read_calibration(args.source)
    print(json.dumps([reading.model_dump() for reading in readings], indent=2))


if __name__ == "__main__":
    main()
