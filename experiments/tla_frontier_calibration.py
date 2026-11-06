"""Task c5fba4 calibration: per-update cost, memory and a rehearsed evaluation pass.

Each size runs its frozen regime for a few updates on the GPU chosen for the
frontier runs, so a size that cannot fit, or would reach too few updates, changes
in a new protocol version before any scoring. Calibration is not a first attempt
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
gate reading depends on its host, so it shows its pod's step speed beside it, and
`read_screen` (`--screen`) applies the protocol's host screen to calibration's
pods and a launch's.

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

from etude.eval.query import statsdb
from etude.eval.records import Benchmark, Environment, Matchup
from etude.experiment import ExperimentState, TrainingRun
from etude.launch import client
from etude.launch.records import Job, Run
from etude.run.listing import as_storage
from etude.run.storage import Storage
from etude.train.read import RunData, read_all

from experiments.tla_frontier import (
    EXPERIMENT as FRONTIER,
    GAMES_PER_PAIR,
    HOST_UPDATES,
    LARGE,
    SMALL,
    HostSpeed,
    Protocol,
    SizeLabel,
    capped_runs,
    checkpoint_inference,
    evaluation_pass,
    frozen,
    host_speed,
    last_update,
    paired_matchup,
    pass_name,
    with_launcher_ends,
)
from experiments.tla_pilot_analysis import stored_game_columns
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.infra.runtime import regime_tag
from manabot.training.regime import TrainingRegime, TrainSelfPlay

EXPERIMENT = Path(__file__).stem
UPDATES = 5
# 2,400 s stay usable under the 600 s stop reserve: one update well past the
# protocol's 1,500 s launch gate, and all five at MTG-213's 150 s.
WALL_SECONDS = 3600.0
# USD2 in total, inside the protocol's reserve. Etude's 15-minute setup allowance
# and these hours fit one size's ceiling at the USD0.67/hour the training budget
# admits.
MAX_HOURS = 1.2
MAX_USD_PER_SIZE = Decimal("1")
# A pod reads slow above this multiple of the lowest step speed in the store. A
# rough line from Elemental's fast and slow bands; TLA has no anchor of its own.
SLOW_HOST_RATIO = 1.3
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
    # What the frontier wall budget would reach at that cost.
    updates_in_wall_budget: int | None
    peak_allocated_cuda_bytes: int | None
    peak_reserved_cuda_bytes: int | None
    # Sampled once per update, so a peak inside an update can exceed it.
    peak_rss_bytes: int | None
    median_rows_per_update: float | None
    # Calibration's five updates reach only part of the host window.
    host: HostSpeed


class ScreenReading(StrictModel):
    """One pod against the fastest in the store; `slow` None means not read yet.

    A run is read once it has passed the host window or ended.
    """

    run: str
    job: str | None
    experiment: str
    state: str
    updates: int
    host: HostSpeed
    ratio: float | None
    slow: bool | None


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


def _regime_id(label: SizeLabel) -> str:
    return f"tla-frontier-c5fba4-calibration-{label}"


def calibration_regime(regime: TrainingRegime, label: SizeLabel) -> TrainingRegime:
    """The frozen regime, changed only in its identity, update target and wall."""
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    return regime.model_copy(
        update={
            "id": _regime_id(label),
            "wall_seconds": WALL_SECONDS,
            "stages": [
                stage.model_copy(
                    update={
                        "updates": UPDATES,
                        "execution": stage.execution.model_copy(
                            update={"wall_seconds": WALL_SECONDS}
                        ),
                    }
                )
            ],
        }
    )


def runs(revision: str) -> list[Run]:
    """One capped allocation per size; a size that fails is itself the reading."""
    regimes, protocol = frozen()
    return [
        run
        for label, regime in regimes.items()
        for run in capped_runs(
            calibration_regime(regime, label),
            experiment=EXPERIMENT,
            revision=revision,
            seed=protocol.training_seed,
            max_hours=MAX_HOURS,
            max_usd=MAX_USD_PER_SIZE,
        )
    ]


# -- Evaluation rehearsal -----------------------------------------------------


def _furthest_run(state: ExperimentState, label: SizeLabel) -> TrainingRun:
    """The size's run that got furthest, known by its tag since stored names are ids.

    A failed calibration can be launched again, and any of its runs may have
    stored checkpoints; the rehearsal wants the heaviest evidence of the size.
    """
    tag = regime_tag(_regime_id(label))
    found = [run for run in state.runs if run.checkpoints.tagged(tag)]
    if not found:
        raise ValueError(f"no {label} calibration checkpoint: {label} fails its gate")
    return max(found, key=lambda run: last_update(run) or 0)


def rehearsal_matchups(
    state: ExperimentState, protocol: Protocol
) -> tuple[Matchup, Matchup]:
    """W256 against its own start, then against w64; both last checkpoints.

    W256 in both seats is the slowest game the frontier schedules. A w256 that
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


def _rehearsal_environment(protocol: Protocol) -> Environment:
    return Environment(
        name="tla-frontier-c5fba4-calibration-v1",
        files=canonical_sha256(protocol.model_dump(mode="json")),
        config={"purpose": "evaluation-rehearsal-not-scored"},
    )


def eval(state: ExperimentState) -> list[Job]:
    """The rehearsal, once no calibration run can still save a later checkpoint."""
    if state.experiment != EXPERIMENT:
        raise ValueError("rehearsal requires calibration's runs")
    closed = with_launcher_ends(state)
    if not closed.runs or not all(run.ended for run in closed.runs):
        return []
    _, protocol = frozen()
    return [
        evaluation_pass(
            Benchmark(
                name=pass_name("tla-frontier-c5fba4-rehearsal-v1"),
                environment=_rehearsal_environment(protocol),
                matchups=rehearsal_matchups(closed, protocol),
            )
        )
    ]


# -- Readings -----------------------------------------------------------------


def _peak(data: RunData, metric: str) -> int | None:
    rows = data.points(metric, "updates")
    return int(rows.y.max()) if len(rows) else None


def _updates(data: RunData) -> int:
    rows = data.points("rows", "updates")
    return int(rows.x[-1]) if len(rows) else 0


def _job(data: RunData) -> str | None:
    return data.file.declaration.job_name if data.file else None


def _state(data: RunData) -> str:
    return data.state.state if data.state else "running"


def _calibration(data: RunData, storage: Storage, wall_seconds: float) -> Calibration:
    collection = data.points("time/collection", "updates")
    learning = data.points("time/learning", "updates")
    seconds = [float(a + b) for a, b in zip(collection.y, learning.y, strict=True)]
    cost = median(seconds[1:] or seconds) if seconds else None
    rows = data.points("rows", "updates")
    return Calibration(
        run=data.name,
        job=_job(data),
        state=_state(data),
        updates=_updates(data),
        seconds_per_update=cost,
        updates_in_wall_budget=int(wall_seconds / cost) if cost else None,
        peak_allocated_cuda_bytes=_peak(data, "memory/peak_allocated_cuda_bytes"),
        peak_reserved_cuda_bytes=_peak(data, "memory/peak_reserved_cuda_bytes"),
        peak_rss_bytes=_peak(data, "coordinates/rss_bytes"),
        median_rows_per_update=float(median(rows.y)) if len(rows) else None,
        host=host_speed(data, storage),
    )


def _training_data(storage: Storage, *experiments: str) -> list[RunData]:
    """The experiments' training runs; the rehearsal's passes are runs here too."""
    return [
        data
        for data in read_all(storage)
        if data.experiment in experiments
        and (data.file is None or data.file.declaration.kind != "evaluation")
    ]


def read_calibration(source: Storage | Path | str) -> list[Calibration]:
    """Every calibration run under `source`, retries included."""
    storage = as_storage(source)
    wall_seconds = frozen()[0][SMALL].wall_seconds
    return [
        _calibration(data, storage, wall_seconds)
        for data in _training_data(storage, EXPERIMENT)
    ]


def read_screen(source: Storage | Path | str) -> list[ScreenReading]:
    """Calibration's and the frontier's pods, retired ones included, per the host screen.

    Replacing a slow pod leaves the lowest speed in place, so a replacement is
    read against the same line as the pod it replaced.
    """
    storage = as_storage(source)
    found = [
        (data, host_speed(data, storage))
        for data in _training_data(storage, EXPERIMENT, FRONTIER)
    ]
    speeds = [
        host.step_us_per_row for _, host in found if host.step_us_per_row is not None
    ]
    lowest = min(speeds, default=None)

    def reading(data: RunData, host: HostSpeed) -> ScreenReading:
        updates = _updates(data)
        read = data.state is not None or updates >= HOST_UPDATES[-1]
        ratio = (
            host.step_us_per_row / lowest
            if lowest is not None and host.step_us_per_row is not None
            else None
        )
        return ScreenReading(
            run=data.name,
            job=_job(data),
            experiment=data.experiment,
            state=_state(data),
            updates=updates,
            host=host,
            ratio=ratio,
            slow=ratio > SLOW_HOST_RATIO if read and ratio is not None else None,
        )

    return [reading(data, host) for data, host in found]


def read_rehearsal(source: Storage | Path | str) -> Rehearsal:
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
    reading = parser.add_mutually_exclusive_group()
    reading.add_argument(
        "--screen", action="store_true", help="read the pre-scoring host screen"
    )
    reading.add_argument(
        "--rehearsal", action="store_true", help="read the evaluation rehearsal"
    )
    args = parser.parse_args()
    if args.rehearsal:
        print(read_rehearsal(args.source).model_dump_json(indent=2))
        return
    readings = (read_screen if args.screen else read_calibration)(args.source)
    print(json.dumps([reading.model_dump() for reading in readings], indent=2))


if __name__ == "__main__":
    main()
