"""Task c5fba4 calibration: per-update cost and memory of both frontier sizes.

Each size runs its frozen regime for a few updates on the GPU chosen for the
frontier runs, so a size that cannot fit, or would reach too few updates, changes
in a new protocol version before any scoring. Calibration is not a first attempt
at the frozen runs: its regime ids and this file's experiment keep it out of
`experiments.tla_frontier.size_run`, and it plays no evaluation.

Runs after the protocol merges, before `experiments.tla_frontier.runs`:

    doppler run -p etude -c prd -- doppler run -p manabot -c prd \
        --preserve-env=RUNPOD_API_KEY -- uv run --extra artifacts --extra evaluation \
        --with "etude-sdk[s3,eval]==0.2.14" etude launch \
        experiments/tla_frontier_calibration.py --revision COMMIT --rent <gpu> \
        --cohort tla-frontier-c5fba4-calibration --parallel 2 --max-usd 2
    doppler run -p manabot -c prd -- uv run --extra artifacts --extra evaluation \
        python -m experiments.tla_frontier_calibration --source <store>

Update timing starts from a fresh policy, so its games need not last as long as
a trained policy's. The reading is a median that sets aside the first update's
warm-up, not a forecast of late-run cost. The protocol's launch gate compares
the w256 reading against 1,500 s/update. Host memory follows rows per update,
and an update is about one game per stream, so longer late-run games raise it:
`peak_rss_bytes` and `median_rows_per_update` show how far the regime's 40 GiB
stop is from a fresh policy's reading. Neither is a gate.

A stored run is named by its launcher id, so each reading shows its job name.
A gate reading depends on its host: a slow pod reads up to 1.8× above a fast one,
so a reading shows its pod's step speed beside it. `read_screen` applies the
protocol's host screen to calibration's pods and a launch's, reading the same
step speed; `--screen` prints it:

    doppler run -p manabot -c prd -- uv run --extra artifacts --extra evaluation \
        python -m experiments.tla_frontier_calibration --source <store> --screen
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from statistics import median

from etude.launch.records import Run
from etude.run.listing import as_storage
from etude.run.storage import Storage
from etude.train.read import RunData, read_all

from experiments.tla_frontier import (
    EXPERIMENT as FRONTIER,
    HOST_UPDATES,
    SMALL,
    HostSpeed,
    SizeLabel,
    capped_runs,
    frozen,
    host_speed,
)
from manabot.infra.identity import StrictModel
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


def calibration_regime(regime: TrainingRegime, label: SizeLabel) -> TrainingRegime:
    """The frozen regime, changed only in its identity, update target and wall."""
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    return regime.model_copy(
        update={
            "id": f"tla-frontier-c5fba4-calibration-{label}",
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


def read_calibration(source: Storage | Path | str) -> list[Calibration]:
    """Every calibration run under `source`, retries included."""
    storage = as_storage(source)
    wall_seconds = frozen()[0][SMALL].wall_seconds
    return [
        _calibration(data, storage, wall_seconds)
        for data in read_all(storage)
        if data.experiment == EXPERIMENT
    ]


def read_screen(source: Storage | Path | str) -> list[ScreenReading]:
    """Calibration's and the frontier's pods, retired ones included, per the host screen.

    Replacing a slow pod leaves the lowest speed in place, so a replacement is
    read against the same line as the pod it replaced.
    """
    storage = as_storage(source)
    found = [
        (data, host_speed(data, storage))
        for data in read_all(storage)
        if data.experiment in (EXPERIMENT, FRONTIER)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Etude training store")
    parser.add_argument(
        "--screen", action="store_true", help="read the pre-scoring host screen"
    )
    args = parser.parse_args()
    readings = (read_screen if args.screen else read_calibration)(args.source)
    print(json.dumps([reading.model_dump() for reading in readings], indent=2))


if __name__ == "__main__":
    main()
