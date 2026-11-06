"""Task c5fba4 calibration: per-update cost and CUDA peak of both frontier sizes.

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
the w256 reading against 1,500 s/update.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from statistics import median

from etude.launch.records import Run
from etude.run.storage import Storage
from etude.train.read import RunData, read_all

from experiments.tla_frontier import ROOT, SMALL, SizeLabel, frozen
from manabot.experiment.job import TrainingJob
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


class Calibration(StrictModel):
    """One run's reading; None means it logged no such row."""

    run: str
    state: str
    updates: int
    seconds_per_update: float | None
    # What the frontier wall budget would reach at that cost.
    updates_in_wall_budget: int | None
    peak_allocated_cuda_bytes: int | None
    peak_reserved_cuda_bytes: int | None


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
        for run in TrainingJob(
            regime=calibration_regime(regime, label),
            experiment=EXPERIMENT,
            finalization_seconds=600,
            stop_reserve_seconds=600,
            validate_numerics=True,
        )
        .job(
            repo=ROOT,
            revision=revision,
            max_hours=MAX_HOURS,
            max_usd=MAX_USD_PER_SIZE,
        )
        .runs(protocol.training_seed)
    ]


def _peak(data: RunData, metric: str) -> int | None:
    rows = data.points(metric, "updates")
    return int(rows.y.max()) if len(rows) else None


def _calibration(data: RunData, wall_seconds: float) -> Calibration:
    collection = data.points("time/collection", "updates")
    learning = data.points("time/learning", "updates")
    seconds = [float(a + b) for a, b in zip(collection.y, learning.y, strict=True)]
    cost = median(seconds[1:] or seconds) if seconds else None
    return Calibration(
        run=data.name,
        state=data.state.state if data.state else "running",
        updates=int(collection.x[-1]) if len(collection) else 0,
        seconds_per_update=cost,
        updates_in_wall_budget=int(wall_seconds / cost) if cost else None,
        peak_allocated_cuda_bytes=_peak(data, "memory/peak_allocated_cuda_bytes"),
        peak_reserved_cuda_bytes=_peak(data, "memory/peak_reserved_cuda_bytes"),
    )


def read_calibration(source: Storage | Path | str) -> list[Calibration]:
    """Every calibration run under `source`, retries included."""
    wall_seconds = frozen()[0][SMALL].wall_seconds
    return [
        _calibration(data, wall_seconds)
        for data in read_all(source)
        if data.experiment == EXPERIMENT
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Etude training store")
    args = parser.parse_args()
    readings = read_calibration(args.source)
    print(json.dumps([reading.model_dump() for reading in readings], indent=2))


if __name__ == "__main__":
    main()
