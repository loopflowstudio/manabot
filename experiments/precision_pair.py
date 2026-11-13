"""BF16 against float32 over the first 50 updates at w384l8, one shared seed (MTG-345).

`runs` launches precision_ladder's two arms unchanged except for the update
target: the damping horizon stays 300 updates, so updates 1-50 are the same
schedule MTG-324's four runs walked. `eval` plays the two final checkpoints
head-to-head. `curves` reads the pair, the MTG-345 proof run and MTG-324's
runs from storage and writes the per-update table and chart the report cites.
The question, measurements and cost are in `experiments/precision_pair.md`.

    uv run --with "etude-sdk[s3,eval]==0.2.15" etude launch \\
        experiments/precision_pair.py --rent mtg324-l4-ab --parallel 2 \\
        --cohort NAME --max-usd 5 --revision COMMIT
    uv run etude eval experiments/precision_pair.py --rent cpu32 --source r2://manabot/jobs
    uv run --extra artifacts --extra reports python -m experiments.precision_pair
"""

from __future__ import annotations

import csv
from pathlib import Path

from etude.eval.records import Inference, Matchup
from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report
from etude.run.storage import storage_from_uri
from etude.train.read import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.learning_signal import (  # noqa: E402
    ELEMENTAL_PAIRS,
    Conditions,
    benchmark_jobs,
    training_runs,
)
from experiments.precision_ladder import ARMS, arm, require_single_axis  # noqa: E402
from manabot.infra.runtime import regime_tag  # noqa: E402
from manabot.training.panels import ML, PERF  # noqa: E402
from manabot.training.regime import (  # noqa: E402
    Precision,
    TrainingRegime,
    TrainSelfPlay,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
DATA = ROOT / "experiments/data" / EXPERIMENT
SEED = 32401
UPDATES = 50
# Float32 paced 134 s/update in MTG-324 (188 updates in 7 h): 50 updates in
# 1.9 h. Both arms get the same 3-hour window, which neither should reach.
HOURS = 3.0
USD_PER_RUN = 2.5
EVALUATION = "precision-pair-v1"


def regime_id(precision: Precision) -> str:
    return f"precision-pair-w384l8-{precision}-u{UPDATES}"


def pair_arm(precision: Precision) -> TrainingRegime:
    """precision_ladder's arm, stopped at `UPDATES` on its 300-update horizon."""
    regime = arm(precision)
    regime.id = regime_id(precision)
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = UPDATES
    return TrainingRegime.model_validate(regime.authored_dump())


def runs(
    revision: str, hours: float = HOURS, usd_per_run: float = USD_PER_RUN
) -> list[Run]:
    """Both arms from the same seed: same weights, pairings and deals at the start."""
    reference, candidate = (pair_arm(precision) for precision in ARMS)
    require_single_axis(reference, candidate)
    return training_runs(
        (reference, candidate),
        updates=UPDATES,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=(SEED,),
        hours=hours,
        usd_per_run=usd_per_run,
    )


def eval(state: ExperimentState) -> list[Job]:
    """The two update-50 checkpoints head-to-head, float32 first."""
    final = []
    for precision in ARMS:
        tag = regime_tag(regime_id(precision))
        found = [
            run.final
            for run in state.runs
            if run.checkpoints.tagged(tag)
            and run.final is not None
            and int(run.final.clocks["updates"]) == UPDATES
        ]
        if len(found) != 1:
            return []
        final.append(found[0])
    inference = Inference(
        source=[
            *sorted((ROOT / "manabot").rglob("*.py")),
            ROOT / "experiments/learning_signal.py",
        ]
    )
    matchup = Matchup(
        name=f"{ARMS[0]} v {ARMS[1]} u{UPDATES}",
        players=(final[0].model(inference), final[1].model(inference)),
        conditions=tuple(
            Conditions(design=ELEMENTAL_PAIRS.name, **pairing.model_dump()).model_dump()
            for pairing in ELEMENTAL_PAIRS.pairings
        ),
        games=ELEMENTAL_PAIRS.legs,
        tags=("head-to-head",),
    )
    return benchmark_jobs((matchup,), EVALUATION, ELEMENTAL_PAIRS, hours=0.5, usd=1.0)


def report() -> Report:
    return Report(clock="updates", baseline=regime_id(ARMS[0]), perf=PERF, ml=ML)


# --------------------------------------------------------------------------- curves

# MTG-324's runs and the MTG-345 proof walked the same schedule from the same
# recipe; seed 32401's three BF16 runs differ only in revision and length.
EARLIER: dict[str, tuple[str, str]] = {
    "07f84834e617": ("precision_ladder", "float32 32401 (MTG-324)"),
    "aa9518f6fc4b": ("precision_ladder", "float32 32402 (MTG-324)"),
    "2be568a65518": ("precision_ladder", "bf16 32401 (MTG-324)"),
    "6aa5ba4d5fac": ("precision_ladder", "bf16 32402 (MTG-324)"),
    "b00f4500b78d": ("bf16_behavior_logs", "bf16 32401 (proof)"),
}
# Chart title by metric name; seconds are per update.
METRICS = {
    "loss": "Loss",
    "policy_loss": "Policy loss",
    "value_loss": "Value loss",
    "collection_kl": "KL from the collecting policy",
    "numerical/clip_fraction": "Clip fraction",
    "value_scale/explained_variance": "Value explained variance",
    "entropy": "Entropy",
    "time/learning": "Learning seconds",
    "time/collection": "Collection seconds",
}


def curves(source: str = "r2://manabot/jobs", pair: str = "") -> None:
    """Write `updates.csv` and `curves.png` for updates 1-50 of every run.

    `pair` is "FLOAT32_RUN,BF16_RUN"; without it only the earlier runs are read.
    """
    storage = storage_from_uri(source)
    wanted = dict(EARLIER)
    if pair:
        for precision, run in zip(ARMS, pair.split(","), strict=True):
            wanted[run] = (EXPERIMENT, f"{precision} {SEED} (pair)")
    rows: list[dict[str, str | float]] = []
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for run, (experiment, label) in wanted.items():
        data = read(storage, experiment, run)
        series: dict[str, np.ndarray] = {}
        for metric in METRICS:
            points = data.points(metric, "updates")
            keep = (points.x >= 1) & (points.x <= UPDATES)
            series[metric] = points.y[keep]
            if len(series[metric]) != int(keep.sum()) or not keep.any():
                raise ValueError(f"{run} has no {metric}")
            updates = points.x[keep]
        for index, update in enumerate(updates):
            rows.append(
                {"run": run, "label": label, "update": int(update)}
                | {metric: float(series[metric][index]) for metric in METRICS}
            )
        style = {
            "color": "tab:blue" if label.startswith("float32") else "tab:orange",
            "linestyle": "-"
            if "(pair)" in label
            else ":"
            if "32402" in label
            else "--",
            "linewidth": 2.0 if "(pair)" in label else 1.0,
        }
        for axis, metric in zip(axes.flat, METRICS, strict=True):
            axis.plot(updates, series[metric], label=label, **style)
            axis.set_title(METRICS[metric])
    for axis in axes[-1]:
        axis.set_xlabel("update")
    axes.flat[0].legend(fontsize=7)
    figure.tight_layout()
    DATA.mkdir(parents=True, exist_ok=True)
    figure.savefig(DATA / "curves.png", dpi=110)
    with (DATA / "updates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import sys

    curves(pair=sys.argv[1] if len(sys.argv) > 1 else "")
