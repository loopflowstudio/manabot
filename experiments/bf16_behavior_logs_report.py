"""Charts and tables for `experiments/bf16_behavior_logs.md` (MTG-345).

Reads the proof run and MTG-324's four runs from Etude storage: the same
w384l8 recipe and 300-update damping schedule, two seeds, float32 and BF16.
`main` writes per-update rows (`updates.csv`), a per-run summary
(`summary.json`) and two charts under `experiments/data/bf16_behavior_logs/`.

    uv run --extra artifacts --extra reports python -m experiments.bf16_behavior_logs_report
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from etude.run.storage import storage_from_uri
from etude.train.read import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path(__file__).resolve().parent / "data/bf16_behavior_logs"
PROOF = "b00f4500b78d"
# Run -> (experiment, precision, seed, note). MTG-324's runs predate the fix
# (revision 783ccdba); the proof ran this branch at 34b380a554.
RUNS: dict[str, tuple[str, str, int, str]] = {
    "07f84834e617": ("precision_ladder", "float32", 32401, "MTG-324"),
    "aa9518f6fc4b": ("precision_ladder", "float32", 32402, "MTG-324"),
    "2be568a65518": ("precision_ladder", "bf16", 32401, "MTG-324, stopped at 141"),
    "6aa5ba4d5fac": ("precision_ladder", "bf16", 32402, "MTG-324"),
    PROOF: ("bf16_behavior_logs", "bf16", 32401, "with the fix"),
}
LEARNING = {
    "loss": "Loss",
    "collection_kl": "KL from the collecting policy",
    "numerical/clip_fraction": "Clip fraction",
    "value_scale/explained_variance": "Value explained variance",
}
SECONDS = {
    "time/collection": "Collection seconds per update",
    "time/learning": "Learning seconds per update",
}
NUMERICAL = (
    "numerical/legal_probability_min",
    "numerical/legal_logit_gap_max",
    "numerical/legal_underflow_count",
    "numerical/rejected_steps",
    "numerical/nonfinite_count",
)
# Per-update learning metrics are noisy; charts show this trailing mean.
WINDOW = 10
FLOAT32_MIN_NORMAL = float(np.finfo(np.float32).tiny)
# Below this probability one float32 spacing exceeds the old rule's 1e-5.
AT_RISK = 1.4e-40


@dataclass(frozen=True)
class Summary:
    run: str
    precision: str
    seed: int
    note: str
    updates: int
    rejected_steps: int
    nonfinite: int
    largest_logit_gap: float
    smallest_legal_probability: float
    subnormal_updates: int
    at_risk_updates: int
    zero_updates: int
    # Medians over the updates every run reached.
    collection_seconds: float
    learning_seconds: float


def _trailing_mean(values: np.ndarray) -> np.ndarray:
    sums = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), WINDOW)
    return (sums[1:] - sums[np.arange(1, len(values) + 1) - counts]) / counts


def main(source: str = "r2://manabot/jobs") -> None:
    storage = storage_from_uri(source)
    metrics = (*LEARNING, *SECONDS, *NUMERICAL)
    series: dict[str, dict[str, np.ndarray]] = {}
    for run, (experiment, *_rest) in RUNS.items():
        data = read(storage, experiment, run)
        series[run] = {}
        for metric in metrics:
            points = data.points(metric, "updates")
            if not len(points) or points.x[0] != 1 or np.any(np.diff(points.x) != 1):
                raise ValueError(f"{run} lacks a complete {metric} series")
            series[run][metric] = points.y
        if len({len(values) for values in series[run].values()}) != 1:
            raise ValueError(f"{run} metrics end at different updates")
    common = min(len(values["loss"]) for values in series.values())

    DATA.mkdir(parents=True, exist_ok=True)
    summaries = []
    with (DATA / "updates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "precision", "seed", "update", *metrics))
        for run, (_experiment, precision, seed, note) in RUNS.items():
            values = series[run]
            for index in range(len(values["loss"])):
                writer.writerow(
                    (run, precision, seed, index + 1)
                    + tuple(float(values[metric][index]) for metric in metrics)
                )
            smallest = values["numerical/legal_probability_min"]
            summaries.append(
                Summary(
                    run,
                    precision,
                    seed,
                    note,
                    updates=len(smallest),
                    rejected_steps=int(values["numerical/rejected_steps"].sum()),
                    nonfinite=int(values["numerical/nonfinite_count"].sum()),
                    largest_logit_gap=float(
                        values["numerical/legal_logit_gap_max"].max()
                    ),
                    smallest_legal_probability=float(smallest.min()),
                    subnormal_updates=int((smallest < FLOAT32_MIN_NORMAL).sum()),
                    at_risk_updates=int((smallest < AT_RISK).sum()),
                    zero_updates=int((smallest == 0).sum()),
                    collection_seconds=float(
                        np.median(values["time/collection"][:common])
                    ),
                    learning_seconds=float(np.median(values["time/learning"][:common])),
                )
            )
    (DATA / "summary.json").write_text(
        json.dumps(
            {"common_updates": common, "runs": [asdict(row) for row in summaries]},
            indent=2,
        )
        + "\n"
    )

    for name, wanted, smooth in (
        ("learning.png", LEARNING, True),
        ("seconds.png", SECONDS, False),
    ):
        figure, axes = plt.subplots(
            1, len(wanted), figsize=(5.5 * len(wanted), 4.2), squeeze=False
        )
        for axis, (metric, title) in zip(axes[0], wanted.items(), strict=True):
            for run, (_experiment, precision, seed, note) in RUNS.items():
                values = series[run][metric]
                axis.plot(
                    np.arange(1, len(values) + 1),
                    _trailing_mean(values) if smooth else values,
                    color="tab:blue" if precision == "float32" else "tab:orange",
                    linestyle="-" if seed == 32401 else ":",
                    linewidth=2.2 if run == PROOF else 1.2,
                    label=f"{precision}, seed {seed} ({note})",
                )
            axis.set_title(
                f"{title}, {WINDOW}-update trailing mean" if smooth else title
            )
            axis.set_xlabel("update")
        axes[0][0].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(DATA / name, dpi=110)
        plt.close(figure)


if __name__ == "__main__":
    main()
