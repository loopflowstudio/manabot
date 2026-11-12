"""MTG-319: live-token attention against bucketed shapes on one rented L4.

Exploratory pilot of `ActorCritic(..., live_tokens=True)` (`manabot.model.live_tokens`).
It reuses MTG-301's profile (`experiments/l4_step_profile.py`): one in-situ w160l8
update keeps collection observations and learner minibatches, and those inputs replay
through four arms in separate processes, interleaved with baseline repeats:
`baseline`, `bucketed`, `live` (packed attention over the full tables) and
`bucketed-live` (packed attention over bucket-trimmed tables), at w160l8 and w384l8.
Each arm also writes fixed-weight numerics compared against the baseline.

Decision rule (report: experiments/live_token_attention.md): build live tokens out only
if `bucketed-live` takes at most 1/1.5 of `bucketed`'s learner step at w384l8 with
outputs equal to within float tolerance.

    uv run etude launch experiments/live_token_attention.py --rent l4-etude \\
        --revision COMMIT

The report's figures come from the retained evidence:

    uv run --extra reports python -m experiments.live_token_attention \\
        experiments/data/live_token_attention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments import l4_step_profile


def runs(revision: str, hours: str = "2.0", usd: str = "3.00") -> Any:
    """One L4 job of the `live` candidate at `revision`."""
    return l4_step_profile.runs(
        revision,
        candidate="live",
        hours=hours,
        usd=usd,
        name="mtg319-live-token-attention",
    )


# ---------------------------------------------------------------------------
# Report figures

_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("baseline", "Padded (default)"),
    ("bucketed", "Bucketed shapes"),
    ("live", "Live tokens, full tables"),
    ("bucketed-live", "Live tokens over buckets"),
)
_SIZES: tuple[str, ...] = ("w160l8", "w384l8")
_COLORS: dict[str, str] = {
    "baseline": "#898781",
    "bucketed": "#2a78d6",
    "live": "#7fb2ea",
    "bucketed-live": "#eb6834",
}
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
_BAR = 1.5


def _learner_step_ms(folder: Path, arm: str, size: str) -> float:
    """Median over every timed learner step (minibatches × passes) of one arm."""
    receipt = json.loads(
        (folder / "replay" / f"{arm}-{size}" / "receipt.json").read_text()
    )
    return 1000 * float(np.median(np.concatenate(receipt["learner_seconds"])))


def _max_deviation(numerics: dict[str, Any], phase: str, metric: str) -> float:
    """Largest absolute deviation from the padded path over a phase's inputs."""
    return max(float(row[metric]["max_abs"]) for row in numerics[phase])


def charts(folder: Path) -> None:
    """Write `learner_step.svg` and `equivalence.svg` into `folder` (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": _INK,
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _INK,
            "xtick.color": _MUTED,
            "ytick.color": _INK,
            "svg.fonttype": "none",
            # Fixed ids and no date: redrawn figures keep their manifest digests.
            "svg.hashsalt": "live_token_attention",
        }
    )

    def frame(axis: Any) -> None:
        axis.set_facecolor(_SURFACE)
        axis.grid(axis="x", color=_GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(axis="y", length=0)

    # Learner step: median ms per layout, one panel per model size. The dashed
    # line on each panel is the pre-registered bar, bucketed / 1.5.
    figure, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
    figure.patch.set_facecolor(_SURFACE)
    rows = list(range(len(_LAYOUTS)))
    for axis, size in zip(axes, _SIZES, strict=True):
        frame(axis)
        steps = [_learner_step_ms(folder, arm, size) for arm, _ in _LAYOUTS]
        axis.barh(rows, steps, height=0.6, color=[_COLORS[arm] for arm, _ in _LAYOUTS])
        bar = steps[1] / _BAR
        axis.axvline(bar, color=_MUTED, linewidth=1, linestyle="--")
        limit = max(steps) * 1.18
        axis.text(
            bar + limit * 0.015,
            -0.75,
            f"bucketed ÷ {_BAR}",
            ha="left",
            color=_MUTED,
            fontsize=8,
        )
        for row, ms in zip(rows, steps, strict=True):
            axis.text(ms + limit * 0.015, row, f"{ms:.0f} ms", va="center", fontsize=8)
        axis.set_xlim(0, limit)
        axis.set_title(size, loc="left", fontsize=9)
        axis.set_xlabel("median learner step (ms)")
    axes[0].set_yticks(rows, [label for _, label in _LAYOUTS])
    axes[0].set_ylim(len(_LAYOUTS) - 0.5, -1.1)
    figure.tight_layout()
    figure.savefig(folder / "learner_step.svg", metadata={"Date": None})
    plt.close(figure)

    # Equivalence: largest absolute deviation from the padded path per metric,
    # on a log axis because the metrics differ by orders of magnitude.
    summary = json.loads((folder / "summary.json").read_text())["numerics"]
    metrics = (
        ("inference", "logits", "Collection logits"),
        ("inference", "values", "Values"),
        ("learner", "loss", "Losses"),
        ("learner", "grad", "Gradients"),
    )
    compared = _LAYOUTS[1:]
    figure, axis = plt.subplots(figsize=(7, 2.8))
    figure.patch.set_facecolor(_SURFACE)
    frame(axis)
    axis.grid(axis="x", which="minor", color=_GRID, linewidth=0.3)
    for index, (arm, label) in enumerate(compared):
        offset = (index - (len(compared) - 1) / 2) * 0.22
        numerics = summary[f"{arm}-w160l8"]
        axis.scatter(
            [_max_deviation(numerics, phase, metric) for phase, metric, _ in metrics],
            [row + offset for row in range(len(metrics))],
            s=36,
            color=_COLORS[arm],
            edgecolors=_SURFACE,
            linewidths=1.5,
            label=label,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xlim(1e-9, 1e-4)
    axis.set_yticks(range(len(metrics)), [label for _, _, label in metrics])
    axis.invert_yaxis()
    axis.set_xlabel("largest absolute deviation from the padded path (w160l8, CUDA)")
    axis.legend(frameon=False, loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "equivalence.svg", metadata={"Date": None})
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the report's figures.")
    parser.add_argument("folder", type=Path, help="the run's evidence folder")
    folder = parser.parse_args(argv).folder
    if not (folder / "summary.json").is_file():
        parser.error(
            f"{folder}/summary.json is missing; fetch summary.json and "
            "replay/<arm>-<size>/receipt.json with "
            "`uv run --extra artifacts python -m manabot.data fetch`"
        )
    charts(folder)


if __name__ == "__main__":
    main()
