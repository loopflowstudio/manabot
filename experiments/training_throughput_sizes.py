"""Step 3 of the training-throughput experiment: model size, GPU and precision (Task b015d5).

Protocol: experiments/protocols/training-throughput-b015d5-v1.md, "Step 3". Every
cell is the control recipe at another network size or precision, 256 streams, both
seats, one producer, CUDA. A leg is one cell, `<gpu>-<size>-k<K>-<precision>`, and
runs its K seeds packed on one hand-held rental through Etude `--slots 4`, as step 2's
legs did. Speed cells run 4 updates. The learning-metrics check is the 12-update
w160l8 legs on the third GPU: float32 twice (the second is `...-float32-repeat`),
TF32 and bf16, each one seed.

Run limits follow the size's expected pace, the previous size's K=1 seconds per update
on that GPU (`--pace`, read from the last cell before launching the next). The watchdog
is the larger of 1,500 s and 6 x pace x updates; the stall watcher cancels a run whose
update rows stop for the larger of 600 s and 3 x pace (`limits`).

    etude rent b015d5-a4000x4 --hours 6
    etude queue experiments/training_throughput_sizes.py --rent b015d5-a4000x4 \\
        --revision COMMIT --leg a4000-w128l4-k1-float32
    etude queue experiments/training_throughput_sizes.py --rent b015d5-a4000x4 \\
        --revision COMMIT --leg a4000-w160l8-k2-float32 --pace 52 --cohort ...

Print a leg's limits at a pace:

    uv run python -m experiments.training_throughput_sizes a4000-a384l8-k1-bf16 140
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Literal

from etude.launch.records import Run, Variable

from experiments.elemental_variants import elemental_recipe, self_play_stage, sized
from experiments.learning_signal import (
    FINALIZATION_SECONDS,
    SETUP_SECONDS,
    training_runs,
)
from experiments.training_throughput_packing import GPU_SAMPLER, LEG_VARIABLE
from manabot.training.regime import Precision, TrainingRegime

EXPERIMENT = "training_throughput_sizes"
SEEDS = (26301, 26302, 26303, 26304)
SPEED_UPDATES = 4
CHECK_UPDATES = 12
# Step 2's solo A4000 reading at w64; the pace of the ladder's first size.
FIRST_PACE_SECONDS = 37.0
WATCHDOG_FLOOR_SECONDS = 1500.0
WATCHDOG_PACES = 6
STALL_FLOOR_SECONDS = 600.0
STALL_PACES = 3
# Etude asks a run's dollars to cover its hours plus setup at the machine's ceiling.
SETUP_HOURS = 0.25

type Gpu = Literal["a4000", "l4", "fast"]
type Size = Literal["w64", "w128l4", "w160l8", "a384l8"]

# Each on-demand's hourly ceiling in the isolated Etude home.
CEILING_USD: dict[Gpu, float] = {"a4000": 0.40, "l4": 0.60, "fast": 0.95}


@dataclass(frozen=True)
class Shape:
    width: int
    layers: int
    heads: int
    feedforward: int | None = None


# Feedforward None is the model's default, heads x width.
SIZES: dict[Size, Shape] = {
    "w64": Shape(64, 2, 4),
    "w128l4": Shape(128, 4, 4),
    "w160l8": Shape(160, 8, 8),
    "a384l8": Shape(384, 8, 8, feedforward=1536),
}


def recipe(size: Size, precision: Precision, updates: int) -> TrainingRegime:
    """The control at `size` and `precision`; w64 float32 is the control itself."""
    check = "-check" if updates == CHECK_UPDATES else ""
    regime_id = f"elemental-throughput-{size}-{precision}{check}-v1"
    if size == "w64":
        regime = elemental_recipe(regime_id, updates)
    else:
        shape = SIZES[size]
        regime = sized(
            regime_id,
            updates,
            width=shape.width,
            layers=shape.layers,
            heads=shape.heads,
            feedforward=shape.feedforward,
        )
    self_play_stage(regime).execution.precision = precision
    return TrainingRegime.model_validate(regime.model_dump())


@dataclass(frozen=True)
class Leg:
    gpu: Gpu
    size: Size
    packed: Literal[1, 2, 4]
    precision: Precision
    updates: int = SPEED_UPDATES
    repeat: bool = False

    @property
    def name(self) -> str:
        suffix = "-repeat" if self.repeat else ""
        return f"{self.gpu}-{self.size}-k{self.packed}-{self.precision}{suffix}"


def _legs() -> dict[str, Leg]:
    legs: list[Leg] = []
    for gpu in ("a4000", "l4", "fast"):
        # A4000's w64 comes from step 2 and L4's from step 1.
        if gpu == "fast":
            legs += [Leg(gpu, "w64", k, "float32") for k in (1, 4)]
        for size in ("w128l4", "w160l8", "a384l8"):
            legs += [Leg(gpu, size, k, "float32") for k in (1, 2, 4)]
        for size in ("w160l8", "a384l8"):
            legs += [Leg(gpu, size, 1, p) for p in ("tf32", "bf16")]
    legs = [
        Leg(leg.gpu, leg.size, 1, leg.precision, CHECK_UPDATES)
        if (leg.gpu, leg.size, leg.packed) == ("fast", "w160l8", 1)
        else leg
        for leg in legs
    ]
    legs.append(Leg("fast", "w160l8", 1, "float32", CHECK_UPDATES, repeat=True))
    return {leg.name: leg for leg in legs}


LEGS = _legs()


@dataclass(frozen=True)
class Limits:
    watchdog_seconds: float
    max_hours: float
    max_usd: float
    stall_seconds: float


def limits(leg: Leg, pace: float) -> Limits:
    """A cell's run limits at `pace`, the previous size's K=1 seconds per update."""
    if pace <= 0:
        raise ValueError("pace must be positive seconds per update")
    watchdog = max(WATCHDOG_FLOOR_SECONDS, WATCHDOG_PACES * pace * leg.updates)
    # Quarter hours keep the job's limits exact; the watchdog takes the slack.
    quarters = math.ceil((watchdog + SETUP_SECONDS + FINALIZATION_SECONDS) / 900)
    hours = quarters / 4
    usd = math.ceil((hours + SETUP_HOURS) * CEILING_USD[leg.gpu] * 100) / 100
    return Limits(
        watchdog_seconds=hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS,
        max_hours=hours,
        max_usd=usd,
        stall_seconds=max(STALL_FLOOR_SECONDS, STALL_PACES * pace),
    )


def runs(revision: str, leg: str, seeds: str = "", pace: str = "") -> list[Run]:
    """One cell's runs. `seeds`, comma-separated, picks some of its seeds; `pace`
    is the previous size's K=1 seconds per update on this GPU (default: w64's)."""
    if leg not in LEGS:
        raise ValueError(f"unknown leg {leg!r}; choose from {sorted(LEGS)}")
    chosen = LEGS[leg]
    declared = SEEDS[: chosen.packed]
    selected = tuple(int(seed) for seed in seeds.split(",")) if seeds else declared
    if unknown := set(selected) - set(declared):
        raise ValueError(f"leg {leg} declares no seeds {sorted(unknown)}")
    limit = limits(chosen, float(pace) if pace else FIRST_PACE_SECONDS)
    regime = recipe(chosen.size, chosen.precision, chosen.updates)
    job = training_runs(
        (regime,),
        updates=chosen.updates,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=selected,
        hours=limit.max_hours,
        usd_per_run=limit.max_usd,
    )[0].job
    return job.model_copy(
        update={
            "name": f"{regime.id}-{leg}",
            "environment": (Variable(name=LEG_VARIABLE, value=leg),),
            "command": GPU_SAMPLER + job.command,
        }
    ).runs(*selected)


def main() -> None:
    leg, pace = sys.argv[1], float(sys.argv[2])
    print(limits(LEGS[leg], pace))


if __name__ == "__main__":
    main()
