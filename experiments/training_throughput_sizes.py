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

C1 legs are prefixed `c1-` and include their endpoint (`-u8`, `-u12`).
`-overlap` is a separately identified, conditional twelve-update retry;
`-u2-admission` runs mandatory CUDA tests before its two-update BF16 worker.
C1 uses seeds 26401–26404 (26400 for admission) and its own experiment name.
The operator admits cells against the inclusive budget and rental deadline
using `training_throughput_readings.admit_cost`; job limits alone are not a
campaign budget. Neither declarations nor discovery rent a machine.

Print a leg's limits at a pace:

    uv run python -m experiments.training_throughput_sizes a4000-a384l8-k1-bf16 140
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import sys
from typing import Literal

from etude.launch.records import Run, Variable

from experiments.elemental_variants import self_play_stage, sized
from experiments.learning_signal import (
    FINALIZATION_SECONDS,
    SETUP_SECONDS,
    training_runs,
)
from experiments.training_throughput_packing import GPU_SAMPLER, LEG_VARIABLE
from manabot.training.regime import Precision, TrainingRegime

EXPERIMENT = "training_throughput_sizes"
C1_EXPERIMENT = "training-throughput-sizes-c1"
C1_SEEDS = (26401, 26402, 26403, 26404)
CUDA_ADMISSION = (
    "MANABOT_REQUIRE_CUDA=1 uv run --extra dev pytest "
    "tests/model/test_semantic_card_precision.py -k cuda_semantic_policy_admission -q -s\n"
)
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
    shape = SIZES[size]
    regime = sized(
        f"elemental-throughput-{size}-{precision}{check}-v1",
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
    continuation: bool = False
    admission: bool = False
    overlap_retry: bool = False

    @property
    def name(self) -> str:
        suffix = "-repeat" if self.repeat else ""
        name = f"{self.gpu}-{self.size}-k{self.packed}-{self.precision}{suffix}"
        if not self.continuation:
            return name
        name = f"c1-{name}-u{self.updates}"
        if self.admission:
            return name + "-admission"
        return name + "-overlap" if self.overlap_retry else name


def _leg(gpu: Gpu, size: Size, packed: Literal[1, 2, 4], precision: Precision) -> Leg:
    # The third GPU's solo w160l8 cells are the learning-metrics check.
    check = (gpu, size, packed) == ("fast", "w160l8", 1)
    return Leg(gpu, size, packed, precision, CHECK_UPDATES if check else SPEED_UPDATES)


def _legs() -> dict[str, Leg]:
    legs: list[Leg] = []
    for gpu in ("a4000", "l4", "fast"):
        # A4000's w64 comes from step 2 and L4's from step 1.
        if gpu == "fast":
            legs += [_leg(gpu, "w64", k, "float32") for k in (1, 4)]
        for size in ("w128l4", "w160l8", "a384l8"):
            legs += [_leg(gpu, size, k, "float32") for k in (1, 2, 4)]
        for size in ("w160l8", "a384l8"):
            legs += [_leg(gpu, size, 1, p) for p in ("tf32", "bf16")]
    legs.append(Leg("fast", "w160l8", 1, "float32", CHECK_UPDATES, repeat=True))
    return {leg.name: leg for leg in legs}


LEGS = _legs()


def _continuation_legs() -> dict[str, Leg]:
    # Original declarations remain frozen. C1 has its own endpoints and identities.
    legs = [
        replace(leg, updates=12 if leg.updates == 12 else 8, continuation=True)
        for leg in LEGS.values()
    ]
    legs += [
        Leg(gpu, "w64", k, "float32", 8, continuation=True)
        for gpu in ("a4000", "l4", "fast")
        for k in (1, 2, 4)
        if not any(
            existing.gpu == gpu and existing.size == "w64" and existing.packed == k
            for existing in legs
        )
    ]
    # One whole-cell overlap rerun, plus the matching solo endpoint. Execution
    # admits these only after insufficient overlap; the reader never pools them.
    legs += [
        replace(leg, updates=12, overlap_retry=True)
        for leg in legs
        if leg.updates == 8 and leg.precision == "float32"
    ]
    legs += [
        Leg(gpu, "w160l8", 1, "bf16", 2, continuation=True, admission=True)
        for gpu in ("a4000", "l4", "fast")
    ]
    return {leg.name: leg for leg in legs}


C1_LEGS = _continuation_legs()


def leg_recipe(leg: Leg) -> TrainingRegime:
    regime = recipe(leg.size, leg.precision, leg.updates)
    if leg.continuation:
        regime.id = (
            f"elemental-throughput-c1-{leg.size}-{leg.precision}-u{leg.updates}-v1"
        )
        if (
            regime.observation.policy_input_version != 6
            or regime.observation.max_facts != 512
        ):
            raise ValueError("C1 requires the declared v6 / 512-fact input")
        if regime.agent.catalog_records(regime.observation.policy_input_version):
            raise ValueError("C1 requires effective hashed records")
    return regime


@dataclass(frozen=True)
class Limits:
    watchdog_seconds: float
    max_hours: float
    max_usd: float
    stall_seconds: float


def limits(leg: Leg, pace: float) -> Limits:
    """A cell's run limits at `pace`, the previous size's K=1 seconds per update."""
    if not math.isfinite(pace) or pace <= 0:
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
    declarations = LEGS | C1_LEGS
    if leg not in declarations:
        raise ValueError(f"unknown leg {leg!r}; choose from {sorted(declarations)}")
    chosen = declarations[leg]
    declared = (C1_SEEDS if chosen.continuation else SEEDS)[: chosen.packed]
    if chosen.admission:
        declared = (26400,)
    selected = tuple(int(seed) for seed in seeds.split(",")) if seeds else declared
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("seeds must be nonempty and distinct")
    if unknown := set(selected) - set(declared):
        raise ValueError(f"leg {leg} declares no seeds {sorted(unknown)}")
    limit = limits(chosen, float(pace) if pace else FIRST_PACE_SECONDS)
    regime = leg_recipe(chosen)
    job = training_runs(
        (regime,),
        updates=chosen.updates,
        revision=revision,
        experiment=C1_EXPERIMENT if chosen.continuation else EXPERIMENT,
        seeds=selected,
        hours=limit.max_hours,
        usd_per_run=limit.max_usd,
    )[0].job
    command = GPU_SAMPLER + job.command
    if chosen.admission:
        command = (
            CUDA_ADMISSION.rstrip()
            + " || exit $?\n"
            + command
            + " || exit $?\nuv run --locked python scripts/check_throughput_admission.py"
        )
    return job.model_copy(
        update={
            "name": f"{regime.id}-{leg}",
            "environment": (Variable(name=LEG_VARIABLE, value=leg),),
            "command": command,
        }
    ).runs(*selected)


def main() -> None:
    leg, pace = sys.argv[1], float(sys.argv[2])
    print(limits((LEGS | C1_LEGS)[leg], pace))


if __name__ == "__main__":
    main()
