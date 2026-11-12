"""Step 2 of the training-throughput experiment: does packing change learning? (Task b015d5)

Protocol: experiments/protocols/training-throughput-b015d5-v1.md, "Step 2". Four
seeds share one machine through Etude `--slots 4`. Each seed is its own process, so
a packed seed should learn as a solo seed does unless something leaks between them.
Every leg runs through `TrainingJob` on one rented machine, one leg at a time:

- D1 and D2 run the digest variant on CPU, solo and beside a neighbor. Seed 26201
  must give the same `training_digest` in both.
- M1 packs four seeds of the control recipe to 20 updates on CUDA; M2 repeats them
  solo, and M3 repeats one solo seed for the scale of run-to-run CUDA noise.
- S14, S22 and S44 read speed alone at K seeds of P producers each, 8 updates, as
  regime variants differing from the control only in `execution.workers`. M1 and M2
  are the (4, 1) and (1, 1) shapes.

`LEGS` gives each leg a job identity of its own while the regime bytes stay equal
across M1, M2 and M3, so a repeated seed is a separate run. Launch M2's seeds one at
a time, because a free slot would otherwise take the next seed early. Every CUDA leg
samples the GPU into its output; a run's CPU use is already in its update rows.

    etude rent b015d5-a4000x4 --hours 5
    etude launch experiments/training_throughput_packing.py --rent b015d5-a4000x4 \\
        --revision COMMIT --leg d1
    etude wait ...
    etude launch experiments/training_throughput_packing.py --rent b015d5-a4000x4 \\
        --revision COMMIT --leg d2 --cohort b015d5-d2

Compare two runs' `training/` folders, which prints any difference by path:

    uv run python -m experiments.training_throughput_packing D1/training D2/training
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from etude.launch.records import Run, Variable
import torch

from experiments.elemental_variants import elemental_recipe, self_play_stage
from experiments.learning_signal import training_runs
from manabot.infra.identity import file_sha256
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime

EXPERIMENT = "training_throughput_packing"
DIGEST_REGIME = "elemental-throughput-digest-v1"
METRIC_REGIME = "elemental-throughput-pack-check-v1"
DIGEST_STREAMS = 24
DIGEST_UPDATES = 3
METRIC_UPDATES = 20
METRIC_SEEDS = (26211, 26212, 26213, 26214)
SPEED_UPDATES = 8
GPU_SAMPLES = "gpu-samples.csv"
# Names the leg in the job's identity; nothing reads it at run time.
LEG_VARIABLE = "TRAINING_THROUGHPUT_LEG"
# Reads the whole GPU, so runs sharing a machine record the same device; the rows'
# timestamps give each run's span. Etude ends an attempt by process group, and the
# trap covers an ordinary exit, whose status it leaves unchanged.
GPU_SAMPLER = (
    'mkdir -p "$ETUDE_OUTPUT_DIR"\n'
    "nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used "
    "--format=csv,noheader,nounits -l 1 "
    f'> "$ETUDE_OUTPUT_DIR/{GPU_SAMPLES}" &\n'
    "sampler=$!\n"
    "trap 'kill \"$sampler\"' EXIT\n"
)


def digest_recipe(
    streams: int = DIGEST_STREAMS, updates: int = DIGEST_UPDATES
) -> TrainingRegime:
    """The control recipe, small and single-threaded on CPU: a check, not a shape."""
    regime = elemental_recipe(DIGEST_REGIME, updates)
    stage = self_play_stage(regime)
    stage.streams = streams
    stage.execution.device = "cpu"
    stage.execution.threads = 1
    return regime


def metric_recipe() -> TrainingRegime:
    return elemental_recipe(METRIC_REGIME, METRIC_UPDATES)


def speed_recipe(producers: Literal[2, 4]) -> TrainingRegime:
    """The control recipe at another producer count: a speed reading, not an arm."""
    regime = elemental_recipe(
        f"elemental-throughput-producers-{producers}-v1", SPEED_UPDATES
    )
    self_play_stage(regime).execution.workers = producers
    return regime


@dataclass(frozen=True)
class Leg:
    recipe: Callable[[], TrainingRegime]
    seeds: tuple[int, ...]
    hours: float
    usd_per_run: float


def _digest_leg(*seeds: int) -> Leg:
    return Leg(digest_recipe, seeds, hours=0.5, usd_per_run=0.45)


def _metric_leg(*seeds: int) -> Leg:
    return Leg(metric_recipe, seeds, hours=1.0, usd_per_run=0.75)


def _speed_leg(seeds: int, producers: Literal[2, 4]) -> Leg:
    return Leg(
        partial(speed_recipe, producers),
        METRIC_SEEDS[:seeds],
        hours=0.75,
        usd_per_run=0.60,
    )


LEGS: dict[str, Leg] = {
    "d1": _digest_leg(26201),
    "d2": _digest_leg(26201, 26202),
    "m1": _metric_leg(*METRIC_SEEDS),
    "m2": _metric_leg(*METRIC_SEEDS),
    "m3": _metric_leg(METRIC_SEEDS[0]),
    "s14": _speed_leg(seeds=1, producers=4),
    "s22": _speed_leg(seeds=2, producers=2),
    "s44": _speed_leg(seeds=4, producers=4),
}


def runs(revision: str, leg: str, seeds: str = "") -> list[Run]:
    """One leg's runs. `seeds`, comma-separated, picks some of the leg's seeds."""
    if leg not in LEGS:
        raise ValueError(f"unknown leg {leg!r}; choose from {sorted(LEGS)}")
    chosen = LEGS[leg]
    selected = tuple(int(seed) for seed in seeds.split(",")) if seeds else chosen.seeds
    if unknown := set(selected) - set(chosen.seeds):
        raise ValueError(f"leg {leg} declares no seeds {sorted(unknown)}")
    regime = chosen.recipe()
    stage = self_play_stage(regime)
    job = training_runs(
        (regime,),
        updates=stage.updates,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=selected,
        hours=chosen.hours,
        usd_per_run=chosen.usd_per_run,
    )[0].job
    sampled = stage.execution.device == "cuda"
    return job.model_copy(
        update={
            "name": f"{regime.id}-{leg}",
            "environment": (Variable(name=LEG_VARIABLE, value=leg),),
            "command": (GPU_SAMPLER if sampled else "") + job.command,
        }
    ).runs(*selected)


# ---------------------------------------------------------------------------
# Training digest
#
# Whole-file hashes differ between same-seed runs through identity and timing
# fields, so a digest maps every recorded value to its path, leaving out only the
# fields below. A new volatile field therefore shows up as a difference.

Leaf = str | int | float | bool | None

# The policy files' only field that names the run rather than its learning.
_VOLATILE_POLICY = frozenset({"bc/run_id"})
# Game-attempt fields that read the clock, the host or the run's identity.
_VOLATILE_ATTEMPT = frozenset(
    {
        "attempt/seconds",
        "context/run",
        "context/coordinates/training_seconds",
        "context/coordinates/active_training_seconds",
        "context/coordinates/host_load_1m",
        "context/coordinates/rss_bytes",
        "context/coordinates/process_cpu_seconds",
    }
)
_POLICY_FILES = (
    "self-play-initial-raw.pt",
    "self-play-raw.pt",
    "self-play-ema.pt",
)
_OPTIMIZER_FILE = "self-play-optimizer.pt"
_ATTEMPTS_FILE = "game-attempts.jsonl"


def _flatten(value: object, path: str = "") -> Iterator[tuple[str, Leaf]]:
    """Every leaf under `value` by slash-joined path; a tensor is its exact bytes."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten(child, f"{path}/{key}" if path else str(key))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{path}/{index}" if path else str(index))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        exact = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        yield path, f"{tensor.dtype}{list(tensor.shape)}:{exact}"
    elif value is None or isinstance(value, str | int | float | bool):
        yield path, value
    else:
        raise TypeError(f"no digest for {type(value).__name__} at {path}")


def training_digest(training: Path) -> dict[str, Leaf]:
    """What a run learned and played, by path, from its `training/` folder."""
    digest: dict[str, Leaf] = {}
    for name in _POLICY_FILES:
        # These are the run's own exports, whose metadata needs full unpickling.
        policy = torch.load(training / name, map_location="cpu", weights_only=False)
        for path, leaf in _flatten(policy):
            if path not in _VOLATILE_POLICY:
                digest[f"{name}/{path}"] = leaf
    digest[_OPTIMIZER_FILE] = file_sha256(training / _OPTIMIZER_FILE)
    rows = (training / _ATTEMPTS_FILE).read_text().splitlines()
    digest[f"{_ATTEMPTS_FILE}/rows"] = len(rows)
    for index, line in enumerate(rows):
        for path, leaf in _flatten(json.loads(line)):
            if path not in _VOLATILE_ATTEMPT:
                digest[f"{_ATTEMPTS_FILE}/{index}/{path}"] = leaf
    receipt = TrainingRun.model_validate_json((training / "run.json").read_bytes())
    stage = receipt.stages[0]
    digest["receipt/updates"] = stage.updates
    digest["receipt/games"] = stage.games
    digest["receipt/environment_decisions"] = stage.environment_decisions
    digest["receipt/learner_transitions"] = stage.learner_transitions
    return digest


def digest_differences(
    left: Mapping[str, Leaf], right: Mapping[str, Leaf]
) -> list[str]:
    """Every path whose value differs or that only one side records."""
    return sorted(
        path
        for path in left.keys() | right.keys()
        if path not in left or path not in right or left[path] != right[path]
    )


def main() -> None:
    left, right = (Path(argument) for argument in sys.argv[1:3])
    differences = digest_differences(training_digest(left), training_digest(right))
    print("\n".join(differences) if differences else "digests equal")
    raise SystemExit(1 if differences else 0)


if __name__ == "__main__":
    main()
