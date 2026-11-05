"""MTG-252 critic readout arms, derived from the MTG-183 Elemental recipe.

The base is `elemental-long-183-v1.json` (ten Elemental decks, uniform
per-game pairings, two-seat self-play) with only its world re-pinned to the
native runtime's. The comparison then changes one serialized field,
`agent.value_aggregation`, and nothing else: depth, width, heads, capacities,
learning rule, filter, lambdas and execution stay those of that base. The
study design lives in `experiments/value-readout-252.md`; this module owns the
arm derivation, its single-axis check and the static per-arm model costs that
the protocol reports before any training, and `runs()`, the Etude launch of
every arm under one shared seed list (3 arms x 3 seeds x 150 updates inside a
USD20 study cap that also covers evaluation):

    doppler run -p etude -c prd -- doppler run -p manabot -c prd \\
        --preserve-env=RUNPOD_API_KEY -- \\
        uv run --with "etude-sdk[s3,eval]==0.2.14" etude launch \\
        experiments/value_readout_252.py --rent 07c490-l4 --cohort mtg252 \\
        --parallel 9 --max-usd 13.05 --revision COMMIT
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from etude.launch.records import Run as EtudeRun
import torch

from experiments.learning_signal import training_runs
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ValueAggregation
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.builders import with_value_aggregation
from manabot.training.execution import validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym

ROOT = Path(__file__).resolve().parents[1]
BASE_REGIME = ROOT / "experiments/regimes/elemental-long-183-v1.json"
# Canonical order; the baseline is the 183 recipe's own masked mean.
ARMS: tuple[ValueAggregation, ...] = ("masked_mean", "value_token", "attention_readout")
EXPERIMENT = "value_readout_252"
SEEDS = "25201,25202,25203"
UPDATES = 150
# MTG-186 trained this two-seat Elemental recipe at about 44 s/update on an L4
# (w13); 150 updates is about 1.8 h, plus 15 min setup and 5 min finalization.
# The watchdog stops a slower run with its exports; the L4 ceiling is $0.60/h.
HOURS = 2.4
USD_PER_RUN = 1.45


def base() -> TrainingRegime:
    """MTG-183's recipe in the native world; the frozen 183 file is unchanged.

    Re-pinning resolves the saved observation at the current fresh policy
    input. The protocol pins the resulting world, shapes and source.
    """
    saved = TrainingRegime.model_validate_json(BASE_REGIME.read_bytes())
    if saved.agent.value_aggregation != ARMS[0]:
        raise ValueError("183 baseline is no longer the masked-mean recipe")
    regime = saved.model_copy(
        update={"id": "value-readout-252-base", "world": managym.WORLD_VERSION}
    )
    return validate_regime(regime)


def arm(aggregation: ValueAggregation, updates: int | None = None) -> TrainingRegime:
    """The re-pinned 183 recipe with only its critic readout replaced.

    `updates` sets the stage's absolute target for every arm alike. Damped
    learning rates follow the absolute iteration, so a shorter target leaves
    the learning schedule unchanged.
    """
    regime = base()
    if updates is not None:
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = updates
    derived = with_value_aggregation(
        regime, id=f"value-readout-252-{aggregation}", aggregation=aggregation
    )
    require_single_axis(regime, derived)
    return derived


@dataclass(frozen=True)
class Run:
    """One training run: an arm under one shared run seed.

    The run seed fixes initialization and the collection seed
    (`run seed + 10000`). Native game `k` on stream `s` is seeded
    `collection + s + k * streams`, and its Elemental pairing and deal are a
    function of that seed alone, so every arm under one run seed meets the same
    pairing sequence on every stream whatever its policy does. Arms differ only
    in how far through that sequence they are at a given update.
    """

    aggregation: ValueAggregation
    seed: int
    regime: TrainingRegime


def grid(seeds: Sequence[int], updates: int | None = None) -> list[Run]:
    """Every arm under every seed: one seed list, never per-arm seeds."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("readout runs need distinct seeds shared by every arm")
    regimes = {aggregation: arm(aggregation, updates) for aggregation in ARMS}
    return [
        Run(aggregation=aggregation, seed=seed, regime=regimes[aggregation])
        for seed in seeds
        for aggregation in ARMS
    ]


def runs(
    revision: str,
    seeds: str = SEEDS,
    updates: int = UPDATES,
    hours: float = HOURS,
    usd_per_run: float = USD_PER_RUN,
) -> list[EtudeRun]:
    """Etude training runs: every arm under the same comma-separated seeds."""
    chosen = tuple(int(seed) for seed in seeds.split(","))
    planned = grid(chosen, updates)
    return training_runs(
        (arm(aggregation, updates) for aggregation in ARMS),
        updates=updates,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=tuple(sorted({run.seed for run in planned})),
        hours=hours,
        usd_per_run=usd_per_run,
    )


def require_single_axis(baseline: TrainingRegime, candidate: TrainingRegime) -> None:
    """Reject any resolved difference other than the label and value readout."""
    left = baseline.model_dump(mode="json")
    right = candidate.model_dump(mode="json")
    for dump in (left, right):
        dump.pop("id")
        dump["agent"].pop("value_aggregation")
    if left != right:
        raise ValueError("readout arm changes more than agent.value_aggregation")


@dataclass(frozen=True)
class ArmCost:
    """Static model cost of one arm at the recipe's capacities.

    `attention_pairs` counts query-key products per observation across all
    object-attention blocks plus any readout query; it is a size measure, not
    measured time. Shared parameters are identical to the baseline's under the
    same initialization seed except where `shares_initialization` is false.
    """

    aggregation: ValueAggregation
    parameters: int
    added_parameters: int
    objects: int
    attention_pairs: int
    shares_initialization: bool


def costs(seed: int = 252) -> list[ArmCost]:
    regime = base()
    space = ObservationSpace(regime.observation)
    models: dict[ValueAggregation, ActorCritic] = {}
    for aggregation in ARMS:
        torch.manual_seed(seed)
        models[aggregation] = ActorCritic(space, arm(aggregation).agent)
    baseline = models[ARMS[0]].state_dict()
    baseline_size = sum(p.numel() for p in models[ARMS[0]].parameters())
    layers = regime.agent.attention_layers
    # Visible object sequence: both players, then each seat's cards and permanents.
    capacity = regime.observation
    objects = 2 + 2 * (
        capacity.max_cards_per_player + capacity.max_permanents_per_player
    )
    result = []
    for aggregation, model in models.items():
        state = model.state_dict()
        size = sum(p.numel() for p in model.parameters())
        sequence = objects + (aggregation == "value_token")
        result.append(
            ArmCost(
                aggregation=aggregation,
                parameters=size,
                added_parameters=size - baseline_size,
                objects=objects,
                attention_pairs=layers * sequence**2
                + (objects if aggregation == "attention_readout" else 0),
                shares_initialization=all(
                    torch.equal(state[key], value) for key, value in baseline.items()
                ),
            )
        )
    return result
