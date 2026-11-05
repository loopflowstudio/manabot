"""MTG-252 critic readout arms, derived from the frozen v10 TLA pilot recipe.

The comparison changes one serialized field, `agent.value_aggregation`, and
nothing else: depth, width, heads, capacities, action support, learning rule,
filter, lambdas and execution stay those of `tla-trophy-213-v10.json`. The
study design lives in `experiments/value-readout-252.md`; this module owns the
arm derivation, its single-axis check and the static per-arm model costs that
the protocol reports before any training.

No launch path lives here. Execution, budget and the frozen scoring protocol
remain separate decisions recorded in that document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ValueAggregation
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.builders import with_value_aggregation
from manabot.training.regime import TrainingRegime

ROOT = Path(__file__).resolve().parents[1]
BASE_REGIME = ROOT / "experiments/regimes/tla-trophy-213-v10.json"
# Canonical order; the baseline is the v10 recipe's own masked mean.
ARMS: tuple[ValueAggregation, ...] = ("masked_mean", "value_token", "attention_readout")


def base() -> TrainingRegime:
    regime = TrainingRegime.model_validate_json(BASE_REGIME.read_bytes())
    if regime.agent.value_aggregation != ARMS[0]:
        raise ValueError("v10 baseline is no longer the masked-mean recipe")
    return regime


def arm(aggregation: ValueAggregation) -> TrainingRegime:
    """The v10 recipe with only its critic readout replaced."""
    regime = base()
    derived = with_value_aggregation(
        regime, id=f"value-readout-252-{aggregation}", aggregation=aggregation
    )
    require_single_axis(regime, derived)
    return derived


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
