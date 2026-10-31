"""Ataraxos treatment ablation arms (experiments/ataraxos-mtg-ablations.md).

Each arm is `direct_self_play`'s setup with only some Ataraxos treatments kept:
separate policy/value estimators (value lambda 1), advantage filtering (half the
samples retained) and coordinated learning-rate/regularization decay.

`uv run --extra notebook python -m experiments.ataraxos_ablation --out .runs/ablations`
runs the five-arm study (smoke by default; see `--help`).
"""

from manabot.experiment.study import main
from manabot.infra.config import ModelConfig
from manabot.recipes.builders import PACK, ppo_checkpoints
from manabot.sim.matchup import selected_match
from manabot.training.regime import Learning, Schedule, TrainingRegime
import managym


def _arm(
    id: str,
    *,
    separate_estimators: bool,
    advantage_filtering: bool,
    coordinated_decay: bool,
) -> TrainingRegime:
    decay = 9 if coordinated_decay else 0
    learning = Learning(
        value_lambda=1.0 if separate_estimators else 0.95,
        retained_fraction=0.5 if advantage_filtering else 1.0,
        learning_rate=Schedule(initial=2.5e-4, decay=decay),
        tau=Schedule(initial=0.01, decay=decay),
    )
    return TrainingRegime(
        id=id,
        world=managym.WORLD_VERSION,
        match=selected_match(),
        agent=ModelConfig(world=PACK),
        stages=ppo_checkpoints(2, learning),
        wall_seconds=900,
    )


def rl_control() -> TrainingRegime:
    """No Ataraxos treatment: the PPO control."""
    return _arm(
        "rl-control",
        separate_estimators=False,
        advantage_filtering=False,
        coordinated_decay=False,
    )


def separate_estimators() -> TrainingRegime:
    return _arm(
        "separate-estimators",
        separate_estimators=True,
        advantage_filtering=False,
        coordinated_decay=False,
    )


def advantage_filtering() -> TrainingRegime:
    return _arm(
        "advantage-filtering",
        separate_estimators=False,
        advantage_filtering=True,
        coordinated_decay=False,
    )


def coordinated_decay() -> TrainingRegime:
    return _arm(
        "coordinated-decay",
        separate_estimators=False,
        advantage_filtering=False,
        coordinated_decay=True,
    )


def combined() -> TrainingRegime:
    """Every Ataraxos treatment."""
    return _arm(
        "combined",
        separate_estimators=True,
        advantage_filtering=True,
        coordinated_decay=True,
    )


# Arm order is the study's; the first arm is the paired reference.
ARMS = (
    rl_control,
    separate_estimators,
    advantage_filtering,
    coordinated_decay,
    combined,
)

if __name__ == "__main__":
    main("ataraxos-ablations", ARMS)
