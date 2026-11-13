"""Bounded real-engine recipe execution, admission and iteration contracts."""

from collections.abc import Callable
from pathlib import Path

import pytest

from experiments.ataraxos_move import ataraxos_move, ataraxos_move_scalar
from manabot.training.regime import DampedPolicyLearning, TrainingRegime
from manabot.training.validation import validate_regime
import managym

ROOT = Path(__file__).resolve().parents[2]


def _recipe(build: Callable[[], TrainingRegime] = ataraxos_move) -> TrainingRegime:
    recipe = build()
    recipe.world = managym.WORLD_VERSION
    recipe.id = f"test-{recipe.world}-{recipe.id}"
    return recipe


def test_iteration_schedule_and_invalid_config() -> None:
    learning = DampedPolicyLearning(gradient="damped_policy")
    assert learning.rates(1) == (1e-4, 0.05)
    rate, tau = learning.rates(10000)
    assert rate == pytest.approx(0.5 / 10000**1.1)
    assert tau == pytest.approx(0.05 / 10000**0.3)
    assert learning.rates(10**9)[0] == 5e-6
    with pytest.raises(ValueError, match="positive"):
        learning.rates(0)
    with pytest.raises(ValueError):
        DampedPolicyLearning.model_validate({"gradient": "damped_policy", "epochs": 4})
    bad = _recipe().model_dump()
    bad["stages"][0]["learning"] = {"gradient": "ppo"}
    with pytest.raises(ValueError, match="categorical"):
        validate_regime(bad)
    bad = _recipe(ataraxos_move_scalar).model_dump()
    bad["stages"][1]["learning"] = {"gradient": "ppo", "ema": 0.999}
    with pytest.raises(ValueError, match="gradient"):
        validate_regime(bad)
