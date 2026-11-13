"""Active-time endpoints count completed learner work, excluding slow exports."""

import pytest

from experiments.ataraxos_move import ataraxos_move_scalar
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from tests.training.test_ataraxos_regime import _recipe


def recipe(updates: int = 10) -> TrainingRegime:
    result = _recipe(ataraxos_move_scalar)
    result.stages = result.stages[:1]
    result.agent.hidden_dim = 8
    stage = result.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.transitions = 4
    stage.updates = updates
    stage.active_seconds = 15
    stage.execution.wall_seconds = 900
    result.wall_seconds = 1000
    return TrainingRegime.model_validate(result.model_dump())


def test_old_recipe_bytes_and_invalid_endpoint() -> None:
    old = _recipe(ataraxos_move_scalar)
    assert all("active_seconds" not in stage for stage in old.model_dump()["stages"])
    payload = recipe().model_dump()
    payload["stages"][0]["active_seconds"] = 900
    with pytest.raises(ValueError, match="watchdog reserve"):
        TrainingRegime.model_validate(payload)
    payload = recipe().model_dump()
    payload["recovery_max_microsteps"] = 10000
    payload["schedule_clock"] = "iteration_fraction"
    with pytest.raises(ValueError, match="active-time process recovery"):
        TrainingRegime.model_validate(payload)
