"""The long curriculum changes duration and recovery, preserving the intervention."""

import json
from pathlib import Path

from experiments.runners.experiment_mirrors import (
    ARMS,
    declaration,
    long_recipe,
    recipe,
)
from manabot.training.models import TrainSelfPlay


def test_long_recipe_preserves_model_and_update_meaning() -> None:
    for arm in ARMS:
        pilot = recipe(arm, 120, 4000)
        long = long_recipe(arm, 400000)
        assert long.agent == pilot.agent
        assert long.match == pilot.match
        assert long.observation == pilot.observation
        assert (
            sum(s.updates for s in long.stages if isinstance(s, TrainSelfPlay)) == 10000
        )
        assert long.recovery_max_microsteps == 20000000
        assert long.recovery_every_updates == 250
        previous: str | None = None
        for s in long.stages:
            assert isinstance(s, TrainSelfPlay)
            assert s.initial == previous
            assert s.matchup_curriculum == arm
            assert s.learning == pilot.stages[0].learning
            assert (s.streams, s.transitions, s.execution.threads) == (12, 32, 1)
            previous = s.id


def test_long_monitoring_has_four_precise_cells_and_fixed_paired_seeds(
    tmp_path: Path,
) -> None:
    plan = {
        "scope": "two-seed-10k",
        "seeds": [12551, 12552],
        "updates": 10000,
        "run_allowance_seconds": 400000,
        "wall_seconds": 2000000,
        "process_seconds": 1800000,
        "checkpoint_seconds": 2000000,
        "evaluation_reserve_seconds": 150000,
        "attempt_seconds": 18000,
        "greedy_games_per_cell": 100,
        "random_games_per_cell": 10,
        "greedy_deal_start": 1912610000,
        "random_deal_start": 1912620000,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    experiment = declaration(tmp_path)
    schedule = experiment.schedule
    assert schedule is not None
    assert schedule.seeds == (12551, 12552)
    assert schedule.order == ((0, 1), (1, 0))
    assert len(schedule.monitoring.protocol.deal_seeds) == 50
    assert schedule.monitoring.protocol.include_mirrors
    assert len(schedule.monitoring.additional_protocols[0].deal_seeds) == 5
    assert schedule.monitoring.include_initial
    assert schedule.checkpoint_seconds > plan["run_allowance_seconds"]
    for regime in experiment.resolve().regimes.values():
        assert [s.updates for s in regime.stages] == [2500, 2500, 5000]
