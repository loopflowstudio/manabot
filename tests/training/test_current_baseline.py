"""Positive-control engine outcomes and real collector boundary checks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.runners import current_baseline as baseline
from experiments.runners.current_baseline import declaration
from experiments.runners.current_baseline_evidence import Result
from manabot.env import Match, ObservationSpace, Reward
from manabot.env.target_practice import resolve_target, target_root
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.sim.net_opponent import SeatRoutedCollector, transition_gae
from manabot.training.models import TrainingRegime, TrainingRun
from manabot.verify.store import VerifyStore


def test_targets_reverse_with_seat_and_resolve_in_engine() -> None:
    regime = declaration().resolve().cases[0].regime
    space = ObservationSpace(regime.observation)
    for seed in (118, 119, 22243):
        for seat in (0, 1):
            for action in (0, 1):
                env = target_root(Match(regime.match), space, seed, seat)
                assert resolve_target(env, action) == 1 - action


def test_real_collector_terminal_rewards_and_advantages() -> None:
    torch.set_num_threads(1)
    regime = declaration().resolve().cases[0].regime
    space = ObservationSpace(regime.observation)
    agent = Agent(space, regime.agent)
    collector = SeatRoutedCollector(
        space,
        Match(regime.match),
        Reward(RewardHypers()),
        num_envs=4,
        seed=118,
        root="lethal-target-v1",
    )
    batch = collector.collect(agent, 4)
    assert batch.dones.all()
    expected = np.where(1 - batch.actions == np.arange(4) % 2, 1, -1)
    np.testing.assert_array_equal(batch.rewards, expected)
    advantages, returns = transition_gae(
        torch.tensor(batch.rewards),
        torch.zeros_like(torch.tensor(batch.values)),
        torch.tensor(batch.dones),
        torch.full((4,), 100.0),
        1.0,
        0.5,
    )
    torch.testing.assert_close(advantages, torch.tensor(batch.rewards))
    torch.testing.assert_close(returns, torch.tensor(batch.rewards))
    assert np.all(batch.probabilities[:, :, 2:] == 0)


def test_failed_debugging_attempt_retains_typed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(
        regime: TrainingRegime, seed: int, out: Path, store: VerifyStore
    ) -> TrainingRun:
        raise RuntimeError("retained fixture failure")

    monkeypatch.setattr(baseline, "execute_regime", fail)
    monkeypatch.setattr(baseline.signal, "signal", lambda *args: None)
    monkeypatch.setattr(baseline.signal, "setitimer", lambda *args: None)
    prior = tmp_path / "prior.json"
    prior.write_text(
        Result(status="failed", seconds=2, prior_seconds=3).model_dump_json()
    )
    with pytest.raises(RuntimeError, match="retained fixture failure"):
        baseline.run(tmp_path / "attempt", prior)
    result = Result.model_validate_json((tmp_path / "attempt/result.json").read_text())
    assert result.status == "failed" and result.prior_seconds == 5
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "failed"
    assert (
        result.attempts[0].error
        == result.error
        == "RuntimeError: retained fixture failure"
    )
    assert result.seconds >= result.attempts[0].seconds > 0
