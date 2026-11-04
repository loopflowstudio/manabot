"""The explicit policy prefix survives collection and evaluation encodings."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.env.env import Env
from manabot.env.failures import CaptureScope, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.model.actor_critic import ActorCritic
from manabot.training.rollout.collection import SeatRoutedCollector


@pytest.mark.parametrize("workers", [1, 2])
def test_collector_uses_configured_prefix_and_one_masked_distribution(
    workers: int, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    agent = ActorCritic(space, ModelConfig(hidden_dim=8, attention_on=False))
    match = Match(
        MatchConfig(hero_deck={"Mountain": 30}, villain_deck={"Mountain": 30})
    )
    with capture_scope(CaptureScope(tmp_path)):
        collector = SeatRoutedCollector(
            space,
            match,
            Reward(RewardConfig()),
            num_envs=2,
            seed=213,
            workers=workers,
            opponent_mode="self",
            learning_seats="both",
        )
        try:
            batch = collector.collect(agent, 2)
        finally:
            collector.close()
    valid = batch.obs["actions_valid"].astype(bool)
    assert batch.probabilities.shape == valid.shape
    assert np.all(batch.probabilities[~valid] == 0)
    assert np.all(batch.probabilities[valid] > 0)
    np.testing.assert_allclose(batch.probabilities.sum(-1), 1, atol=1e-6)
    assert np.all(np.take_along_axis(valid, batch.actions[..., None], -1))
    reports = [
        json.loads(path.read_bytes()) for path in tmp_path.glob("action-cutoff-*.json")
    ]
    assert len(reports) == workers
    assert (
        sum(
            sum(counts["overflows"] for counts in report["counts_by_kind"].values())
            for report in reports
        )
        > 0
    )


def test_scalar_evaluation_cutoff_is_diagnostic_not_game_truncation(
    tmp_path: Path,
) -> None:
    space = ObservationSpace(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    match = Match(
        MatchConfig(hero_deck={"Mountain": 30}, villain_deck={"Mountain": 30})
    )
    with capture_scope(CaptureScope(tmp_path)):
        env = Env(match, space, Reward(RewardConfig()), seed=213)
        try:
            obs, _ = env.reset()
            assert obs["actions_valid"].sum() == 2
            assert len(env.last_raw_obs.action_space.actions) > 2
            _, _, _, truncated, info = env.step(0)
            assert not truncated and not info["action_space_truncated"]
        finally:
            env.close()
    (path,) = tmp_path.glob("action-cutoff-*.json")
    report = json.loads(path.read_bytes())
    assert report["samples"][0]["game"]["seed"] == 213
    assert report["samples"][0]["retained_count"] == 2
