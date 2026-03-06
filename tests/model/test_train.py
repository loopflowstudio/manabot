"""
test_train.py
Tests for PPO trainer rollout collection and flat-buffer updates.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Local imports
from manabot.env import (
    Match,
    ObservationSpace,
    Reward,
    RustVectorEnv,
)
from manabot.infra import (
    AgentHypers,
    Experiment,
    ExperimentHypers,
    Hypers,
    ObservationSpaceHypers,
    RewardHypers,
    TrainHypers,
)
from manabot.model import Agent, Trainer
from manabot.model.train import build_training_components


@contextmanager
def temp_run_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test runs."""
    original_runs = os.environ.get("MANABOT_RUNS_DIR")
    temp_dir = tempfile.mkdtemp(prefix="manabot_test_")
    os.environ["MANABOT_RUNS_DIR"] = temp_dir
    try:
        yield temp_dir
    finally:
        if original_runs:
            os.environ["MANABOT_RUNS_DIR"] = original_runs
        else:
            os.environ.pop("MANABOT_RUNS_DIR", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def run_dir() -> Generator[str, None, None]:
    with temp_run_dir() as d:
        yield d


@pytest.fixture
def observation_space():
    return ObservationSpace(ObservationSpaceHypers())


@pytest.fixture
def experiment(run_dir):
    return Experiment(
        ExperimentHypers(
            exp_name="test_run",
            seed=42,
            wandb=False,
            device="cpu",
            runs_dir=Path(run_dir),
        )
    )


@pytest.fixture
def trainer(observation_space, experiment):
    env = RustVectorEnv(
        num_envs=2,
        match=Match(),
        observation_space=observation_space,
        reward=Reward(RewardHypers()),
        device=experiment.device,
        opponent_policy="passive",
    )
    agent = Agent(
        observation_space,
        AgentHypers(hidden_dim=4, num_attention_heads=2),
    )
    hypers = TrainHypers(
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        total_timesteps=32,
        opponent_policy="passive",
    )
    t = Trainer(agent, experiment, env, hypers)
    yield t
    t.experiment.close()
    t.env.close()


class TestRollout:
    def test_rollout_step_shapes(self, trainer):
        next_obs, _ = trainer.env.reset()
        new_obs, reward, done, action, logprob, value = trainer._rollout_step(next_obs)

        assert isinstance(new_obs, dict)
        assert reward.shape == (trainer.hypers.num_envs,)
        assert done.shape == (trainer.hypers.num_envs,)
        assert action.shape == (trainer.hypers.num_envs,)
        assert logprob.shape == (trainer.hypers.num_envs,)
        assert value.shape == (trainer.hypers.num_envs,)

    def test_flatten_rollout_shapes(self, trainer):
        next_obs, _ = trainer.env.reset()
        (
            obs_buf,
            actions_buf,
            logprobs_buf,
            rewards_buf,
            dones_buf,
            values_buf,
        ) = trainer._init_rollout_buffers(next_obs)

        for step in range(trainer.hypers.num_steps):
            for key in obs_buf:
                obs_buf[key][step] = next_obs[key]
            actions_buf[step] = step
            logprobs_buf[step] = float(step)
            rewards_buf[step] = 1.0
            dones_buf[step] = False
            values_buf[step] = 0.5

        advantages = torch.ones_like(rewards_buf)
        returns = advantages + values_buf

        flat = trainer._flatten_rollout(
            obs_buf,
            actions_buf,
            logprobs_buf,
            advantages,
            returns,
            values_buf,
        )
        obs, logprobs, actions, advantages, returns, values = flat

        expected_batch = trainer.hypers.num_steps * trainer.hypers.num_envs
        assert logprobs.shape == (expected_batch,)
        assert actions.shape == (expected_batch,)
        assert advantages.shape == (expected_batch,)
        assert returns.shape == (expected_batch,)
        assert values.shape == (expected_batch,)
        for tensor in obs.values():
            assert tensor.shape[0] == expected_batch


class TestGAE:
    def test_gae_matches_reference_simple_case(self, trainer):
        trainer.hypers.num_steps = 4
        rewards = torch.ones((4, 2), dtype=torch.float32)
        values = torch.zeros((4, 2), dtype=torch.float32)
        dones = torch.zeros((4, 2), dtype=torch.bool)
        next_value = torch.zeros(2, dtype=torch.float32)
        next_done = torch.zeros(2, dtype=torch.bool)

        advantages, returns = trainer._compute_gae(
            rewards,
            values,
            dones,
            next_value,
            next_done,
            gamma=1.0,
            gae_lambda=1.0,
        )

        expected = torch.tensor(
            [[4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(advantages, expected)
        assert torch.allclose(returns, expected)


class TestPPOCorrectnessFixes:
    def test_truncated_episode_marks_done(self, trainer, monkeypatch):
        next_obs, _ = trainer.env.reset()
        truncated = torch.tensor(
            [True, False], dtype=torch.bool, device=trainer.experiment.device
        )
        terminated = torch.zeros_like(truncated)
        reward = torch.zeros(
            trainer.hypers.num_envs,
            dtype=torch.float32,
            device=trainer.experiment.device,
        )

        def fake_step(_action):
            return (
                next_obs,
                reward,
                terminated,
                truncated,
                {"action_space_truncated": np.array([False, False])},
            )

        monkeypatch.setattr(trainer.env, "step", fake_step)
        trainer.logger.warning = MagicMock()
        _, _, done, _, _, _ = trainer._rollout_step(next_obs)

        assert done.tolist() == [True, False]
        trainer.logger.warning.assert_called_once_with(
            "Truncation in 1/2 envs (no value bootstrap)"
        )

    def test_advantage_normalization(self, trainer):
        advantages = torch.tensor([1.0, 2.0, 3.0], device=trainer.experiment.device)

        trainer.hypers.norm_adv = True
        normalized = trainer._maybe_normalize_advantages(advantages)
        assert normalized.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert normalized.std().item() == pytest.approx(1.0, abs=1e-5)

        trainer.hypers.norm_adv = False
        untouched = trainer._maybe_normalize_advantages(advantages)
        assert torch.equal(untouched, advantages)

    def test_actual_batch_size_used(self, trainer):
        inds, minibatch_size = trainer._build_minibatch_plan(actual_batch_size=7)
        assert minibatch_size == 3
        assert len(inds) == 7
        assert inds[-1] == 6


def test_training_loop_runs_100_steps(observation_space, experiment):
    env = RustVectorEnv(
        num_envs=2,
        match=Match(),
        observation_space=observation_space,
        reward=Reward(RewardHypers()),
        device=experiment.device,
        opponent_policy="passive",
    )
    agent = Agent(observation_space, AgentHypers(hidden_dim=4, num_attention_heads=2))
    hypers = TrainHypers(
        num_envs=2,
        num_steps=10,
        num_minibatches=2,
        update_epochs=1,
        total_timesteps=100,
        opponent_policy="passive",
    )

    trainer = Trainer(agent, experiment, env, hypers)
    assert np.isnan(trainer.last_explained_variance)
    trainer.train()
    assert trainer.global_step == 100
    assert not np.isnan(trainer.last_explained_variance)


def test_build_training_components_uses_rust_vector_env(run_dir):
    hypers = Hypers(
        experiment=ExperimentHypers(wandb=False, device="cpu", runs_dir=Path(run_dir)),
        train=TrainHypers(
            total_timesteps=8,
            num_envs=2,
            num_steps=2,
        ),
    )
    experiment, env, _ = build_training_components(hypers)
    try:
        assert isinstance(env, RustVectorEnv)
    finally:
        env.close()
        experiment.close()
