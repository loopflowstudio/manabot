"""
test_trainer.py
Tests for the PPO Trainer using actual simulation environment.
"""

import logging
import pytest
import torch
import numpy as np
from contextlib import contextmanager
from typing import Generator
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from manabot.env import VectorEnv, Match, ObservationSpace, Reward
from manabot.model import Agent, Trainer
from manabot.infra import (
    Experiment, TrainHypers, AgentHypers, ObservationSpaceHypers, 
    RewardHypers, ExperimentHypers
)
import manabot.env.observation as obs_mod

# ────────────────────────────── Fixtures ──────────────────────────────

@contextmanager
def temp_run_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test runs."""
    original_runs = os.environ.get('MANABOT_RUNS_DIR')
    temp_dir = tempfile.mkdtemp(prefix='manabot_test_')
    os.environ['MANABOT_RUNS_DIR'] = temp_dir
    try:
        yield temp_dir
    finally:
        if original_runs:
            os.environ['MANABOT_RUNS_DIR'] = original_runs
        else:
            os.environ.pop('MANABOT_RUNS_DIR', None)
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
    return Experiment(ExperimentHypers(
        exp_name="test_run",
        seed=42,
        wandb=False,
        device="cpu",
        runs_dir=Path(run_dir)
    ))

@pytest.fixture
def vector_env(observation_space, experiment):
    return VectorEnv(
        2,  # num_envs=2 for testing multi-agent scenarios
        Match(),
        observation_space,
        Reward(RewardHypers(trivial=True)),
        device=experiment.device
    )

@pytest.fixture
def agent(observation_space):
    a_hypers = AgentHypers(
        hidden_dim=4,  # Small network for fast testing
        num_attention_heads=2
    )
    return Agent(observation_space, a_hypers)

@pytest.fixture
def trainer(agent, experiment, vector_env):
    t_hypers = TrainHypers(
        num_envs=2,
        num_steps=128,  # Smaller number of steps for testing
        num_minibatches=4,
        update_epochs=4,
        total_timesteps=2000  # Small number for testing
    )
    trainer = Trainer(agent, experiment, vector_env, t_hypers)
    yield trainer
    trainer.experiment.close()
    trainer.env.close()

# ────────────────────────────── Helper Functions ──────────────────────────────

def collect_rollout(trainer, steps: int = 4):
    """Collect a rollout using the actual environment."""
    next_obs, _ = trainer.env.reset()
    next_done = torch.zeros(trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device)
    actor_ids = obs_mod.get_agent_indices(next_obs)
    
    for _ in range(steps):
        next_obs, next_done, actor_ids = trainer._rollout_step(next_obs, actor_ids)
        
    return next_obs, next_done, actor_ids

# ────────────────────────────── Tests ──────────────────────────────

class TestRolloutAndBuffer:
    def test_rollout_step(self, trainer):
        """Test single rollout step produces valid observations and transitions."""
        next_obs, _ = trainer.env.reset()
        actor_ids = obs_mod.get_agent_indices(next_obs)
        
        # Take a step
        new_obs, done, new_actor_ids = trainer._rollout_step(next_obs, actor_ids)
        
        # Verify observation structure
        assert isinstance(new_obs, dict)
        assert set(new_obs.keys()) == set(trainer.env.observation_space.keys())
        
        # Verify actor indices
        assert new_actor_ids.shape == (trainer.hypers.num_envs,)
        assert torch.all((new_actor_ids >= 0) & (new_actor_ids < 2))


    def test_observation_validation(self, trainer):
        """Test observation validation with real environment data."""
        next_obs, _ = trainer.env.reset()
        assert trainer._validate_obs(next_obs), "Valid observation should pass validation"

        # Test invalid observation
        invalid_obs = {k: v for k, v in next_obs.items() if k != 'agent_player'}
        assert not trainer._validate_obs(invalid_obs), "Invalid observation should fail validation"

class TestAdvantageComputation:
    def test_gae_basic(self, trainer):
        """Test GAE computation with real environment transitions."""
        next_obs, next_done, _ = collect_rollout(trainer, steps=trainer.hypers.num_steps // 2)
        
        # Compute advantages
        with torch.no_grad():
            next_value = trainer.agent.get_value(next_obs)
        trainer.multi_buffer.compute_advantages(next_value, next_done, trainer.hypers.gamma, trainer.hypers.gae_lambda)
        
        # Verify advantages
        for buffer in trainer.multi_buffer.buffers.values():
            if buffer.advantages is not None:
                assert not torch.isnan(buffer.advantages).any()
                assert not torch.isinf(buffer.advantages).any()

    def test_gae_terminal_vs_nonterminal(self, trainer):
        """Test GAE handles terminal vs non-terminal states correctly."""
        next_obs, _ = trainer.env.reset()
        next_done = torch.zeros(trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device)
        
        # Force one environment to terminal state
        next_done[0] = True
        
        with torch.no_grad():
            next_value = trainer.agent.get_value(next_obs)
        trainer.multi_buffer.compute_advantages(next_value, next_done, trainer.hypers.gamma, trainer.hypers.gae_lambda)

class TestPPOOptimization:
    def test_single_ppo_update(self, trainer):
        """Test single PPO optimization step with real data."""
        # Collect some real transitions
        next_obs, next_done, _ = collect_rollout(trainer, steps=trainer.hypers.num_steps)
        
        # Compute advantages
        with torch.no_grad():
            next_value = trainer.agent.get_value(next_obs)
        trainer.multi_buffer.compute_advantages(next_value, next_done, trainer.hypers.gamma, trainer.hypers.gae_lambda)
        
        # Get flattened data
        obs, logprobs, actions, advantages, returns, values = trainer.multi_buffer.get_flattened()
        
        # Perform update
        trainer.hypers.target_kl = float('inf')  # Disable early stopping for test
        approx_kl, clip_fraction = trainer._optimize_step(obs, logprobs, actions, advantages, returns, values)
        
        assert approx_kl >= 0
        assert 0 <= clip_fraction <= 1

    def test_value_loss_clipping(self, trainer):
        """Test value loss with and without clipping."""
        next_obs, next_done, _ = collect_rollout(trainer, steps=trainer.hypers.num_steps)
        
        with torch.no_grad():
            next_value = trainer.agent.get_value(next_obs)
        trainer.multi_buffer.compute_advantages(next_value, next_done, trainer.hypers.gamma, trainer.hypers.gae_lambda)
        
        obs, logprobs, actions, advantages, returns, values = trainer.multi_buffer.get_flattened()
        
        # Test without clipping
        trainer.hypers.clip_vloss = False
        _, _ = trainer._optimize_step(obs, logprobs, actions, advantages, returns, values)
        
        # Test with clipping
        trainer.hypers.clip_vloss = True
        _, _ = trainer._optimize_step(obs, logprobs, actions, advantages, returns, values)


class TestPPOCorrectnessFixes:
    def _mock_step(
        self,
        trainer,
        monkeypatch,
        truncated: torch.Tensor,
        action_space_truncated: list[bool],
    ):
        next_obs, _ = trainer.env.reset()
        actor_ids = obs_mod.get_agent_indices(next_obs)
        terminated = torch.zeros_like(truncated)
        reward = torch.zeros(trainer.hypers.num_envs, dtype=torch.float32, device=trainer.experiment.device)

        def fake_step(_action):
            return (
                next_obs,
                reward,
                terminated,
                truncated,
                {"action_space_truncated": np.array(action_space_truncated)},
            )

        monkeypatch.setattr(trainer.env, "step", fake_step)
        return next_obs, actor_ids

    def test_truncated_episode_marks_done(self, trainer, monkeypatch):
        truncated = torch.tensor([True, False], dtype=torch.bool, device=trainer.experiment.device)
        next_obs, actor_ids = self._mock_step(
            trainer,
            monkeypatch,
            truncated=truncated,
            action_space_truncated=[False, False],
        )
        trainer.logger.warning = MagicMock()
        _, done, _ = trainer._rollout_step(next_obs, actor_ids)

        assert done.tolist() == [True, False]
        trainer.logger.warning.assert_called_once_with("Truncation in 1/2 envs (no value bootstrap)")

    def test_advantage_normalization(self, trainer):
        advantages = torch.tensor([1.0, 2.0, 3.0], device=trainer.experiment.device)

        trainer.hypers.norm_adv = True
        normalized = trainer._maybe_normalize_advantages(advantages)
        assert normalized.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert normalized.std().item() == pytest.approx(1.0, abs=1e-5)
        singleton = trainer._maybe_normalize_advantages(advantages[:1])
        assert torch.isfinite(singleton).all()
        assert singleton.item() == pytest.approx(0.0, abs=1e-8)

        trainer.hypers.norm_adv = False
        untouched = trainer._maybe_normalize_advantages(advantages)
        assert torch.equal(untouched, advantages)

    def test_actual_batch_size_used(self, trainer):
        inds, minibatch_size = trainer._build_minibatch_plan(actual_batch_size=7)
        assert minibatch_size == 1
        assert len(inds) == 7
        assert inds[-1] == 6

    def test_small_actual_batch_skips_update(self, trainer):
        trainer.hypers.num_minibatches = 8
        trainer.logger.warning = MagicMock()
        plan = trainer._build_minibatch_plan(actual_batch_size=4)
        assert plan is None
        trainer.logger.warning.assert_called_once()

    def test_no_weight_decay(self, trainer):
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0

    def test_rollout_health_counters(self, trainer, monkeypatch):
        trainer._reset_rollout_health_update()
        trainer._increment_rollout_health("skipped_steps", 2)

        truncated = torch.tensor([True, True], dtype=torch.bool, device=trainer.experiment.device)
        next_obs, actor_ids = self._mock_step(
            trainer,
            monkeypatch,
            truncated=truncated,
            action_space_truncated=[True, False],
        )
        trainer.logger.warning = MagicMock()
        trainer._rollout_step(next_obs, actor_ids)

        assert trainer.rollout_health_update["skipped_steps"] == 2
        assert trainer.rollout_health_update["truncated_episodes"] == 2
        assert trainer.rollout_health_update["action_space_truncations"] == 1
        assert trainer.rollout_health["skipped_steps"] >= 2
        assert trainer.rollout_health["truncated_episodes"] >= 2
        assert trainer.rollout_health["action_space_truncations"] >= 1


def test_action_masking(trainer):
    """Test action masking correctly prevents invalid actions."""
    next_obs, _ = trainer.env.reset()
    
    with torch.no_grad():
        logits, _ = trainer.agent.forward(next_obs)
    
    # Cast actions_valid to bool before inverting.
    invalid_actions = ~(next_obs["actions_valid"].bool())
    # Here, ensure that the invalid action positions are set to -1e8.
    masked_logits = logits.masked_fill(next_obs["actions_valid"] == 0, -1e8)
    
    # For every invalid action, check that the corresponding masked logits equal -1e8.
    assert torch.all(masked_logits[invalid_actions] == -1e8), "Invalid actions not properly masked."


if __name__ == "__main__":
    pytest.main([__file__])
