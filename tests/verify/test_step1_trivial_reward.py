"""Integration test for step1: PPO optimizer sanity with trivial +1 reward."""

import pytest
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.model.train import run_training
from manabot.verify.util import build_hypers


def _estimate_value(agent, obs_space, match, reward, seed: int) -> float:
    env = Env(match, obs_space, reward, seed=seed, auto_reset=False)
    try:
        obs, _ = env.reset(seed=seed)
    finally:
        env.close()

    device = next(agent.parameters()).device
    tensor_obs = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs.items()
    }
    with torch.no_grad():
        value = agent.get_value(tensor_obs).mean().item()
    return float(value)


@pytest.mark.slow
@pytest.mark.training
def test_trivial_reward_explained_variance():
    """PPO with constant +1 reward should converge: explained variance > 0.8."""
    num_envs = 4
    num_steps = 128
    updates = 500
    total_timesteps = num_envs * num_steps * updates

    hypers = build_hypers(
        experiment={"seed": 1, "exp_name": "test-step1-trivial-reward"},
        reward={"trivial": True},
        train={
            "num_envs": num_envs,
            "num_steps": num_steps,
            "total_timesteps": total_timesteps,
        },
    )

    trainer = run_training(hypers)

    assert trainer.last_explained_variance > 0.8, (
        f"explained_variance={trainer.last_explained_variance:.3f}, expected > 0.8"
    )


@pytest.mark.slow
@pytest.mark.training
def test_trivial_reward_value_estimate_direction():
    """Value estimate should be positive and heading toward 1/(1-gamma)."""
    num_envs = 4
    num_steps = 128
    updates = 500
    total_timesteps = num_envs * num_steps * updates

    hypers = build_hypers(
        experiment={"seed": 2, "exp_name": "test-step1-value-direction"},
        reward={"trivial": True},
        train={
            "num_envs": num_envs,
            "num_steps": num_steps,
            "total_timesteps": total_timesteps,
        },
    )

    trainer = run_training(hypers)

    obs_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)

    value_target = 1.0 / (1.0 - hypers.train.gamma)
    value_estimate = _estimate_value(trainer.agent, obs_space, match, reward, seed=2)

    # Should be positive and at least 25% of target.
    assert value_estimate > value_target * 0.25, (
        f"value_estimate={value_estimate:.1f}, expected > {value_target * 0.25:.1f}"
    )
