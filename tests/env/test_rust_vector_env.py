"""
test_rust_vector_env.py
Coverage for Rust-backed vectorized environment wrapper.
"""

import torch

from manabot.env import Match, ObservationSpace, PassivePolicy, Reward, RustVectorEnv, VectorEnv
from manabot.infra.hypers import RewardHypers


def _build_envs(seed: int = 19):
    match = Match()
    observation_space = ObservationSpace()
    reward = Reward(RewardHypers())
    legacy = VectorEnv(
        num_envs=1,
        match=match,
        observation_space=observation_space,
        reward=reward,
        device="cpu",
        seed=seed,
        opponent_policy=PassivePolicy(),
    )
    rust = RustVectorEnv(
        num_envs=1,
        match=match,
        observation_space=observation_space,
        reward=reward,
        device="cpu",
        seed=seed,
        opponent_policy="passive",
    )
    return legacy, rust


def test_rust_vector_env_tensor_outputs():
    env = RustVectorEnv(
        num_envs=3,
        match=Match(),
        observation_space=ObservationSpace(),
        reward=Reward(RewardHypers()),
        device="cpu",
        seed=7,
        opponent_policy="passive",
    )

    obs, info = env.reset()
    assert isinstance(obs, dict)
    for tensor in obs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape[0] == 3
    assert isinstance(info, dict)

    actions = torch.zeros(3, dtype=torch.int64)
    next_obs, rewards, terminated, truncated, _ = env.step(actions)
    for tensor in next_obs.values():
        assert tensor.shape[0] == 3
    assert rewards.dtype == torch.float32
    assert terminated.dtype == torch.bool
    assert truncated.dtype == torch.bool
    assert rewards.shape == (3,)
    assert terminated.shape == (3,)
    assert truncated.shape == (3,)

    env.close()


def test_terminal_step_returns_post_reset_observation():
    env = RustVectorEnv(
        num_envs=1,
        match=Match(),
        observation_space=ObservationSpace(),
        reward=Reward(RewardHypers()),
        device="cpu",
        seed=11,
        opponent_policy="passive",
    )
    env.reset()

    found_terminal = False
    for _ in range(4000):
        _, _, terminated, truncated, _ = env.step(torch.zeros(1, dtype=torch.int64))
        if bool(terminated[0] or truncated[0]):
            found_terminal = True
            assert env._last_raw_obs[0].game_over is False
            break

    assert found_terminal
    env.close()


def test_observation_parity_with_existing_vector_env():
    legacy, rust = _build_envs(seed=31)
    try:
        legacy_obs, _ = legacy.reset()
        rust_obs, _ = rust.reset()
        assert legacy_obs.keys() == rust_obs.keys()
        for key in legacy_obs:
            assert torch.equal(legacy_obs[key], rust_obs[key])

        actions = torch.zeros(1, dtype=torch.int64)
        for _ in range(100):
            legacy_obs, legacy_reward, legacy_term, legacy_trunc, _ = legacy.step(actions)
            rust_obs, rust_reward, rust_term, rust_trunc, _ = rust.step(actions)

            for key in legacy_obs:
                assert torch.equal(legacy_obs[key], rust_obs[key])
            assert torch.equal(legacy_reward, rust_reward)
            assert torch.equal(legacy_term, rust_term)
            assert torch.equal(legacy_trunc, rust_trunc)
    finally:
        legacy.close()
        rust.close()
