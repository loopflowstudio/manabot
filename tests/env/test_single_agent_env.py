"""
test_single_agent_env.py
Tests for single-agent wrapper and fixed-opponent policies.
"""

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

# Local imports
from manabot.env.single_agent_env import (
    PassivePolicy,
    RandomPolicy,
    SingleAgentEnv,
    build_opponent_policy,
)


def make_obs(
    player_index: int,
    actions_valid: list[float] | None = None,
    pass_priority_slots: list[int] | None = None,
):
    actions_valid = actions_valid or [1.0, 1.0, 0.0, 0.0]
    pass_priority_slots = pass_priority_slots or []

    actions = np.zeros((len(actions_valid), 6), dtype=np.float32)
    for slot in pass_priority_slots:
        actions[slot, 2] = 1.0  # PRIORITY_PASS_PRIORITY

    agent_player = np.zeros((1, 26), dtype=np.float32)
    agent_player[0, 0] = 1.0
    return {
        "agent_player": agent_player,
        "actions": actions,
        "actions_valid": np.array(actions_valid, dtype=np.float32),
        "_player_index": player_index,
    }


class FakeInnerEnv:
    def __init__(self, reset_result, step_results):
        self._reset_result = deepcopy(reset_result)
        self._step_results = [deepcopy(step) for step in step_results]
        self.step_actions = []
        self.last_raw_obs = self._obs_to_raw(self._reset_result[0])
        self.observation_space = gym.spaces.Dict(
            {
                "agent_player": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, 26), dtype=np.float32
                ),
                "actions": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4, 6), dtype=np.float32
                ),
                "actions_valid": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(4,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(4)

    def _strip_internal_keys(self, obs):
        clean_obs = deepcopy(obs)
        clean_obs.pop("_player_index")
        return clean_obs

    def _obs_to_raw(self, obs):
        return SimpleNamespace(agent=SimpleNamespace(player_index=obs["_player_index"]))

    def reset(self, *, seed=None, options=None):
        obs, info = deepcopy(self._reset_result)
        self.last_raw_obs = self._obs_to_raw(obs)
        return self._strip_internal_keys(obs), info

    def step(self, action):
        self.step_actions.append(int(action))
        if not self._step_results:
            raise RuntimeError("No scripted step results left")
        obs, reward, terminated, truncated, info = deepcopy(self._step_results.pop(0))
        self.last_raw_obs = self._obs_to_raw(obs)
        return self._strip_internal_keys(obs), reward, terminated, truncated, info

    def close(self):
        return None


def test_passive_policy_picks_pass_priority():
    obs = make_obs(player_index=1, actions_valid=[1, 1, 0, 0], pass_priority_slots=[1])
    action = PassivePolicy()(obs)
    assert action == 1


def test_random_policy_picks_only_valid_actions():
    obs = make_obs(player_index=1, actions_valid=[1, 0, 1, 0], pass_priority_slots=[])
    policy = RandomPolicy()
    chosen = {policy(obs) for _ in range(100)}
    assert chosen.issubset({0, 2})
    assert chosen


def test_build_opponent_policy_none():
    assert build_opponent_policy("none") is None


def test_reset_skips_initial_opponent_turn():
    fake_inner = FakeInnerEnv(
        reset_result=(make_obs(player_index=1), {"action_space_truncated": False}),
        step_results=[
            (
                make_obs(player_index=0),
                0.0,
                False,
                False,
                {
                    "action_space_truncated": True,
                    "true_terminated": False,
                    "true_truncated": False,
                },
            )
        ],
    )
    env = SingleAgentEnv(
        match=None,
        obs_space=None,
        reward=None,
        opponent_policy=PassivePolicy(),
        inner_env=fake_inner,
    )

    obs, info = env.reset()

    assert fake_inner.last_raw_obs.agent.player_index == 0
    assert info["action_space_truncated"] is True
    assert fake_inner.step_actions == [0]


def test_terminal_on_opponent_turn_negates_reward():
    fake_inner = FakeInnerEnv(
        reset_result=(make_obs(player_index=0), {"action_space_truncated": False}),
        step_results=[
            (
                make_obs(player_index=1),
                0.0,
                False,
                False,
                {
                    "action_space_truncated": False,
                    "true_terminated": False,
                    "true_truncated": False,
                },
            ),
            (
                make_obs(player_index=0),
                1.0,
                False,
                False,
                {
                    "action_space_truncated": False,
                    "true_terminated": True,
                    "true_truncated": False,
                },
            ),
        ],
    )
    env = SingleAgentEnv(
        match=None,
        obs_space=None,
        reward=None,
        opponent_policy=PassivePolicy(),
        inner_env=fake_inner,
    )

    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    assert fake_inner.last_raw_obs.agent.player_index == 0
    assert reward == -1.0
    assert terminated is False
    assert truncated is False
    assert info["true_terminated"] is True


def test_terminal_flags_latched_after_post_reset_skip():
    fake_inner = FakeInnerEnv(
        reset_result=(make_obs(player_index=0), {"action_space_truncated": False}),
        step_results=[
            (
                make_obs(player_index=1),
                1.0,
                False,
                False,
                {
                    "action_space_truncated": False,
                    "true_terminated": True,
                    "true_truncated": False,
                },
            ),
            (
                make_obs(player_index=0),
                0.0,
                False,
                False,
                {
                    "action_space_truncated": False,
                    "true_terminated": False,
                    "true_truncated": False,
                },
            ),
        ],
    )
    env = SingleAgentEnv(
        match=None,
        obs_space=None,
        reward=None,
        opponent_policy=PassivePolicy(),
        inner_env=fake_inner,
    )

    env.reset()
    obs, reward, _, _, info = env.step(0)

    assert fake_inner.last_raw_obs.agent.player_index == 0
    assert reward == 1.0
    assert info["true_terminated"] is True
    assert info["true_truncated"] is False
