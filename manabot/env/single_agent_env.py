"""
single_agent_env.py
Single-agent environment wrapper that auto-steps a fixed opponent policy.
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np

# Local directory imports
from .env import Env
from .observation import ActionEnum


class PassivePolicy:
    """Always pass priority when possible."""

    def __call__(self, obs: dict[str, np.ndarray]) -> int:
        valid_indices = np.flatnonzero(obs["actions_valid"] > 0)
        if len(valid_indices) == 0:
            raise ValueError("No valid opponent actions available.")

        pass_priority_type = int(ActionEnum.PRIORITY_PASS_PRIORITY)
        for idx in valid_indices:
            if obs["actions"][idx, pass_priority_type] > 0:
                return int(idx)
        return int(valid_indices[0])


class RandomPolicy:
    """Choose uniformly from valid actions."""

    def __call__(self, obs: dict[str, np.ndarray]) -> int:
        valid_indices = np.flatnonzero(obs["actions_valid"] > 0)
        if len(valid_indices) == 0:
            raise ValueError("No valid opponent actions available.")
        return int(np.random.choice(valid_indices))


def build_opponent_policy(name: str):
    if name == "passive":
        return PassivePolicy()
    if name == "random":
        return RandomPolicy()
    raise ValueError(f"Unsupported opponent_policy: {name}")


class SingleAgentEnv(gym.Env):
    """Wrap Env so hero (player 0) is the only external actor."""

    def __init__(
        self,
        match,
        obs_space,
        reward,
        opponent_policy,
        seed: int = 0,
        enable_profiler: bool = False,
        enable_behavior_tracking: bool = False,
        inner_env: Optional[Env] = None,
    ):
        super().__init__()
        self.hero_player_index = 0
        self.opponent_policy = opponent_policy
        self.inner = inner_env or Env(
            match,
            obs_space,
            reward,
            seed=seed,
            auto_reset=True,
            enable_profiler=enable_profiler,
            enable_behavior_tracking=enable_behavior_tracking,
        )
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.inner.reset(seed=seed, options=options)
        return self._skip_opponent(obs, info)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        action_space_truncated = bool(info.get("action_space_truncated", False))

        if info.get("true_terminated") or info.get("true_truncated"):
            latched_terminated = bool(info.get("true_terminated", False))
            latched_truncated = bool(info.get("true_truncated", False))
            obs, info = self._skip_opponent(obs, info)
            info["true_terminated"] = latched_terminated
            info["true_truncated"] = latched_truncated
            info["action_space_truncated"] = action_space_truncated or bool(
                info.get("action_space_truncated", False)
            )
            return obs, reward, terminated, truncated, info

        while self._is_opponent(obs):
            opponent_action = self.opponent_policy(obs)
            obs, opponent_reward, terminated, truncated, info = self.inner.step(
                opponent_action
            )
            action_space_truncated = action_space_truncated or bool(
                info.get("action_space_truncated", False)
            )

            if info.get("true_terminated") or info.get("true_truncated"):
                latched_terminated = bool(info.get("true_terminated", False))
                latched_truncated = bool(info.get("true_truncated", False))
                obs, info = self._skip_opponent(obs, info)
                info["true_terminated"] = latched_terminated
                info["true_truncated"] = latched_truncated
                info["action_space_truncated"] = action_space_truncated or bool(
                    info.get("action_space_truncated", False)
                )
                return obs, -opponent_reward, terminated, truncated, info

        info["action_space_truncated"] = action_space_truncated
        return obs, reward, terminated, truncated, info

    def _is_opponent(self, obs: dict[str, np.ndarray]) -> bool:
        return int(obs["agent_player"][0, 0]) != self.hero_player_index

    def _skip_opponent(self, obs: dict[str, np.ndarray], info: dict[str, Any]):
        action_space_truncated = bool(info.get("action_space_truncated", False))
        while self._is_opponent(obs):
            opponent_action = self.opponent_policy(obs)
            obs, _, _, _, info = self.inner.step(opponent_action)
            action_space_truncated = action_space_truncated or bool(
                info.get("action_space_truncated", False)
            )
        info["action_space_truncated"] = action_space_truncated
        return obs, info

    def close(self):
        self.inner.close()
