"""
rust_vector_env.py
Vectorized environment wrapper backed by managym.VectorEnv.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import managym

from .env import add_truncation_flags, stack_encoded_observations
from .match import Match, Reward
from .observation import ObservationSpace
from .single_agent_env import PassivePolicy, RandomPolicy


def _opponent_policy_to_name(opponent_policy: Optional[Any]) -> str:
    if opponent_policy is None:
        return "none"
    if isinstance(opponent_policy, str):
        return opponent_policy
    if isinstance(opponent_policy, PassivePolicy):
        return "passive"
    if isinstance(opponent_policy, RandomPolicy):
        return "random"
    raise ValueError(f"Unsupported opponent_policy: {type(opponent_policy)}")


class RustVectorEnv:
    def __init__(
        self,
        num_envs: int,
        match: Match,
        observation_space: ObservationSpace,
        reward: Reward,
        device: str,
        seed: int = 0,
        opponent_policy: Optional[Any] = None,
    ):
        self._base_seed = seed
        self._skip_trivial = True
        self._policy_name = _opponent_policy_to_name(opponent_policy)
        self._player_configs = match.to_rust()
        self.observation_space = observation_space
        self.reward = reward
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._rust_env = self._build_rust_env(seed=seed)
        self._last_raw_obs: list[managym.Observation] = []

    def _build_rust_env(self, seed: int) -> "managym.VectorEnv":
        return managym.VectorEnv(
            num_envs=self.num_envs,
            seed=seed,
            skip_trivial=self._skip_trivial,
            opponent_policy=self._policy_name,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if seed is not None and seed != self._base_seed:
            self._base_seed = seed
            self._rust_env = self._build_rust_env(seed=seed)
        if options and "match" in options:
            self._player_configs = options["match"].to_rust()

        results = self._rust_env.reset_all(self._player_configs)
        raw_obs = [obs for obs, _ in results]
        infos = [dict(info) for _, info in results]
        self._last_raw_obs = raw_obs
        return self._process_obs(raw_obs), self._stack_infos(infos)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        results = self._rust_env.step(actions.cpu().tolist())

        raw_obs: list[managym.Observation] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, Any]] = []

        for env_index, (obs, raw_reward, done, cut, info) in enumerate(results):
            info_dict = dict(info)
            reward = self.reward.compute(raw_reward, self._last_raw_obs[env_index], obs)
            add_truncation_flags(obs, info_dict, self.observation_space.encoder)

            raw_obs.append(obs)
            rewards.append(float(reward))
            terminated.append(bool(done))
            truncated.append(bool(cut))
            infos.append(info_dict)

        self._last_raw_obs = raw_obs
        return (
            self._process_obs(raw_obs),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(terminated, dtype=torch.bool, device=self.device),
            torch.tensor(truncated, dtype=torch.bool, device=self.device),
            self._stack_infos(infos),
        )

    def _process_obs(
        self, raw_obs: list[managym.Observation]
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.observation_space.encode(obs) for obs in raw_obs]
        return stack_encoded_observations(encoded, self.device)

    def _stack_infos(self, infos: list[dict[str, Any]]) -> Dict[str, Any]:
        if not infos:
            return {}

        keys = set().union(*(info.keys() for info in infos))
        stacked: dict[str, Any] = {}
        for key in keys:
            values = [info.get(key, False) for info in infos]
            if all(isinstance(v, bool) for v in values):
                stacked[key] = np.array(values, dtype=bool)
            else:
                stacked[key] = np.array(values, dtype=object)
        return stacked

    def to(self, device: str) -> "RustVectorEnv":
        self.device = torch.device(device)
        return self

    def close(self):
        return None
