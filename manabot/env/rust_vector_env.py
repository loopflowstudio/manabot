"""
rust_vector_env.py
Vectorized environment wrapper backed by managym.VectorEnv.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import managym

from .match import Match, Reward
from .observation import ObservationSpace

VALID_OPPONENT_POLICIES = ("none", "passive", "random")


class RustVectorEnv:
    def __init__(
        self,
        num_envs: int,
        match: Match,
        observation_space: ObservationSpace,
        reward: Reward,
        device: str,
        seed: int = 0,
        opponent_policy: str = "none",
    ):
        if opponent_policy not in VALID_OPPONENT_POLICIES:
            raise ValueError(
                f"Unsupported opponent_policy: {opponent_policy!r}. "
                f"Must be one of {VALID_OPPONENT_POLICIES}"
            )
        self._base_seed = seed
        self._skip_trivial = True
        self._policy_name = opponent_policy
        self._player_configs = match.to_rust()
        self.observation_space = observation_space
        self.reward = reward
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._rust_env = self._build_rust_env(seed=seed)

        self._buffers = self._allocate_buffers()
        self._rust_env.set_buffers(self._buffers)
        self._build_tensor_views()

    def _build_rust_env(self, seed: int) -> "managym.VectorEnv":
        return managym.VectorEnv(
            num_envs=self.num_envs,
            seed=seed,
            skip_trivial=self._skip_trivial,
            opponent_policy=self._policy_name,
        )

    def _allocate_buffers(self) -> Dict[str, np.ndarray]:
        buffers = self.observation_space.encoder.allocate(self.num_envs)
        buffers.update(
            rewards=np.zeros((self.num_envs,), dtype=np.float64),
            terminated=np.zeros((self.num_envs,), dtype=np.uint8),
            truncated=np.zeros((self.num_envs,), dtype=np.uint8),
        )
        return buffers

    def _build_tensor_views(self) -> None:
        self._obs_keys = tuple(self.observation_space.keys())
        self._obs_cpu_views = {
            key: torch.from_numpy(self._buffers[key]) for key in self._obs_keys
        }
        self._reward_cpu_view = torch.from_numpy(self._buffers["rewards"])
        self._terminated_cpu_view = torch.from_numpy(self._buffers["terminated"])
        self._truncated_cpu_view = torch.from_numpy(self._buffers["truncated"])

        self._obs_tensors = {}
        for key in self._obs_keys:
            cpu_view = self._obs_cpu_views[key]
            if self.device.type == "cpu" and key != "action_focus":
                self._obs_tensors[key] = cpu_view
            else:
                self._obs_tensors[key] = torch.empty(
                    cpu_view.shape,
                    dtype=torch.float32 if key == "action_focus" else cpu_view.dtype,
                    device=self.device,
                )

        self._terminated_tensor = torch.empty(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._truncated_tensor = torch.empty(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._reward_tensor = torch.empty(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

    def _sync_tensors_from_buffers(self) -> None:
        for key in self._obs_keys:
            src = self._obs_cpu_views[key]
            dst = self._obs_tensors[key]
            if dst.data_ptr() != src.data_ptr():
                dst.copy_(src)

        self._reward_tensor.copy_(self._reward_cpu_view)

        self._terminated_tensor.copy_(self._terminated_cpu_view)
        self._truncated_tensor.copy_(self._truncated_cpu_view)

    def _apply_reward_policy(self) -> None:
        if self.reward.hypers.managym:
            return

        if self.reward.hypers.trivial:
            self._reward_tensor.fill_(1.0)
            return

        done = self._terminated_tensor | self._truncated_tensor
        if not bool(done.any()):
            return

        win_reward = torch.full_like(self._reward_tensor, self.reward.hypers.win_reward)
        lose_reward = torch.full_like(
            self._reward_tensor, self.reward.hypers.lose_reward
        )
        terminal_shaped = torch.where(self._reward_tensor > 0, win_reward, lose_reward)
        self._reward_tensor.copy_(
            torch.where(done, terminal_shaped, self._reward_tensor)
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if seed is not None and seed != self._base_seed:
            self._base_seed = seed
            self._rust_env = self._build_rust_env(seed=seed)
            self._rust_env.set_buffers(self._buffers)
        if options and "match" in options:
            self._player_configs = options["match"].to_rust()

        self._rust_env.reset_all_into_buffers(self._player_configs)
        self._sync_tensors_from_buffers()
        return self._obs_tensors, {}

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        self._rust_env.step_into_buffers(actions.cpu().tolist())
        self._sync_tensors_from_buffers()
        self._apply_reward_policy()
        return (
            self._obs_tensors,
            self._reward_tensor,
            self._terminated_tensor,
            self._truncated_tensor,
            {},
        )

    def get_last_info(self) -> Dict[str, Any]:
        infos = [dict(info) for info in self._rust_env.get_last_info()]
        return self._stack_infos(infos)

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
        self._build_tensor_views()
        self._sync_tensors_from_buffers()
        return self

    def close(self):
        return None
