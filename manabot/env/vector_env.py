"""
vector_env.py
Vectorized environment wrapper backed by managym.VectorEnv.
"""

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch

from manabot.env.failures import RecordedVectorEnv
from manabot.env.match import Match, Reward
from manabot.env.observation import (
    ObservationSpace,
    allocate_vector_buffers,
)
import managym


class VectorEnv:
    def __init__(
        self,
        num_envs: int,
        match: Match,
        observation_space: ObservationSpace,
        reward: Reward,
        device: str,
        seed: int = 0,
        *,
        player_policies: tuple[str, str] | None = None,
    ) -> None:
        self._skip_trivial = True
        self._player_policies = player_policies
        self._player_configs = match.to_rust()
        self.observation_space = observation_space
        self.reward = reward
        self.num_envs = num_envs
        self._rust_env = self._build_rust_env(seed=seed)
        self._terminal_infos: list[dict[str, Any]] | None = None

        # Dense buffer registration; migrate or remove in 0.3.0 with VectorEnv.set_buffers.
        self._buffers = allocate_vector_buffers(self.observation_space, self.num_envs)
        self._rust_env.set_buffers(
            self._buffers,
            self.observation_space.config.action_overflow,
            layout=self.observation_space.layout,
        )
        self._obs_cpu_views = {
            key: torch.from_numpy(self._buffers[key])
            for key in self.observation_space.keys()
        }
        self._reward_cpu_view = torch.from_numpy(self._buffers["rewards"])
        self._terminated_cpu_view = torch.from_numpy(self._buffers["terminated"])
        self._truncated_cpu_view = torch.from_numpy(self._buffers["truncated"])
        # Outcome tensors own the completed step even after a refill clears its buffers.
        self._reward_tensor = self._reward_cpu_view.to(torch.float32)
        self._terminated_tensor = self._terminated_cpu_view.to(torch.bool)
        self._truncated_tensor = self._truncated_cpu_view.to(torch.bool)
        self.to(device)

        # Land and creature counts read the battlefield: input v11's permanent
        # rows, or the deprecated v9/v10 battlefield card rows.
        self._battlefield_zone_index: int | None
        arrays = observation_space.layout.arrays
        self._life_index = arrays["agent_player"]["columns"]["life"]["index"]
        if self.observation_space.one_token:
            self._board_keys = ("agent_permanents", "opponent_permanents")
            self._battlefield_zone_index = None
        else:
            self._board_keys = ("agent_cards", "opponent_cards")
            self._battlefield_zone_index = arrays["agent_cards"]["columns"][
                "zone_battlefield"
            ]["index"]
        board = arrays[self._board_keys[0]]["columns"]
        self._land_index, self._creature_index, self._valid_index = (
            board[key]["index"] for key in ("is_land", "is_creature", "valid")
        )
        self._prev_obs_keys: Tuple[str, ...] = (self._board_keys[0], "opponent_player")
        if self.reward.config.potential_enabled:
            self._prev_obs_keys = (
                *self._board_keys,
                "agent_player",
                "opponent_player",
            )

    def _build_rust_env(self, seed: int) -> RecordedVectorEnv:
        return RecordedVectorEnv(
            num_envs=self.num_envs,
            decision_memory=self.observation_space.config.max_decision_records > 0,
            decision_memory_include_own=self.observation_space.config.decision_memory_include_own,
            seed=seed,
            skip_trivial=self._skip_trivial,
            player_policies=self._player_policies,
        )

    def _sync_observations(self) -> None:
        for key, src in self._obs_cpu_views.items():
            dst = self._obs_tensors[key]
            if dst.data_ptr() != src.data_ptr():
                dst.copy_(src)

    def _sync_outcomes(self) -> None:
        self._reward_tensor.copy_(self._reward_cpu_view)
        self._terminated_tensor.copy_(self._terminated_cpu_view)
        self._truncated_tensor.copy_(self._truncated_cpu_view)

    def _apply_reward_policy(self) -> None:
        if self.reward.config.managym:
            return

        if self.reward.config.trivial:
            self._reward_tensor.fill_(1.0)
            return

        shaping = self._compute_progress_shaping()
        if shaping is not None:
            self._reward_tensor.add_(shaping)

        potential_shaping = self._compute_potential_shaping()

        done = self._terminated_tensor | self._truncated_tensor
        if bool(done.any()):
            win_reward = torch.full_like(
                self._reward_tensor, self.reward.config.win_reward
            )
            lose_reward = torch.full_like(
                self._reward_tensor, self.reward.config.lose_reward
            )
            terminal_shaped = torch.where(
                self._reward_tensor > 0,
                win_reward,
                torch.where(
                    self._reward_tensor < 0,
                    lose_reward,
                    torch.zeros_like(self._reward_tensor),
                ),
            )
            self._reward_tensor.copy_(
                torch.where(done, terminal_shaped, self._reward_tensor)
            )

        # Potential-based shaping is added after the terminal win/lose
        # overwrite: the terminal step still carries its -Phi(s_last)
        # correction (Phi(terminal) = 0), which is required for the
        # potential to telescope out and preserve policy invariance.
        if potential_shaping is not None:
            self._reward_tensor.add_(potential_shaping)

    def _compute_progress_shaping(self) -> torch.Tensor | None:
        config = self.reward.config
        if (
            config.land_play_reward == 0.0
            and config.creature_play_reward == 0.0
            and config.opponent_life_loss_reward == 0.0
        ):
            return None

        prev_agent_cards = self._prev_obs[self._board_keys[0]]
        next_agent_cards = self._obs_tensors[self._board_keys[0]]
        prev_opponent_player = self._prev_obs["opponent_player"]
        next_opponent_player = self._obs_tensors["opponent_player"]

        shaping = torch.zeros_like(self._reward_tensor)

        if config.land_play_reward != 0.0:
            land_delta = self._battlefield_card_count(
                next_agent_cards,
                self._land_index,
            ) - self._battlefield_card_count(
                prev_agent_cards,
                self._land_index,
            )
            shaping += config.land_play_reward * torch.clamp(land_delta, min=0.0)

        if config.creature_play_reward != 0.0:
            creature_delta = self._battlefield_card_count(
                next_agent_cards,
                self._creature_index,
            ) - self._battlefield_card_count(
                prev_agent_cards,
                self._creature_index,
            )
            shaping += config.creature_play_reward * torch.clamp(
                creature_delta, min=0.0
            )

        if config.opponent_life_loss_reward != 0.0:
            prev_life = prev_opponent_player[:, 0, self._life_index]
            next_life = next_opponent_player[:, 0, self._life_index]
            shaping += config.opponent_life_loss_reward * torch.clamp(
                prev_life - next_life,
                min=0.0,
            )

        shaping = shaping.masked_fill(
            self._terminated_tensor | self._truncated_tensor, 0.0
        )
        return shaping

    def _compute_potential_shaping(self) -> torch.Tensor | None:
        """Potential-based shaping term gamma * Phi(s') - Phi(s) per env.

        Phi(terminal) is treated as 0: on done steps the buffers already hold
        the next episode's reset observation, which must not leak into the
        shaping, and zeroing the terminal potential is what makes the shaping
        telescope out over an episode (Ng, Harada & Russell 1999).
        """
        config = self.reward.config
        if not config.potential_enabled:
            return None

        prev_phi = self._potential(
            self._prev_obs[self._board_keys[0]],
            self._prev_obs[self._board_keys[1]],
            self._prev_obs["agent_player"],
            self._prev_obs["opponent_player"],
        )
        next_phi = self._potential(
            self._obs_tensors[self._board_keys[0]],
            self._obs_tensors[self._board_keys[1]],
            self._obs_tensors["agent_player"],
            self._obs_tensors["opponent_player"],
        )
        done = self._terminated_tensor | self._truncated_tensor
        next_phi = next_phi.masked_fill(done, 0.0)
        return config.potential_gamma * next_phi - prev_phi

    def _potential(
        self,
        agent_cards: torch.Tensor,
        opponent_cards: torch.Tensor,
        agent_player: torch.Tensor,
        opponent_player: torch.Tensor,
    ) -> torch.Tensor:
        """Hero-perspective board-state potential Phi(s).

        Life features are already encoded as life/20, matching the
        (hero_life - villain_life)/20 term in the Phi definition.
        """
        config = self.reward.config
        phi = torch.zeros_like(self._reward_tensor)

        if config.potential_land_weight != 0.0:
            phi += config.potential_land_weight * (
                self._battlefield_card_count(agent_cards, self._land_index)
                - self._battlefield_card_count(opponent_cards, self._land_index)
            )
        if config.potential_creature_weight != 0.0:
            phi += config.potential_creature_weight * (
                self._battlefield_card_count(agent_cards, self._creature_index)
                - self._battlefield_card_count(opponent_cards, self._creature_index)
            )
        if config.potential_life_weight != 0.0:
            phi += config.potential_life_weight * (
                agent_player[:, 0, self._life_index]
                - opponent_player[:, 0, self._life_index]
            )
        return phi

    def _battlefield_card_count(
        self,
        cards: torch.Tensor,
        type_feature_index: int,
    ) -> torch.Tensor:
        """Valid battlefield rows of `cards` [B, rows, width] with the type flag."""
        valid = cards[..., self._valid_index] > 0.5
        if self._battlefield_zone_index is not None:
            valid = valid & (cards[..., self._battlefield_zone_index] > 0.5)
        matches_type = cards[..., type_feature_index] > 0.5
        return (valid & matches_type).float().sum(dim=1)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Restart every slot at ``seed + slot`` when a seed is given, so equal
        seeds reproduce deals; without one, continue native seed succession."""
        if seed is not None:
            self._rust_env.close()
            self._rust_env = self._build_rust_env(seed=seed)
            self._rust_env.set_buffers(
                self._buffers,
                self.observation_space.config.action_overflow,
                layout=self.observation_space.layout,
            )
        if options and "match" in options:
            self._player_configs = options["match"].to_rust()

        self._rust_env.reset_all_into_buffers(self._player_configs)
        self._terminal_infos = None
        self._sync_outcomes()
        self._sync_observations()
        return self._obs_tensors, {}

    def _step_and_refill(self, actions: torch.Tensor) -> None:
        self._prev_obs = {
            key: self._obs_tensors[key].clone() for key in self._prev_obs_keys
        }
        self._rust_env.step_into_buffers(actions.cpu().tolist())
        self._sync_outcomes()
        done = np.flatnonzero(self._buffers["terminated"] | self._buffers["truncated"])
        self._terminal_infos = self._rust_env.get_last_info() if done.size else None
        if done.size:
            self._rust_env.load_slots_into_buffers(
                [managym.SlotLoad.fresh(int(slot)) for slot in done]
            )

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        self._step_and_refill(actions)
        self._sync_observations()
        self._apply_reward_policy()
        return (
            self._obs_tensors,
            self._reward_tensor,
            self._terminated_tensor,
            self._truncated_tensor,
            {},
        )

    def get_last_info(self) -> Dict[str, Any]:
        infos = (
            self._terminal_infos
            if self._terminal_infos is not None
            else self._rust_env.get_last_info()
        )
        return self._stack_infos(infos)

    def write_observations_for_players(
        self,
        players: list[int],
        buffers: dict[str, np.ndarray],
        action_overflow: Literal["error", "first-legal"] = "error",
        *,
        layout: managym.ObservationLayout | None = None,
    ) -> list[managym.ObservationRead]:
        """Write viewer-safe NumPy rows; discard every output row after an error.

        Execution and capture buffers stay unchanged. Seats are native indices.
        """
        return self._rust_env.write_observations_for_players(
            players,
            buffers,
            action_overflow=action_overflow,
            layout=self.observation_space.layout if layout is None else layout,
        )

    def hand_snapshots(self, players: list[int]) -> list[managym.HandSnapshot]:
        """Privileged CPU hand snapshots of the current games, including refilled slots."""
        return self._rust_env.hand_snapshots(players)

    def current_agent_indices(self) -> list[int]:
        """Native seat choosing in each slot, including under control effects."""
        return self._rust_env.current_agent_indices()

    def skip_trivial_counts(self) -> np.ndarray:
        """Per-env trivial-decision collapse counters for the current games."""
        return np.asarray(self._rust_env.skip_trivial_counts(), dtype=np.int64)

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

    def to(self, device: str) -> "VectorEnv":
        self.device = torch.device(device)
        self._obs_tensors = {
            key: src.to(self.device) for key, src in self._obs_cpu_views.items()
        }
        self._reward_tensor = self._reward_tensor.to(self.device)
        self._terminated_tensor = self._terminated_tensor.to(self.device)
        self._truncated_tensor = self._truncated_tensor.to(self.device)
        return self

    def close(self) -> None:
        self._rust_env.close()
