"""Engine-backed lethal-target roots for bounded learning controls.

The native scenario API owns state construction. The ordinary encoder, collector,
learner and terminal winner own observations, actions, updates and rewards. A
fixed pass continuation resolves the selected target; it never chooses a target
or invents a reward. These injected roots are not complete-game strength evidence.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.infra.config import RewardConfig
from manabot.verify.scenarios import find_cast_action, pass_index
import managym


def target_root(match: Match, space: ObservationSpace, seed: int, seat: int) -> Env:
    """Prepare a real Igneous Inspiration target prompt, with both players lethal.

    The UR setup must contain four Mountains and Igneous Inspiration. Seat and
    seeded library/visible land count vary; absolute target index is not a label.
    Preparation is outside the learner trajectory, like tactical evaluation roots.
    """
    env = Env(
        match if seat == 0 else match.swapped(),
        space,
        Reward(RewardConfig()),
        seed=seed,
    )
    env.reset()
    # skip_trivial can surface cleanup Discard before the first priority offer.
    # Preserve the sampled deal; execute its ordinary choice instead of reseeding.
    for _ in range(100):
        if int(env.last_raw_obs.action_space.action_space_type) == int(
            managym.ActionSpaceEnum.PRIORITY
        ):
            break
        _, _, ended, truncated, _ = env.step(0)
        if ended or truncated:
            raise RuntimeError(f"root preparation ended before priority: {seed}")
    else:
        raise RuntimeError(f"root preparation never reached priority: {seed}")
    for player in (0, 1):
        env._engine.scenario_clear_hand(player)
        env._engine.scenario_set_life(player, 1 + seed % 3)
    for _ in range(3 + seed % 2):
        env._engine.scenario_force_battlefield(seat, "Mountain", True)
    env._engine.scenario_force_card_in_hand(seat, "Igneous Inspiration")
    env.scenario_refresh()
    for _ in range(100):
        raw = env.last_raw_obs
        cast = find_cast_action(raw, "Igneous Inspiration")
        if int(raw.agent.player_index) == seat and cast is not None:
            env.step(cast)
            raw = env.last_raw_obs
            if len(raw.action_space.actions) != 2 or any(
                int(action.action_type) != 5 for action in raw.action_space.actions
            ):
                raise ValueError("lethal-target-v1 requires exactly two player targets")
            return env
        _, _, ended, truncated, _ = env.step(pass_index(raw))
        if ended or truncated:
            break
    raise RuntimeError("could not prepare lethal-target-v1 root")


def resolve_target(env: Env, action: int) -> int:
    """Execute the selected target and fixed pass continuation to actual terminal."""
    _, _, ended, truncated, _ = env.step(action)
    for _ in range(20):
        if ended or truncated:
            break
        _, _, ended, truncated, _ = env.step(pass_index(env.last_raw_obs))
    winner = env._engine.winner_index()
    if truncated or not ended or winner is None:
        raise RuntimeError("lethal-target-v1 did not resolve to a terminal winner")
    return int(winner)


class TargetPracticeVector:
    """Serial native roots with the collector's existing in-place buffer contract.

    Each action resolves a terminal transition and parks its row until the
    collector requests another root. Only loaded rows spend a seed.
    """

    # Every root is an `Env`, which always skips forced decisions.
    skip_trivial = True

    def __init__(
        self, space: ObservationSpace, match: Match, count: int, seed: int
    ) -> None:
        self.space, self.match, self.count = space, match, count
        self.seeds = [seed + row for row in range(count)]
        self.envs: list[Env | None] = [None] * count
        self.buffers: dict[str, np.ndarray] = {}
        self.rows: list[tuple[managym.ObservationBatch, int] | None] = [None] * count
        self.infos: list[dict[str, Any]] = [{} for _ in range(count)]

    def set_observation_layout(
        self,
        buffers: dict[str, np.ndarray],
        action_overflow: str = "error",
        *,
        layout: managym.ObservationLayout,
    ) -> None:
        if action_overflow != self.space.config.action_overflow:
            raise ValueError("target practice action policy differs from its encoder")
        if layout.shapes != self.space.shapes:
            raise ValueError("target practice layout differs from its encoder")
        self.buffers = buffers

    def observation_rows(self) -> list[tuple[managym.ObservationBatch, int]]:
        rows = [entry for entry in self.rows if entry is not None]
        if len(rows) != self.count:
            raise RuntimeError(
                "target practice slots have no observations before reset"
            )
        return rows

    def _encode(self, slots: list[int], games: list[Env]) -> None:
        """Publish these projections together, retaining every unselected slot."""
        batch = encode_policy_batch(self.space, [game.last_raw_obs for game in games])
        for index, slot in enumerate(slots):
            self.rows[slot] = (batch, index)

    def reset_all_into_buffers(self, configs: object) -> None:
        self.load_slots_into_buffers(
            [managym.SlotLoad.fresh(row) for row in range(self.count)]
        )

    def load_slots_into_buffers(self, requests: Sequence[managym.SlotLoad]) -> None:
        """Prepare every requested target before publishing any root or spending a seed."""
        rows = [request.slot for request in requests]
        if len(set(rows)) != len(rows) or any(row >= self.count for row in rows):
            raise ValueError("target practice requires distinct slots in range")
        if not all(request.uses_slot_setup for request in requests):
            raise ValueError(
                "target practice requires fresh loads with its configured match"
            )

        roots = [
            target_root(self.match, self.space, self.seeds[row], row % 2)
            for row in rows
        ]
        self._encode(rows, roots)
        for row, env in zip(rows, roots, strict=True):
            self.envs[row] = env
            self.seeds[row] += self.count
            self.infos[row] = {}
            for key in ("terminated", "truncated", "rewards"):
                self.buffers[key][row] = 0

    def current_agent_indices(self) -> list[int]:
        return [
            int(env.last_raw_obs.agent.player_index) if env is not None else -1
            for env in self.envs
        ]

    def get_last_info(self) -> list[dict[str, Any]]:
        return self.infos

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        playing: list[tuple[int, Env]] = []
        for row, enabled in enumerate(active):
            if enabled:
                env = self.envs[row]
                if env is None:
                    raise RuntimeError(f"cannot step parked target practice slot {row}")
                playing.append((row, env))
        for key in ("terminated", "truncated", "rewards"):
            self.buffers[key].fill(0)
        for row, env in playing:
            winner = resolve_target(env, actions[row])
            self._encode([row], [env])
            self.buffers["rewards"][row] = 1.0 if winner == row % 2 else -1.0
            self.infos[row] = {"winner_index": winner}
            self.envs[row] = None
            self.buffers["terminated"][row] = 1
