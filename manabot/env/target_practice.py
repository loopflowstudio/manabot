"""Engine-backed lethal-target roots for bounded learning controls.

The native scenario API owns state construction. The ordinary encoder, collector,
learner and terminal winner own observations, actions, updates and rewards. A
fixed pass continuation resolves the selected target; it never chooses a target
or invents a reward. These injected roots are not complete-game strength evidence.
"""

from typing import Any

import numpy as np

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import RewardHypers
from manabot.verify.competency import _find_cast_action, _pass_index
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
        Reward(RewardHypers()),
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
        cast = _find_cast_action(raw, "Igneous Inspiration")
        if int(raw.agent.player_index) == seat and cast is not None:
            env.step(cast)
            raw = env.last_raw_obs
            if len(raw.action_space.actions) != 2 or any(
                int(action.action_type) != 5 for action in raw.action_space.actions
            ):
                raise ValueError("lethal-target-v1 requires exactly two player targets")
            return env
        _, _, ended, truncated, _ = env.step(_pass_index(raw))
        if ended or truncated:
            break
    raise RuntimeError("could not prepare lethal-target-v1 root")


def resolve_target(env: Env, action: int) -> int:
    """Execute the selected target and fixed pass continuation to actual terminal."""
    _, _, ended, truncated, _ = env.step(action)
    for _ in range(20):
        if ended or truncated:
            break
        _, _, ended, truncated, _ = env.step(_pass_index(env.last_raw_obs))
    winner = env._engine.winner_index()
    if truncated or not ended or winner is None:
        raise RuntimeError("lethal-target-v1 did not resolve to a terminal winner")
    return int(winner)


class TargetPracticeVector:
    """Serial native roots with the collector's existing in-place buffer contract.

    Every row is one genuine terminal transition. The next root replaces only
    the next observation; terminal flags and the previous winner remain available
    to SeatRoutedCollector. Paused rows neither advance seeds nor consume actions.
    """

    def __init__(
        self, space: ObservationSpace, match: Match, count: int, seed: int
    ) -> None:
        self.space, self.match, self.count = space, match, count
        self.seeds = [seed + row for row in range(count)]
        self.envs: list[Env] = []
        self.buffers: dict[str, np.ndarray] = {}
        self.infos: list[dict[str, Any]] = [{} for _ in range(count)]

    def set_buffers(self, buffers: dict[str, np.ndarray]) -> None:
        self.buffers = buffers

    def _reset(self, row: int) -> Env:
        env = target_root(self.match, self.space, self.seeds[row], row % 2)
        self.seeds[row] += self.count
        encoded = self.space.encode(env.last_raw_obs)
        for key, value in encoded.items():
            self.buffers[key][row] = value
        return env

    def reset_all_into_buffers(self, configs: object) -> None:
        self.envs = [self._reset(row) for row in range(self.count)]

    def current_agent_indices(self) -> list[int]:
        return [int(env.last_raw_obs.agent.player_index) for env in self.envs]

    def get_last_info(self) -> list[dict[str, Any]]:
        return self.infos

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        self.buffers["terminated"].fill(0)
        self.buffers["truncated"].fill(0)
        self.buffers["rewards"].fill(0)
        for row, enabled in enumerate(active):
            if enabled:
                winner = resolve_target(self.envs[row], actions[row])
                self.infos[row] = {"winner_index": winner}
                self.envs[row] = self._reset(row)
                self.buffers["terminated"][row] = 1
