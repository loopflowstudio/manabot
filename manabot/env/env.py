"""
env.py
Environment wrapper around the managym Rust engine that conforms to the Gymnasium API.
"""

from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium import spaces

# Local imports
from manabot.infra.log import getLogger
import managym

# Local directory imports
from .match import Match, Reward
from .observation import ObservationSpace


def add_truncation_flags(
    raw_obs: managym.Observation,
    info: dict[str, Any],
    encoder: Any,
) -> None:
    info["action_space_truncated"] = (
        len(raw_obs.action_space.actions) > encoder.max_actions
    )
    info["card_space_truncated"] = (
        len(raw_obs.agent_cards) > encoder.cards_per_player
        or len(raw_obs.opponent_cards) > encoder.cards_per_player
    )
    info["permanent_space_truncated"] = (
        len(raw_obs.agent_permanents) > encoder.perms_per_player
        or len(raw_obs.opponent_permanents) > encoder.perms_per_player
    )


class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        match: Match,
        obs_space: ObservationSpace,
        reward: Reward,
        seed: int = 0,
        auto_reset: bool = False,
        enable_profiler: bool = False,
        enable_behavior_tracking: bool = False,
    ):
        """
        Gymnasium-compatible Env wrapper around the managym Rust engine.

        Args:
            observation_space: The ObservationSpace we use to encode raw observations.
            skip_trivial: Passed to the underlying managym.Env constructor.
            render_mode: Gymnasium render mode, e.g. "human" or None.
        """
        super().__init__()
        self.seed = seed
        self.skip_trivial = True
        self.enable_profiler = enable_profiler
        self.enable_behavior_tracking = enable_behavior_tracking
        logger = getLogger(__name__)
        logger.info(
            f"Initializing Env with seed={self.seed}, skip_trivial={self.skip_trivial}, enable_profiler={self.enable_profiler}, enable_behavior_tracking={self.enable_behavior_tracking}"
        )
        self._engine = managym.Env(
            seed=self.seed,
            skip_trivial=self.skip_trivial,
            enable_profiler=self.enable_profiler,
            enable_behavior_tracking=self.enable_behavior_tracking,
        )

        # For when we need manabot.ObservationSpace
        self.obs_space: ObservationSpace = obs_space
        # Type: gymnasium.Space
        self.observation_space = self.obs_space
        self.action_space = spaces.Discrete(self.obs_space.encoder.max_actions)

        self.match = match
        self.reward = reward
        self.auto_reset = auto_reset

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict, dict]:
        """
        Resets the environment to an initial state and returns (observation, info).

        Args:
            seed: Optional seed for environment’s RNG.
            options: Must contain `player_configs` (list of player configs) if needed.

        Returns:
            observation: A dictionary of numpy arrays (encoded from managym.Observation).
            info: Additional debug info from managym, as a dict of string->string.
        """
        # Gymnasium requires calling this for seeding (if you use self.np_random)
        super().reset(seed=seed)

        match = self.match if not options else options.get("match", self.match)

        # Get the initial managym observation
        raw_obs, raw_info = self._engine.reset(match.to_rust())
        self._last_obs = raw_obs
        # Encode to our dictionary-of-numpy format
        py_obs = self.obs_space.encode(raw_obs)

        return py_obs, raw_info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        Step the environment by applying `action` (int).
        Automatically

        Args:
            action: Chosen action index (within our placeholder discrete space).

        Returns:
            observation: Dictionary of numpy arrays.
            reward: Float reward for this step.
            terminated: Whether the episode ended because the game ended in a terminal state.
            truncated: Whether the episode ended due to a timelimit or external condition.
            info: Additional debug info from managym, e.g. partial game logs.
        """
        log = getLogger(__name__).getChild("step")
        raw_obs, raw_reward, terminated, truncated, info = self._engine.step(action)
        reward = self.reward.compute(raw_reward, self._last_obs, raw_obs)
        info["true_terminated"] = terminated
        info["true_truncated"] = truncated
        add_truncation_flags(raw_obs, info, self.obs_space.encoder)

        log.debug(
            f"Stepped env. Step output: reward={raw_reward}, terminated={terminated}, truncated={truncated}"
        )
        if terminated or truncated:
            log.info(f"Episode terminated: {terminated}, truncated: {truncated}")
            if self.auto_reset:
                raw_obs, _ = self._engine.reset(self.match.to_rust())
                terminated = False
                truncated = False

        py_obs = self.obs_space.encode(raw_obs)
        self._last_obs = raw_obs
        return py_obs, reward, terminated, truncated, info

    def render(self):
        pass

    def info(self) -> Dict[str, Any]:
        return self._engine.info()

    def close(self):
        pass

    @property
    def last_raw_obs(self):
        return self._last_obs
