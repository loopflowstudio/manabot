"""
env.py
Environment wrapper around the C++ managym.Env that conforms to the Gymnasium API.
"""

from ast import Match
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any, Tuple, Dict
import torch
import numpy as np

import managym
from .observation import ObservationSpace
from .match import Match, Reward
from manabot.infra.log import getLogger

class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        match: Match,
        obs_space: ObservationSpace,
        reward: Reward,
        # Whether to automatically reset the environment when it terminates/truncates.
        # If False, the environment will not be reset until the agent calls reset().
        # If True, when a terminated/truncated event occurs, the environment returns:
        # - the post-reset observation,
        # - the pre-reset reward,
        # - false, # terminate is never true so that AsyncVectorEnv doesnt call reset()
        # - false, # truncated is never true so that AsyncVectorEnv doesnt call reset()
        # - info["true_terminated"]: true
        seed: int = 0,
        auto_reset: bool = False,
        enable_profiler: bool = False,
        enable_behavior_tracking: bool = False
    ):
        """
        Gymnasium-compatible Env wrapper around the managym.Env C++ class.

        Args:
            observation_space: The ObservationSpace (manabot.data) we use to encode C++ observations.
            skip_trivial: Passed to the underlying managym.Env constructor.
            render_mode: Gymnasium render mode, e.g. "human" or None.
        """
        super().__init__()
        self.seed = seed
        self.skip_trivial = True
        self.enable_profiler = enable_profiler
        self.enable_behavior_tracking = enable_behavior_tracking
        logger = getLogger(__name__)
        logger.info(f"Initializing Env with seed={self.seed}, skip_trivial={self.skip_trivial}, enable_profiler={self.enable_profiler}, enable_behavior_tracking={self.enable_behavior_tracking}")
        self._cpp_env = managym.Env(seed=self.seed, skip_trivial=self.skip_trivial, enable_profiler=self.enable_profiler, enable_behavior_tracking=self.enable_behavior_tracking)

        # For when we need manabot.ObservationSpace
        self.obs_space: ObservationSpace = obs_space
        # Type: gymnasium.Space
        self.observation_space = self.obs_space
        self.action_space = spaces.Discrete(self.obs_space.encoder.max_actions)

        self.match = match
        self.reward = reward
        self.auto_reset = auto_reset

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
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

        match = self.match
        if options:
            if "match" in options:
                match = options["match"]

        # Get the initial managym observation
        cpp_obs, cpp_info = self._cpp_env.reset(match.to_cpp())
        self._last_obs = cpp_obs
        # Encode to our dictionary-of-numpy format
        py_obs = self.obs_space.encode(cpp_obs)

        return py_obs, cpp_info

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
        cpp_obs, cpp_reward, terminated, truncated, info = self._cpp_env.step(action)
        reward = self.reward.compute(cpp_reward, self._last_obs, cpp_obs)
        info["true_terminated"] = terminated
        info["true_truncated"] = truncated


        log.debug(f"Stepped env. Step output: reward={cpp_reward}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            log.info(f"Episode terminated: {terminated}, truncated: {truncated}")
            if self.auto_reset:
                # TODO: merge infos? Hopefully we remove this code soon and use gymnasium's autoreset when its released
                cpp_obs, _ = self._cpp_env.reset(self.match.to_cpp())
                terminated = False
                truncated = False
        
        py_obs = self.obs_space.encode(cpp_obs)
        self._last_obs = cpp_obs
        return py_obs, reward, terminated, truncated, info

    def render(self):
        pass

    def info(self) -> Dict[str, Any]:
        return self._cpp_env.info()

    def close(self):
        pass

    @property
    def last_cpp_obs(self):
        return self._last_obs



class VectorEnv:
    """
    Vector environment that automatically batches observations from multiple environments
    and converts them to PyTorch tensors. The first dimension is always the number of 
    environments.
    """
    def __init__(self, num_envs: int, match: Match, observation_space: ObservationSpace, reward: Reward, device: str):
        self._env = gym.vector.AsyncVectorEnv(
            [lambda: Env(match, observation_space, reward, auto_reset=True) for _ in range(num_envs)],
            shared_memory=False

        )
        self.observation_space = observation_space
        self.num_envs = num_envs
        self.device = torch.device(device)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Reset all environments and return batched observations as tensors.

        Returns:
            observations: Dict where each value is a tensor with shape (num_envs, ...)
            info: Dict of additional information
        """
        obs_tuple, info = self._env.reset(seed=seed, options=options)
        return self._process_obs(obs_tuple), info

    def _parse_bool(self, val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Step all environments and return batched observations and rewards as tensors.

        Args:
            actions: Tensor of shape (num_envs,) containing action indices

        Returns:
            observations: Dict where each value is a tensor with shape (num_envs, ...)
            rewards: Tensor of shape (num_envs,)
            terminated: Tensor of shape (num_envs,) containing true termination state
            truncated: Tensor of shape (num_envs,) containing true truncation state
            info: Dict of additional information
        """
        actions_np = actions.cpu().numpy()
        obs_tuple, rewards, terminated, truncated, info = self._env.step(actions_np)
        
        # Get true termination states from info arrays
        true_terminated = info.get('true_terminated', terminated)
        true_truncated = info.get('true_truncated', truncated)
        
        # Convert everything to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminated_tensor = torch.tensor(true_terminated, dtype=torch.bool).to(self.device)
        truncated_tensor = torch.tensor(true_truncated, dtype=torch.bool).to(self.device)
        
        return (
            self._process_obs(obs_tuple),
            rewards_tensor,
            terminated_tensor,
            truncated_tensor,
            info
        )

    def _process_obs(self, obs_tuple: Tuple[Dict[str, np.ndarray], ...]) -> Dict[str, torch.Tensor]:
        """
        Convert tuple of observation dicts into dict of batched tensors.

        Args:
            obs_tuple: Tuple of length num_envs, where each element is a dict
                      of observations for a single environment.

        Returns:
            Dict where each value is a tensor with leading dimension num_envs.
        """
        # Get keys from first observation
        keys = obs_tuple[0].keys()
        
        # Initialize dict to store batched tensors
        batched = {}
        
        # For each key, stack the arrays and convert to tensor
        for key in keys:
            arrays = [obs[key] for obs in obs_tuple]
            stacked = np.stack(arrays)
            batched[key] = torch.tensor(stacked, dtype=torch.float32).to(self.device)
            
        return batched
    
    def to(self, device: str) -> 'VectorEnv':
        """
        Move the environment to the specified device.

        Args:
            device: The target device (e.g., "cpu", "cuda")

        Returns:
            self for chaining
        """
        self.device = torch.device(device)
        return self

    def close(self):
        """Close the environment."""
        self._env.close()

