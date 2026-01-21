"""
test_env.py
"""

import numpy as np
import pytest
import logging
import torch
from manabot.env.observation import ObservationSpace
from manabot.env import Env, VectorEnv, Reward
from manabot.env.match import Match
from manabot.infra.hypers import RewardHypers
import manabot.env.observation
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


SAMPLE_DECKS = {
    "basic": {"Mountain": 12, "Forest": 12},
    "creature": {"Mountain": 24, "Lightning Bolt": 36},
    "default": {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Grey Ogre": 18,
    }
}

@pytest.fixture
def observation_space() -> ObservationSpace:
    """Create an observation space for testing."""
    return ObservationSpace()

@pytest.fixture
def sample_match() -> Match:
    """Create a match with a known configuration for testing."""
    return Match()

@pytest.fixture
def reward() -> Reward:
    """Create a reward for testing."""
    return Reward(RewardHypers(trivial=True))

@pytest.fixture
def env(sample_match, observation_space, reward) -> Env:
    """Create a fresh environment instance for each test."""
    return Env(sample_match, observation_space, reward, auto_reset=False)

@pytest.fixture
def vector_env(sample_match, observation_space, reward) -> VectorEnv:
    """Create a vectorized environment for testing."""
    return VectorEnv(
        num_envs=7,
        match=sample_match,
        observation_space=observation_space,
        reward=reward,
        device="cpu"
    )

class TestEnvironment:
    """Tests for the ManaBot environment wrapper."""

    def test_initialization(self, env):
        """Test basic environment initialization."""
        obs, info = env.reset()
        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert all(isinstance(v, (int, float, bool, dict)) for v in info.values()), \
            "Info values should be basic Python types"

    def test_step_types(self, env):
        """Test types of environment step outputs."""
        obs, info = env.reset()
        next_obs, reward, done, truncated, step_info = env.step(0)
        
        assert isinstance(next_obs, dict), "Observation should be a dictionary"
        assert isinstance(reward, float), "Reward should be a float"
        assert isinstance(done, bool), "Done should be a boolean"
        assert isinstance(truncated, bool), "Truncated should be a boolean"
        assert isinstance(step_info, dict), "Step info should be a dictionary"

    def test_game_completion(self, env):
        """Test that a game can successfully complete."""
        logger.debug("Starting game completion test")
        obs, info = env.reset()
        max_steps = 10000
        steps_taken = 0
        terminated = False
        next_obs = None

        while not terminated and steps_taken < max_steps:
            logger.debug(f"Step {steps_taken}: Taking action")
            next_obs, reward, terminated, truncated, info = env.step(0)
            logger.debug(f"Step {steps_taken}: Got reward {reward}, terminated={terminated}")
            steps_taken += 1
            assert isinstance(reward, float), f"Invalid reward type at step {steps_taken}"
            assert not (terminated and truncated), "Cannot be both terminated and truncated"
            
            if steps_taken % 1000 == 0:
                print(f"Game still running after {steps_taken} steps...")

        assert terminated, f"Game did not complete within {max_steps} steps"
        assert steps_taken < max_steps, "Possible infinite loop detected"
        assert isinstance(next_obs, dict), "Final observation should be a dictionary"

    @pytest.mark.parametrize("steps", [1, 10, 100])
    def test_consistent_stepping(self, env, steps):
        """Test environment consistency over multiple steps."""
        obs, info = env.reset()
        
        for step in range(steps):
            next_obs, reward, done, truncated, info = env.step(0)
            assert set(next_obs.keys()) == set(obs.keys()), \
                f"Observation keys changed at step {step}"
            obs = next_obs
            
            if done or truncated:
                break

    def test_vectorenv_tensor_outputs(self, vector_env):
        """Test that VectorEnv properly converts everything to tensors."""
        num_envs = vector_env.num_envs
        device = vector_env.device
        
        # Test reset
        observations, info = vector_env.reset()
        
        # Verify we get tensors with correct dtypes
        assert isinstance(observations, dict)
        for key, tensor in observations.items():
            assert isinstance(tensor, torch.Tensor), f"{key} is not a tensor"
            assert tensor.dtype == torch.float32, f"{key} has wrong dtype"
            assert tensor.shape[0] == num_envs, f"{key} missing batch dimension"
        
        # Test step with tensor action
        action = torch.zeros(num_envs, dtype=torch.int64, device=device)
        next_obs, reward, terminated, truncated, info = vector_env.step(action)
        
        # Verify observation tensors maintain properties
        for key, tensor in next_obs.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.shape[0] == num_envs
        
        # Verify reward and flags are correct tensor types
        assert isinstance(reward, torch.Tensor)
        assert reward.dtype == torch.float32
        assert reward.shape == (num_envs,)
        
        assert isinstance(terminated, torch.Tensor)
        assert terminated.dtype == torch.bool
        assert terminated.shape == (num_envs,)
        
        assert isinstance(truncated, torch.Tensor)
        assert truncated.dtype == torch.bool
        assert truncated.shape == (num_envs,)
        
        # Test device movement
        if torch.cuda.is_available():
            vector_env = vector_env.to("cuda")
            observations, info = vector_env.reset()
            for key, tensor in observations.items():
                assert tensor.device.type == "cuda"
        if torch.backends.mps.is_available():
            vector_env = vector_env.to("mps")
            observations, info = vector_env.reset()
            for key, tensor in observations.items():
                assert tensor.device.type == "mps"

        vector_env.close()

    def test_agent_turns_distribution(self, env):
        """Test that agent indices match between observation encoding and managym."""
        obs, info = env.reset()
        
        # Convert the observation to a "vectorized" format using torch:
        obs_vec = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        
        # Print shapes to understand what we're working with
        print("obs_vec['agent_player'] shape:", obs_vec["agent_player"].shape)
        
        max_steps = 1000
        steps = 0
        turn_counts = {}
        
        while steps < max_steps:
            # Extract agent indices using the vectorized observation.
            actor_idx = manabot.env.observation.get_agent_indices(obs_vec)
            # Get the single scalar value using item() to convert to Python int
            actor_idx = actor_idx[0].item()
            
            raw_obs = env.last_cpp_obs
            cpp_player_idx = raw_obs.agent.player_index
            
            assert actor_idx == cpp_player_idx, (
                f"Actor index {actor_idx} doesn't match managym player index {cpp_player_idx}"
            )
            
            # Record the turn count.
            turn_counts[actor_idx] = turn_counts.get(actor_idx, 0) + 1
            
            # Take a step in the environment
            obs, reward, done, truncated, info = env.step(0)
            
            # Re-wrap the new observation.
            obs_vec = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            steps += 1
            
            # Check termination
            if done or truncated:
                break

        print(f"Agent turn counts: {turn_counts}")
        assert len(turn_counts) >= 2, (
            f"Expected at least 2 different agent indices, got {turn_counts}"
        )


if __name__ == "__main__":
    pytest.main([__file__])