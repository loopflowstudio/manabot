"""
test_agent.py

This module provides comprehensive testing for the ManaBot Agent architecture.
Tests are organized into logical groups that mirror the agent's processing pipeline:

1. Basic functionality (forward pass, gradient flow)
2. Component-specific tests (embeddings, attention, action processing)
3. Integration tests (action selection, policy outputs)
4. Numerical stability tests

Each test aims to verify a specific aspect of the agent's behavior while
maintaining robustness to architectural changes.
"""

import logging
import pytest
import torch
import torch.nn as nn
import copy
from typing import Dict, Tuple, Generator

from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers, ObservationSpaceHypers
from manabot.model import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Test Fixtures and Utilities
# -----------------------------------------------------------------------------

@pytest.fixture
def observation_space() -> ObservationSpace:
    """Create a minimal but complete observation space for testing."""
    return ObservationSpace(
        ObservationSpaceHypers(
            max_cards_per_player=3,       # Small number for testing
            max_permanents_per_player=2,  # Small number for testing
            max_actions=5,               # Enough actions to test selection
            max_focus_objects=2          # Standard focus object count
        )
    )

@pytest.fixture
def agent_hypers() -> AgentHypers:
    """Create agent hyperparameters optimized for testing."""
    return AgentHypers(
        hidden_dim=4,           # Small embedding dimension for fast testing
        num_attention_heads=2    # Multiple heads but keep it small
    )

@pytest.fixture
def agent(observation_space: ObservationSpace, agent_hypers: AgentHypers) -> Agent:
    """Create an agent instance with controlled randomization."""
    torch.manual_seed(42)  # Ensure reproducible initialization
    return Agent(observation_space, agent_hypers)

@pytest.fixture
def real_observation(observation_space: ObservationSpace) -> Generator[Dict[str, torch.Tensor], None, None]:
    """
    Create a real observation using a VectorEnv and the actual manabot observation encoder.
    
    Returns a dictionary of torch.Tensors with a batch dimension (here, batch_size=2)
    as produced by the real environment.
    """
    from manabot.env.match import Match
    from manabot.env import Reward
    from manabot.infra.hypers import RewardHypers
    reward = Reward(RewardHypers(trivial=True))
    match = Match()
    from manabot.env import VectorEnv
    vec_env = VectorEnv(num_envs=2, match=match, observation_space=observation_space, reward=reward, device="cpu")
    obs, _ = vec_env.reset()
    yield obs
    vec_env.close()

# -----------------------------------------------------------------------------
# Basic Functionality Tests
# -----------------------------------------------------------------------------

def test_forward_pass_shapes(agent: Agent, real_observation: Dict[str, torch.Tensor], observation_space: ObservationSpace):
    """
    Verify that a basic forward pass produces outputs of the correct shape.
    This is our most fundamental sanity check.
    """
    obs = real_observation  # Provided by the real_observation fixture (batch_size=2)
    
    # Execute forward pass
    logits, value = agent(obs)
    batch_size = obs[next(iter(obs))].shape[0]
    
    # Check output shapes
    assert logits.shape == (batch_size, observation_space.encoder.max_actions), \
        f"Expected logits shape {(batch_size, observation_space.encoder.max_actions)}, got {logits.shape}"
    assert value.shape == (batch_size,), \
        f"Expected value shape {(batch_size,)}, got {value.shape}"
    
    # Check for any degenerate values
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "fInf in logits"
    assert not torch.isnan(value).any(), "NaN in value"
    assert not torch.isinf(value).any(), "Inf in value"

def test_gradient_flow(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify that gradients flow properly through the entire network.
    """
    obs = real_observation
    logits, value = agent(obs)
    loss = logits.mean() + value.mean()
    loss.backward()
    
    for name, param in agent.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in parameter {name}"
        assert not torch.isinf(param.grad).any(), f"Infinite gradient in parameter {name}"

# -----------------------------------------------------------------------------
# Component-Specific Tests
# -----------------------------------------------------------------------------

def test_object_embeddings(agent: Agent, real_observation: Dict[str, torch.Tensor], observation_space: ObservationSpace):
    """
    Verify that object embeddings maintain proper dimensionality and masking.
    """
    obs = real_observation
    objects, is_agent, validity = agent._gather_object_embeddings(obs)
    
    expected_objects = (
        2 +  # Two players
        observation_space.encoder.cards_per_player * 2 +  # Cards for both players
        observation_space.encoder.perms_per_player * 2  # Permanents for both players
    )
    batch_size = obs[next(iter(obs))].shape[0]
    assert objects.shape == (batch_size, expected_objects, agent.hypers.hidden_dim), \
        f"Wrong shape for objects tensor: {objects.shape}"
    assert is_agent.shape == (batch_size, expected_objects), \
        f"Wrong shape for is_agent tensor: {is_agent.shape}"
    assert validity.shape == (batch_size, expected_objects), \
        f"Wrong shape for validity tensor: {validity.shape}"
    
    # Players should always be valid
    assert torch.all(validity[:, :2] == 1), "Players should always be valid"
    
    # Invalid objects should have zero embeddings
    invalid_mask = ~validity.bool()
    if invalid_mask.any():
        assert torch.all(objects[invalid_mask] == 0), "Invalid objects should have zero embeddings"

def test_attention_mechanism(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify the attention mechanism's behavior and output properties.
    """
    obs = real_observation
    objects, is_agent, validity = agent._gather_object_embeddings(obs)
    key_padding_mask = (validity == 0)
    attended = agent.attention(objects, is_agent, key_padding_mask=key_padding_mask)

    assert attended.shape == objects.shape, \
        f"Attention changed shape from {objects.shape} to {attended.shape}"
    assert not torch.isnan(attended).any(), "NaN in attention output"
    assert not torch.isinf(attended).any(), "Inf in attention output"
    
    if key_padding_mask.any():
        masked_positions = key_padding_mask.unsqueeze(-1).expand_as(attended)
        assert torch.all(attended[masked_positions] == 0), "Attention leaked through mask"

def test_focus_object_incorporation(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify that focus objects are correctly incorporated into action representations.
    """
    obs = real_observation
    objects, is_agent, validity = agent._gather_object_embeddings(obs)
    key_padding_mask = (validity == 0)
    post_attention = agent.attention(objects, is_agent, key_padding_mask=key_padding_mask)
    
    actions = agent.action_embedding(obs["actions"][..., :-1])
    actions_with_focus = agent._add_focus(actions, post_attention, obs["action_focus"])
    
    batch_size = obs["actions"].shape[0]
    expected_dim = (1 + agent.max_focus_objects) * agent.hypers.hidden_dim
    assert actions_with_focus.shape == (batch_size, obs["actions"].shape[1], expected_dim), \
        f"Wrong shape for actions with focus: {actions_with_focus.shape}"
    
    # Check that invalid focus indices (-1) yield zero embeddings
    invalid_focus = (obs["action_focus"] == -1)
    if invalid_focus.any():
        focus_start = agent.hypers.hidden_dim
        for i in range(agent.max_focus_objects):
            focus_slice = slice(focus_start + i * agent.hypers.hidden_dim,
                                focus_start + (i + 1) * agent.hypers.hidden_dim)
            invalid_positions = invalid_focus[..., i].unsqueeze(-1).expand(-1, obs["actions"].shape[1], agent.hypers.hidden_dim)
            focus_embeddings = actions_with_focus[..., focus_slice]
            assert torch.all(focus_embeddings[invalid_positions] == 0), f"Invalid focus index {i} produced non-zero embedding"

# -----------------------------------------------------------------------------
# Action Selection Tests
# -----------------------------------------------------------------------------

def test_action_masking(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """Verify that action masking properly prevents selection of invalid actions."""
    obs = real_observation
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(obs)
    
    action_mask = obs["actions_valid"]
    batch_size = action_mask.shape[0]
    for i in range(batch_size):
        chosen_action = action[i].item()
        assert action_mask[i, chosen_action].item() == 1, f"Selected invalid action {chosen_action} in batch {i}"


# -----------------------------------------------------------------------------
# Numerical Stability Tests
# -----------------------------------------------------------------------------

def test_value_estimate_range(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify that value estimates stay within reasonable bounds.
    """
    obs = real_observation
    with torch.no_grad():
        _, value = agent(obs)
    assert not torch.isnan(value).any(), "NaN in value estimates"
    assert not torch.isinf(value).any(), "Inf in value estimates"
    assert torch.all(value > -1000) and torch.all(value < 1000), "Value estimates outside reasonable range"

def test_probability_distribution(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify that action probabilities form a valid distribution.
    """
    obs = real_observation
    with torch.no_grad():
        logits, _ = agent(obs)
        probs = torch.softmax(logits, dim=-1)
    assert not torch.isnan(probs).any(), "NaN in probabilities"
    assert not torch.isinf(probs).any(), "Inf in probabilities"
    assert torch.all(probs >= 0), "Negative probabilities found"
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums)), "Probabilities do not sum to 1"
    
    valid_mask = obs["actions_valid"].bool()
    invalid_mask = ~valid_mask
    if invalid_mask.any():
        assert torch.all(probs[invalid_mask] == 0), "Non-zero probability for invalid actions"

def test_attention_stability(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify that the attention mechanism produces stable outputs across multiple forward passes.
    """
    obs = real_observation
    objects, is_agent, validity = agent._gather_object_embeddings(obs)
    key_padding_mask = (validity == 0)
    outputs = []
    for _ in range(5):
        attended = agent.attention(objects, is_agent, key_padding_mask=key_padding_mask)
        outputs.append(attended)
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], rtol=1e-5), "Attention outputs not consistent across forward passes"

# -----------------------------------------------------------------------------
# Policy Output Tests
# -----------------------------------------------------------------------------

def test_policy_output_consistency(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify consistency between direct forward pass and action sampling interface.
    """
    obs = real_observation
    with torch.no_grad():
        logits_direct, value_direct = agent(obs)
        _, _, _, value_action = agent.get_action_and_value(obs)
    assert torch.allclose(value_direct, value_action), "Inconsistent values between forward and get_action_and_value"

def test_action_entropy(agent: Agent, real_observation: Dict[str, torch.Tensor]):
    """
    Verify that action distribution entropy behaves reasonably.
    """
    obs = real_observation
    with torch.no_grad():
        _, _, entropy, _ = agent.get_action_and_value(obs)
    assert torch.all(entropy >= 0), "Negative entropy found"
    valid_actions = obs["actions_valid"].sum(dim=-1)
    max_entropy = torch.log(valid_actions)
    assert torch.all(entropy <= max_entropy + 1e-5), "Entropy exceeds theoretical maximum"

if __name__ == "__main__":
    pytest.main([__file__])
