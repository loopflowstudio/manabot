"""Small self-play captures shared by backend conformance and numerical parity tests."""

from typing import TYPE_CHECKING

import pytest
import torch

from manabot.backends import resolve
from manabot.backends.parity import BACKENDS, installed_backends
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.policy.checkpoint import PolicyCheckpoint, from_torch_state
from manabot.policy.protocol import BackendPolicy, Precision
from manabot.training.rollout.collection import (
    RolloutBatch,
    close_collector,
    collect,
    open_collector,
)
from manabot.training.rollout.numerics import CollectionSession
import managym

if TYPE_CHECKING:
    from manabot.backends.mlx.checkpoint import MLXBackend
    from manabot.backends.torch.checkpoint import TorchBackend


def installed_backend(
    name: str, *, precision: Precision | None = None
) -> "TorchBackend | MLXBackend":
    if name not in installed_backends():
        pytest.skip(f"the {BACKENDS[name]} framework is not installed")
    return resolve(name, precision=precision)


def policy_weights(agent: ActorCritic) -> PolicyCheckpoint:
    return from_torch_state(
        agent.config,
        agent.observation_space.config,
        {name: tensor.numpy().copy() for name, tensor in agent.state_dict().items()},
        parameter_names=dict(agent.named_parameters()),
        readings=agent.readings,
    )


def self_play_batch(
    policy: BackendPolicy,
    match: Match,
    numerics: CollectionSession,
    decisions: int,
    *,
    num_envs: int = 2,
) -> RolloutBatch:
    collector = open_collector(
        policy.observation_space,
        match,
        Reward(RewardConfig()),
        num_envs=num_envs,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
        numerics=numerics,
    )
    try:
        return collect(collector, policy, decisions)
    finally:
        close_collector(collector)


def legacy_capture(
    *, seed: int, num_envs: int, decisions: int
) -> tuple[PolicyCheckpoint, RolloutBatch]:
    torch.manual_seed(seed)
    with pytest.warns(DeprecationWarning):
        space = ObservationSpace(ObservationConfig(policy_input_version=9))
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
    )
    return policy_weights(agent), self_play_batch(
        agent,
        Match(),
        TorchCollectionSession(device="cpu", seed=0),
        decisions,
        num_envs=num_envs,
    )
