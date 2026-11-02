"""Serving preparation preserves policy meaning across collection and learning."""

from copy import deepcopy
from dataclasses import dataclass, fields
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import PolicyInference
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from tests.training.test_recovery import recipe


def _batch_tensors(batch: RolloutBatch) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, np.ndarray):
            result[field.name] = torch.from_numpy(value)
        elif isinstance(value, dict):
            for key, array in value.items():
                result[f"{field.name}/{key}"] = torch.from_numpy(array)
        elif value is not None:
            raise TypeError(f"unexpected rollout field {field.name}")
    return result


@dataclass
class PolicyFixture:
    agent: ActorCritic
    collector: SeatRoutedCollector
    obs: dict[str, torch.Tensor]


def _policy(*, semantic: bool = True) -> PolicyFixture:
    torch.set_num_threads(1)
    torch.manual_seed(819)
    config = recipe()
    config.agent.value_aggregation = "masked_mean"
    if not semantic:
        config.agent.world = None
    space = ObservationSpace(config.observation)
    agent = ActorCritic(space, config.agent).eval()
    collector = SeatRoutedCollector(
        space,
        Match(config.match),
        Reward(RewardConfig()),
        num_envs=2,
        seed=91,
        opponent_mode="self",
        recovery_max_microsteps=10000,
    )
    return PolicyFixture(agent, collector, collector._slice_obs_tensors(np.arange(2)))


@pytest.mark.parametrize("semantic", [False, True])
def test_preparation_preserves_forward_and_training_gradients(semantic: bool) -> None:
    case = _policy(semantic=semantic)
    reference = deepcopy(case.agent)
    with PolicyInference(case.agent) as scope, torch.no_grad():
        for _ in range(2):
            prepared = scope.prepare(case.obs)
            actual = case.agent.forward_distribution(prepared)
            expected = reference.forward_distribution(case.obs)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Serving must not detach or reuse a catalog during a subsequent update.
    for agent in (case.agent, reference):
        logits, values = agent.forward_distribution(case.obs)
        loss = (
            logits[case.obs["actions_valid"].bool()].square().mean()
            + values.square().mean()
        )
        loss.backward()
    for a, b in zip(case.agent.parameters(), reference.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@pytest.mark.parametrize("bad", [-1.0, 0.5, float("nan"), 1e6])
@pytest.mark.parametrize("semantic", [False, True])
def test_cpu_admission_rejects_invalid_definition(bad: float, semantic: bool) -> None:
    case = _policy(semantic=semantic)
    case.obs["semantic_cards"][0, 0, 0] = bad
    with PolicyInference(case.agent) as scope:
        with pytest.raises(ValueError, match="unadmitted"):
            scope.prepare(case.obs)


def test_preparation_preserves_object_validity_and_mode_checks() -> None:
    case = _policy()
    with PolicyInference(case.agent) as scope:
        case.agent.train()
        with pytest.raises(RuntimeError, match="mode/configuration"):
            scope.prepare(case.obs)
    for key in (
        "agent_player_valid",
        "opponent_player_valid",
        "agent_cards_valid",
        "opponent_cards_valid",
        "agent_permanents_valid",
        "opponent_permanents_valid",
    ):
        case.obs[key].zero_()
    with PolicyInference(case.agent) as scope:
        with pytest.raises(ValueError, match="valid object"):
            scope.prepare(case.obs)
    with torch.no_grad(), pytest.raises(ValueError, match="valid object"):
        case.agent(case.obs)


def test_scope_rejects_autograd_closed_and_changed_policy() -> None:
    case = _policy()
    with PolicyInference(case.agent) as scope:
        prepared = scope.prepare(case.obs)
        with pytest.raises(RuntimeError, match="autograd"):
            case.agent(prepared)
        with torch.no_grad():
            case.agent(prepared)
            next(case.agent.parameters()).add_(0.1)
            with pytest.raises(RuntimeError, match="policy changed"):
                case.agent(prepared)
    with torch.no_grad(), pytest.raises(RuntimeError, match="closed"):
        case.agent(prepared)
    with PolicyInference(case.agent) as fresh, torch.no_grad():
        expected = case.agent(case.obs)
        actual = case.agent(fresh.prepare(case.obs))
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_prepared_collection_and_restored_continuation_match_reference() -> None:
    optimized, reference = _policy(), _policy()

    def plain_inputs(
        scope: PolicyInference, obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {key: value.to(scope.device) for key, value in obs.items()}

    actual = optimized.collector.collect(optimized.agent, 8)
    with patch.object(PolicyInference, "prepare", plain_inputs):
        expected = reference.collector.collect(reference.agent, 8)
    for key, value in _batch_tensors(actual).items():
        torch.testing.assert_close(value, _batch_tensors(expected)[key], atol=0, rtol=0)
    snapshot = optimized.collector.snapshot()
    assert snapshot.journal == reference.collector.snapshot().journal
    torch.testing.assert_close(
        snapshot.sampling_rng,
        reference.collector.snapshot().sampling_rng,
        atol=0,
        rtol=0,
    )
    restored = _policy()
    restored.collector.restore(snapshot, lambda: None)
    continued = optimized.collector.collect(optimized.agent, 8)
    resumed = restored.collector.collect(restored.agent, 8)
    for key, value in _batch_tensors(continued).items():
        torch.testing.assert_close(value, _batch_tensors(resumed)[key], atol=0, rtol=0)
