"""Serving preparation preserves policy meaning across collection and learning."""

from copy import deepcopy
from dataclasses import dataclass, fields
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import PERMANENT_DEFINITION_INDEX, ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import PolicyInference
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from tests.training.test_recovery import recipe


def _batch_tensors(batch: RolloutBatch) -> dict[str, torch.Tensor]:
    rows = batch.observations()
    result = {
        "probabilities": torch.from_numpy(rows.probabilities),
        "log_probabilities": torch.from_numpy(rows.log_probabilities),
        **{
            f"obs/{key}": torch.from_numpy(value)
            for key, value in rows.observation.items()
        },
    }
    for field in fields(batch):
        if field.name in {"recorded", "action_data"}:
            continue  # Compared above through the complete materialized evidence.
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


def _policy(*, semantic: bool = True, history: bool = False) -> PolicyFixture:
    torch.set_num_threads(1)
    torch.manual_seed(819)
    config = recipe()
    config.agent.value_aggregation = "masked_mean"
    if history:
        config.observation.policy_history_version = 1
        config.agent.recent_events = True
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


@pytest.mark.parametrize("history", [False, True])
def test_training_encodes_only_referenced_definitions_without_changing_learning(
    history: bool,
) -> None:
    case = _policy(history=history)
    if history:
        # Later decisions carry public history.
        batch = case.collector.collect(case.agent, 16)
        case.obs = {
            key: torch.from_numpy(value[-8:, 0]).float()
            if key == "action_focus"
            else torch.from_numpy(value[-8:, 0])
            for key, value in batch.observations().observation.items()
        }
    semantic = case.agent.semantic_cards
    assert semantic is not None
    _plant_unseen_definitions(case.obs, semantic.definition_rows, history)
    full = deepcopy(case.agent)
    encode = semantic.definitions
    referenced: list[int] = []

    def poisoned(rows: torch.Tensor | None = None) -> torch.Tensor:
        """Unreferenced rows read NaN, so any missed reference source fails."""
        assert rows is not None
        referenced.append(int(rows.unique().numel()))
        table = encode(rows)
        unreferenced = torch.ones(len(table), dtype=torch.bool)
        unreferenced[0] = False
        unreferenced[rows] = False
        return table.masked_fill(unreferenced[:, None], float("nan"))

    semantic.definitions = poisoned  # type: ignore[method-assign]
    # The reference encodes the complete catalog.
    full._referenced_definitions = lambda obs, ids: None  # type: ignore[method-assign,assignment,return-value]
    outputs = []
    for agent in (case.agent, full):
        logits, values = agent.forward_distribution(case.obs)
        outputs.append((logits, values))
        (
            logits[case.obs["actions_valid"].bool()].square().mean()
            + values.square().mean()
        ).backward()
    assert 0 < referenced[0] < semantic.catalog_shape.definitions
    for a, b in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(case.agent.parameters(), full.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def _plant_unseen_definitions(
    obs: dict[str, torch.Tensor], lookup: torch.Tensor, history: bool
) -> None:
    """Give every reference source a definition no other source names, so
    the encoded subset must draw on each source separately."""
    named = set(lookup[obs["semantic_cards"].long()].flatten().tolist())
    unseen = iter(
        transport
        for transport in range(1, len(lookup))
        if int(lookup[transport]) > 0 and int(lookup[transport]) not in named
    )
    obs["agent_permanents"][0, 0, PERMANENT_DEFINITION_INDEX] = next(unseen)
    obs["known_hand"][0, 0, 0, 0] = next(unseen)
    obs["decklists"][0, 0, 0, 0] = next(unseen)
    obs["relations"][0, 0, 5] = next(unseen)
    if history:
        batch, event = obs["events_valid"].nonzero()[0].tolist()
        obs["events"][batch, event, 2] = next(unseen)


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
