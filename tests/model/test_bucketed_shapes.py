"""Bucketed shapes agree with the default path on native observation batches."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.policy.buckets import BucketSizes, bucket, sizes_of
from manabot.policy.prepare import MIN_LIVE_FACT_ROWS
from manabot.policy.spans import ObservationSelection, observation_spans
from manabot.training.rollout.collection import Collector, collect, open_collector
from tests.model.policy_fixture import recipe
from tests.model.prepared_routes import device_input


@dataclass
class PolicyFixture:
    agent: ActorCritic
    collector: Collector


def _policy(*, history: bool = False, bucketed_shapes: bool = False) -> PolicyFixture:
    torch.set_num_threads(1)
    torch.manual_seed(819)
    config = recipe()
    config.agent.value_aggregation = "masked_mean"
    if history:
        config.observation.policy_history_version = 1
        config.agent.recent_events = True
    space = ObservationSpace(config.observation)
    agent = ActorCritic(space, config.agent, bucketed_shapes=bucketed_shapes).eval()
    collector = open_collector(
        space,
        Match(config.match),
        Reward(RewardConfig()),
        num_envs=2,
        seed=91,
        opponent_mode="self",
        numerics=CollectionSession(device="cpu", seed=91, precision="float32"),
    )
    return PolicyFixture(agent, collector)


def _bucketed(reference: ActorCritic) -> ActorCritic:
    """A bucketed copy of a policy with identical weights."""
    agent = ActorCritic(
        reference.observation_space, reference.config, bucketed_shapes=True
    ).eval()
    agent.load_state_dict(reference.state_dict())
    # Fresh fact and link outputs are zero; give those joins real signal.
    with torch.no_grad():
        for original, copy in zip(
            reference.parameters(), agent.parameters(), strict=True
        ):
            noise = torch.randn_like(original) * 0.05
            original.add_(noise)
            copy.add_(noise)
    return agent


def _later_decisions(case: PolicyFixture) -> ObservationSelection:
    """Decisions from played turns: permanents, facts and public history."""
    batch = collect(case.collector, case.agent, 24)
    steps, streams = batch.actions.shape
    rows = np.arange(steps - 8, steps) * streams
    return batch.observations(rows).observation


def _buckets(
    obs: ObservationSelection, space: ObservationSpace, *, objects: bool
) -> BucketSizes:
    spans = observation_spans(obs, space, bucketed=True, objects=objects)
    assert spans.buckets is not None
    return spans.buckets


def _needs(obs: ObservationSelection, space: ObservationSpace) -> np.ndarray:
    needs = observation_spans(obs, space, bucketed=True, objects=True).needs
    assert needs is not None
    return needs


def _legal(obs: ObservationSelection) -> torch.Tensor:
    return torch.from_numpy(obs.project("actions_valid")).bool()


def test_bucket_rule() -> None:
    assert [bucket(n, 512) for n in (0, 1, 8, 9, 59, 133, 512)] == [
        8,
        8,
        8,
        16,
        64,
        256,
        512,
    ]
    assert bucket(31, 96) == 32
    assert bucket(70, 96) == 96
    assert bucket(0, 4) == 4


@pytest.mark.parametrize(
    ("history", "streams"), [(False, False), (True, False), (True, True)]
)
def test_outputs_and_gradients_match_default(history: bool, streams: bool) -> None:
    """`streams` compares the `stream_norm` and `focus_film` layout."""
    torch.manual_seed(302)
    case = _policy(history=history)
    obs = _later_decisions(case)
    reference = case.agent
    if streams:
        reference = ActorCritic(
            reference.observation_space,
            reference.config.model_copy(
                update={"stream_norm": True, "focus_film": True}
            ),
        ).eval()
    agent = _bucketed(reference)
    buckets = _buckets(obs, agent.observation_space, objects=True)
    assert buckets.trims_objects
    outputs = []
    for model in (reference, agent):
        logits, values = model.forward_distribution(device_input(model, obs))
        outputs.append((logits, values))
        legal = _legal(obs)
        (logits[legal].square().mean() + values.square().mean()).backward()
    (expected_logits, expected_values), (logits, values) = outputs
    assert logits.shape == expected_logits.shape
    torch.testing.assert_close(logits, expected_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values, expected_values, atol=1e-5, rtol=1e-5)
    for (name, a), b in zip(
        reference.named_parameters(), agent.parameters(), strict=True
    ):
        if a.grad is None:
            assert b.grad is None, name
        else:
            assert b.grad is not None, name
            torch.testing.assert_close(b.grad, a.grad, atol=1e-5, rtol=1e-4)
    with torch.no_grad(), PolicyInference(agent) as scope:
        prepared = scope.prepare(obs)
        assert isinstance(prepared.bucket_needs, np.ndarray)
        assert (
            sizes_of(
                prepared.bucket_needs,
                agent.observation_space,
                objects=True,
            )
            == buckets
        )
        assert prepared.facts.kinds.shape == (
            len(obs.coordinates),
            buckets.sizes["facts"],
        )
        served, _ = agent.forward_distribution(prepared)
    torch.testing.assert_close(served, expected_logits, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("objects", [True, False])
def test_row_needs_regroup_to_the_measured_buckets(objects: bool) -> None:
    """Needs measured once per row give any grouping of those rows the buckets
    measuring the group directly gives, so later epochs need not measure."""
    case = _policy(history=True)
    space = case.agent.observation_space
    batch = collect(case.collector, case.agent, 24)
    obs = batch.observations(np.arange(batch.actions.size)).observation
    needs = _needs(obs, space)
    assert len(needs) == len(obs.coordinates)
    rng = np.random.default_rng(0)
    for _ in range(20):
        rows = rng.choice(len(needs), size=rng.integers(1, len(needs)), replace=False)
        assert sizes_of(needs[rows], space, objects=objects) == _buckets(
            obs.select(rows), space, objects=objects
        )


def test_collection_records_each_rows_measured_needs() -> None:
    """These decisions retain all measured extents after legal-action packing."""
    case = _policy(history=True, bucketed_shapes=True)
    batch = collect(case.collector, case.agent, 12)
    rows = np.arange(batch.actions.size)
    recorded = batch.recorded.gather_needs(rows)
    assert recorded is not None
    expected = _needs(
        batch.observations(rows).observation, case.agent.observation_space
    )
    assert np.array_equal(recorded, expected)


@pytest.mark.parametrize("bucketed", [False, True])
def test_mps_live_fact_routes_match_padded_outputs_and_gradients(
    bucketed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MPS preparation routes live facts only; run here on CPU against padding."""
    torch.manual_seed(303)
    case = _policy(history=False)
    obs = _later_decisions(case)
    reference = _bucketed(case.agent)
    if not bucketed:
        reference.bucketed_shapes = False
    padded_input = device_input(reference, obs)
    monkeypatch.setattr(ActorCritic, "device_type", "mps")
    live_input = device_input(reference, obs)
    live = int(obs.spans.tables["facts"].counts.sum())
    assert live > 0 and padded_input.facts.programs is not None
    assert live_input.facts.compact
    assert live_input.facts.kinds.shape == (
        1,
        max(MIN_LIVE_FACT_ROWS, 1 << (live - 1).bit_length()),
    )
    gradients = []
    outputs = []
    for prepared in (padded_input, live_input):
        reference.zero_grad()
        logits, values = reference.forward_distribution(prepared)
        legal = _legal(obs)
        (logits[legal].square().mean() + values.square().mean()).backward()
        outputs.append((logits.detach(), values.detach()))
        gradients.append([p.grad for p in reference.parameters()])
    torch.testing.assert_close(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)
    for (name, _), a, b in zip(reference.named_parameters(), *gradients, strict=True):
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(b, a, atol=1e-5, rtol=1e-4)
    with torch.no_grad(), PolicyInference(reference) as scope:
        served = scope.prepare(obs)
        assert served.facts.compact
        if not bucketed:
            assert served.attention_columns is not None
            assert len(served.attention_columns) == bucket(
                int(served.validity.bool().any(0).sum()), served.validity.shape[1]
            )
        served_outputs = reference.forward_distribution(served)
    torch.testing.assert_close(served_outputs, outputs[0], atol=1e-5, rtol=1e-5)
