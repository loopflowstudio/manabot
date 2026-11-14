"""Bucketed shapes agree with the default path on admitted inputs."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import FACT_OBJECT_INDEX
from manabot.model.admission import admit_observation
from manabot.model.inference import PolicyInference
from manabot.policy.buckets import (
    ONE_BASED_REFERENCES,
    ZERO_BASED_REFERENCES,
    bucket,
    bucket_sizes,
    row_needs,
    sizes_of,
)
from manabot.training.rollout.collection import collect
import managym
from tests.model.test_inference import PolicyFixture, _policy


def _bucketed(case: PolicyFixture) -> ActorCritic:
    """A bucketed copy of the fixture's policy with identical weights."""
    agent = ActorCritic(
        case.agent.observation_space, case.agent.config, bucketed_shapes=True
    ).eval()
    agent.load_state_dict(case.agent.state_dict())
    # Fresh fact and link outputs are zero; give those joins real signal.
    with torch.no_grad():
        for reference, copy in zip(
            case.agent.parameters(), agent.parameters(), strict=True
        ):
            noise = torch.randn_like(reference) * 0.05
            reference.add_(noise)
            copy.add_(noise)
    return agent


def _host(obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.numpy() for key, value in obs.items()}


def _later_decisions(case: PolicyFixture) -> dict[str, torch.Tensor]:
    """Decisions from played turns: permanents, facts and public history."""
    batch = collect(case.collector, case.agent, 24)
    return {
        key: torch.from_numpy(value[-8:, 0]).float()
        for key, value in batch.observations().observation.items()
    }


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


def test_remapped_references_are_every_object_table_column() -> None:
    """A new object-table reference must be remapped, not trimmed silently."""
    layout = managym.observation_layout(policy_history=True)
    one_based: set[tuple[str, int]] = set()
    zero_based: set[tuple[str, int]] = set()
    for key, array in layout.items():
        for column in array["columns"].values():
            encoding = column["encoding"].lower()
            if column["kind"] != "reference" or "object" not in encoding:
                continue
            target = zero_based if encoding.startswith("zero-based") else one_based
            target.add((key, column["index"]))
        element = array.get("element")
        if element is not None and element["kind"] == "reference":
            assert key == "action_focus", key
    assert one_based == set(ONE_BASED_REFERENCES)
    assert zero_based == set(ZERO_BASED_REFERENCES)


@pytest.mark.parametrize("history", [False, True])
def test_outputs_and_gradients_match_default(
    history: bool,
) -> None:
    torch.manual_seed(302)
    case = _policy(history=history)
    obs = _later_decisions(case)
    agent = _bucketed(case)
    buckets = bucket_sizes(_host(obs), agent.observation_space.encoder, objects=True)
    assert buckets is not None and buckets.trims_objects
    outputs = []
    for model in (case.agent, agent):
        logits, values = model.forward_distribution(admit_observation(model, obs))
        outputs.append((logits, values))
        legal = obs["actions_valid"].bool()
        (logits[legal].square().mean() + values.square().mean()).backward()
    (expected_logits, expected_values), (logits, values) = outputs
    assert logits.shape == expected_logits.shape
    torch.testing.assert_close(logits, expected_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values, expected_values, atol=1e-5, rtol=1e-5)
    for (name, a), b in zip(
        case.agent.named_parameters(), agent.parameters(), strict=True
    ):
        if a.grad is None:
            # The complete catalog may add exactly-zero program gradients.
            assert b.grad is None or not b.grad.any(), name
        else:
            assert b.grad is not None, name
            torch.testing.assert_close(b.grad, a.grad, atol=1e-5, rtol=1e-4)
    with torch.no_grad(), PolicyInference(agent) as scope:
        prepared = scope.prepare(obs)
        assert prepared.buckets == buckets
        assert prepared["facts"].shape == obs["facts"].shape
        served, _ = agent.forward_distribution(prepared)
    torch.testing.assert_close(served, expected_logits, atol=1e-5, rtol=1e-5)


def _set_fact_object(value: float) -> Callable[[dict[str, torch.Tensor]], None]:
    def plant(obs: dict[str, torch.Tensor]) -> None:
        obs["facts"][0, 0] = 0
        obs["facts"][0, 0, 11] = 1
        obs["facts"][0, 0, 1] = 1
        obs["facts"][0, 0, FACT_OBJECT_INDEX] = value

    return plant


def _set_focus(value: float) -> Callable[[dict[str, torch.Tensor]], None]:
    def plant(obs: dict[str, torch.Tensor]) -> None:
        slot = int(obs["actions_valid"][0].nonzero()[0])
        obs["action_focus"][0, slot, 0] = value

    return plant


def _shrink_facts(obs: dict[str, torch.Tensor]) -> None:
    obs["facts"] = obs["facts"][:, :100]


@pytest.mark.parametrize(
    ("plant", "message"),
    [
        (_set_fact_object(10_000), "invalid fact object reference"),
        (_set_fact_object(2.5), "invalid fact object reference"),
        (_set_focus(10_000), "invalid action focus reference"),
        (_shrink_facts, "facts"),
    ],
)
def test_rejections_match_default(
    plant: Callable[[dict[str, torch.Tensor]], None],
    message: str,
) -> None:
    case = _policy()
    obs = _later_decisions(case)
    agent = _bucketed(case)
    plant(obs)
    errors = []
    for model in (case.agent, agent):
        with pytest.raises(ValueError, match=message) as error:
            model.forward_distribution(admit_observation(model, obs))
        errors.append(str(error.value))
    assert errors[0] == errors[1]


def test_reference_to_padding_object_is_kept() -> None:
    """A focus on the last permanent slot keeps that slot, as the default reads it."""
    case = _policy()
    obs = _later_decisions(case)
    agent = _bucketed(case)
    objects = case.agent._is_agent_template.shape[1]
    _set_focus(objects - 1)(obs)
    buckets = bucket_sizes(_host(obs), agent.observation_space.encoder, objects=True)
    assert buckets is not None
    assert buckets.sizes["permanents"] == buckets.capacities["permanents"]
    with torch.no_grad():
        expected, _ = case.agent.forward_distribution(
            admit_observation(case.agent, obs)
        )
        actual, _ = agent.forward_distribution(admit_observation(agent, obs))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("objects", [True, False])
def test_row_needs_regroup_to_the_measured_buckets(objects: bool) -> None:
    """Needs measured once per row give any grouping of those rows the buckets
    measuring the group directly gives, so later epochs need not measure."""
    case = _policy(history=True)
    encoder = case.agent.observation_space.encoder
    batch = collect(case.collector, case.agent, 24)
    obs = {
        key: torch.from_numpy(value[:, 0]).float()
        for key, value in batch.observations().observation.items()
    }
    # A fractional object reference in one row, which only that row needs.
    _set_fact_object(3.7)(obs)
    host = _host(obs)
    needs = row_needs(host, encoder)
    assert needs is not None and len(needs) == len(host["actions_valid"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        rows = rng.choice(len(needs), size=rng.integers(1, len(needs)), replace=False)
        group = {key: value[rows] for key, value in host.items()}
        assert sizes_of(needs[rows], encoder, objects=objects) == bucket_sizes(
            group, encoder, objects=objects
        )


def test_collection_records_each_rows_measured_needs() -> None:
    """A bucketed collector records the needs it measured for every learner row,
    equal to measuring the recorded rows again, so learning reuses them."""
    case = _policy(history=True)
    agent = _bucketed(case)
    batch = collect(case.collector, agent, 12)
    rows = np.arange(batch.actions.size)
    recorded = batch.recorded.gather_needs(rows)
    assert recorded is not None
    host = batch.observations(rows).observation
    encoder = agent.observation_space.encoder
    assert np.array_equal(recorded, row_needs(host, encoder))
