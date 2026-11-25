"""Serving preparation preserves policy meaning across collection and learning."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.env.match import Match, Reward
from manabot.env.observation import (
    ObservationSpace,
)
from manabot.infra.config import RewardConfig
from manabot.policy.prepare import CatalogPrograms, Prepared
from manabot.policy.spans import ObservationSelection, observation_spans
from manabot.training.rollout import collection
from manabot.training.rollout.collection import (
    Collector,
    RolloutBatch,
    collect,
    open_collector,
)
from tests.fixtures.rollout_arrays import projected_fields, trajectory_rows
from tests.model.policy_fixture import recipe
from tests.model.prepared_routes import device_input, program_input


def _batch_tensors(batch: RolloutBatch) -> dict[str, torch.Tensor]:
    rows = trajectory_rows(batch)
    result = {
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
        elif isinstance(value, (dict, ObservationSelection)):
            for key, array in projected_fields(value).items():
                result[f"{field.name}/{key}"] = torch.from_numpy(array)
        elif value is not None:
            raise TypeError(f"unexpected rollout field {field.name}")
    return result


@dataclass
class PolicyFixture:
    agent: ActorCritic
    collector: Collector
    obs: ObservationSelection


def _legal(obs: ObservationSelection) -> torch.Tensor:
    counts = torch.from_numpy(obs.action_counts.astype(np.int64))
    return torch.arange(obs.space.config.max_actions) < counts[:, None]


def _policy(
    *,
    semantic: bool = True,
    history: bool = False,
    bucketed_shapes: bool = False,
) -> PolicyFixture:
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
    agent = ActorCritic(space, config.agent, bucketed_shapes=bucketed_shapes).eval()
    collector = open_collector(
        space,
        Match(config.match),
        Reward(RewardConfig()),
        num_envs=2,
        seed=91,
        opponent_mode="self",
        numerics=TorchCollectionSession(device="cpu", seed=91, precision="float32"),
    )
    return PolicyFixture(
        agent,
        collector,
        collection._retain_obs_rows(collector, np.arange(2)),
    )


def test_half_precision_head_keeps_finite_public_logits() -> None:
    """The output boundary handles half scores without overflowing -1e8."""
    case = _policy()

    def half_scores(
        module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        return output.half()

    # Inject the autocast head's output dtype without depending on a GPU or
    # running unrelated encoder operators in CPU half precision.
    handle = case.agent.policy_head.register_forward_hook(half_scores)
    try:
        logits, _ = case.agent(device_input(case.agent, case.obs))
        legal = _legal(case.obs)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert (logits[~legal] == -1e8).all()
        logits[legal].sum().backward()
        parameter = case.agent.policy_head[-1].weight
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    finally:
        handle.remove()


@pytest.mark.parametrize("semantic", [False, True])
def test_preparation_preserves_forward_and_training_gradients(semantic: bool) -> None:
    case = _policy(semantic=semantic)
    reference = deepcopy(case.agent)
    with PolicyInference(case.agent) as scope, torch.no_grad():
        for _ in range(2):
            prepared = scope.prepare(case.obs)
            actual = case.agent.forward_distribution(prepared)
            expected = reference.forward_distribution(device_input(reference, case.obs))
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Serving must not detach or reuse a catalog during a subsequent update.
    for agent in (case.agent, reference):
        logits, values = agent.forward_distribution(device_input(agent, case.obs))
        loss = logits[_legal(case.obs)].square().mean() + values.square().mean()
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
    # Later decisions (stream 0's last eight) hold permanents and, with
    # history, public events.
    batch = collect(case.collector, case.agent, 16)
    steps, streams = batch.actions.shape
    case.obs = batch.observations(np.arange(steps - 8, steps) * streams).observation
    semantic = case.agent.semantic_cards
    assert semantic is not None
    planted = _plant_unseen_definitions(case.agent, case.obs, history)
    full = deepcopy(case.agent)
    encode = semantic.definitions
    referenced: list[int] = []

    def poisoned(programs: CatalogPrograms[torch.Tensor]) -> torch.Tensor:
        """Unreferenced rows read NaN, so any missed reference source fails."""
        rows = programs.rows
        referenced.append(int(rows.unique().numel()))
        table = encode(programs)
        unreferenced = torch.ones(len(table), dtype=torch.bool)
        unreferenced[0] = False
        unreferenced[rows] = False
        return table.masked_fill(unreferenced[:, None], float("nan"))

    semantic.definitions = poisoned  # type: ignore[method-assign]
    # The reference encodes the complete catalog.
    assert full.semantic_cards is not None
    full_encode = full.semantic_cards.definitions
    full.semantic_cards.definitions = lambda programs: full_encode(
        program_input(full.semantic_cards)
    )  # type: ignore[method-assign]
    outputs = []
    for agent in (case.agent, full):
        logits, values = agent.forward_distribution(device_input(agent, planted))
        outputs.append((logits, values))
        (logits[_legal(case.obs)].square().mean() + values.square().mean()).backward()
    assert 0 < referenced[0] < semantic.catalog_shape.definitions
    for a, b in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(case.agent.parameters(), full.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def _plant_unseen_definitions(
    agent: ActorCritic, obs: ObservationSelection, history: bool
) -> Prepared[np.ndarray]:
    """Learner rows where every reference source names a definition no other
    source names, so the encoded subset must draw on each source separately.

    The definitions are planted in the native rows before preparation, which
    scans them for the catalog subset.
    """
    assert agent.semantic_cards is not None
    lookup = agent.semantic_cards.definition_rows
    spans = observation_spans(
        obs, agent.observation_space, bucketed=False, objects=False
    )
    named = set(
        lookup[torch.from_numpy(spans.tables["semantic_cards"].integers).long()]
        .flatten()
        .tolist()
    )
    unseen = iter(
        transport
        for transport in range(1, len(lookup))
        if int(lookup[transport]) > 0 and int(lookup[transport]) not in named
    )
    sources = [
        ("agent_permanents", "definition"),
        ("known_hand", "definition"),
        ("decklists", "definition"),
        ("relations", "source_definition"),
    ]
    if history:
        sources.append(("events", "source_definition"))
    tables = dict(spans.tables)
    for key, field in sources:
        rows = tables[key]
        assert len(rows.records), f"no live {key} row to plant"
        records = rows.records.copy()
        records[field][0] = next(unseen)
        tables[key] = replace(rows, records=records)
    return agent._prepare_rows(
        replace(spans, tables=tables),
        agent._admission_spec().tables,
        False,
        None,
        None,
    )


def test_preparation_preserves_object_validity_and_mode_checks() -> None:
    case = _policy()
    with PolicyInference(case.agent) as scope:
        case.agent.train()
        with pytest.raises(RuntimeError, match="mode/configuration"):
            scope.prepare(case.obs)


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
        expected = case.agent(device_input(case.agent, case.obs))
        actual = case.agent(fresh.prepare(case.obs))
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_prepared_collection_matches_reference() -> None:
    optimized, reference = _policy(), _policy()

    @torch.no_grad()
    def plain_inputs(
        scope: PolicyInference, obs: ObservationSelection
    ) -> Prepared[torch.Tensor]:
        """Admitted outside the scope: no cached catalog or scope checks."""
        return device_input(scope.agent, obs)

    actual = collect(optimized.collector, optimized.agent, 8)
    with patch.object(PolicyInference, "prepare", plain_inputs):
        expected = collect(reference.collector, reference.agent, 8)
    for key, value in _batch_tensors(actual).items():
        torch.testing.assert_close(value, _batch_tensors(expected)[key], atol=0, rtol=0)
    torch.testing.assert_close(
        optimized.collector.numerics.snapshot_rng(),
        reference.collector.numerics.snapshot_rng(),
        atol=0,
        rtol=0,
    )
