"""Packed seat trajectories bootstrap at their viewer's frozen game revision."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends import resolve
import manabot.backends.torch as backend
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.observation import (
    ObservationSpace,
    allocate_vector_buffers,
    encode_observation,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.checkpoint import from_torch_state
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.rollout.actions import ACTION_FIELDS, PackedActions
from manabot.training.rollout.collection import (
    RecordedObservations,
    RolloutBatch,
    TrajectorySpan,
)
from manabot.training.updates.damped_policy import (
    compute_targets,
    update_damped_iteration,
)
from manabot.training.updates.ppo import update_iteration
import managym
from tests.model.prepared_routes import device_input


@pytest.fixture(autouse=True)
def bounded_torch() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            yield
    finally:
        torch.set_num_threads(previous)


@torch.no_grad()
def three_choices(agent: ActorCritic) -> tuple[RolloutBatch, torch.Tensor]:
    """Play 0, 0, 1 and compare fixed-seat critic reads to independent scalar views."""
    space = agent.observation_space
    setup = managym.GameSetup(
        [
            managym.PlayerConfig("a", {"Mountain": 8, "Gray Ogre": 4}),
            managym.PlayerConfig("b", {"Island": 8, "Wind Drake": 4}),
        ],
        decklists="closed",
    )
    vector = managym.VectorEnv(1, seed=42)
    scalar = managym.Env(seed=42)
    scalar.reset(setup)
    buffers = allocate_vector_buffers(space, 1)
    vector.set_buffers(buffers, layout=space.layout)
    vector.reset_all_into_buffers(setup)
    for _ in range(4):
        vector.step_into_buffers([0])
        scalar.step(0)
    rows = space.allocate(3)
    for row, seat in enumerate((0, 0, 1)):
        assert vector.current_agent_indices() == [seat]
        for key in rows:
            rows[key][row] = buffers[key][0]
        vector.step_into_buffers([0])
        scalar.step(0)
    prepared = device_input(agent, rows)
    logits, values = agent.forward_distribution(prepared)
    logs = logits.masked_fill(~prepared.legal, -torch.inf).log_softmax(-1).numpy()
    outcomes = None
    if agent.config.value_kind == "categorical_wdl":
        outcomes = values.softmax(-1).numpy()[:, None, :]
        values = backend.rl.expected_outcome(values.softmax(-1))[:, None]

    frozen = space.allocate(2)
    expected = []
    # Store seat 1 first: span order is independent of bootstrap storage order.
    for index, seat in enumerate((1, 0)):
        target = {key: value[index : index + 1] for key, value in frozen.items()}
        [read] = vector.write_observations_for_players(
            [seat], target, layout=space.layout
        )
        assert (read.slot, read.seed, read.player) == (0, 42, seat)
        assert bool(target["actions_valid"].any()) == (
            seat == scalar.current_agent_index()
        )
        encoded = encode_observation(space, scalar.observation_for_player(seat))
        _, prediction = agent.forward_distribution(
            device_input(agent, {key: value[None] for key, value in encoded.items()})
        )
        expected.append(prediction[0])

    batch = RolloutBatch(
        recorded=RecordedObservations(
            {key: value for key, value in rows.items() if key not in ACTION_FIELDS},
            np.arange(3),
        ),
        actions=np.zeros((3, 1), np.int64),
        logprobs=logs[:, :1],
        rewards=np.zeros((3, 1), np.float32),
        dones=np.zeros((3, 1), bool),
        values=values.numpy(),
        next_obs=frozen,
        next_done=np.zeros(2, bool),
        action_data=PackedActions.pack(rows, logs, validator=backend.rollout),
        outcome_probabilities=outcomes,
        trajectories=(TrajectorySpan(0, 2, 0, 1), TrajectorySpan(2, 3, 1, 0)),
    )
    return batch, torch.stack(expected)


def terminal_batch(batch: RolloutBatch) -> RolloutBatch:
    """Assign the terminal outcome of a seat-0 win to the two fixture spans."""
    return replace(
        batch,
        next_obs={},
        next_done=np.zeros(0, bool),
        dones=np.array([[False], [True], [True]]),
        rewards=np.array([[0], [1], [-1]], np.float32),
        trajectories=(TrajectorySpan(0, 2, 0), TrajectorySpan(2, 3, 1)),
    )


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_packed_cut_updates_use_the_frozen_view_of_each_seat(
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(attention_on=False, hidden_dim=8, value_kind=kind),
    )
    batch, raw = three_choices(agent)
    learner = Learner(agent, "cpu", minibatch_rows=3)
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        policy_lambda=0.25,
        value_lambda=0.75,
        advantage_quantile=0,
        min_advantage=0,
    )
    targets = compute_targets(backend, learner, AdmittedRollout(agent, batch), learning)
    predictions = raw.softmax(-1) if kind == "categorical_wdl" else raw
    expected = predictions[[1, 1, 0]]
    # The two native views must make distinguishable predictions for this control.
    assert not torch.allclose(predictions[0], predictions[1])
    signed = (
        backend.rl.expected_outcome(expected)
        if kind == "categorical_wdl"
        else expected[:, 0].clone()
    )
    signed[0] = 0.75 * batch.values[1, 0].item() + 0.25 * signed[1]
    collected = (
        batch.outcome_probabilities if kind == "categorical_wdl" else batch.values
    )
    assert collected is not None
    expected[0] = (
        0.25 * torch.from_numpy(collected[1, 0:1]).flatten() + 0.75 * expected[1]
    )
    torch.testing.assert_close(
        targets.values, expected[:, None] if kind == "categorical_wdl" else expected
    )
    torch.testing.assert_close(
        targets.advantages[:, 0], signed - torch.from_numpy(batch.values[:, 0])
    )

    result = update_damped_iteration(backend, learner, batch, learning, 1)
    assert result["optimizer_exposures"] == 3
    assert result["bootstrapped_tail_fraction"] == 1
    assert {g["terminal_distance"] for g in result["selection_groups"]} == {"censored"}


def test_mixed_ppo_discounts_each_cut_and_censors_only_its_own_span() -> None:
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    batch, raw = three_choices(agent)
    batch = replace(
        batch,
        dones=np.array([[False], [False], [True]]),
        rewards=np.array([[0], [0], [-1]], np.float32),
        next_obs={key: value[1:] for key, value in batch.next_obs.items()},
        next_done=np.zeros(1, bool),
        trajectories=(TrajectorySpan(0, 2, 0, 0), TrajectorySpan(2, 3, 1)),
    )
    diagnostics = update_iteration(
        backend,
        Learner(agent, "cpu", minibatch_rows=3),
        batch,
        Learning(gamma=0.5, epochs=1, minibatches=1, retained_fraction=1),
        0.0,
        np.random.default_rng(2),
    )
    assert diagnostics["optimizer_exposures"] == 3
    assert diagnostics["bootstrapped_tail_fraction"] == 0.5
    assert diagnostics["return_mean"] == pytest.approx(
        (0.75 * raw[1].item() - 1) / 3, abs=1e-6
    )
    assert (
        sum(
            g["rows"]
            for g in diagnostics["selection_groups"]
            if g["terminal_distance"] == "censored"
        )
        == 2
    )


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_packed_terminal_spans_need_no_bootstrap_input(
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(attention_on=False, hidden_dim=8, value_kind=kind),
    )
    batch, _ = three_choices(agent)
    batch = terminal_batch(batch)
    admitted = AdmittedRollout(agent, batch)
    assert admitted.next_observation(torch.device("cpu")) is None
    targets = compute_targets(
        backend,
        Learner(agent, "cpu", minibatch_rows=3),
        admitted,
        DampedPolicyLearning(gradient="damped_policy", value_lambda=1),
    )
    signed = (
        backend.rl.expected_outcome(targets.values)
        if kind == "categorical_wdl"
        else targets.values
    )
    torch.testing.assert_close(signed[:, 0], torch.tensor([1.0, 1.0, -1.0]))


def test_mlx_scalar_cut_and_terminal_targets() -> None:
    pytest.importorskip("mlx.core")
    mlx = resolve("mlx")
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
    batch, raw = three_choices(agent)
    checkpoint = from_torch_state(
        agent.config,
        space.config,
        {k: v.detach().numpy() for k, v in agent.state_dict().items()},
        parameter_names=dict(agent.named_parameters()),
        readings=agent.readings,
    )
    learner = mlx.learner.Learner(mlx.load(checkpoint)[0], "cpu", minibatch_rows=3)
    learning = DampedPolicyLearning(
        gradient="damped_policy", policy_lambda=1, value_lambda=1
    )
    for rollout, expected in (
        (batch, raw.numpy()[[1, 1, 0], 0]),
        (terminal_batch(batch), np.array([1, 1, -1])),
    ):
        targets = compute_targets(
            mlx, learner, mlx.rollout.AdmittedRollout(learner.agent, rollout), learning
        )
        np.testing.assert_allclose(
            mlx.arrays.to_numpy(targets.values)[:, 0], expected, atol=1e-5
        )


def test_invalid_packed_boundaries_are_rejected_at_construction() -> None:
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    batch, _ = three_choices(agent)
    for spans in (
        (TrajectorySpan(0, 1, 0, 1), TrajectorySpan(2, 3, 1, 0)),  # missing row
        (TrajectorySpan(0, 2, 0, 2), TrajectorySpan(2, 3, 1, 0)),  # missing observation
        (TrajectorySpan(0, 2, 0), TrajectorySpan(2, 3, 1, 0)),  # cut lacks a bootstrap
    ):
        with pytest.raises(ValueError, match="trajectory|nonterminal spans"):
            replace(batch, trajectories=spans)
    with pytest.raises(ValueError, match="cross terminals"):
        replace(batch, dones=np.array([[True], [False], [False]]))
