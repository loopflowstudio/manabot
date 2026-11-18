"""Execution chunks preserve full prefixes, RNG partition and logical updates."""

from copy import deepcopy
from typing import Literal

import numpy as np
import pytest
import torch

import manabot.backends.torch as backend
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.rollout import AdmittedRollout
import manabot.env.memory_store as storage
from manabot.env.memory_store import (
    MEMORY_FIELDS,
    MemoryStore,
    Prefix,
    PrefixIdentity,
    PrefixObservation,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    ModelConfig,
)
from manabot.infra.profiler import Profiler
from manabot.policy.prepare import Prepared
from manabot.sim.rollout import BatchedSampler
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.rollout.actions import ACTION_FIELDS, PackedActions
from manabot.training.rollout.collection import (
    RecordedObservations,
    RolloutBatch,
)
from manabot.training.updates.bootstrap import bootstrap_predictions
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
from tests.model.test_decision_memory import memory_input, policy


def mixed_rows() -> tuple[ObservationSpace, PrefixObservation]:
    space, dense = memory_input()
    store = MemoryStore((4, 1024, 1024, 16384))
    prefix = store.capture(dense, 0, PrefixIdentity("game", 0, "root"))
    empty = Prefix(PrefixIdentity("empty", 1, "root"), None)
    # Duplicated, shuffled and empty prefixes; unequal final execution chunk.
    return space, PrefixObservation(
        {k: np.repeat(v, 5, axis=0) for k, v in dense.items()},
        [prefix, empty, prefix, prefix, empty],
        (4, 1024, 1024, 16384),
    )


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
@pytest.mark.parametrize("summaries", [1, 4])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("filter_scope", ["actor", "actor_critic"])
def test_memory_execution_preserves_sampling_bootstrap_and_adam(
    encoder: Literal["transformer", "gru"],
    summaries: int,
    categorical: bool,
    filter_scope: Literal["actor", "actor_critic"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space, rows = mixed_rows()
    model = policy(space, encoder, summaries)
    if categorical:
        model = type(model)(space, replace_config(model.config))
    session = CollectionSession(device="cpu", seed=111)
    with session.scope(model) as scope:
        rng = session.snapshot_rng()
        expected = scope.sample(rows, Profiler())
        expected_rng = session.snapshot_rng()
        session.rng.set_state(rng)
        sizes: list[int] = []
        forward = model.forward_distribution

        def bounded_forward(
            obs: Prepared[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sizes.append(len(obs.legal))
            return forward(obs)

        with monkeypatch.context() as patch:
            patch.setattr(model, "forward_distribution", bounded_forward)
            actual = scope.sample(rows, Profiler(), execution_rows=2)
        assert sizes == [2, 2, 1]
    np.testing.assert_array_equal(actual.actions, expected.actions)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6)
    np.testing.assert_allclose(
        actual.log_probabilities, expected.log_probabilities, atol=1e-6
    )
    assert torch.equal(session.snapshot_rng(), expected_rng)
    # Frozen opponent sampling keeps one global backend RNG call, too.
    torch.manual_seed(42)
    frozen = BatchedSampler(model, execution_rows=None).select(rows, np.arange(5))
    torch.manual_seed(42)
    frozen_chunked = BatchedSampler(model, execution_rows=2).select(rows, np.arange(5))
    np.testing.assert_array_equal(frozen, frozen_chunked)

    packed = PackedActions.pack(
        {k: v for k, v in rows.items() if k in ACTION_FIELDS},
        actual.log_probabilities,
        validator=backend.rollout,
    )
    batch = RolloutBatch(
        RecordedObservations(
            {k: v for k, v in rows.items() if k not in ACTION_FIELDS},
            np.arange(5),
            memory=rows.prefixes,
            memory_capacities=(4, 1024, 1024, 16384),
        ),
        actual.actions[None],
        actual.logprobs[None],
        np.array([[1.0, -1.0, 0.0, -1.0, 0.0]]),
        np.array([[True, True, False, True, True]]),
        actual.values[None],
        rows,
        np.zeros(5, bool),
        packed,
        outcome_probabilities=None
        if actual.outcomes is None
        else actual.outcomes[None],
    )
    full = Learner(deepcopy(model), "cpu", minibatch_rows=5)
    chunked = Learner(
        deepcopy(model), "cpu", minibatch_rows=5, execution_rows=2, bootstrap_rows=2
    )
    with torch.no_grad():
        a = bootstrap_predictions(
            backend, full, AdmittedRollout(full.agent, batch), full.agent
        )
        b = bootstrap_predictions(
            backend, chunked, AdmittedRollout(chunked.agent, batch), chunked.agent
        )
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    # Bound materialization, including admission; no full-group dense gather.
    gather = storage.gather_memory
    gathered: list[int] = []

    def bounded(prefixes: tuple[Prefix, ...]) -> dict[str, np.ndarray]:
        gathered.append(len(prefixes))
        assert len(prefixes) <= 2
        return gather(prefixes)

    for rule in ("damped", "ppo"):
        if categorical and rule == "ppo":
            continue  # PPO's existing contract is scalar.
        full = Learner(deepcopy(model), "cpu", minibatch_rows=5)
        chunked = Learner(
            deepcopy(model),
            "cpu",
            minibatch_rows=5,
            execution_rows=1 if filter_scope == "actor_critic" else 2,
            bootstrap_rows=2,
        )
        if rule == "damped":
            learning = DampedPolicyLearning(
                gradient="damped_policy",
                filter_scope=filter_scope,
                advantage_quantile=0,
                min_advantage=0,
                max_grad_norm=0.01,
            )
            expected_diag = update_damped_iteration(backend, full, batch, learning, 1)
            with monkeypatch.context() as patch:
                patch.setattr(storage, "gather_memory", bounded)
                actual_diag = update_damped_iteration(
                    backend, chunked, batch, learning, 1
                )
        else:
            learning = Learning(
                filter_scope=filter_scope,
                epochs=1,
                minibatches=1,
                max_grad_norm=0.01,
                retained_fraction=1,
            )
            expected_diag = update_iteration(
                backend, full, batch, learning, 0, np.random.default_rng(3)
            )
            with monkeypatch.context() as patch:
                patch.setattr(storage, "gather_memory", bounded)
                actual_diag = update_iteration(
                    backend, chunked, batch, learning, 0, np.random.default_rng(3)
                )

        for name, parameter in full.agent.named_parameters():
            torch.testing.assert_close(
                parameter,
                chunked.agent.get_parameter(name),
                atol=2e-6,
                rtol=1e-4,
                msg=lambda message: f"{rule} {name}: {message}",
            )
        for state in chunked.optimizer.state.values():
            assert state["step"].item() == 1
        for key in ("loss", "policy_loss", "value_loss", "gradient_norm", "entropy"):
            assert actual_diag[key] == pytest.approx(
                expected_diag[key], abs=2e-6, rel=1e-4
            )
        assert (
            actual_diag["optimizer_exposures"] == expected_diag["optimizer_exposures"]
        )
    assert not gathered  # cached fragments replace dense history gathers
    assert not set(MEMORY_FIELDS).intersection(rows)


def replace_config(config: ModelConfig) -> ModelConfig:
    return config.model_copy(update={"value_kind": "categorical_wdl"})
