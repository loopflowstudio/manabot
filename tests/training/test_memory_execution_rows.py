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
from manabot.env.match import Match, Reward
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
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.infra.profiler import Profiler
from manabot.policy.prepare import Prepared
from manabot.sim.rollout import BatchedSampler
from manabot.training.regime import (
    DampedPolicyLearning,
    Execution,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout.actions import ACTION_FIELDS, PackedActions
from manabot.training.rollout.collection import (
    RecordedObservations,
    RolloutBatch,
    SeatRoutedCollector,
)
from manabot.training.updates.bootstrap import bootstrap_predictions
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
import managym
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
        session.restore_rng(rng)
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
    assert gathered and max(gathered) == 2
    assert not set(MEMORY_FIELDS).intersection(rows)


def replace_config(config: ModelConfig) -> ModelConfig:
    return config.model_copy(update={"value_kind": "categorical_wdl"})


@pytest.mark.parametrize("include_own", [False, True])
def test_memory_chunked_collector_restore_keeps_sampling(include_own: bool) -> None:

    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            decision_memory_include_own=include_own,
            max_decision_reference_details=65536,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    model = policy(space, "gru", 4)

    def collector() -> SeatRoutedCollector:
        return SeatRoutedCollector(
            space,
            Match(),
            Reward(RewardConfig()),
            num_envs=3,
            execution_rows=2,
            opponent_mode="self",
            learning_seats="one",
            batch_closure="exact-quota",
            recovery_max_game_microsteps=1000,
            numerics=CollectionSession(device="cpu", seed=111),
        )

    original, resumed = collector(), collector()
    try:
        original.collect(model, 3)
        snapshot = original.snapshot()
        resumed.restore(snapshot, lambda: None)
        rng = torch.get_rng_state()
        expected = original.collect(model, 3)
        torch.set_rng_state(rng)
        actual = resumed.collect(model, 3)
        np.testing.assert_array_equal(actual.actions, expected.actions)
        np.testing.assert_allclose(actual.logprobs, expected.logprobs, atol=1e-6)
        for key in MEMORY_FIELDS:
            np.testing.assert_array_equal(
                actual.next_obs.materialize()[key], expected.next_obs.materialize()[key]
            )
    finally:
        original.close()
        resumed.close()


def test_memory_regime_requires_all_execution_limits_and_retains_them() -> None:
    space, _ = memory_input()
    model = policy(space, "gru", 4)
    stage = TrainSelfPlay(
        id="train", operation="train_self_play", streams=3, updates=1, transitions=3
    )
    fields = dict(
        id="memory-limits",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=model.config,
        observation=space.encoder.config,
    )
    with pytest.raises(ValueError, match="explicit collection_rows"):
        TrainingRegime(**fields, stages=[stage])
    limits = Execution(collection_rows=2, bootstrap_rows=2, learner_rows=2)
    regime = TrainingRegime(
        **fields, stages=[stage.model_copy(update={"execution": limits})]
    )
    restored = TrainingRegime.model_validate_json(regime.model_dump_json())
    assert restored.stages[0].execution.collection_rows == 2
    assert restored.stages[0].execution.bootstrap_rows == 2
    assert restored.stages[0].execution.learner_rows == 2
    assert not {"collection_rows", "bootstrap_rows", "learner_rows"}.intersection(
        Execution().model_dump()
    )
    with pytest.raises(ValueError):
        Execution(learner_rows=0)
