"""Observation admission: checked once where observations enter, never in forward."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pytest
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.admission import AdmittedObservation, admit_observation
from manabot.model.inference import PolicyInference
from manabot.training.learning import learner as learner_module
from manabot.training.learning.learner import Learner
from manabot.training.loops.damped_policy import update_damped_iteration
from manabot.training.loops.ppo import update_iteration
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from tests.training.test_recovery import recipe


def _policy(*, history: bool = True) -> tuple[ActorCritic, RolloutBatch]:
    torch.set_num_threads(1)
    torch.manual_seed(299)
    config = recipe()
    config.agent.value_aggregation = "masked_mean"
    if history:
        config.observation.policy_history_version = 1
        config.agent.recent_events = True
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
    return agent, collector.collect(agent, 8)


def _observation(batch: RolloutBatch) -> dict[str, torch.Tensor]:
    rows = batch.observations(np.arange(batch.actions.size))
    return {
        key: torch.from_numpy(value).float()
        if key == "action_focus"
        else torch.from_numpy(value)
        for key, value in rows.observation.items()
    }


def test_forward_rejects_unadmitted_and_foreign_observations() -> None:
    agent, batch = _policy()
    obs = _observation(batch)
    with pytest.raises(TypeError, match="admit_observation"):
        agent(obs)
    with pytest.raises(TypeError, match="admit_observation"):
        agent.get_action_and_value(obs)
    admitted = admit_observation(agent, obs)
    # A copy with the same configuration (a restored or bootstrap model) reads it.
    agent(admitted)
    deepcopy(agent)(admitted)
    other = deepcopy(agent)
    other.config = other.config.model_copy(update={"policy_film": True})
    with pytest.raises(ValueError, match="different policy configuration"):
        other(admitted)


@contextmanager
def _validation_reads() -> Iterator[list[str]]:
    """Record host reads that only validation used: scalar and equality reads.

    Compaction keeps nonzero/tolist/cpu reads to choose its work.
    """
    reads: list[str] = []
    patched: list[tuple[object, str, Callable[..., object]]] = [
        (torch.Tensor, "__bool__", torch.Tensor.__bool__),
        (torch.Tensor, "item", torch.Tensor.item),
        (torch, "equal", torch.equal),
    ]

    def recording(name: str, original: Callable[..., object]) -> Callable[..., object]:
        def read(*args: object, **kwargs: object) -> object:
            reads.append(name)
            return original(*args, **kwargs)

        return read

    for owner, name, original in patched:
        setattr(owner, name, recording(name, original))
    try:
        yield reads
    finally:
        for owner, name, original in patched:
            setattr(owner, name, original)


@pytest.mark.parametrize("grad", [False, True])
def test_admitted_forward_performs_no_validation_reads(grad: bool) -> None:
    agent, batch = _policy()
    obs = _observation(batch)
    if grad:
        agent.train()
        admitted = admit_observation(agent, obs)
        with _validation_reads() as reads:
            agent.forward_distribution(admitted)
    else:
        with PolicyInference(agent) as scope, torch.no_grad():
            prepared = scope.prepare(obs)
            agent(prepared)  # The first forward encodes the cached catalog.
            with _validation_reads() as reads:
                agent(prepared)
    assert reads == []


@pytest.mark.parametrize("rule", ["ppo", "damped"])
def test_learning_admits_each_trained_row_once(
    monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    admit = learner_module.admit_observation

    def admitted_rows(epochs: int) -> list[int]:
        agent, batch = _policy(history=False)
        rows: list[int] = []

        def counting(
            model: ActorCritic, obs: dict[str, torch.Tensor]
        ) -> AdmittedObservation:
            rows.append(len(obs["actions_valid"]))
            return admit(model, obs)

        monkeypatch.setattr(learner_module, "admit_observation", counting)
        learner = Learner(agent, "cpu", minibatch_rows=2)
        if rule == "ppo":
            update_iteration(
                learner,
                batch,
                Learning(epochs=epochs, minibatches=2),
                0.5,
                np.random.default_rng(3),
            )
        else:
            update_damped_iteration(
                learner, batch, DampedPolicyLearning(gradient="damped_policy"), 1
            )
        # Bootstrap rows first, then trained rows in minibatch-sized chunks.
        assert rows[0] == len(batch.next_done)
        assert sum(rows[1:]) <= batch.actions.size
        return rows

    once = admitted_rows(1)
    if rule == "ppo":
        assert admitted_rows(3) == once
