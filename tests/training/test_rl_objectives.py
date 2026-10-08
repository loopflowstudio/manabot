"""Independent RL estimator, policy-reference and update-clock contracts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manabot.sim.net_opponent import NetOpponentTrainer, RolloutBatch, transition_gae
from manabot.training.models import Learning, Schedule
from manabot.training.objectives import (
    reference_distribution,
    update_ema,
    update_iteration,
)
from manabot.training.selection import selected_rows


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.1, 0.4, -0.2, 0.0]))
        self.value = torch.nn.Parameter(torch.tensor(0.2))
        self.register_buffer("identity", torch.tensor([7]))

    def forward(self, obs):
        valid = obs["actions_valid"] > 0
        return self.logits.expand(valid.shape).masked_fill(
            ~valid, -torch.inf
        ), self.value.expand(len(valid), 1)

    def get_value(self, obs):
        return self(obs)[1]


def fixture():
    agent = TinyPolicy()
    obs = {
        "actions_valid": np.tile([1, 1, 1, 0], (4, 1)).astype(np.float32),
        "actions": np.tile(
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], (4, 1, 1)
        ).astype(np.float32),
    }
    with torch.no_grad():
        logits, values = agent({k: torch.as_tensor(v) for k, v in obs.items()})
        probabilities = logits.softmax(-1).numpy()
    actions = np.array([[0], [1], [2], [0]])
    batch = RolloutBatch(
        obs={k: v[:, None] for k, v in obs.items()},
        actions=actions,
        logprobs=np.log(probabilities[np.arange(4), actions[:, 0]])[:, None],
        rewards=np.array([[0], [1], [0], [-1]], dtype=np.float32),
        dones=np.array([[False], [True], [False], [True]]),
        values=values.detach().numpy().copy(),
        next_obs={k: v[:1] for k, v in obs.items()},
        next_done=np.array([True]),
        probabilities=probabilities[:, None],
        log_probabilities=logits.log_softmax(-1).numpy()[:, None],
    )
    trainer = SimpleNamespace(
        agent=agent,
        experiment=SimpleNamespace(device="cpu"),
        optimizer=torch.optim.Adam(agent.parameters(), lr=0.01),
        _obs_to_tensors=NetOpponentTrainer._obs_to_tensors,
    )
    return trainer, batch


def control(**changes):
    return Learning(
        retained_fraction=1,
        epochs=1,
        minibatches=1,
        learning_rate=Schedule(initial=0.01),
        tau=Schedule(initial=0),
        collection_kl=0,
        policy_lambda=0.95,
        value_lambda=0.95,
    ).model_copy(update=changes)


def run(trainer, batch, learning):
    return update_iteration(trainer, batch, learning, 0.5, np.random.default_rng(9))


def test_nonzero_values_adjacent_episodes_and_exact_tail():
    rewards = torch.tensor([[0.0], [1.0], [-1.0], [0.0]])
    values = torch.tensor([[0.2], [0.3], [0.4], [0.5]])
    ends = torch.tensor([[False], [True], [True], [False]])
    advantages, returns = transition_gae(
        rewards, values, ends, torch.tensor([0.8]), 0.9, 1
    )
    torch.testing.assert_close(returns[:, 0], torch.tensor([0.9, 1, -1, 0.72]))
    torch.testing.assert_close(advantages, returns - values)
    _, one_step = transition_gae(rewards, values, ends, torch.tensor([0.8]), 0.9, 0)
    torch.testing.assert_close(one_step[:, 0], torch.tensor([0.27, 1, -1, 0.72]))


def test_reference_and_filter_support_order_and_threshold():
    _, batch = fixture()
    obs = {k: torch.as_tensor(v[:, 0]) for k, v in batch.obs.items()}
    for kind, expected in [
        ("uniform", [1 / 3, 1 / 3, 1 / 3, 0]),
        ("action_type_uniform", [0.25, 0.25, 0.5, 0]),
    ]:
        result = reference_distribution(obs, kind)
        torch.testing.assert_close(result[0], torch.tensor(expected))
        assert torch.equal(result > 0, obs["actions_valid"] > 0)
        permutation = torch.tensor([2, 3, 0, 1])
        moved = reference_distribution(
            {k: v[:, permutation] for k, v in obs.items()}, kind
        )
        torch.testing.assert_close(moved, result[:, permutation])
    assert selected_rows(torch.tensor([0.1, -3, 2, 0]), 0.5, 2.5).tolist() == [1]
    obs["actions_valid"].zero_()
    with pytest.raises(ValueError, match="legal offer"):
        reference_distribution(obs, "uniform")


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": 0.99},
        {"policy_lambda": 0.5},
        {"value_lambda": 0.8},
        {"retained_fraction": 0.5},
        {"retained_fraction": 0.25, "min_advantage": 0.01},
        {"tau": Schedule(initial=0.1)},
        {"tau": Schedule(initial=0.1), "reference": "action_type_uniform"},
        {"collection_kl": 0.1},
        {"learning_rate": Schedule(initial=0.01, decay=9)},
        {"tau": Schedule(initial=0.1, decay=9)},
    ],
)
def test_independent_treatments_execute_optimizer(changes):
    trainer, batch = fixture()
    before = deepcopy(trainer.agent.state_dict())
    diagnostic = run(trainer, batch, control(**changes))
    assert diagnostic["optimizer_exposures"] == diagnostic["retained"]
    assert np.isfinite(diagnostic["loss"])
    assert any(
        not torch.equal(before[k], v) for k, v in trainer.agent.state_dict().items()
    )
    assert sum(diagnostic["selected_action_types"]) == diagnostic["retained"]
    assert sum(diagnostic["action_types"]) == diagnostic["rows"]
    assert diagnostic["schedule_progress"] == 0.5


def test_value_estimator_changes_targets_without_changing_filter():
    first, batch = fixture()
    second, _ = fixture()
    a = run(first, batch, control(value_lambda=0))
    b = run(second, batch, control(value_lambda=1))
    assert a["selected_action_types"] == b["selected_action_types"]
    assert a["value_residual_abs_mean"] != b["value_residual_abs_mean"]


def test_empty_filter_preserves_parameters_and_optimizer_state():
    trainer, batch = fixture()
    before = deepcopy(trainer.agent.state_dict())
    result = run(trainer, batch, control(min_advantage=100))
    assert result["optimizer_exposures"] == 0
    assert result["skipped"] == "empty advantage filter"
    assert not trainer.optimizer.state
    for name, value in trainer.agent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_collection_kl_uses_stored_full_distribution():
    trainer, batch = fixture()
    original = batch.probabilities.copy()
    same = run(trainer, batch, control(learning_rate=Schedule(initial=0)))
    assert same["collection_kl"] == pytest.approx(0, abs=1e-7)
    with torch.no_grad():
        trainer.agent.logits[0] += 1
    moved = run(
        trainer, batch, control(learning_rate=Schedule(initial=0), collection_kl=0.1)
    )
    current = trainer.agent.logits[:3].softmax(-1).detach().numpy()
    expected = (current * np.log(current / original[0, 0, :3])).sum()
    assert moved["collection_kl"] == pytest.approx(expected, abs=1e-7)
    np.testing.assert_array_equal(batch.probabilities, original)


def test_ema_is_iteration_clocked_and_never_overwrites_learner():
    trainer, batch = fixture()
    averaged = deepcopy(trainer.agent)
    initial = deepcopy(averaged.state_dict())
    run(trainer, batch, control(epochs=3, minibatches=2))
    for name, value in averaged.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    raw = deepcopy(trainer.agent.state_dict())
    update_ema(averaged, trainer.agent, 0.75)
    for name, value in averaged.named_parameters():
        torch.testing.assert_close(value, 0.75 * initial[name] + 0.25 * raw[name])
    # Even an empty-filter iteration advances the evaluation average once.
    run(trainer, batch, control(min_advantage=100))
    update_ema(averaged, trainer.agent, 0.75)
    for name, value in averaged.named_parameters():
        torch.testing.assert_close(
            value, 0.75**2 * initial[name] + (1 - 0.75**2) * raw[name]
        )
    for name, value in trainer.agent.state_dict().items():
        torch.testing.assert_close(value, raw[name], rtol=0, atol=0)
    trainer.agent.identity.fill_(8)
    update_ema(averaged, trainer.agent, 0)
    assert averaged.identity.item() == 8


def test_actor_only_empty_filter_trains_only_critic() -> None:
    trainer, batch = fixture()
    before_policy = trainer.agent.logits.detach().clone()
    before_value = trainer.agent.value.detach().clone()
    diagnostics = run(trainer, batch, control(min_advantage=100, filter_scope="actor"))
    torch.testing.assert_close(trainer.agent.logits, before_policy, rtol=0, atol=0)
    assert not torch.equal(trainer.agent.value, before_value)
    assert diagnostics["actor_exposures"] == 0
    assert diagnostics["critic_exposures"] == 4
    assert diagnostics["optimizer_exposures"] == 4
    assert sum(group["rows"] for group in diagnostics["selection_groups"]) == 4


def test_bootstrap_uses_collection_model() -> None:
    trainer, batch = fixture()
    batch.dones[:] = False
    batch.rewards[:] = 0
    behavior = deepcopy(trainer.agent)
    with torch.no_grad():
        behavior.value.fill_(3)
    expected, _ = transition_gae(
        torch.tensor(batch.rewards),
        torch.tensor(batch.values),
        torch.tensor(batch.dones),
        torch.tensor([3.0]),
        1,
        0.95,
    )
    diagnostics = update_iteration(
        trainer,
        batch,
        control(min_advantage=100),
        0.5,
        np.random.default_rng(9),
        bootstrap_agent=behavior,
    )
    assert diagnostics["advantage_abs_mean"] == pytest.approx(
        float(expected.abs().mean())
    )
