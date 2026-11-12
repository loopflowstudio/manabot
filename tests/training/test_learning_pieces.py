"""Directed checks of the public loss and update-step pieces on small tensors."""

from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
import importlib
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor

from manabot.model.policy_distribution import policy_logs
from manabot.rl.losses import categorical_value_loss, entropy, squared_value_loss
from manabot.rl.numerics import NumericalError
from manabot.rl.returns import standardized_advantages
from manabot.rl.step import StepRecorder, gradient_step, update_ema
from manabot.training.learning import objectives
from manabot.training.learning.health import NumericalHealth, OptimizerHealth
from manabot.training.learning.learner import Learner
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import RolloutBatch


def test_squared_value_loss_accepts_column_predictions() -> None:
    targets = torch.tensor([1.0, -1.0, 0.0])
    expected = torch.tensor([0.25, 4.0, 0.0])
    flat = torch.tensor([0.5, 1.0, 0.0])
    torch.testing.assert_close(squared_value_loss(flat, targets), expected)
    torch.testing.assert_close(squared_value_loss(flat[:, None], targets), expected)
    with pytest.raises(ValueError):
        squared_value_loss(torch.zeros(2), targets)


def test_categorical_value_loss_is_cross_entropy_against_mixtures() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, -1.0]])
    targets = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    logs = logits.log_softmax(-1)
    expected = torch.stack([-logs[0, 2], -(0.5 * logs[1, 0] + 0.5 * logs[1, 1])])
    torch.testing.assert_close(categorical_value_loss(logits, targets), expected)
    with pytest.raises(ValueError):
        categorical_value_loss(logits[:, :2], targets[:, :2])


def test_entropy_reads_only_legal_offers() -> None:
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    logs = policy_logs(torch.tensor([[0.0, 0.0, 9.0, -9.0], [3.0, 1.0, 0, 0]]), valid)
    torch.testing.assert_close(
        entropy(logs, valid), torch.tensor([torch.log(torch.tensor(2.0)), 0.0])
    )


def _health() -> NumericalHealth:
    return NumericalHealth(
        optimizer_steps=0, rejected_steps=0, skipped_steps=0, sampled_steps=0
    )


def test_standardized_advantages_take_statistics_from_the_selected_rows() -> None:
    advantages = torch.tensor([1.0, 3.0, 100.0, -50.0])
    everywhere = standardized_advantages(advantages)
    assert torch.allclose(everywhere.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(everywhere.std(unbiased=False), torch.tensor(1.0))

    selected = torch.tensor([0, 1])
    subset = standardized_advantages(advantages, selected)
    # Mean 2 and population std 1 over the selected rows rescale every row.
    assert torch.equal(subset, torch.tensor([-1.0, 1.0, 98.0, -52.0]))


def test_standardized_advantages_centre_a_constant_batch() -> None:
    advantages = torch.full((5,), 0.25)
    assert torch.equal(standardized_advantages(advantages), torch.zeros(5))


def test_gradient_step_clips_steps_and_records_when_observed() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    control = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loss = model(torch.full((1, 2), 10.0)).sum()
    health = _health()
    norm = gradient_step(
        loss, model, optimizer, 1.0, recorder=OptimizerHealth(health, 1)
    )
    control(torch.full((1, 2), 10.0)).sum().backward()
    expected = torch.nn.utils.get_total_norm([p.grad for p in control.parameters()])
    torch.testing.assert_close(norm, expected)
    # One SGD step of a gradient clipped to unit norm moves parameters by one.
    moved = torch.cat(
        [(a - b).flatten() for a, b in zip(model.parameters(), control.parameters())]
    )
    torch.testing.assert_close(moved.norm(), torch.tensor(1.0))
    assert health["optimizer_steps"] == 1 and health["sampled_steps"] == 1


class _Sections:
    """A recorder for a model with no critic: it only counts what it sees."""

    def __init__(self) -> None:
        self.sections: list[str] = []
        self.sampled_norms: list[float] = []
        self.steps = 0

    def track(self, section: str) -> AbstractContextManager[object]:
        self.sections.append(section)
        return nullcontext()

    def rejecting(self) -> AbstractContextManager[object]:
        return nullcontext()

    def sampling(self) -> bool:
        return not self.sampled_norms

    def before_clip(self, model: torch.nn.Module) -> None:
        pass

    def record_gradients(self, model: torch.nn.Module, norm_before: Tensor) -> None:
        self.sampled_norms.append(float(norm_before))

    def record_step(
        self, parameters: list[torch.nn.Parameter], before: list[Tensor]
    ) -> None:
        self.steps += 1


def test_gradient_step_serves_any_model_through_a_recorder() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters())
    recorder: StepRecorder = _Sections()
    for _ in range(2):
        loss = model(torch.randn(5, 3)).logsumexp(-1).mean()
        gradient_step(loss, model, optimizer, 0.5, recorder=recorder)
    assert isinstance(recorder, _Sections)
    assert recorder.steps == 2 and len(recorder.sampled_norms) == 1
    assert (
        recorder.sections
        == [
            "Backward and gradient clipping",
            "Optimizer step",
        ]
        * 2
    )


def test_gradient_step_refuses_detached_loss_without_stepping() -> None:
    model = torch.nn.Linear(1, 1)
    before = deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    with pytest.raises(NumericalError, match="detached_objective"):
        gradient_step(model(torch.ones(1, 1)).sum().detach(), model, optimizer, 1.0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_update_ema_moves_parameters_by_one_minus_rate() -> None:
    model = torch.nn.Linear(1, 1)
    averaged = deepcopy(model)
    with torch.no_grad():
        model.weight.add_(4.0)
    start = averaged.weight.detach().clone()
    update_ema(averaged, model, rate=0.75)
    torch.testing.assert_close(averaged.weight, start + 1.0)
    with pytest.raises(ValueError):
        update_ema(averaged, model, rate=1.0)


# Names published before the generic/player split, and where they live now.
_MOVED = [
    ("manabot.training.learning.objectives", "update_ema", "manabot.rl.step"),
    (
        "manabot.training.learning.damped_policy",
        "update_damped_iteration",
        "manabot.training.loops.damped_policy",
    ),
    (
        "manabot.training.learning.damped_policy",
        "damped_policy_loss",
        "manabot.training.loops.damped_policy",
    ),
    (
        "manabot.training.learning.damped_policy",
        "categorical_lambda_returns",
        "manabot.rl.returns",
    ),
    ("manabot.training.learning.returns", "transition_gae", "manabot.rl.returns"),
    ("manabot.training.learning.selection", "selection_mask", "manabot.rl.selection"),
    ("manabot.training.learning.selection", "selected_mean", "manabot.rl.selection"),
    ("manabot.model.policy_distribution", "NumericalError", "manabot.rl.numerics"),
    ("manabot.model.policy_distribution", "require", "manabot.rl.numerics"),
    ("manabot.model.policy_distribution", "clipped_surrogate", "manabot.rl.losses"),
    ("manabot.model.policy_distribution", "reverse_kl", "manabot.rl.losses"),
    ("manabot.model.numerics", "float64_operand", "manabot.rl.numerics"),
]


@pytest.mark.parametrize(("old", "name", "new"), _MOVED)
def test_moved_names_resolve_through_deprecated_aliases(
    old: str, name: str, new: str
) -> None:
    module = importlib.import_module(old)
    with pytest.warns(DeprecationWarning, match=rf"{new}\.{name}.*0\.3\.0"):
        alias = getattr(module, name)
    assert alias is getattr(importlib.import_module(new), name)
    with pytest.raises(AttributeError):
        getattr(module, "not_a_name")


def test_deprecated_update_iteration_dispatches_on_the_learning_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(objectives.ppo, "update_iteration", lambda *a, **k: {"rows": 1})
    monkeypatch.setattr(
        objectives, "update_damped_iteration", lambda *a, **k: {"rows": 2}
    )
    rng = np.random.default_rng(0)
    learner, batch = cast(Learner, None), cast(RolloutBatch, None)
    damped = DampedPolicyLearning(gradient="damped_policy")
    with pytest.warns(DeprecationWarning, match=r"loops\.ppo.*0\.3\.0"):
        assert objectives.update_iteration(learner, batch, Learning(), 0, rng) == {
            "rows": 1
        }
    with pytest.warns(DeprecationWarning, match=r"update_damped_iteration.*0\.3\.0"):
        assert objectives.update_iteration(learner, batch, damped, 0, rng) == {
            "rows": 2
        }
