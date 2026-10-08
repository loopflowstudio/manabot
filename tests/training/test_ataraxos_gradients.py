"""Independent checks of supplement S3.4 equations 5–6 and MTG boundaries.

The paper's move update uses a clipped likelihood ratio and two reverse KLs.
These analytic checks establish objective fidelity, not equilibrium or strength.
Return tests use the MTG adaptation of consecutive decisions by one learner.
Stored outcome order is loss/draw/win (a permutation of the paper prose order).
"""

from collections.abc import Iterator

import pytest
import torch

from manabot.training.ataraxos import (
    categorical_lambda_returns,
    damped_policy_loss,
)
from manabot.training.selection import selected_moves


@pytest.fixture(autouse=True)
def _one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "advantages", [[0.7, -0.3, 0.2], [-0.7, 0.3, -0.2], [0.0, 0.0, 0.0]]
)
def test_enumerated_mixed_policy_gradient(advantages: list[float]) -> None:
    """Integrate sampled-action gradients under the actual behavior distribution."""
    p = torch.tensor([0.65, 0.1, 0.25], dtype=torch.float64)
    behavior = torch.tensor([0.3, 0.4, 0.3], dtype=torch.float64)
    reference = torch.tensor([0.2, 0.2, 0.6], dtype=torch.float64)
    advantage = torch.tensor(advantages, dtype=torch.float64, requires_grad=True)
    logits = p.log().requires_grad_()
    saved_behavior = behavior.expand(3, -1).clone().requires_grad_()
    saved_reference = reference.expand(3, -1).clone().requires_grad_()
    losses = damped_policy_loss(
        logits.expand(3, -1),
        torch.arange(3),
        advantage,
        saved_behavior.log(),
        saved_reference,
        torch.ones((3, 3), dtype=torch.bool),
        clip=0.2,
        collection_kl=0.17,
        tau=0.23,
    )
    assert losses.shape == (3,)
    (losses * behavior).sum().backward()

    ratio = p / behavior
    active = ~(
        ((advantage.detach() > 0) & (ratio > 1.2))
        | ((advantage.detach() < 0) & (ratio < 0.8))
    )
    effective = advantage.detach() * active
    expected = p * (-effective + (p * effective).sum())
    for coefficient, target in [(0.17, behavior), (0.23, reference)]:
        log_ratio = (p / target).log()
        expected += coefficient * p * (log_ratio - (p * log_ratio).sum())
    torch.testing.assert_close(logits.grad, expected, rtol=1e-10, atol=1e-12)
    assert advantage.grad is None
    assert saved_behavior.grad is None
    assert saved_reference.grad is None


def test_padding_and_action_permutation_preserve_loss_and_gradient() -> None:
    logits = torch.tensor(
        [[0.3, -0.4, 900.0, 0.7]], dtype=torch.float64, requires_grad=True
    )
    valid = torch.tensor([[True, True, False, True]])
    behavior = torch.tensor([[0.2, 0.5, 0.0, 0.3]], dtype=torch.float64)
    reference = torch.tensor([[0.4, 0.2, 0.0, 0.4]], dtype=torch.float64)
    action = torch.tensor([3])
    advantages = torch.tensor([0.6], dtype=torch.float64)
    original = damped_policy_loss(
        logits,
        action,
        advantages,
        behavior.log(),
        reference,
        valid,
        clip=0.2,
        collection_kl=0.1,
        tau=0.05,
    )
    original.sum().backward()
    assert logits.grad is not None
    assert logits.grad[0, 2] == 0
    assert torch.isfinite(logits.grad).all()
    permutation = torch.tensor([3, 2, 0, 1])
    moved_logits = logits.detach()[:, permutation].requires_grad_()
    moved = damped_policy_loss(
        moved_logits,
        torch.tensor([0]),
        advantages,
        behavior[:, permutation].log(),
        reference[:, permutation],
        valid[:, permutation],
        clip=0.2,
        collection_kl=0.1,
        tau=0.05,
    )
    moved.sum().backward()
    torch.testing.assert_close(moved, original)
    torch.testing.assert_close(moved_logits.grad, logits.grad[:, permutation])


@pytest.mark.parametrize(
    "bad", [[0.0, 1.0], [0.2, 0.2], [-0.1, 1.1], [float("nan"), 0.5]]
)
@pytest.mark.parametrize("which", ["behavior", "reference"])
def test_invalid_distribution_rejected(bad: list[float], which: str) -> None:
    valid_distribution = torch.tensor([[0.4, 0.6]])
    invalid = torch.tensor([bad])
    with pytest.raises(ValueError):
        damped_policy_loss(
            torch.zeros((1, 2)),
            torch.tensor([0]),
            torch.ones(1),
            (invalid if which == "behavior" else valid_distribution).log(),
            invalid if which == "reference" else valid_distribution,
            torch.ones((1, 2), dtype=torch.bool),
            clip=0.2,
            collection_kl=0.1,
            tau=0.05,
        )


def test_mass_on_padding_rejected() -> None:
    with pytest.raises(ValueError):
        damped_policy_loss(
            torch.zeros((1, 2)),
            torch.tensor([0]),
            torch.ones(1),
            torch.tensor([[0.9, 0.1]]).log(),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, False]]),
            clip=0.2,
            collection_kl=0.1,
            tau=0.05,
        )


@pytest.mark.parametrize("trace_lambda", [0.0, 0.4, 1.0])
def test_categorical_terminal_reset_and_tail(trace_lambda: float) -> None:
    """A draw cuts the trace; the next episode cannot change its preceding row."""
    probabilities = torch.tensor(
        [
            [[0.2, 0.3, 0.5]],
            [[0.6, 0.1, 0.3]],
            [[0.1, 0.8, 0.1]],
            [[0.2, 0.2, 0.6]],
            [[0.4, 0.4, 0.2]],
        ],
        requires_grad=True,
    )
    bootstrap = torch.tensor([[0.7, 0.2, 0.1]], requires_grad=True)
    rewards = torch.tensor([[0.0], [0.0], [-1.0], [0.0], [0.0]])
    ends = torch.tensor([[False], [True], [True], [False], [False]])
    result = categorical_lambda_returns(
        probabilities, rewards, ends, bootstrap, trace_lambda
    )
    draw = torch.tensor([0.0, 1.0, 0.0])
    loss = torch.tensor([1.0, 0.0, 0.0])
    expected = torch.stack(
        [
            (1 - trace_lambda) * probabilities.detach()[1, 0] + trace_lambda * draw,
            draw,
            loss,
            (1 - trace_lambda) * probabilities.detach()[4, 0]
            + trace_lambda * bootstrap.detach()[0],
            bootstrap.detach()[0],
        ]
    )[:, None]
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.sum(-1), torch.ones((5, 1)))
    assert torch.all(result >= 0)
    assert not result.requires_grad


def test_terminal_outcomes_ignore_bootstrap_and_other_streams() -> None:
    probabilities = torch.full((1, 3, 3), 1 / 3)
    result = categorical_lambda_returns(
        probabilities,
        torch.tensor([[1.0, -1.0, 0.0]]),
        torch.ones((1, 3), dtype=torch.bool),
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        1.0,
    )
    torch.testing.assert_close(result[0], torch.eye(3)[torch.tensor([2, 0, 1])])


def test_quantile_filter_keeps_ties_without_advantage_normalization() -> None:
    advantages = torch.tensor([0.0, -0.2, 0.2, 0.2])
    assert selected_moves(advantages, 0.75, 0.01).tolist() == [False, True, True, True]
    assert not selected_moves(torch.zeros(4), 0.75, 0.01).any()
    assert selected_moves(advantages, 0.75, 0.3).sum() == 0
