"""Hand-calculated transition-end credit counterexamples from the audit."""

import pytest
import torch

from manabot.training.learning.damped_policy import categorical_lambda_returns
from manabot.training.learning.returns import (
    reverse_waves,
    terminal_distances,
    transition_gae,
)


def _row_by_row_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ends: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """The recursion as stated: one row of every column at a time."""
    advantages = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    following = next_value.reshape(-1)
    for step in reversed(range(len(rewards))):
        live = (~ends[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * live * following - values[step]
        tail = delta + gamma * lam * live * tail
        advantages[step] = tail
        following = values[step]
    return advantages


def _row_by_row_outcomes(
    probabilities: torch.Tensor,
    rewards: torch.Tensor,
    ends: torch.Tensor,
    bootstrap: torch.Tensor,
    trace_lambda: float,
) -> torch.Tensor:
    targets = torch.empty_like(probabilities)
    tail = bootstrap
    for step in reversed(range(len(rewards))):
        following = bootstrap if step == len(rewards) - 1 else probabilities[step + 1]
        continuation = (1 - trace_lambda) * following + trace_lambda * tail
        terminal = torch.nn.functional.one_hot((rewards[step] + 1).long(), 3).float()
        tail = torch.where(ends[step, :, None], terminal, continuation)
        targets[step] = tail
    return targets


def _row_by_row_distances(ends: torch.Tensor) -> torch.Tensor:
    distances = torch.full_like(ends, -1, dtype=torch.long)
    tail = distances[-1].clone()
    for step in reversed(range(len(ends))):
        tail = torch.where(ends[step], 0, torch.where(tail >= 0, tail + 1, -1))
        distances[step] = tail
    return distances


# Packed complete games, rectangular unfinished tails, single-row trajectories.
TRAJECTORIES = [(257, 1, 0.05), (64, 5, 0.1), (40, 3, 0.9), (1, 4, 0.5), (30, 2, 0.0)]


class TestTrajectoryParallelism:
    """Trajectories advanced together must equal the row-by-row recursion exactly."""

    @pytest.mark.parametrize("rows, columns, cut", TRAJECTORIES)
    def test_every_row_appears_once_after_the_row_it_depends_on(
        self, rows: int, columns: int, cut: float
    ) -> None:
        ends = (
            torch.rand(rows, columns, generator=torch.Generator().manual_seed(1)) < cut
        )
        order = torch.full((rows, columns), -1)
        for wave, (steps, cols) in enumerate(reverse_waves(ends)):
            assert (order[steps, cols] == -1).all()
            order[steps, cols] = wave
        assert (order >= 0).all()
        for step in range(rows - 1):
            continues = ~ends[step]
            assert (order[step][continues] == order[step + 1][continues] + 1).all()
        assert (order[ends] == 0).all()

    @pytest.mark.parametrize("rows, columns, cut", TRAJECTORIES)
    @pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
    def test_gae_is_bit_identical_to_the_row_recursion(
        self, rows: int, columns: int, cut: float, lam: float
    ) -> None:
        generator = torch.Generator().manual_seed(rows + columns)
        rewards = torch.randn(rows, columns, generator=generator)
        values = torch.randn(rows, columns, generator=generator)
        ends = torch.rand(rows, columns, generator=generator) < cut
        following = torch.randn(columns, generator=generator)
        advantages, returns = transition_gae(
            rewards, values, ends, following, 0.99, lam
        )
        expected = _row_by_row_gae(rewards, values, ends, following, 0.99, lam)
        assert torch.equal(advantages, expected)
        assert torch.equal(returns, expected + values)

    @pytest.mark.parametrize("rows, columns, cut", TRAJECTORIES)
    def test_outcome_targets_are_bit_identical_to_the_row_recursion(
        self, rows: int, columns: int, cut: float
    ) -> None:
        generator = torch.Generator().manual_seed(rows * columns)
        probabilities = torch.rand(rows, columns, 3, generator=generator).softmax(-1)
        bootstrap = torch.rand(columns, 3, generator=generator).softmax(-1)
        ends = torch.rand(rows, columns, generator=generator) < cut
        signs = torch.randint(-1, 2, (rows, columns), generator=generator).float()
        rewards = torch.where(ends, signs, 0)
        assert torch.equal(
            categorical_lambda_returns(probabilities, rewards, ends, bootstrap, 0.8),
            _row_by_row_outcomes(probabilities, rewards, ends, bootstrap, 0.8),
        )

    @pytest.mark.parametrize("rows, columns, cut", TRAJECTORIES)
    def test_terminal_distances_match_the_row_recursion(
        self, rows: int, columns: int, cut: float
    ) -> None:
        ends = (
            torch.rand(rows, columns, generator=torch.Generator().manual_seed(2)) < cut
        )
        assert torch.equal(terminal_distances(ends), _row_by_row_distances(ends))


class TestGAE:
    def test_transition_end_credit_does_not_cross_reset(self) -> None:
        # Two two-decision games: win, then loss. Values deliberately differ
        # across the reset; terminal rows must never borrow the next game's V.
        rewards = torch.tensor([[0.0], [1.0], [0.0], [-1.0]])
        values = torch.tensor([[0.2], [0.4], [-0.3], [0.6]])
        ends = torch.tensor([[False], [True], [False], [True]])
        advantages, returns = transition_gae(
            rewards, values, ends, torch.tensor([99.0]), 1.0, 0.5
        )
        # R0 = .5*.4 + .5*1 = .7; R2 = .5*.6 + .5*(-1) = -.2.
        torch.testing.assert_close(returns[:, 0], torch.tensor([0.7, 1, -0.2, -1]))
        torch.testing.assert_close(
            advantages[:, 0], torch.tensor([0.5, 0.6, 0.1, -1.6])
        )

    def test_nonterminal_tail_bootstraps_only_its_own_game(self) -> None:
        rewards = torch.tensor([[1.0], [0.0], [0.0]])
        values = torch.tensor([[0.2], [0.4], [-0.3]])
        ends = torch.tensor([[True], [False], [False]])
        _, returns = transition_gae(
            rewards, values, ends, torch.tensor([0.8]), 0.9, 0.5
        )
        # Terminal win; then R2=.9*.8, R1=.9*(.5*(-.3)+.5*.72).
        torch.testing.assert_close(returns[:, 0], torch.tensor([1.0, 0.189, 0.72]))

    def test_gae_matches_reference_simple_case(self) -> None:
        rewards = torch.ones((4, 2), dtype=torch.float32)
        values = torch.zeros((4, 2), dtype=torch.float32)
        dones = torch.zeros((4, 2), dtype=torch.bool)
        next_value = torch.zeros(2, dtype=torch.float32)

        advantages, returns = transition_gae(
            rewards,
            values,
            dones,
            next_value,
            gamma=1.0,
            lam=1.0,
        )

        expected = torch.tensor(
            [[4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(advantages, expected)
        assert torch.allclose(returns, expected)
