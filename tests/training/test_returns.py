"""Hand-calculated transition-end credit counterexamples from the audit."""

import torch

from manabot.training.learning.returns import transition_gae


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
