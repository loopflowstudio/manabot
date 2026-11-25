"""
test_potential_shaping.py

Tests for potential-based reward shaping (Ng, Harada & Russell 1999):
shaping of the form gamma * Phi(s') - Phi(s) with Phi(terminal) = 0.

Covers:
- sign conventions of the hero-perspective potential Phi,
- terminal handling (Phi(terminal) treated as 0),
- the telescoping property: with gamma = 1 the total shaped episode return
  equals the unshaped episode return plus -Phi(s_0), and Phi(s_0) = 0 for a
  symmetric start.
"""

from collections.abc import Sequence
from types import SimpleNamespace

from manabot.env.match import Reward
from manabot.infra.config import RewardConfig
import managym

BATTLEFIELD = int(managym.ZoneEnum.BATTLEFIELD)
HAND = int(managym.ZoneEnum.HAND)


def _stub_card(
    *, zone: int, is_land: bool = False, is_creature: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        zone=zone,
        card_types=SimpleNamespace(is_land=is_land, is_creature=is_creature),
    )


def _stub_obs(
    *,
    agent_life: int = 20,
    opponent_life: int = 20,
    agent_cards: Sequence[SimpleNamespace] = (),
    opponent_cards: Sequence[SimpleNamespace] = (),
    game_over: bool = False,
    won: bool = False,
) -> SimpleNamespace:
    # Battlefield fixtures expose characteristics on their permanent.
    def permanents(cards: Sequence[SimpleNamespace]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(characteristics=card)
            for card in cards
            if card.zone == BATTLEFIELD
        ]

    return SimpleNamespace(
        agent=SimpleNamespace(life=agent_life),
        opponent=SimpleNamespace(life=opponent_life),
        agent_cards=[card for card in agent_cards if card.zone != BATTLEFIELD],
        opponent_cards=[card for card in opponent_cards if card.zone != BATTLEFIELD],
        agent_permanents=permanents(agent_cards),
        opponent_permanents=permanents(opponent_cards),
        game_over=game_over,
        won=won,
    )


class TestSingleEnvRewardPotential:
    def _config(self, gamma: float = 1.0) -> RewardConfig:
        return RewardConfig(
            potential_enabled=True,
            potential_gamma=gamma,
            potential_land_weight=0.03,
            potential_creature_weight=0.06,
            potential_life_weight=0.2,
        )

    def test_potential_sign_conventions(self):
        reward = Reward(self._config())
        symmetric = _stub_obs()
        assert reward._potential(symmetric) == 0.0

        hero_land = _stub_obs(agent_cards=[_stub_card(zone=BATTLEFIELD, is_land=True)])
        assert abs(reward._potential(hero_land) - 0.03) < 1e-9

        villain_board = _stub_obs(
            opponent_cards=[
                _stub_card(zone=BATTLEFIELD, is_land=True),
                _stub_card(zone=BATTLEFIELD, is_creature=True),
            ]
        )
        assert abs(reward._potential(villain_board) - (-0.03 - 0.06)) < 1e-9

        # Hand cards do not count toward the board potential.
        hand_only = _stub_obs(agent_cards=[_stub_card(zone=HAND, is_land=True)])
        assert reward._potential(hand_only) == 0.0

        life_lead = _stub_obs(agent_life=20, opponent_life=10)
        assert abs(reward._potential(life_lead) - 0.2 * 0.5) < 1e-9

    def test_step_reward_is_gamma_phi_next_minus_phi_prev(self):
        reward = Reward(self._config(gamma=0.99))
        s0 = _stub_obs()
        s1 = _stub_obs(agent_cards=[_stub_card(zone=BATTLEFIELD, is_land=True)])
        value = reward.compute(0.0, s0, s1)
        assert abs(value - (0.99 * 0.03 - 0.0)) < 1e-9

    def test_terminal_phi_is_zero_and_episode_telescopes(self):
        """gamma = 1 telescoping: total shaped return over an episode equals
        the terminal return plus -Phi(s_0) (= terminal return here)."""

        reward = Reward(self._config(gamma=1.0))
        s0 = _stub_obs()
        s1 = _stub_obs(agent_cards=[_stub_card(zone=BATTLEFIELD, is_land=True)])
        s2 = _stub_obs(
            agent_cards=[
                _stub_card(zone=BATTLEFIELD, is_land=True),
                _stub_card(zone=BATTLEFIELD, is_creature=True),
            ],
            opponent_life=17,
        )
        s3 = _stub_obs(game_over=True, won=True)

        total = (
            reward.compute(0.0, s0, s1)
            + reward.compute(0.0, s1, s2)
            + reward.compute(0.0, s2, s3)
        )
        unshaped = Reward(RewardConfig())
        unshaped_total = (
            unshaped.compute(0.0, s0, s1)
            + unshaped.compute(0.0, s1, s2)
            + unshaped.compute(0.0, s2, s3)
        )
        # Phi(s0) = 0 for the symmetric start, so the sums must match exactly.
        assert abs(total - unshaped_total) < 1e-9
        assert abs(unshaped_total - 1.0) < 1e-9

        # The terminal step itself carries -Phi(s2), not Phi(terminal board).
        terminal_value = reward.compute(0.0, s2, s3)
        phi_s2 = reward._potential(s2)
        assert abs(terminal_value - (1.0 - phi_s2)) < 1e-9

    def test_disabled_by_default(self):
        reward = Reward(RewardConfig())
        s0 = _stub_obs()
        s1 = _stub_obs(agent_cards=[_stub_card(zone=BATTLEFIELD, is_land=True)])
        assert reward.compute(0.0, s0, s1) == 0.0
