"""
test_potential_shaping.py

Tests for potential-based reward shaping (Ng, Harada & Russell 1999):
shaping of the form gamma * Phi(s') - Phi(s) with Phi(terminal) = 0.

Covers:
- sign conventions of the hero-perspective potential Phi (both reward paths),
- terminal handling (Phi(terminal) treated as 0),
- the telescoping property: with gamma = 1 the total shaped episode return
  equals the unshaped episode return plus -Phi(s_0), and Phi(s_0) = 0 for a
  symmetric start.
"""

from types import SimpleNamespace

import torch

import managym
from manabot.env import Reward, VectorEnv
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.hypers import RewardHypers

BATTLEFIELD = int(managym.ZoneEnum.BATTLEFIELD)
HAND = int(managym.ZoneEnum.HAND)


def make_vector_env(reward_hypers: RewardHypers, seed: int = 0) -> VectorEnv:
    return VectorEnv(
        num_envs=2,
        match=Match(),
        observation_space=ObservationSpace(),
        reward=Reward(reward_hypers),
        device="cpu",
        seed=seed,
        opponent_policy="random",
    )


def first_valid_actions(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Deterministic policy: first valid action index per env."""
    return obs["actions_valid"].float().argmax(dim=1).long()


# -----------------------------------------------------------------------------
# VectorEnv path
# -----------------------------------------------------------------------------
class TestVectorEnvPotential:
    def _blank_tensors(self, env: VectorEnv):
        encoder = env.observation_space.encoder
        cards = torch.zeros(
            (env.num_envs, encoder.cards_per_player, encoder.card_dim)
        )
        players = torch.zeros((env.num_envs, 1, encoder.player_dim))
        return cards, players

    def _add_card(
        self,
        cards: torch.Tensor,
        env_index: int,
        slot: int,
        *,
        zone: int,
        is_land: bool = False,
        is_creature: bool = False,
        env_ref: VectorEnv | None = None,
    ) -> None:
        assert env_ref is not None
        cards[env_index, slot, zone] = 1.0
        if is_land:
            cards[env_index, slot, env_ref._card_land_index] = 1.0
        if is_creature:
            cards[env_index, slot, env_ref._card_creature_index] = 1.0
        cards[env_index, slot, -1] = 1.0  # validity

    def test_potential_sign_conventions(self):
        hypers = RewardHypers(
            potential_enabled=True,
            potential_land_weight=0.03,
            potential_creature_weight=0.06,
            potential_life_weight=0.2,
        )
        env = make_vector_env(hypers)
        env.reset()

        agent_cards, agent_player = self._blank_tensors(env)
        opponent_cards, opponent_player = self._blank_tensors(env)

        # Equal life (20/20 encoded as 1.0), empty boards -> Phi = 0.
        agent_player[:, 0, 0] = 1.0
        opponent_player[:, 0, 0] = 1.0
        phi = env._potential(
            agent_cards, opponent_cards, agent_player, opponent_player
        )
        assert torch.allclose(phi, torch.zeros_like(phi))

        # Hero battlefield land -> +land_weight.
        self._add_card(
            agent_cards, 0, 0, zone=BATTLEFIELD, is_land=True, env_ref=env
        )
        phi = env._potential(
            agent_cards, opponent_cards, agent_player, opponent_player
        )
        assert abs(phi[0].item() - 0.03) < 1e-6
        assert phi[1].item() == 0.0

        # Villain battlefield creature -> -creature_weight (subtracts).
        self._add_card(
            opponent_cards, 0, 0, zone=BATTLEFIELD, is_creature=True, env_ref=env
        )
        phi = env._potential(
            agent_cards, opponent_cards, agent_player, opponent_player
        )
        assert abs(phi[0].item() - (0.03 - 0.06)) < 1e-6

        # Cards outside the battlefield do not count.
        self._add_card(
            agent_cards, 1, 1, zone=HAND, is_creature=True, env_ref=env
        )
        phi_after = env._potential(
            agent_cards, opponent_cards, agent_player, opponent_player
        )
        assert torch.allclose(phi, phi_after)

        # Hero life advantage -> positive life term: hero 20, villain 10
        # gives potential_life_weight * (20 - 10)/20 = 0.2 * 0.5 = 0.1.
        opponent_player[1, 0, 0] = 0.5
        phi = env._potential(
            agent_cards, opponent_cards, agent_player, opponent_player
        )
        assert abs(phi[1].item() - 0.1) < 1e-6

    def test_disabled_by_default(self):
        env = make_vector_env(RewardHypers())
        env.reset()
        assert env._compute_potential_shaping() is None

    def test_telescoping_episode_returns_match_unshaped(self):
        """With gamma = 1 and Phi(terminal) = 0, shaping telescopes to
        -Phi(s_0), which is 0 for the symmetric start — so per-episode
        shaped and unshaped returns must be identical."""

        base = RewardHypers()
        shaped = RewardHypers(potential_enabled=True, potential_gamma=1.0)

        episode_sums: dict[str, list[float]] = {"base": [], "shaped": []}
        step_rewards: dict[str, list[torch.Tensor]] = {"base": [], "shaped": []}
        dones: dict[str, list[torch.Tensor]] = {"base": [], "shaped": []}

        for name, hypers in (("base", base), ("shaped", shaped)):
            env = make_vector_env(hypers, seed=123)
            obs, _ = env.reset()
            running = torch.zeros(env.num_envs)
            for _ in range(3000):
                actions = first_valid_actions(obs)
                obs, reward, terminated, truncated, _ = env.step(actions)
                step_rewards[name].append(reward.clone())
                done = terminated | truncated
                dones[name].append(done.clone())
                running += reward
                for i in range(env.num_envs):
                    if done[i]:
                        episode_sums[name].append(float(running[i]))
                        running[i] = 0.0
                if len(episode_sums[name]) >= 4:
                    break
            env.close()

        assert len(episode_sums["base"]) >= 4
        assert len(episode_sums["shaped"]) >= 4

        # Identical seeds + deterministic policy -> identical trajectories.
        n = min(len(dones["base"]), len(dones["shaped"]))
        for t in range(n):
            assert torch.equal(dones["base"][t], dones["shaped"][t]), (
                "trajectories diverged; telescoping comparison invalid"
            )

        # Per-step rewards differ (dense signal is present)...
        deltas = [
            (step_rewards["shaped"][t] - step_rewards["base"][t]).abs().max()
            for t in range(n)
        ]
        assert max(float(d) for d in deltas) > 1e-4

        # ...but per-episode sums are identical: the potential telescopes out.
        k = min(len(episode_sums["base"]), len(episode_sums["shaped"]))
        for a, b in zip(episode_sums["base"][:k], episode_sums["shaped"][:k]):
            assert abs(a - b) < 1e-3, (a, b)

    def test_terminal_step_carries_minus_prev_phi(self):
        """On a done step the shaping must be exactly -Phi(s_prev): the
        buffers hold the next episode's reset obs, and Phi(terminal) = 0."""

        env = make_vector_env(
            RewardHypers(potential_enabled=True, potential_gamma=1.0), seed=7
        )
        obs, _ = env.reset()
        for _ in range(3000):
            actions = first_valid_actions(obs)
            prev_obs = {
                key: env._obs_tensors[key].clone() for key in env._prev_obs_keys
            }
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if bool(done.any()):
                prev_phi = env._potential(
                    prev_obs["agent_cards"],
                    prev_obs["opponent_cards"],
                    prev_obs["agent_player"],
                    prev_obs["opponent_player"],
                )
                shaping = env._compute_potential_shaping()
                for i in range(env.num_envs):
                    if done[i]:
                        assert abs(
                            float(shaping[i]) - (-float(prev_phi[i]))
                        ) < 1e-6
                        # Terminal reward is win/lose plus that correction.
                        assert (
                            abs(abs(float(reward[i]) + float(prev_phi[i])) - 1.0)
                            < 1e-6
                        )
                env.close()
                return
        raise AssertionError("no episode terminated within the step budget")


# -----------------------------------------------------------------------------
# Single-env Match.Reward path
# -----------------------------------------------------------------------------
def _stub_card(*, zone: int, is_land: bool = False, is_creature: bool = False):
    return SimpleNamespace(
        zone=zone,
        card_types=SimpleNamespace(is_land=is_land, is_creature=is_creature),
    )


def _stub_obs(
    *,
    agent_life: int = 20,
    opponent_life: int = 20,
    agent_cards=(),
    opponent_cards=(),
    game_over: bool = False,
    won: bool = False,
):
    return SimpleNamespace(
        agent=SimpleNamespace(life=agent_life),
        opponent=SimpleNamespace(life=opponent_life),
        agent_cards=list(agent_cards),
        opponent_cards=list(opponent_cards),
        game_over=game_over,
        won=won,
    )


class TestSingleEnvRewardPotential:
    def _hypers(self, gamma: float = 1.0) -> RewardHypers:
        return RewardHypers(
            potential_enabled=True,
            potential_gamma=gamma,
            potential_land_weight=0.03,
            potential_creature_weight=0.06,
            potential_life_weight=0.2,
        )

    def test_potential_sign_conventions(self):
        reward = Reward(self._hypers())
        symmetric = _stub_obs()
        assert reward._potential(symmetric) == 0.0

        hero_land = _stub_obs(
            agent_cards=[_stub_card(zone=BATTLEFIELD, is_land=True)]
        )
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
        reward = Reward(self._hypers(gamma=0.99))
        s0 = _stub_obs()
        s1 = _stub_obs(agent_cards=[_stub_card(zone=BATTLEFIELD, is_land=True)])
        value = reward.compute(0.0, s0, s1)
        assert abs(value - (0.99 * 0.03 - 0.0)) < 1e-9

    def test_terminal_phi_is_zero_and_episode_telescopes(self):
        """gamma = 1 telescoping: total shaped return over an episode equals
        the terminal return plus -Phi(s_0) (= terminal return here)."""

        reward = Reward(self._hypers(gamma=1.0))
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
        unshaped = Reward(RewardHypers())
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
        reward = Reward(RewardHypers())
        s0 = _stub_obs()
        s1 = _stub_obs(agent_cards=[_stub_card(zone=BATTLEFIELD, is_land=True)])
        assert reward.compute(0.0, s0, s1) == 0.0
