"""Tests for batched policy rollouts (exp-07 / wave C7).

Covers the RolloutPool bindings (slot layout, batched stepping, scoring),
the batched sampler (valid-action guarantees), the K-stream vector driver
(seat balancing).
"""

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.sim.rollout import (
    BatchedSampler,
    PolicyRolloutMCPlayer,
    RandomBatchController,
    run_vector_games,
)
from manabot.sim.search.flat_mc import aggregate_records
from manabot.verify.games import INTERACTIVE_DECK
import managym


def make_env(seed: int = 3) -> tuple[Env, managym.Observation]:
    obs_space = ObservationSpace()
    match = Match(
        MatchConfig(
            hero="a",
            villain="b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(match, obs_space, Reward(RewardConfig()), seed=seed)
    obs, _ = env.reset(seed=seed)
    return env, obs


class TestRolloutPool:
    def test_slot_layout_and_random_drain(self):
        env, _ = make_env()
        num_actions = env._engine.action_count()
        pool = env._engine.rollout_pool(2, 3, 42, 2000)
        assert pool.num_actions == num_actions
        assert pool.num_slots == 2 * 3 * num_actions

        pool.set_observation_layout(ObservationSpace().layout)

        rng = np.random.default_rng(0)
        active, batch = pool.encode_active_batch()
        plies = 0
        while active:
            plies += 1
            assert plies < 5000, "rollouts failed to drain"
            assert batch.batch_size == len(active)
            counts = np.asarray(batch.action_counts)
            assert (counts > 0).all(), "active slot with no valid action"
            actions = np.zeros(pool.num_slots, dtype=np.int64)
            for slot, count in zip(active, counts, strict=True):
                actions[slot] = rng.integers(count)
            active, batch = pool.step_active_batch(actions.tolist())

        scores, simulations, cap_hits = pool.scores()
        assert len(scores) == num_actions
        assert simulations == pool.num_slots
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert cap_hits == 0

    def test_pool_does_not_mutate_source_env(self):
        env, _ = make_env(seed=9)
        before = env._engine.action_count()
        pool = env._engine.rollout_pool(2, 2, 11, 2000)
        pool.set_observation_layout(ObservationSpace().layout)
        active, _ = pool.encode_active_batch()
        guard = 0
        while active and guard < 3000:
            guard += 1
            actions = np.zeros(pool.num_slots, dtype=np.int64)
            active, _ = pool.step_active_batch(actions.tolist())
        assert env._engine.action_count() == before
        assert not env._engine.is_game_over()


class TestBatchedSampler:
    def _obs_batch(self, batch: int = 8) -> managym.ObservationBatch:
        return managym.encode_observation_batch(
            [make_env(seed=20 + row)[1] for row in range(batch)],
            ObservationSpace().layout,
        )

    def test_selects_valid_actions_only(self):
        agent = ActorCritic(ObservationSpace(), ModelConfig())
        buffers = self._obs_batch()
        rows = np.arange(8)
        for kwargs in (
            {"deterministic": True},
            {"deterministic": False},
            {"deterministic": False, "epsilon": 1.0},
            {"deterministic": False, "temperature": 0.3},
        ):
            sampler = BatchedSampler(agent, seed=2, **kwargs)
            actions = sampler.select(buffers, rows)
            assert actions.shape == (8,)
            assert ((actions >= 0) & (actions < buffers.action_counts)).all()

    @pytest.mark.parametrize(
        "temperature,deterministic", [(1.0, False), (0.3, False), (1.0, True)]
    )
    def test_backend_sampling_preserves_actions_and_rng(
        self, temperature: float, deterministic: bool
    ) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(17)
        agent = ActorCritic(ObservationSpace(), ModelConfig(hidden_dim=8))
        sampler = BatchedSampler(
            agent, temperature=temperature, deterministic=deterministic
        )
        buffers = self._obs_batch(3)
        rows = np.arange(3)
        prepared = on_device(
            agent.prepare(
                buffers,
                inference=True,
            ),
            "cpu",
        )
        state = torch.random.get_rng_state()
        with torch.inference_mode():
            logits, _ = agent(prepared)
            expected = (
                logits.argmax(-1)
                if deterministic
                else torch.multinomial(
                    torch.softmax(logits / temperature, dim=-1), 1
                ).squeeze(-1)
            )
        expected_state = torch.random.get_rng_state()
        torch.random.set_rng_state(state)
        actual = sampler.select(buffers, rows)
        np.testing.assert_array_equal(actual, expected.numpy())
        assert torch.equal(torch.random.get_rng_state(), expected_state)

    def test_random_controller_valid(self):
        buffers = self._obs_batch()
        rows = np.arange(8)
        controller = RandomBatchController(seed=3)
        actions = controller.select(buffers, rows)
        assert ((actions >= 0) & (actions < buffers.action_counts)).all()


class TestVectorDriver:
    def test_only_unfinished_seat_quota_keeps_playing(self) -> None:
        records, stats = run_vector_games(
            RandomBatchController(1),
            RandomBatchController(2),
            num_games=3,
            num_streams=2,
            deck={"Forest": 8},
            max_env_steps=100,
        )
        assert sorted(game.hero_seat for game in records) == [0, 0, 1]
        assert all(game.winner in (0, 1) for game in records)
        assert stats["total_env_steps"] == sum(game.steps for game in records)

    def test_seat_balanced_records(self):
        torch.manual_seed(0)
        agent = ActorCritic(ObservationSpace(), ModelConfig())
        sampler = BatchedSampler(agent, seed=1)
        records, stats = run_vector_games(
            sampler,
            RandomBatchController(seed=2),
            num_games=8,
            num_streams=4,
            seed=3,
        )
        metrics = aggregate_records(records)
        assert metrics["num_games"] == 8
        assert metrics["games_on_play"] == 4
        assert metrics["games_on_draw"] == 4
        assert stats["total_env_steps"] > 0


class TestPolicyRolloutPlayer:
    def test_plays_and_scores(self):
        torch.manual_seed(0)
        env, obs = make_env(seed=13)
        agent = ActorCritic(ObservationSpace(), ModelConfig())
        sampler = BatchedSampler(agent, epsilon=0.2, seed=5)
        player = PolicyRolloutMCPlayer(4, sampler, seed=17)
        action = player.act(env, obs)
        assert 0 <= action < env._engine.action_count()
        assert player.stats.decisions == 1
        assert player.stats.simulations == player.sims * env._engine.action_count()
        assert player.last_scores is not None
        assert player.last_scores.min() >= 0.0
        assert player.last_scores.max() <= 1.0
        # The chosen action must be playable in the real env.
        env.step(action)
