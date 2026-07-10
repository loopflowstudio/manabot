"""Tests for batched policy rollouts (exp-07 / wave C7).

Covers the RolloutPool bindings (slot layout, batched stepping, scoring),
the batched sampler (valid-action guarantees), the K-stream vector driver
(seat balancing), and soft distillation targets.
"""

import numpy as np
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.sim.distill import soft_targets_from_scores
from manabot.sim.flat_mc import aggregate_records
from manabot.sim.rollout import (
    BatchedSampler,
    PolicyRolloutMCPlayer,
    RandomBatchController,
    _allocate_buffers,
    run_vector_games,
)
from manabot.verify.util import INTERACTIVE_DECK


def make_env(seed: int = 3) -> tuple[Env, dict]:
    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero="a",
            villain="b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(match, obs_space, Reward(RewardHypers()), seed=seed, auto_reset=False)
    obs, _ = env.reset(seed=seed)
    return env, obs


class TestRolloutPool:
    def test_slot_layout_and_random_drain(self):
        env, _ = make_env()
        num_actions = env._engine.action_count()
        pool = env._engine.rollout_pool(2, 3, 42, 2000)
        assert pool.num_actions == num_actions
        assert pool.num_slots == 2 * 3 * num_actions

        obs_space = ObservationSpace()
        buffers = _allocate_buffers(obs_space, pool.num_slots)
        pool.set_buffers(buffers, pool.num_slots)

        rng = np.random.default_rng(0)
        active = pool.encode_active()
        plies = 0
        while active:
            plies += 1
            assert plies < 5000, "rollouts failed to drain"
            rows = np.asarray(active)
            valid = buffers["actions_valid"][rows] > 0
            assert valid.any(axis=1).all(), "active slot with no valid action"
            actions = np.zeros(pool.num_slots, dtype=np.int64)
            for r, v in zip(rows, valid):
                actions[r] = rng.choice(np.flatnonzero(v))
            active = pool.step_active(actions.tolist())

        scores, simulations, cap_hits = pool.scores()
        assert len(scores) == num_actions
        assert simulations == pool.num_slots
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert cap_hits == 0

    def test_buffer_capacity_can_exceed_slots(self):
        env, _ = make_env(seed=5)
        pool = env._engine.rollout_pool(1, 1, 7, 2000)
        capacity = pool.num_slots + 8
        buffers = _allocate_buffers(ObservationSpace(), capacity)
        pool.set_buffers(buffers, capacity)
        assert pool.encode_active() is not None

    def test_pool_does_not_mutate_source_env(self):
        env, _ = make_env(seed=9)
        before = env._engine.action_count()
        pool = env._engine.rollout_pool(2, 2, 11, 2000)
        buffers = _allocate_buffers(ObservationSpace(), pool.num_slots)
        pool.set_buffers(buffers, pool.num_slots)
        active = pool.encode_active()
        guard = 0
        while active and guard < 3000:
            guard += 1
            actions = np.zeros(pool.num_slots, dtype=np.int64)
            active = pool.step_active(actions.tolist())
        assert env._engine.action_count() == before
        assert not env._engine.is_game_over()


class TestBatchedSampler:
    def _obs_batch(self, batch: int = 8):
        obs_space = ObservationSpace()
        buffers = _allocate_buffers(obs_space, batch)
        rng = np.random.default_rng(1)
        for key, arr in buffers.items():
            if key in ("rewards", "terminated", "truncated"):
                continue
            if arr.dtype == np.float32:
                arr[:] = rng.random(arr.shape, dtype=np.float32)
        valid = np.zeros_like(buffers["actions_valid"])
        valid[:, :5] = (rng.random((batch, 5)) < 0.6).astype(np.float32)
        valid[np.arange(batch), rng.integers(0, 5, batch)] = 1.0
        buffers["actions_valid"][:] = valid
        buffers["actions"][..., -1] = valid
        buffers["action_focus"][:] = -1
        return buffers

    def test_selects_valid_actions_only(self):
        agent = Agent(ObservationSpace(), AgentHypers())
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
            chosen_valid = buffers["actions_valid"][rows, actions]
            assert (chosen_valid > 0).all(), f"invalid action for {kwargs}"

    def test_random_controller_valid(self):
        buffers = self._obs_batch()
        rows = np.arange(8)
        controller = RandomBatchController(seed=3)
        actions = controller.select(buffers, rows)
        assert (buffers["actions_valid"][rows, actions] > 0).all()


class TestVectorDriver:
    def test_seat_balanced_records(self):
        torch.manual_seed(0)
        agent = Agent(ObservationSpace(), AgentHypers())
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
        agent = Agent(ObservationSpace(), AgentHypers())
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


class TestSoftTargets:
    def test_padding_masked_and_temperature_sharpens(self):
        scores = torch.tensor([[0.5, 0.6, -1.0, -1.0]])
        sharp = soft_targets_from_scores(scores, 0.02)
        flat = soft_targets_from_scores(scores, 1.0)
        for probs in (sharp, flat):
            assert probs[0, 2:].sum().item() < 1e-6
            assert abs(probs.sum().item() - 1.0) < 1e-5
        assert sharp[0, 1] > flat[0, 1] > 0.5
