"""Batched policy inference for rollouts: eval, datagen, and search.

Exp-07 (wave/search C7, goal 1). The historical net-in-loop path ran one
forward pass per decision at tiny batch (2.0k SPS at 16 envs, 97% of step
time in torch — reports/sps-closeout.md). This module makes batching the
default for every policy-in-the-loop consumer:

- ``BatchedSampler``: one masked forward pass for many observations at once,
  straight from the zero-copy numpy buffers the Rust envs write into.
- ``run_vector_games``: K parallel game streams on managym.VectorEnv
  (opponent_policy="none", both seats surfaced) with per-seat controllers —
  seat-balanced policy-vs-random and policy-vs-policy evaluation without a
  per-stream Python round trip.
- ``PolicyRolloutMCPlayer``: flat determinized MC search whose rollout
  actions come from a policy net, batched across all simultaneous rollouts
  of the decision via managym.RolloutPool (Env.rollout_pool).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from manabot.env import Env, Match, ObservationSpace
from manabot.infra.hypers import MatchHypers
from manabot.model.agent import Agent
from manabot.sim.flat_mc import (
    DEFAULT_MAX_PLAYOUT_STEPS,
    GameRecord,
    SearchStats,
)
import managym

OBS_KEYS: tuple[str, ...] = tuple(ObservationSpace().shapes.keys())


# -----------------------------------------------------------------------------
# Batched action selection
# -----------------------------------------------------------------------------


class BatchedSampler:
    """Select actions for a batch of encoded observations in one forward.

    ``epsilon`` mixes uniform-over-valid actions in per row (exploration /
    tie-breaking for search rollouts); ``temperature`` scales logits before
    sampling. ``deterministic`` takes the masked argmax instead of sampling.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.agent = agent.to(device)
        self.agent.eval()
        self.deterministic = deterministic
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self.forward_calls = 0
        self.obs_selected = 0

    def select(
        self, buffers: dict[str, np.ndarray], rows: np.ndarray
    ) -> np.ndarray:
        """Actions (len(rows),) for the given rows of the observation buffers."""

        obs = {
            key: torch.from_numpy(buffers[key][rows]).to(self.device)
            for key in OBS_KEYS
        }
        with torch.inference_mode():
            logits, _ = self.agent.forward(obs)
            if self.deterministic:
                actions = logits.argmax(dim=-1)
            else:
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
        actions = actions.cpu().numpy().astype(np.int64)

        if self.epsilon > 0.0:
            explore = self._rng.random(len(rows)) < self.epsilon
            if explore.any():
                valid = buffers["actions_valid"][rows[explore]] > 0
                # Vectorized uniform choice over each row's valid actions.
                weights = valid.astype(np.float64)
                weights /= weights.sum(axis=1, keepdims=True)
                cdf = np.cumsum(weights, axis=1)
                u = self._rng.random((len(cdf), 1))
                actions[explore] = (u < cdf).argmax(axis=1)

        self.forward_calls += 1
        self.obs_selected += len(rows)
        return actions


class RandomBatchController:
    """Uniform-over-valid batched controller (mirrors RandomMatchupPlayer)."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def select(
        self, buffers: dict[str, np.ndarray], rows: np.ndarray
    ) -> np.ndarray:
        valid = buffers["actions_valid"][rows] > 0
        weights = valid.astype(np.float64)
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        weights /= totals
        cdf = np.cumsum(weights, axis=1)
        u = self._rng.random((len(cdf), 1))
        return (u < cdf).argmax(axis=1).astype(np.int64)


# -----------------------------------------------------------------------------
# K-stream vectorized game driver (eval / head-to-head)
# -----------------------------------------------------------------------------


def _allocate_buffers(obs_space: ObservationSpace, n: int) -> dict[str, np.ndarray]:
    buffers = obs_space.encoder.allocate(n)
    buffers.update(
        rewards=np.zeros((n,), dtype=np.float64),
        terminated=np.zeros((n,), dtype=np.uint8),
        truncated=np.zeros((n,), dtype=np.uint8),
    )
    return buffers


def run_vector_games(
    hero: Any,
    villain: Any,
    *,
    num_games: int,
    num_streams: int = 64,
    seed: int = 0,
    deck: dict[str, int] | None = None,
    max_env_steps: int | None = None,
) -> tuple[list[GameRecord], dict[str, float]]:
    """Play seat-balanced games on K parallel streams with batched inference.

    ``hero`` / ``villain`` are batched controllers (BatchedSampler,
    RandomBatchController, or anything with the same ``select`` method).
    Stream ``s`` seats the hero at ``s % 2`` (seat 0 is on the play) for its
    whole life; per-seat game quotas are ``num_games/2`` each, so the result
    is seat-balanced regardless of relative game lengths.

    Mirror-deck only: both seats play ``deck`` (default INTERACTIVE_DECK).
    Returns (records, stats) where stats includes wall seconds and total
    env steps (decisions surfaced).
    """

    from manabot.verify.util import INTERACTIVE_DECK

    deck = dict(deck or INTERACTIVE_DECK)
    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(hero="hero", villain="villain", hero_deck=deck, villain_deck=deck)
    )
    num_streams = min(num_streams, max(2, num_games))
    env = managym.VectorEnv(
        num_envs=num_streams, seed=seed, skip_trivial=True, opponent_policy="none"
    )
    buffers = _allocate_buffers(obs_space, num_streams)
    env.set_buffers(buffers)
    env.reset_all_into_buffers(match.to_rust())

    hero_seat = np.arange(num_streams, dtype=np.int64) % 2
    quota = {0: (num_games + 1) // 2, 1: num_games // 2}
    recorded = {0: 0, 1: 0}
    steps_in_game = np.zeros(num_streams, dtype=np.int64)
    records: list[GameRecord] = []
    total_steps = 0
    game_index = 0
    max_env_steps = max_env_steps or 10_000_000

    start = time.perf_counter()
    while (recorded[0] < quota[0] or recorded[1] < quota[1]) and total_steps < max_env_steps:
        acting = np.asarray(env.current_agent_indices(), dtype=np.int64)
        actions = np.zeros(num_streams, dtype=np.int64)
        hero_rows = np.flatnonzero(acting == hero_seat)
        villain_rows = np.flatnonzero(acting != hero_seat)
        if len(hero_rows):
            actions[hero_rows] = hero.select(buffers, hero_rows)
        if len(villain_rows):
            actions[villain_rows] = villain.select(buffers, villain_rows)

        env.step_into_buffers(actions.tolist())
        total_steps += num_streams
        steps_in_game += 1

        done = (buffers["terminated"] > 0) | (buffers["truncated"] > 0)
        if done.any():
            infos = env.get_last_info()
            for s in np.flatnonzero(done):
                seat = int(hero_seat[s])
                if recorded[seat] < quota[seat]:
                    winner = infos[s].get("winner_index")
                    winner = int(winner) if winner is not None else None
                    records.append(
                        GameRecord(
                            game_index=game_index,
                            hero_seat=seat,
                            hero_won=winner == seat,
                            winner=winner,
                            steps=int(steps_in_game[s]),
                        )
                    )
                    recorded[seat] += 1
                    game_index += 1
                steps_in_game[s] = 0
    wall_seconds = time.perf_counter() - start

    stats = {
        "wall_seconds": wall_seconds,
        "total_env_steps": float(total_steps),
        "steps_per_second": total_steps / wall_seconds if wall_seconds > 0 else 0.0,
        "num_streams": float(num_streams),
    }
    return records, stats


# -----------------------------------------------------------------------------
# Policy rollouts inside flat determinized MC search
# -----------------------------------------------------------------------------


@dataclass
class PolicySearchStats(SearchStats):
    """SearchStats plus batched-inference accounting."""

    net_obs: int = 0
    net_forwards: int = 0
    rollout_steps: int = 0

    def to_dict(self) -> dict[str, float]:
        out = super().to_dict()
        out.update(
            net_obs=float(self.net_obs),
            net_forwards=float(self.net_forwards),
            rollout_steps=float(self.rollout_steps),
        )
        return out


class PolicyRolloutMCPlayer:
    """Flat determinized MC whose rollouts are played by a policy net.

    Same search shape as FlatMCPlayer (``sims`` = worlds x rollouts per legal
    action, worlds shared across actions), but instead of uniformly-random
    playouts, every simulation is stepped by ``sampler`` — one batched
    forward per rollout ply across all simultaneous simulations of the
    decision (managym.RolloutPool). The sampler plays *both* seats of the
    determinized worlds; epsilon-greedy random mixing lives in the sampler.
    """

    def __init__(
        self,
        sims: int,
        sampler: BatchedSampler,
        *,
        rollouts_per_world: int = 1,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        max_rollout_plies: int = 500,
        policy_plies: int | None = None,
        seed: int = 0,
    ):
        if sims < 1:
            raise ValueError("sims must be >= 1")
        self.sampler = sampler
        self.rollouts = max(1, min(rollouts_per_world, sims))
        self.worlds = max(1, sims // self.rollouts)
        self.sims = self.worlds * self.rollouts
        self.max_steps = max_steps
        self.max_rollout_plies = max_rollout_plies
        #: Hybrid rollouts: the policy plays the first `policy_plies` plies
        #: of every simulation (the part adjacent to the decision), then the
        #: engine finishes uniformly-random to terminal at ~0.2 ms/playout
        #: (RolloutPool.finish_random). None = policy to terminal.
        self.policy_plies = policy_plies
        self._seed = seed
        self._calls = 0
        self.stats = PolicySearchStats()
        self.last_scores: np.ndarray | None = None
        self._obs_space = ObservationSpace()
        self._capacity = (
            self._obs_space.encoder.max_actions * self.worlds * self.rollouts
        )
        self._buffers = _allocate_buffers(self._obs_space, self._capacity)

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs  # search reads the raw engine state, not the encoding
        self._calls += 1
        call_seed = (self._seed * 1_000_003 + self._calls) & 0xFFFFFFFFFFFFFFFF
        start = time.perf_counter()

        pool = env._engine.rollout_pool(
            self.worlds, self.rollouts, call_seed, self.max_steps
        )
        pool.set_buffers(self._buffers, self._capacity)
        num_slots = pool.num_slots
        actions_full = np.zeros(num_slots, dtype=np.int64)

        active = pool.encode_active()
        plies = 0
        while active:
            plies += 1
            if self.policy_plies is not None and plies > self.policy_plies:
                # Hybrid: hand the tails to the engine's random playout.
                pool.finish_random()
                break
            rows = np.asarray(active, dtype=np.int64)
            if plies > self.max_rollout_plies:
                # Safety valve: finish stragglers with random actions so a
                # pathological loop cannot pin the net indefinitely.
                controller: Any = _straggler_random
            else:
                controller = self.sampler.select
            actions_full[rows] = controller(self._buffers, rows)
            active = pool.step_active(actions_full.tolist())
            self.stats.rollout_steps += len(rows)

        scores, simulations, cap_hits = pool.scores()
        elapsed = time.perf_counter() - start
        self.stats.decisions += 1
        self.stats.seconds += elapsed
        self.stats.simulations += int(simulations)
        self.stats.cap_hits += int(cap_hits)
        self.stats.decision_seconds.append(elapsed)
        self.stats.net_obs = self.sampler.obs_selected
        self.stats.net_forwards = self.sampler.forward_calls
        self.last_scores = np.asarray(scores, dtype=np.float32)
        return int(np.argmax(scores))


_straggler_rng = np.random.default_rng(0xC7)


def _straggler_random(buffers: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    valid = buffers["actions_valid"][rows] > 0
    weights = valid.astype(np.float64)
    totals = weights.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    weights /= totals
    cdf = np.cumsum(weights, axis=1)
    u = _straggler_rng.random((len(cdf), 1))
    return (u < cdf).argmax(axis=1).astype(np.int64)
