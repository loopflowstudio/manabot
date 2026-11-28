"""Batched policy inference for rollouts: evaluation and search.

Exp-07 (goal 1). The historical net-in-loop path ran one
forward pass per decision at tiny batch (2.0k SPS at 16 envs, 97% of step
time in torch — experiments/reports/sps-closeout.md). This module makes batching the
default for every policy-in-the-loop consumer:

- ``BatchedSampler``: one masked forward pass for many observations at once,
  directly from native `ObservationBatch` or `ObservationSelection` rows.
- ``run_vector_games``: K parallel game streams on managym.VectorEnv
  (player_policies=("agent", "agent"), both seats surfaced) with per-seat controllers —
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

from manabot.env.env import Env
from manabot.env.failures import RecordedVectorEnv
from manabot.env.match import Match
from manabot.env.observation import (
    ObservationSpace,
    step_buffers,
)
from manabot.infra.config import MatchConfig
from manabot.model.world import validate_agent_setup
from manabot.policy.protocol import BackendPolicy
from manabot.policy.spans import ObservationSelection
from manabot.sim.search.flat_mc import (
    DEFAULT_MAX_PLAYOUT_STEPS,
    GameRecord,
    SearchStats,
)
import managym
from managym import SlotLoad

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
        agent: BackendPolicy,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
        execution_rows: int | None = None,
    ) -> None:
        if execution_rows is not None and execution_rows < 1:
            raise ValueError("execution_rows must be positive")
        self.execution_rows = execution_rows
        self.agent = agent.to(device)
        self.agent.eval()
        self.deterministic = deterministic
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = device
        self._rng = np.random.default_rng(seed)
        self.forward_calls = 0
        self.obs_selected = 0

    def select(
        self,
        buffers: managym.ObservationBatch | ObservationSelection,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Actions for the selected rows of a native batch or selection."""
        is_batch = isinstance(buffers, managym.ObservationBatch)
        total = buffers.batch_size if is_batch else len(buffers.coordinates)
        count = total if rows is None else len(rows)
        width = self.execution_rows or max(1, count)
        logits: list[np.ndarray] = []
        with self.agent.inference_scope() as scope:
            for start in range(0, count, width):
                if is_batch:
                    if rows is None and start == 0 and width >= count:
                        chunk: managym.ObservationBatch | ObservationSelection = buffers
                    else:
                        idx = (
                            range(start, min(count, start + width))
                            if rows is None
                            else rows[start : start + width]
                        )
                        chunk = buffers.gather([(buffers, int(i)) for i in idx])
                else:
                    chunk = (
                        buffers.select(slice(start, min(count, start + width)))
                        if rows is None
                        else buffers.select(rows[start : start + width])
                    )
                logits.append(scope.infer(chunk).logits)
                del chunk
        actions = self.agent.sample_logits(
            np.concatenate(logits),
            deterministic=self.deterministic,
            temperature=self.temperature,
            sampling="multinomial",
        )

        if self.epsilon > 0.0:
            explore = self._rng.random(count) < self.epsilon
            if explore.any():
                counts = np.asarray(buffers.action_counts, dtype=np.int64)
                if rows is not None:
                    counts = counts[rows]
                actions[explore] = _uniform_from_counts(counts[explore], self._rng)

        self.forward_calls += 1
        self.obs_selected += count
        return actions


def _uniform_from_counts(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    safe = np.maximum(counts, 1)
    return (rng.random(len(safe)) * safe).astype(np.int64)


class RandomBatchController:
    """Uniform-over-valid batched controller (mirrors RandomMatchupPlayer)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def snapshot_rng(self) -> dict[str, Any]:
        return self._rng.bit_generator.state

    def restore_rng(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = state

    def select(
        self,
        buffers: managym.ObservationBatch | ObservationSelection,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        counts = np.asarray(buffers.action_counts, dtype=np.int64)
        if rows is not None:
            counts = counts[rows]
        return _uniform_from_counts(counts, self._rng)


# -----------------------------------------------------------------------------
# K-stream vectorized game driver (eval / head-to-head)
# -----------------------------------------------------------------------------


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

    from manabot.verify.games import INTERACTIVE_DECK

    deck = dict(deck or INTERACTIVE_DECK)
    obs_space = ObservationSpace()
    match = Match(
        MatchConfig(hero="hero", villain="villain", hero_deck=deck, villain_deck=deck)
    )
    for controller in (hero, villain):
        validate_agent_setup(getattr(controller, "agent", None), match.to_rust())
    num_streams = min(num_streams, max(2, num_games))
    env = RecordedVectorEnv(
        num_envs=num_streams,
        seed=seed,
        skip_trivial=True,
        player_policies=("agent", "agent"),
    )
    outcomes = step_buffers(num_streams)
    env.set_observation_layout(outcomes, layout=obs_space.layout)
    env.reset_all_into_buffers(match.to_rust())

    hero_seat = np.arange(num_streams, dtype=np.int64) % 2
    quota = {0: (num_games + 1) // 2, 1: num_games // 2}
    active = np.array([quota[seat] > 0 for seat in hero_seat])
    recorded = {0: 0, 1: 0}
    steps_in_game = np.zeros(num_streams, dtype=np.int64)
    records: list[GameRecord] = []
    total_steps = 0
    max_env_steps = max_env_steps or 10_000_000

    start = time.perf_counter()
    while active.any() and total_steps < max_env_steps:
        acting = np.asarray(env.current_agent_indices(), dtype=np.int64)
        actions = np.zeros(num_streams, dtype=np.int64)
        selection = ObservationSelection(obs_space, tuple(env.observation_rows()))
        hero_rows = np.flatnonzero(active & (acting == hero_seat))
        villain_rows = np.flatnonzero(active & (acting != hero_seat))
        if len(hero_rows):
            actions[hero_rows] = hero.select(selection, hero_rows)
        if len(villain_rows):
            actions[villain_rows] = villain.select(selection, villain_rows)

        env.context.update(
            requested_games=num_games,
            recorded_games=recorded.copy(),
            environment_steps=total_steps,
            max_env_steps=max_env_steps,
        )
        env.step_into_buffers(actions.tolist(), active.tolist())
        total_steps += int(active.sum())
        steps_in_game[active] += 1

        done = active & ((outcomes["terminated"] > 0) | (outcomes["truncated"] > 0))
        if done.any():
            infos = env.get_last_info()
            for s in np.flatnonzero(done):
                seat = int(hero_seat[s])
                if recorded[seat] < quota[seat]:
                    winner = infos[s].get("winner_index")
                    winner = int(winner) if winner is not None else None
                    records.append(
                        GameRecord(
                            game_index=len(records),
                            hero_seat=seat,
                            hero_won=winner == seat,
                            winner=winner,
                            steps=int(steps_in_game[s]),
                        )
                    )
                    recorded[seat] += 1
                    if recorded[seat] == quota[seat]:
                        active[hero_seat == seat] = False
            refill = np.flatnonzero(done & active)
            if refill.size and total_steps < max_env_steps:
                env.load_slots_into_buffers([SlotLoad.fresh(int(s)) for s in refill])
                steps_in_game[refill] = 0
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

    def act(self, env: Env, obs: managym.Observation) -> int:
        validate_agent_setup(getattr(self.sampler, "agent", None), env.match.to_rust())
        del obs  # search reads the raw engine state, not the encoding
        self._calls += 1
        call_seed = (self._seed * 1_000_003 + self._calls) & 0xFFFFFFFFFFFFFFFF
        start = time.perf_counter()

        pool = env.rollout_pool(self.worlds, self.rollouts, call_seed, self.max_steps)
        pool.set_observation_layout(self._obs_space.layout)
        num_slots = pool.num_slots
        actions_full = np.zeros(num_slots, dtype=np.int64)

        active, batch = pool.encode_active_batch()
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
                actions_full[rows] = _straggler_random(batch)
            else:
                actions_full[rows] = self.sampler.select(batch)
            active, batch = pool.step_active_batch(actions_full.tolist())
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


def _straggler_random(batch: managym.ObservationBatch) -> np.ndarray:
    counts = np.asarray(batch.action_counts, dtype=np.int64)
    return _uniform_from_counts(counts, _straggler_rng)
