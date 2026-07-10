"""In-process cross-game batched datagen for policy-rollout teachers.

Exp-07 finding: MPS does not multiplex across processes — four datagen
workers sharing the GPU deliver roughly one worker's throughput, so
process-parallelism is the wrong axis for a policy-rollout teacher. This
driver plays G self-play games concurrently in ONE process and merges the
rollout batches of every game's current decision into single net forwards:
per ply, one forward covers sum_g(active slots of game g) observations.

Produces shards in exactly the manabot.sim.distill format (observations,
argmax action, raw score vector, provenance tag).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import MatchHypers, RewardHypers
from manabot.sim.distill import OBS_KEYS, SCORE_KEY, _git_commit
from manabot.sim.flat_mc import (
    DEFAULT_MAX_PLAYOUT_STEPS,
    load_checkpoint_agent,
    spec_name,
)
from manabot.sim.rollout import BatchedSampler, _allocate_buffers
from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs


class _GameStream:
    """One self-play game holding at most one live rollout pool."""

    def __init__(self, obs_space: ObservationSpace, match: Match, seed: int):
        self.env = Env(
            match,
            obs_space,
            Reward(RewardHypers()),
            seed=seed,
            auto_reset=False,
            enable_profiler=False,
            enable_behavior_tracking=False,
        )
        self.obs: dict[str, np.ndarray] | None = None
        self.game_index = -1
        self.steps = 0
        self.decision_rows: list[int] = []
        self.pool = None
        self.pool_buffers: dict[str, np.ndarray] | None = None
        self.pool_capacity = 0
        self.active: list[int] = []
        self.plies = 0
        self.actions_full: np.ndarray | None = None

    def start_game(self, game_index: int, seed: int) -> None:
        self.obs, _ = self.env.reset(seed=seed)
        self.game_index = game_index
        self.steps = 0
        self.decision_rows = []


def generate_selfplay_shard_pooled(
    *,
    num_games: int,
    checkpoint: str,
    sims: int = 8,
    rollouts_per_world: int = 1,
    policy_plies: int = 8,
    epsilon: float = 0.1,
    device: str = "mps",
    concurrent_games: int = 16,
    seed: int = 0,
    game_offset: int = 0,
    round_index: int = 1,
    max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
    max_steps_per_game: int = 5000,
    out_path: str | Path | None = None,
    teacher_name: str | None = None,
    log_every: int = 0,
) -> dict[str, Any]:
    """Teacher-vs-teacher self-play with cross-game batched policy rollouts."""

    obs_space = ObservationSpace()
    max_actions = obs_space.encoder.max_actions
    rollouts = max(1, min(rollouts_per_world, sims))
    worlds = max(1, sims // rollouts)
    teacher_spec = {
        "kind": "policy_search",
        "sims": worlds * rollouts,
        "rollouts_per_world": rollouts,
        "checkpoint": checkpoint,
        "epsilon": epsilon,
        "policy_plies": policy_plies,
        "device": device,
        "name": teacher_name or f"psearch-{worlds * rollouts}",
    }
    name = spec_name(teacher_spec)

    agent, _ = load_checkpoint_agent(checkpoint)
    sampler = BatchedSampler(agent, epsilon=epsilon, seed=seed * 2 + 1, device=device)

    match = Match(
        MatchHypers(
            hero=f"{name}-a"[:32],
            villain=f"{name}-b"[:32],
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )

    concurrent_games = max(1, min(concurrent_games, num_games))
    streams = [
        _GameStream(obs_space, match, seed=seed + 977 * s)
        for s in range(concurrent_games)
    ]

    # Dataset accumulators (exact distill shard format).
    obs_buffers: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
    score_rows: list[np.ndarray] = []
    actions: list[int] = []
    game_indices: list[int] = []
    seats: list[int] = []
    num_valids: list[int] = []
    rows_per_game: dict[int, list[int]] = {}
    winners: dict[int, int | None] = {}
    steps_per_game: dict[int, int] = {}

    next_game = 0
    finished_games = 0
    simulations = 0
    cap_hits_total = 0
    net_obs = 0
    net_forwards = 0
    decision_count = 0
    search_seconds = 0.0

    for stream in streams:
        if next_game < num_games:
            stream.start_game(game_offset + next_game, seed + next_game)
            rows_per_game[stream.game_index] = []
            next_game += 1
        else:
            stream.obs = None

    wall_start = time.perf_counter()

    def open_pool(stream: _GameStream) -> None:
        """Create the rollout pool for the stream's current decision."""

        call_seed = (seed * 1_000_003 + decision_count * 31 + stream.game_index) & (
            0xFFFFFFFFFFFFFFFF
        )
        pool = stream.env._engine.rollout_pool(worlds, rollouts, call_seed, max_steps)
        capacity = pool.num_slots
        if stream.pool_buffers is None or stream.pool_capacity < capacity:
            stream.pool_buffers = _allocate_buffers(obs_space, capacity)
            stream.pool_capacity = capacity
        pool.set_buffers(stream.pool_buffers, stream.pool_capacity)
        stream.pool = pool
        stream.active = pool.encode_active()
        stream.plies = 0
        stream.actions_full = np.zeros(pool.num_slots, dtype=np.int64)

    def close_pool_and_step(stream: _GameStream) -> None:
        """Score the pool, record the decision, advance the parent game."""

        nonlocal decision_count, simulations, cap_hits_total, finished_games, next_game

        scores, sims_done, caps = stream.pool.scores()
        simulations += int(sims_done)
        cap_hits_total += int(caps)
        action = int(np.argmax(scores))

        obs = stream.obs
        for key in OBS_KEYS:
            obs_buffers[key].append(np.asarray(obs[key], dtype=np.float32))
        score_row = np.full(max_actions, -1.0, dtype=np.float32)
        k = min(len(scores), max_actions)
        score_row[:k] = np.asarray(scores[:k], dtype=np.float32)
        score_rows.append(score_row)
        actions.append(action)
        game_indices.append(stream.game_index)
        acting = int(stream.env.last_raw_obs.agent.player_index)
        seats.append(acting)
        num_valids.append(int(np.sum(obs["actions_valid"] > 0)))
        rows_per_game[stream.game_index].append(len(actions) - 1)
        decision_count += 1

        stream.pool = None
        new_obs, _, terminated, truncated, info = stream.env.step(action)
        stream.obs = new_obs
        stream.steps += 1
        done = bool(terminated or truncated) or stream.steps >= max_steps_per_game
        if done:
            winner = (
                winner_from_info_or_obs(info, stream.env.last_raw_obs)
                if (terminated or truncated)
                else None
            )
            winners[stream.game_index] = winner
            steps_per_game[stream.game_index] = stream.steps
            finished_games += 1
            if log_every and finished_games % log_every == 0:
                elapsed = time.perf_counter() - wall_start
                print(
                    f"  [pooled] {finished_games}/{num_games} games, "
                    f"{decision_count} decisions, {elapsed:.0f}s "
                    f"({decision_count / max(elapsed, 1e-9):.1f} dec/s)",
                    flush=True,
                )
            if next_game < num_games:
                stream.start_game(game_offset + next_game, seed + next_game)
                rows_per_game[stream.game_index] = []
                next_game += 1
            else:
                stream.obs = None

    while any(stream.obs is not None for stream in streams):
        # 1. Ensure every live stream has a pool at its current decision.
        for stream in streams:
            if stream.obs is not None and stream.pool is None:
                open_pool(stream)
                while stream.pool is not None and not stream.active:
                    # Every simulation ended on the root action.
                    close_pool_and_step(stream)
                    if stream.obs is not None:
                        open_pool(stream)

        live = [s for s in streams if s.obs is not None and s.pool is not None]
        if not live:
            continue

        # 2. One merged forward across every live stream's active slots.
        merged = {
            key: np.concatenate(
                [s.pool_buffers[key][np.asarray(s.active)] for s in live]
            )
            for key in OBS_KEYS
        }
        total_rows = len(merged["actions_valid"])
        chosen = sampler.select(merged, np.arange(total_rows))
        net_obs += total_rows
        net_forwards += 1

        # 3. Scatter actions back, step every pool.
        offset = 0
        for stream in live:
            rows = np.asarray(stream.active)
            stream.actions_full[rows] = chosen[offset : offset + len(rows)]
            offset += len(rows)
            stream.active = stream.pool.step_active(stream.actions_full.tolist())
            stream.plies += 1
            if stream.active and stream.plies >= policy_plies:
                stream.pool.finish_random()
                stream.active = []
            if not stream.active:
                close_pool_and_step(stream)

    wall_seconds = time.perf_counter() - wall_start
    search_seconds = wall_seconds  # single-process: search wall == wall

    winner_column = np.full(len(actions), -1, dtype=np.int8)
    for game_index, rows in rows_per_game.items():
        winner = winners.get(game_index)
        if winner is not None:
            winner_column[rows] = winner

    arrays: dict[str, np.ndarray] = {
        key: (
            np.stack(values)
            if values
            else np.zeros((0, *obs_space.shapes[key]), dtype=np.float32)
        )
        for key, values in obs_buffers.items()
    }
    arrays["action"] = np.asarray(actions, dtype=np.int16)
    arrays["game_index"] = np.asarray(game_indices, dtype=np.int32)
    arrays["seat"] = np.asarray(seats, dtype=np.int8)
    arrays["num_valid"] = np.asarray(num_valids, dtype=np.int16)
    arrays["winner"] = winner_column
    arrays[SCORE_KEY] = (
        np.stack(score_rows)
        if score_rows
        else np.zeros((0, max_actions), dtype=np.float32)
    )
    provenance = {
        "round": round_index,
        "teacher_spec": {k: v for k, v in teacher_spec.items() if k != "device"},
        "teacher_name": name,
        "rollout_policy_checkpoint": checkpoint,
        "generating_opponent": name,
        "git_commit": _git_commit(),
        "seed": seed,
        "game_offset": game_offset,
        "concurrent_games": concurrent_games,
    }
    arrays["provenance"] = np.array(json.dumps(provenance))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)

    ordered_games = sorted(steps_per_game)
    return {
        "teacher": name,
        "provenance": provenance,
        "num_games": finished_games,
        "decisions": len(actions),
        "wall_seconds": wall_seconds,
        "search": {
            "decisions": float(len(actions)),
            "seconds": search_seconds,
            "simulations": float(simulations),
            "cap_hits": float(cap_hits_total),
        },
        "net_obs": net_obs,
        "net_forwards": net_forwards,
        "steps_per_game": [steps_per_game[g] for g in ordered_games],
        "winners": [
            winners[g] if winners.get(g) is not None else -1 for g in ordered_games
        ],
        "out_path": str(out_path) if out_path is not None else None,
    }
