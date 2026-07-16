"""Determinized PUCT search with real root visit-count targets.

Teacher-1 is deliberately a readable reference implementation above the
authoritative ``managym.Env`` clone/determinize/step boundary.  It is genuine
adaptive Monte Carlo tree search, but its priors are uniform and leaves use
random playouts.  Separate trees are built for separate hidden-information
determinizations, so this does not claim information-set-consistent search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np

from manabot.env import Env
from manabot.sim.flat_mc import DEFAULT_MAX_PLAYOUT_STEPS, SearchStats

_U64_MASK = (1 << 64) - 1


def _mix_seed(a: int, b: int) -> int:
    """Match the engine's SplitMix64-style deterministic sub-seed derivation."""

    z = (
        ((a & _U64_MASK) * 0x9E37_79B9_7F4A_7C15)
        + (b & _U64_MASK)
        + 0x9E37_79B9_7F4A_7C15
    ) & _U64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & _U64_MASK
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _U64_MASK
    return (z ^ (z >> 31)) & _U64_MASK


@dataclass(frozen=True)
class PuctResult:
    visit_counts: np.ndarray
    q_values: np.ndarray
    root_value: float
    simulations: int
    cap_hits: int
    worlds: int
    tree_nodes: int
    max_depth: int


@dataclass
class PuctSearchStats(SearchStats):
    """Generic search cost plus evidence that adaptive trees actually grew."""

    tree_nodes: int = 0
    worlds_sampled: int = 0
    max_depth_sum: int = 0
    max_depth_max: int = 0

    def to_dict(self) -> dict[str, float]:
        out = super().to_dict()
        out.update(
            tree_nodes=float(self.tree_nodes),
            worlds_sampled=float(self.worlds_sampled),
            mean_max_depth=(
                self.max_depth_sum / self.decisions if self.decisions else 0.0
            ),
            max_depth_max=float(self.max_depth_max),
        )
        return out


@dataclass
class _Node:
    env: Any
    player: int | None
    visits: np.ndarray
    value_sums: np.ndarray
    priors: np.ndarray
    terminal: bool
    children: dict[int, "_Node"] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Any, *, terminal: bool | None = None) -> "_Node":
        is_terminal = env.is_game_over() if terminal is None else terminal
        if is_terminal:
            return cls(
                env=env,
                player=None,
                visits=np.zeros(0, dtype=np.int64),
                value_sums=np.zeros(0, dtype=np.float64),
                priors=np.zeros(0, dtype=np.float64),
                terminal=True,
            )
        player = env.current_agent_index()
        if player is None:
            raise RuntimeError("nonterminal PUCT node has no acting player")
        action_count = int(env.action_count())
        if action_count < 1:
            raise RuntimeError("nonterminal PUCT node has no legal actions")
        return cls(
            env=env,
            player=int(player),
            visits=np.zeros(action_count, dtype=np.int64),
            value_sums=np.zeros(action_count, dtype=np.float64),
            priors=np.full(action_count, 1.0 / action_count, dtype=np.float64),
            terminal=False,
        )


def _root_score(winner: int | None, hero: int) -> float:
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == hero else 0.0


def _select_action(node: _Node, hero: int, c_puct: float) -> int:
    total = int(node.visits.sum())
    q = np.divide(
        node.value_sums,
        node.visits,
        out=np.full(len(node.visits), 0.5, dtype=np.float64),
        where=node.visits > 0,
    )
    acting_q = q if node.player == hero else 1.0 - q
    exploration = c_puct * node.priors * math.sqrt(total + 1.0) / (node.visits + 1.0)
    return int(np.argmax(acting_q + exploration))


def _evaluate_leaf(
    env: Any, *, hero: int, seed: int, max_steps: int
) -> tuple[float, bool]:
    rollout = env.clone_env()
    winner = rollout.random_playout(seed=seed, max_steps=max_steps)
    hit_cap = winner is None and not rollout.is_game_over()
    return _root_score(winner, hero), hit_cap


def _simulate(
    root: _Node,
    *,
    hero: int,
    c_puct: float,
    rollout_seed: int,
    max_steps: int,
) -> tuple[float, bool, int, int]:
    node = root
    path: list[tuple[_Node, int]] = []
    depth = 0
    added_nodes = 0
    hit_cap = False

    while True:
        if node.terminal:
            value = _root_score(node.env.winner_index(), hero)
            break
        if depth >= max_steps:
            value = 0.5
            hit_cap = True
            break

        action = _select_action(node, hero, c_puct)
        path.append((node, action))
        child = node.children.get(action)
        if child is None:
            child_env = node.env.clone_env()
            _, _, terminated, truncated, _ = child_env.step(action)
            child = _Node.from_env(child_env, terminal=bool(terminated or truncated))
            node.children[action] = child
            added_nodes += 1
            depth += 1
            if terminated:
                value = _root_score(child_env.winner_index(), hero)
            elif truncated:
                value = 0.5
                hit_cap = True
            else:
                value, hit_cap = _evaluate_leaf(
                    child_env,
                    hero=hero,
                    seed=rollout_seed,
                    max_steps=max(1, max_steps - depth),
                )
            break

        node = child
        depth += 1

    for parent, action in path:
        parent.visits[action] += 1
        parent.value_sums[action] += value
    return value, hit_cap, depth, added_nodes


def determinized_puct(
    root_env: Any,
    *,
    simulations: int,
    worlds: int,
    seed: int,
    c_puct: float = 1.5,
    max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
) -> PuctResult:
    """Search one decision and return aggregated root visits and values.

    ``simulations`` is the total tree traversal budget across all worlds—not a
    per-action budget. Every world receives at least one simulation.
    """

    if simulations < 1:
        raise ValueError("simulations must be >= 1")
    if worlds < 1 or worlds > simulations:
        raise ValueError("worlds must be in [1, simulations]")
    if c_puct <= 0:
        raise ValueError("c_puct must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if root_env.is_game_over():
        raise ValueError("cannot search a terminal environment")

    hero = root_env.current_agent_index()
    if hero is None:
        raise RuntimeError("PUCT root has no acting player")
    hero = int(hero)
    action_count = int(root_env.action_count())
    visits = np.zeros(action_count, dtype=np.int64)
    value_sums = np.zeros(action_count, dtype=np.float64)
    total_value = 0.0
    cap_hits = 0
    tree_nodes = 0
    max_depth = 0

    per_world = [simulations // worlds] * worlds
    for world_index in range(simulations % worlds):
        per_world[world_index] += 1

    for world_index, world_simulations in enumerate(per_world):
        world_seed = _mix_seed(seed, world_index)
        world = root_env.clone_env()
        world.determinize(seed=world_seed, perspective=hero)
        if int(world.action_count()) != action_count:
            raise RuntimeError("determinization changed the root legal action count")
        root = _Node.from_env(world)
        tree_nodes += 1
        for simulation_index in range(world_simulations):
            rollout_seed = _mix_seed(world_seed, simulation_index + 1)
            value, hit_cap, depth, added = _simulate(
                root,
                hero=hero,
                c_puct=c_puct,
                rollout_seed=rollout_seed,
                max_steps=max_steps,
            )
            total_value += value
            cap_hits += int(hit_cap)
            tree_nodes += added
            max_depth = max(max_depth, depth)
        visits += root.visits
        value_sums += root.value_sums

    q_values = np.divide(
        value_sums,
        visits,
        out=np.full(action_count, 0.5, dtype=np.float64),
        where=visits > 0,
    )
    return PuctResult(
        visit_counts=visits,
        q_values=q_values.astype(np.float32),
        root_value=total_value / simulations,
        simulations=simulations,
        cap_hits=cap_hits,
        worlds=worlds,
        tree_nodes=tree_nodes,
        max_depth=max_depth,
    )


class DeterminizedPuctPlayer:
    """Matchup player backed by multi-world PUCT and random leaf playouts."""

    def __init__(
        self,
        simulations: int,
        *,
        worlds: int = 4,
        c_puct: float = 1.5,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        seed: int = 0,
    ):
        if simulations < 1 or worlds < 1 or worlds > simulations:
            raise ValueError("simulations must be >= 1 and worlds in [1, simulations]")
        self.simulations = simulations
        self.worlds = worlds
        self.c_puct = c_puct
        self.max_steps = max_steps
        self._seed = seed
        self._calls = 0
        self.stats = PuctSearchStats()
        self.last_visit_counts: np.ndarray | None = None
        self.last_root_value: float | None = None
        self.last_scores: np.ndarray | None = None
        self.last_result: PuctResult | None = None

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs
        self._calls += 1
        call_seed = _mix_seed(self._seed, self._calls)
        started = time.perf_counter()
        result = determinized_puct(
            env._engine,
            simulations=self.simulations,
            worlds=self.worlds,
            seed=call_seed,
            c_puct=self.c_puct,
            max_steps=self.max_steps,
        )
        elapsed = time.perf_counter() - started
        self.last_result = result
        self.last_visit_counts = result.visit_counts.astype(np.float32)
        self.last_root_value = result.root_value
        self.last_scores = result.q_values
        self.stats.decisions += 1
        self.stats.seconds += elapsed
        self.stats.simulations += result.simulations
        self.stats.cap_hits += result.cap_hits
        self.stats.decision_seconds.append(elapsed)
        self.stats.tree_nodes += result.tree_nodes
        self.stats.worlds_sampled += result.worlds
        self.stats.max_depth_sum += result.max_depth
        self.stats.max_depth_max = max(self.stats.max_depth_max, result.max_depth)

        candidates = np.flatnonzero(
            result.visit_counts == int(result.visit_counts.max())
        )
        if len(candidates) == 1:
            return int(candidates[0])
        return int(candidates[np.argmax(result.q_values[candidates])])
