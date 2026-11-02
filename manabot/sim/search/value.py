"""Value-guided players and evaluation of frozen policy checkpoints.

ValueScorer interprets a scalar value head as an acting-player win logit.
VGreedyPlayer scores one-step successors; ValueSearchPlayer uses batched value
estimates at the leaves of determinized rollouts. Terminal rollouts contribute
their true outcome; unfinished leaves use the frozen model. Callers convert
acting-player probabilities to the root player's perspective.

The assessment helpers compare retained predictions with rollout outcomes.
This module performs inference and assessment, not parameter fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from manabot.env.env import Env
from manabot.env.observation import ObservationSpace
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.sim.rollout import OBS_KEYS, _allocate_buffers
from manabot.sim.search.flat_mc import DEFAULT_MAX_PLAYOUT_STEPS, SearchStats

# -----------------------------------------------------------------------------
# Batched value evaluation
# -----------------------------------------------------------------------------


class ValueScorer:
    """Win probability V(obs) for rows of encoded observation buffers.

    One masked forward per call; V = sigmoid(value head), the probability
    that the observation's *agent* (the acting player it is encoded for)
    wins. Callers flip 1-V for non-hero actors.
    """

    def __init__(self, agent: ActorCritic, *, device: str = "cpu"):
        self.agent = agent.to(device)
        self.agent.eval()
        self.device = torch.device(device)
        self.forward_calls = 0
        self.obs_scored = 0

    def score(self, buffers: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
        obs = {
            key: torch.from_numpy(buffers[key][rows]).to(self.device)
            for key in OBS_KEYS
        }
        with torch.inference_mode():
            _, value = self.agent.forward(obs)
            probs = torch.sigmoid(value)
        self.forward_calls += 1
        self.obs_scored += len(rows)
        return probs.float().cpu().numpy()


# -----------------------------------------------------------------------------
# Value-at-leaf action scoring (shared by V-greedy and value-search)
# -----------------------------------------------------------------------------


@dataclass
class ValueSearchStats(SearchStats):
    net_obs: int = 0
    net_forwards: int = 0

    def to_dict(self) -> dict[str, float]:
        out = super().to_dict()
        out.update(net_obs=float(self.net_obs), net_forwards=float(self.net_forwards))
        return out


def _random_valid_actions(
    buffers: dict[str, np.ndarray], rows: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    valid = buffers["actions_valid"][rows] > 0
    weights = valid.astype(np.float64)
    totals = weights.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    weights /= totals
    cdf = np.cumsum(weights, axis=1)
    u = rng.random((len(cdf), 1))
    return (u < cdf).argmax(axis=1).astype(np.int64)


class _ValueLeafScorer:
    """Score every legal root action by determinized value-at-leaf estimates.

    Per act() call: one RolloutPool of ``worlds`` x ``rollouts`` simulations
    per action (root actions pre-applied by the engine), ``depth`` uniformly-
    random plies, then one batched V forward over all surviving leaves.
    Score[a] = (true outcomes of finished sims + hero-perspective V of live
    leaves) / (worlds * rollouts).
    """

    def __init__(
        self,
        scorer: ValueScorer,
        *,
        worlds: int,
        rollouts: int = 1,
        depth: int = 0,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        seed: int = 0,
    ):
        if worlds < 1 or rollouts < 1:
            raise ValueError("worlds and rollouts must be >= 1")
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.scorer = scorer
        self.worlds = worlds
        self.rollouts = rollouts
        self.depth = depth
        self.max_steps = max_steps
        self._seed = seed
        self._rng = np.random.default_rng(seed ^ 0x5EED)
        self._calls = 0
        self.stats = ValueSearchStats()
        self._obs_space = ObservationSpace()
        self._buffers: dict[str, np.ndarray] | None = None
        self._capacity = 0

    def _ensure_buffers(self, capacity: int) -> dict[str, np.ndarray]:
        if self._buffers is None or self._capacity < capacity:
            self._buffers = _allocate_buffers(self._obs_space, capacity)
            self._capacity = capacity
        return self._buffers

    def score_actions(self, env: Env) -> np.ndarray:
        """Hero-perspective score per legal root action (engine order)."""

        validate_agent_setup(self.scorer.agent, env.match.to_rust())
        self._calls += 1
        call_seed = (self._seed * 1_000_003 + self._calls) & 0xFFFFFFFFFFFFFFFF
        start = time.perf_counter()

        pool = env.rollout_pool(self.worlds, self.rollouts, call_seed, self.max_steps)
        num_slots = pool.num_slots
        buffers = self._ensure_buffers(num_slots)
        pool.set_buffers(buffers, self._capacity)
        active = pool.encode_active()

        for _ in range(self.depth):
            if not active:
                break
            rows = np.asarray(active, dtype=np.int64)
            actions_full = np.zeros(num_slots, dtype=np.int64)
            actions_full[rows] = _random_valid_actions(buffers, rows, self._rng)
            active = pool.step_active(actions_full.tolist())

        denominator = float(self.worlds * self.rollouts)
        base_scores, simulations, cap_hits = pool.scores()
        totals = np.asarray(base_scores, dtype=np.float64) * denominator

        if active:
            rows = np.asarray(active, dtype=np.int64)
            values = self.scorer.score(buffers, rows).astype(np.float64)
            acting = np.asarray(pool.acting_players(), dtype=np.int64)[rows]
            hero = int(pool.hero_index)
            # V is the acting player's win prob; flip for the non-hero actor.
            # A missing actor (should not happen for active slots) scores 0.5.
            hero_values = np.where(acting == hero, values, 1.0 - values)
            hero_values = np.where(acting < 0, 0.5, hero_values)
            roots = np.asarray(pool.root_actions(), dtype=np.int64)[rows]
            np.add.at(totals, roots, hero_values)
            self.stats.net_obs = self.scorer.obs_scored
            self.stats.net_forwards = self.scorer.forward_calls

        elapsed = time.perf_counter() - start
        self.stats.decisions += 1
        self.stats.seconds += elapsed
        self.stats.simulations += int(simulations)
        self.stats.cap_hits += int(cap_hits)
        self.stats.decision_seconds.append(elapsed)
        return (totals / denominator).astype(np.float32)


class VGreedyPlayer:
    """Argmax over legal actions of the one-step value.

    Each root action is applied on ONE determinized clone of the current
    state and V is read at the resulting state (hero perspective). This is
    the gate's baseline: search must beat it or V-guided search is not a
    policy improvement operator.
    """

    def __init__(
        self,
        scorer: ValueScorer,
        *,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        seed: int = 0,
    ):
        self._leaf = _ValueLeafScorer(
            scorer, worlds=1, rollouts=1, depth=0, max_steps=max_steps, seed=seed
        )
        self.last_scores: np.ndarray | None = None

    @property
    def stats(self) -> ValueSearchStats:
        return self._leaf.stats

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs
        scores = self._leaf.score_actions(env)
        self.last_scores = scores
        return int(np.argmax(scores))


class ValueSearchPlayer:
    """Flat determinized MC with V at the leaves.

    ``sims`` = worlds x rollouts per legal action (worlds shared across
    actions, common random numbers, exactly like FlatMCPlayer) — but instead
    of playing each simulation to terminal, ``depth`` random plies follow
    the root action and V evaluates the leaf in one batched forward.
    depth=0 with rollouts=1 is pure "N determinized worlds x 1-step-then-V".
    """

    def __init__(
        self,
        sims: int,
        scorer: ValueScorer,
        *,
        rollouts_per_world: int = 1,
        depth: int = 0,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        seed: int = 0,
    ):
        if sims < 1:
            raise ValueError("sims must be >= 1")
        rollouts = max(1, min(rollouts_per_world, sims))
        worlds = max(1, sims // rollouts)
        self.sims = worlds * rollouts
        self._leaf = _ValueLeafScorer(
            scorer,
            worlds=worlds,
            rollouts=rollouts,
            depth=depth,
            max_steps=max_steps,
            seed=seed,
        )
        self.last_scores: np.ndarray | None = None

    @property
    def stats(self) -> ValueSearchStats:
        return self._leaf.stats

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs
        scores = self._leaf.score_actions(env)
        self.last_scores = scores
        return int(np.argmax(scores))


def load_value_scorer(checkpoint: str, *, device: str = "cpu") -> ValueScorer:
    agent, _ = load_checkpoint(checkpoint)
    return ValueScorer(agent, device=device)


# -----------------------------------------------------------------------------
# Assessing V: rollout ground truth + strategic buckets (wave README protocol)
# -----------------------------------------------------------------------------

_HAND_ZONE = 1  # managym.ZoneEnum.HAND
