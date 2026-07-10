"""Value head training, V-greedy play, and value-at-leaf search (exp-10).

Wave/search goal 4 — the pre-registered gate. Exp-07 refuted the policy-
ROLLOUT crank on economics (a policy playout costs ~160x a random playout);
the escape route is a VALUE at the leaves: one forward pass replacing an
entire playout. This module provides the three pieces the gate needs:

- ``train_value`` / ``save_value_checkpoint``: fit the Agent's existing
  scalar value head (shared encoder, hero-perspective win-prob via sigmoid)
  to terminal game outcomes recorded in distillation shards (BCE, split by
  game). Type-error caveat (wave/beliefs): scalar V(observation) is rung-1 —
  the same observation has different values under different opponent ranges;
  it is expected to work at current opponent strength, not in general.
- ``VGreedyPlayer``: argmax over legal actions of the one-step value —
  each root action applied on one determinized clone, V evaluated at the
  resulting state from the deciding player's perspective.
- ``ValueSearchPlayer``: flat determinized MC where the playout is replaced
  (or depth-truncated) by V at the leaves: W worlds x R clones per root
  action, ``depth`` uniformly-random plies after the root action, then one
  batched V evaluation over every still-live leaf. depth=0 is
  "N determinized worlds x 1-step-then-V".

Scoring matches flat_mc/RolloutPool conventions: win 1 / loss 0 / draw-or-
cap 0.5 from the deciding player's perspective; slots that terminate before
the leaf contribute their true outcome (accumulated engine-side by the
pool), so V only ever fills in for genuinely unfinished simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from manabot.env import Env, ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model.agent import Agent
from manabot.sim.distill import OBS_KEYS, save_bc_checkpoint, split_by_game
from manabot.sim.flat_mc import (
    DEFAULT_MAX_PLAYOUT_STEPS,
    SearchStats,
    load_checkpoint_agent,
)
from manabot.sim.rollout import _allocate_buffers


# -----------------------------------------------------------------------------
# Batched value evaluation
# -----------------------------------------------------------------------------


class ValueScorer:
    """Win probability V(obs) for rows of encoded observation buffers.

    One masked forward per call; V = sigmoid(value head), the probability
    that the observation's *agent* (the acting player it is encoded for)
    wins. Callers flip 1-V for non-hero actors.
    """

    def __init__(self, agent: Agent, *, device: str = "cpu"):
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
# Training: value head on terminal outcomes
# -----------------------------------------------------------------------------


@dataclass
class ValueEpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_brier: float
    val_accuracy: float


def outcome_labels(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(usable row indices, win label per usable row) from winner/seat columns.

    Label is 1.0 when the decision-maker (seat) won the source game. Rows
    from games with no recorded winner (draws, caps, unfinished) are dropped.
    """

    winner = dataset["winner"].astype(np.int64)
    seat = dataset["seat"].astype(np.int64)
    usable = np.flatnonzero(winner >= 0)
    labels = (winner[usable] == seat[usable]).astype(np.float32)
    return usable, labels


@torch.no_grad()
def _evaluate_value(
    agent: Agent,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """(mean BCE, Brier score, accuracy) of sigmoid(V) against labels."""

    agent.eval()
    total_bce = 0.0
    total_brier = 0.0
    correct = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        target = torch.as_tensor(
            labels[start : start + batch_size], dtype=torch.float32, device=device
        )
        obs = {
            key: torch.as_tensor(
                dataset[key][batch], dtype=torch.float32, device=device
            )
            for key in OBS_KEYS
        }
        _, value = agent.forward(obs)
        total_bce += float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                value, target, reduction="sum"
            ).item()
        )
        probs = torch.sigmoid(value)
        total_brier += float(((probs - target) ** 2).sum().item())
        correct += int(((probs > 0.5) == (target > 0.5)).sum().item())
    n = max(1, len(indices))
    return total_bce / n, total_brier / n, correct / n


def train_value(
    dataset: dict[str, np.ndarray],
    *,
    init_state: dict[str, Any] | None = None,
    freeze_encoder: bool = False,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 512,
    val_fraction: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    agent_hypers: AgentHypers | None = None,
    log: bool = False,
) -> tuple[Agent, ObservationSpace, list[ValueEpochStats]]:
    """Fit the Agent's value head to terminal outcomes (BCE, split by game).

    ``init_state`` warm-starts from an existing checkpoint (e.g. the BC
    student, so the encoder starts from policy-distillation features).
    ``freeze_encoder`` trains only the value head, leaving every shared
    weight — and therefore the checkpoint's policy behavior — untouched.
    The value loss never reaches the policy head either way (no gradient
    path), so policy weights are only ever moved through the shared encoder.
    """

    torch.manual_seed(seed)
    dev = torch.device(device)
    obs_space = ObservationSpace()
    agent = Agent(obs_space, agent_hypers or AgentHypers()).to(dev)
    if init_state is not None:
        agent.load_state_dict(init_state)
    if freeze_encoder:
        for name, param in agent.named_parameters():
            param.requires_grad = name.startswith("value_head")
    params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    usable, labels = outcome_labels(dataset)
    if len(usable) == 0:
        raise ValueError("dataset has no rows with a recorded winner")
    train_idx, val_idx = split_by_game(dataset, val_fraction=val_fraction, seed=seed)
    usable_set = np.zeros(len(dataset["winner"]), dtype=bool)
    usable_set[usable] = True
    label_by_row = np.full(len(dataset["winner"]), -1.0, dtype=np.float32)
    label_by_row[usable] = labels
    train_idx = train_idx[usable_set[train_idx]]
    val_idx = val_idx[usable_set[val_idx]]

    rng = np.random.default_rng(seed)
    history: list[ValueEpochStats] = []
    for epoch in range(epochs):
        agent.train()
        order = train_idx.copy()
        rng.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            obs = {
                key: torch.as_tensor(
                    dataset[key][batch], dtype=torch.float32, device=dev
                )
                for key in OBS_KEYS
            }
            target = torch.as_tensor(
                label_by_row[batch], dtype=torch.float32, device=dev
            )
            _, value = agent.forward(obs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                value, target
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
        val_bce, val_brier, val_acc = _evaluate_value(
            agent,
            dataset,
            val_idx,
            label_by_row[val_idx],
            batch_size=batch_size,
            device=dev,
        )
        stats = ValueEpochStats(
            epoch=epoch,
            train_loss=total_loss / max(1, len(order)),
            val_loss=val_bce,
            val_brier=val_brier,
            val_accuracy=val_acc,
        )
        history.append(stats)
        if log:
            print(
                f"  epoch {epoch}: train_bce {stats.train_loss:.4f} "
                f"val_bce {stats.val_loss:.4f} val_brier {stats.val_brier:.4f} "
                f"val_acc {stats.val_accuracy:.4f}",
                flush=True,
            )
    return agent, obs_space, history


def save_value_checkpoint(
    agent: Agent,
    obs_space: ObservationSpace,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist in the trainer checkpoint format (load_checkpoint_agent)."""

    save_bc_checkpoint(agent, obs_space, path, extra={"value": extra or {}})


# -----------------------------------------------------------------------------
# Rank correlation (no scipy dependency)
# -----------------------------------------------------------------------------


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties."""

    if len(x) < 2:
        return float("nan")
    rx, ry = _average_ranks(np.asarray(x)), _average_ranks(np.asarray(y))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


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

        self._calls += 1
        call_seed = (self._seed * 1_000_003 + self._calls) & 0xFFFFFFFFFFFFFFFF
        start = time.perf_counter()

        pool = env._engine.rollout_pool(
            self.worlds, self.rollouts, call_seed, self.max_steps
        )
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
    agent, _ = load_checkpoint_agent(checkpoint)
    return ValueScorer(agent, device=device)


# -----------------------------------------------------------------------------
# Assessing V: rollout ground truth + strategic buckets (wave README protocol)
# -----------------------------------------------------------------------------

#: Instants in INTERACTIVE_DECK — "holding interaction" for the aggro-bias
#: tripwire. Ancestral Recall is an instant but not interaction; it is
#: tracked separately.
INTERACTION_NAMES = frozenset({"Lightning Bolt", "Counterspell"})
INSTANT_NAMES = frozenset({"Lightning Bolt", "Counterspell", "Ancestral Recall"})

_HAND_ZONE = 1  # managym.ZoneEnum.HAND


def _state_features(raw_obs: Any) -> dict[str, float]:
    """Bucketing features for the current decision state (acting player)."""

    my_board = sum(
        max(p.power, 0) + max(p.toughness, 0) for p in raw_obs.agent_permanents
    )
    their_board = sum(
        max(p.power, 0) + max(p.toughness, 0) for p in raw_obs.opponent_permanents
    )
    hand_names = [
        card.name for card in raw_obs.agent_cards if int(card.zone) == _HAND_ZONE
    ]
    return {
        "board_adv": float(my_board - their_board),
        "life_adv": float(raw_obs.agent.life - raw_obs.opponent.life),
        "holding_interaction": float(
            any(name in INTERACTION_NAMES for name in hand_names)
        ),
        "holding_instant": float(any(name in INSTANT_NAMES for name in hand_names)),
        "turn": float(raw_obs.turn.turn_number),
    }


def collect_value_assessment(
    *,
    value_checkpoint: str,
    behavior_checkpoint: str,
    num_games: int = 80,
    sample_rate: float = 0.06,
    gt_worlds: int = 16,
    gt_rollouts: int = 4,
    max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
    seed: int = 0,
    device: str = "cpu",
    max_states: int | None = None,
) -> dict[str, np.ndarray]:
    """V(s) vs rollout ground truth on fresh states, with bucket features.

    Plays ``behavior_checkpoint`` self-play (stochastic, both seats); at each
    surfaced decision, with probability ``sample_rate``, records:

    - ``v``: sigmoid value of the current observation (acting player's
      predicted win prob);
    - ``gt``: mean over legal actions of flat_mc_scores(gt_worlds x
      gt_rollouts) — i.e. the mean of gt_worlds*gt_rollouts fresh uniformly-
      random playouts per action from this exact state (the wave README's
      unbiased P(win|s) instrument; >=64 playouts per action at defaults);
    - bucket features (board advantage, holding interaction/instants, life,
      turn) and, post hoc, the game's actual outcome for the acting player.

    States are fresh self-play visits, disjoint by construction from the
    training shards' games.
    """

    from manabot.env import Match, Reward
    from manabot.infra.hypers import MatchHypers, RewardHypers
    from manabot.sim.flat_mc import make_player
    from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs

    scorer = load_value_scorer(value_checkpoint, device=device)
    behavior, obs_space = make_player(
        {"kind": "checkpoint", "path": behavior_checkpoint}, seed=seed * 2 + 1
    )
    obs_space = obs_space or ObservationSpace()
    match = Match(
        MatchHypers(
            hero="assess-a",
            villain="assess-b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(
        match,
        obs_space,
        Reward(RewardHypers()),
        seed=seed,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    rng = np.random.default_rng(seed + 17)

    rows: list[dict[str, float]] = []
    row_game: list[int] = []
    row_seat: list[int] = []
    winners: dict[int, int | None] = {}
    gt_call = 0

    for game_index in range(num_games):
        obs, _ = env.reset(seed=seed + game_index)
        done = False
        info: dict[str, Any] = {}
        while not done:
            acting = int(env.last_raw_obs.agent.player_index)
            num_valid = int(np.sum(obs["actions_valid"] > 0))
            sampled = (
                num_valid > 1
                and rng.random() < sample_rate
                and (max_states is None or len(rows) < max_states)
            )
            if sampled:
                gt_call += 1
                batch = {
                    key: np.asarray(obs[key])[None].astype(
                        np.float32 if key != "action_focus" else np.int32
                    )
                    for key in OBS_KEYS
                }
                v = float(scorer.score(batch, np.array([0]))[0])
                scores, _, _ = env._engine.flat_mc_scores(
                    gt_worlds,
                    gt_rollouts,
                    (seed * 7919 + gt_call) & 0xFFFFFFFFFFFFFFFF,
                    max_steps,
                )
                features = _state_features(env.last_raw_obs)
                features.update(
                    v=v,
                    gt=float(np.mean(scores)),
                    gt_best=float(np.max(scores)),
                    num_valid=float(num_valid),
                )
                rows.append(features)
                row_game.append(game_index)
                row_seat.append(acting)
            action = behavior.act(env, obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
        winners[game_index] = winner_from_info_or_obs(info, env.last_raw_obs)

    out: dict[str, np.ndarray] = {}
    if rows:
        for key in rows[0]:
            out[key] = np.asarray([r[key] for r in rows], dtype=np.float64)
    out["game_index"] = np.asarray(row_game, dtype=np.int64)
    out["seat"] = np.asarray(row_seat, dtype=np.int64)
    out["won"] = np.asarray(
        [
            1.0 if winners.get(g) == s else (0.5 if winners.get(g) is None else 0.0)
            for g, s in zip(row_game, row_seat)
        ],
        dtype=np.float64,
    )
    return out


def bucket_report(assessment: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    """Per-bucket V error vs rollout ground truth.

    Buckets are (board advantage sign) x (holding interaction) — the
    aggro-bias tripwire: rollout-derived value undervalues holding
    interaction, so a V trained on search self-play outcomes is predicted to
    be optimistic in board-ahead states and pessimistic in behind-but-
    holding-interaction states.
    """

    board = assessment["board_adv"]
    holding = assessment["holding_interaction"] > 0.5
    err = assessment["v"] - assessment["gt"]
    rows: list[dict[str, float | str]] = []
    for sign, sign_name in ((1, "ahead"), (0, "even"), (-1, "behind")):
        board_mask = np.sign(board) == sign
        for hold in (True, False):
            mask = board_mask & (holding == hold)
            n = int(mask.sum())
            row: dict[str, float | str] = {
                "bucket": f"{sign_name}/{'holding' if hold else 'no-instants'}",
                "n": n,
            }
            if n > 0:
                row.update(
                    mean_v=float(assessment["v"][mask].mean()),
                    mean_gt=float(assessment["gt"][mask].mean()),
                    bias=float(err[mask].mean()),
                    mae=float(np.abs(err[mask]).mean()),
                    spearman=(
                        spearman(assessment["v"][mask], assessment["gt"][mask])
                        if n >= 8
                        else float("nan")
                    ),
                )
            rows.append(row)
    return rows
