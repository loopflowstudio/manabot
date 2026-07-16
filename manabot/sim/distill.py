"""Search-as-teacher dataset generation and supervised distillation support.

Legacy datasets use flat determinized-MC scores and terminal outcomes. Tree
teachers add root visit counts and root values while retaining the same
viewer-safe observation, legal mask, chosen action, and provenance boundary.

Dataset format (one .npz shard per worker):
    - one array per observation key (shape (D, *obs_shape), float32) — exactly
      the encoded observation dict the Agent consumes at that decision;
    - "action" (D,) int16 — the searcher's argmax action index;
    - "game_index" (D,) int32, "seat" (D,) int8 — provenance;
    - "num_valid" (D,) int16 — count of valid actions at the decision;
    - "winner" (D,) int8 — winner seat of the source game (-1 if none).

Both players are searchers (self-play mirror), so every env step is one
teacher decision and both seats' decisions are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.sim.flat_mc import make_player, spec_name
from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs

OBS_KEYS: tuple[str, ...] = tuple(ObservationSpace().shapes.keys())
META_KEYS = ("action", "game_index", "seat", "num_valid", "winner")
# "scores" (D, max_actions) float32 — raw flat-MC win-probability estimates
# per action (engine order), -1.0 padding on invalid slots. Present in shards
# generated from exp-07 onward; loaders treat it as optional.
SCORE_KEY = "scores"
# Tree-search teachers write per-edge root visit counts and root values under
# these stable columns. Legacy flat-MC shards omit them.
VISIT_COUNT_KEY = "visit_counts"
ROOT_VALUE_KEY = "root_value"


# -----------------------------------------------------------------------------
# Task 1 — dataset generation (teacher self-play)
# -----------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        import subprocess

        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parent,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def generate_selfplay_shard(
    *,
    num_games: int,
    sims: int | None = None,
    teacher_spec: dict[str, Any] | None = None,
    seed: int,
    game_offset: int = 0,
    out_path: str | Path | None = None,
    max_steps_per_game: int = 5000,
    round_index: int = 0,
) -> dict[str, Any]:
    """Play teacher-vs-teacher self-play games, recording every decision.

    The teacher is any search player spec accepted by
    manabot.sim.flat_mc.make_player (flat search, policy search, or
    determinized PUCT); passing ``sims`` alone is shorthand for the exp-03
    random-rollout teacher. Every decision records the encoded observation,
    chosen action, and raw per-action values. Tree teachers additionally emit
    root visit counts and root values.

    Returns a summary dict (games, decisions, wall/search seconds, steps).
    When ``out_path`` is given the decisions are written there as an .npz.
    """

    if teacher_spec is None:
        if sims is None:
            raise ValueError("pass either sims or teacher_spec")
        teacher_spec = {"kind": "search", "sims": sims}
    teacher_name = spec_name(teacher_spec)

    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero=f"{teacher_name}-a"[:32],
            villain=f"{teacher_name}-b"[:32],
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
    players = [
        make_player(teacher_spec, seed=seed * 2 + 1)[0],
        make_player(teacher_spec, seed=seed * 2 + 2)[0],
    ]
    has_tree_targets = all(
        hasattr(player, "last_visit_counts") and hasattr(player, "last_root_value")
        for player in players
    )

    max_actions = obs_space.encoder.max_actions
    obs_buffers: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
    score_rows: list[np.ndarray] = []
    visit_rows: list[np.ndarray] = []
    root_values: list[float] = []
    actions: list[int] = []
    game_indices: list[int] = []
    seats: list[int] = []
    num_valids: list[int] = []
    winners_per_decision: list[list[int]] = []
    steps_per_game: list[int] = []
    winners: list[int | None] = []

    wall_start = time.perf_counter()
    for i in range(num_games):
        game_index = game_offset + i
        obs, _ = env.reset(seed=seed + game_index)
        done = False
        steps = 0
        info: dict[str, Any] = {}
        game_decisions: list[int] = []
        while not done and steps < max_steps_per_game:
            acting = int(env.last_raw_obs.agent.player_index)
            action = players[acting].act(env, obs)
            for key in OBS_KEYS:
                obs_buffers[key].append(np.asarray(obs[key], dtype=np.float32))
            encoded_valid_count = int(np.sum(obs["actions_valid"] > 0))
            if action < 0 or action >= max_actions or obs["actions_valid"][action] <= 0:
                raise RuntimeError(
                    "teacher chose an action outside the encoded legal-action ABI"
                )
            score_row = np.full(max_actions, -1.0, dtype=np.float32)
            raw_scores = players[acting].last_scores
            if raw_scores is not None:
                if len(raw_scores) != encoded_valid_count:
                    raise RuntimeError(
                        "teacher legal surface does not fit the encoded action ABI"
                    )
                score_row[: len(raw_scores)] = raw_scores
            score_rows.append(score_row)
            if has_tree_targets:
                raw_visits = players[acting].last_visit_counts
                root_value = players[acting].last_root_value
                if raw_visits is None or root_value is None:
                    raise RuntimeError("tree teacher did not publish search targets")
                visit_row = np.zeros(max_actions, dtype=np.float32)
                if len(raw_visits) != encoded_valid_count:
                    raise RuntimeError(
                        "tree visits do not fit the encoded legal-action ABI"
                    )
                visit_row[: len(raw_visits)] = raw_visits
                visit_rows.append(visit_row)
                root_values.append(float(root_value))
            actions.append(action)
            game_indices.append(game_index)
            seats.append(acting)
            num_valids.append(encoded_valid_count)
            game_decisions.append(len(actions) - 1)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated or truncated)
        winner = winner_from_info_or_obs(info, env.last_raw_obs) if done else None
        winners.append(winner)
        winners_per_decision.append(game_decisions)
        steps_per_game.append(steps)
    wall_seconds = time.perf_counter() - wall_start

    winner_column = np.full(len(actions), -1, dtype=np.int8)
    for game_decisions, winner in zip(winners_per_decision, winners):
        if winner is not None:
            winner_column[game_decisions] = winner

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
    if has_tree_targets:
        arrays[VISIT_COUNT_KEY] = (
            np.stack(visit_rows)
            if visit_rows
            else np.zeros((0, max_actions), dtype=np.float32)
        )
        arrays[ROOT_VALUE_KEY] = np.asarray(root_values, dtype=np.float32)

    # Provenance tag (expert-iteration staleness accounting, exp-07): who
    # generated these labels, with what rollout policy, at which round, from
    # which code. Self-play mirror, so the generating opponent is the teacher.
    import json as _json

    provenance = {
        "round": round_index,
        "teacher_spec": {k: v for k, v in teacher_spec.items() if k != "device"},
        "teacher_name": teacher_name,
        "rollout_policy_checkpoint": teacher_spec.get("checkpoint"),
        "generating_opponent": teacher_name,
        "git_commit": _git_commit(),
        "seed": seed,
        "game_offset": game_offset,
        "policy_target_kind": (
            "visit_distribution" if has_tree_targets else "score_softmax"
        ),
        "value_target_kind": "root_value" if has_tree_targets else "terminal_outcome",
    }
    arrays["provenance"] = np.array(_json.dumps(provenance))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)

    search_stats = {
        "decisions": players[0].stats.decisions + players[1].stats.decisions,
        "seconds": players[0].stats.seconds + players[1].stats.seconds,
        "simulations": players[0].stats.simulations + players[1].stats.simulations,
        "cap_hits": players[0].stats.cap_hits + players[1].stats.cap_hits,
    }
    if has_tree_targets:
        search_stats.update(
            tree_nodes=players[0].stats.tree_nodes + players[1].stats.tree_nodes,
            worlds_sampled=(
                players[0].stats.worlds_sampled + players[1].stats.worlds_sampled
            ),
            max_depth_sum=(
                players[0].stats.max_depth_sum + players[1].stats.max_depth_sum
            ),
            max_depth_max=max(
                players[0].stats.max_depth_max, players[1].stats.max_depth_max
            ),
        )
    return {
        "teacher": teacher_name,
        "provenance": provenance,
        "num_games": num_games,
        "decisions": len(actions),
        "wall_seconds": wall_seconds,
        "search": search_stats,
        "policy_target_kind": provenance["policy_target_kind"],
        "value_target_kind": provenance["value_target_kind"],
        "steps_per_game": steps_per_game,
        "winners": [w if w is not None else -1 for w in winners],
        "out_path": str(out_path) if out_path is not None else None,
    }


# -----------------------------------------------------------------------------
# Task 2 — behavior cloning
# -----------------------------------------------------------------------------


def load_shards(
    paths: list[str | Path],
    *,
    rounds: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Load and concatenate dataset shards.

    Adds a per-decision "round" column (expert-iteration staleness
    diagnostic): taken from ``rounds`` (one per shard) when given, else from
    each shard's embedded provenance tag, else -1 for legacy shards.
    """

    import json as _json

    shards = [np.load(Path(p)) for p in paths]
    keys = list(OBS_KEYS) + list(META_KEYS)
    for optional_key in (SCORE_KEY, VISIT_COUNT_KEY, ROOT_VALUE_KEY):
        if all(optional_key in shard for shard in shards):
            keys.append(optional_key)
    out = {key: np.concatenate([s[key] for s in shards]) for key in keys}

    round_cols = []
    for i, shard in enumerate(shards):
        if rounds is not None:
            shard_round = int(rounds[i])
        elif "provenance" in shard:
            shard_round = int(_json.loads(str(shard["provenance"])).get("round", -1))
        else:
            shard_round = -1
        round_cols.append(np.full(len(shard["action"]), shard_round, dtype=np.int16))
    out["round"] = np.concatenate(round_cols)

    # Game indices are only unique within a round's shard set; offset them
    # per round so the by-game train/val split never merges games across
    # rounds.
    unique_rounds = np.unique(out["round"])
    if len(unique_rounds) > 1:
        game_index = out["game_index"].astype(np.int64)
        offset = 0
        for r in unique_rounds:
            rows = out["round"] == r
            game_index[rows] += offset
            offset = int(game_index[rows].max()) + 1
        out["game_index"] = game_index
    return out


def split_by_game(
    dataset: dict[str, np.ndarray],
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/val decision indices, split by game to avoid leakage."""

    games = np.unique(dataset["game_index"])
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    num_val = max(1, int(round(len(games) * val_fraction)))
    val_games = set(games[:num_val].tolist())
    is_val = np.isin(dataset["game_index"], list(val_games))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


@dataclass
class BCEpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_accuracy_nontrivial: float
    # Staleness diagnostic (exp-07): validation loss on the freshest round's
    # decisions only, vs the aggregate val_loss above. Only populated when
    # the training data spans multiple rounds; divergence between the two
    # curves is the pre-registered trigger for replay-window tuning.
    val_loss_fresh: float | None = None


def _batch_tensors(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    obs = {
        key: torch.as_tensor(dataset[key][indices], dtype=torch.float32, device=device)
        for key in OBS_KEYS
    }
    target = torch.as_tensor(
        dataset["action"][indices], dtype=torch.long, device=device
    )
    return obs, target


def soft_targets_from_scores(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Distribution over actions from raw flat-MC scores.

    Scores are Monte Carlo win-probability estimates in [0, 1] (padding
    -1.0 marks invalid actions), so p ∝ exp(score / τ) over valid actions.
    τ trades off teacher sharpness against playout noise: at N sims/action
    the per-score standard error is ~sqrt(p(1-p)/N) (≈0.03 at N=256), so τ
    well below that amplifies noise, τ >> score spreads flattens the teacher
    away. Exp-07 sweeps τ; see experiments/exp-07-expert-iteration.md.
    """

    invalid = scores < 0
    logits = (scores / temperature).masked_fill(invalid, -1e9)
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def evaluate_bc(
    agent: Agent,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> tuple[float, float, float]:
    """Return (mean CE loss, accuracy, accuracy on non-trivial decisions)."""

    device = device or torch.device("cpu")
    agent.eval()
    total_loss = 0.0
    correct = 0
    nontrivial_correct = 0
    nontrivial_total = 0
    nontrivial_mask = dataset["num_valid"][indices] > 1
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        obs, target = _batch_tensors(dataset, batch, device)
        logits, _ = agent.forward(obs)
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
        total_loss += float(loss.item())
        pred = logits.argmax(dim=-1)
        hits = (pred == target).cpu().numpy()
        correct += int(hits.sum())
        batch_nontrivial = nontrivial_mask[start : start + batch_size]
        nontrivial_correct += int(hits[batch_nontrivial].sum())
        nontrivial_total += int(batch_nontrivial.sum())
    n = max(1, len(indices))
    return (
        total_loss / n,
        correct / n,
        nontrivial_correct / max(1, nontrivial_total),
    )


def train_bc(
    dataset: dict[str, np.ndarray],
    *,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 512,
    val_fraction: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    agent_hypers: AgentHypers | None = None,
    soft_temperature: float | None = None,
    initial_agent_state: dict[str, Any] | None = None,
    log: bool = False,
) -> tuple[Agent, ObservationSpace, list[BCEpochStats]]:
    """Behavior-clone a fresh Agent on (observation, search action) pairs.

    Cross-entropy on the Agent's masked logits (invalid actions are already
    filled with -1e8 inside Agent.forward, so the distribution is over valid
    actions only). With ``soft_temperature`` set (requires the dataset's
    "scores" column), the target is the softened flat-MC score distribution
    instead of the one-hot argmax. Validation metrics are always measured
    against the teacher argmax so runs stay comparable across temperatures.
    The train/val split is by game. ``initial_agent_state`` warm-starts the
    student (fine-tuning) instead of a fresh init.
    """

    if soft_temperature is not None and SCORE_KEY not in dataset:
        raise ValueError("soft_temperature requires a dataset with scores")

    torch.manual_seed(seed)
    dev = torch.device(device)
    obs_space = ObservationSpace()
    agent = Agent(obs_space, agent_hypers or AgentHypers()).to(dev)
    if initial_agent_state is not None:
        agent.load_state_dict(initial_agent_state)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    train_idx, val_idx = split_by_game(dataset, val_fraction=val_fraction, seed=seed)
    rng = np.random.default_rng(seed)
    history: list[BCEpochStats] = []

    # Staleness diagnostic: freshest-round-only val subset, when the data
    # spans rounds (expert-iteration aggregation).
    fresh_val_idx: np.ndarray | None = None
    if "round" in dataset and len(np.unique(dataset["round"])) > 1:
        fresh_round = int(dataset["round"].max())
        fresh_val_idx = val_idx[dataset["round"][val_idx] == fresh_round]
        if len(fresh_val_idx) == 0:
            fresh_val_idx = None

    for epoch in range(epochs):
        agent.train()
        order = train_idx.copy()
        rng.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            obs, target = _batch_tensors(dataset, batch, dev)
            logits, _ = agent.forward(obs)
            if soft_temperature is not None:
                scores = torch.as_tensor(
                    dataset[SCORE_KEY][batch], dtype=torch.float32, device=dev
                )
                soft = soft_targets_from_scores(scores, soft_temperature)
                loss = torch.nn.functional.cross_entropy(logits, soft)
            else:
                loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
        val_loss, val_acc, val_acc_nontrivial = evaluate_bc(
            agent, dataset, val_idx, batch_size=batch_size, device=dev
        )
        val_loss_fresh = None
        if fresh_val_idx is not None:
            val_loss_fresh, _, _ = evaluate_bc(
                agent, dataset, fresh_val_idx, batch_size=batch_size, device=dev
            )
        stats = BCEpochStats(
            epoch=epoch,
            train_loss=total_loss / max(1, len(order)),
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_accuracy_nontrivial=val_acc_nontrivial,
            val_loss_fresh=val_loss_fresh,
        )
        history.append(stats)
        if log:
            fresh_part = (
                f" val_loss_fresh {stats.val_loss_fresh:.4f}"
                if stats.val_loss_fresh is not None
                else ""
            )
            print(
                f"  epoch {epoch}: train_loss {stats.train_loss:.4f} "
                f"val_loss {stats.val_loss:.4f}{fresh_part} "
                f"val_acc {stats.val_accuracy:.4f} "
                f"val_acc_nontrivial {stats.val_accuracy_nontrivial:.4f}",
                flush=True,
            )

    return agent, obs_space, history


def save_bc_checkpoint(
    agent: Agent,
    obs_space: ObservationSpace,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a BC policy in the trainer checkpoint format.

    The saved file loads through manabot.sim.flat_mc.load_checkpoint_agent, so
    the BC policy plugs into every existing matchup/evaluation harness.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "global_step": 0,
            "hypers": {
                "agent_hypers": agent.hypers.model_dump(),
                "observation_hypers": obs_space.encoder.hypers.model_dump(),
                "train_hypers": {},
            },
            "bc": extra or {},
        },
        path,
    )
