"""Search-as-teacher distillation: dataset generation + behavior cloning.

Exp-03 (wave/search C4). The teacher is the flat determinized MC searcher from
exp-02 (manabot.sim.flat_mc.FlatMCPlayer); the student is a fresh
manabot.model.agent.Agent trained by cross-entropy on the teacher's chosen
action at every recorded decision.

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
from manabot.sim.flat_mc import FlatMCPlayer, spec_name
from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs

OBS_KEYS: tuple[str, ...] = tuple(ObservationSpace().shapes.keys())
META_KEYS = ("action", "game_index", "seat", "num_valid", "winner")


# -----------------------------------------------------------------------------
# Task 1 — dataset generation (teacher self-play)
# -----------------------------------------------------------------------------


def generate_selfplay_shard(
    *,
    num_games: int,
    sims: int,
    seed: int,
    game_offset: int = 0,
    out_path: str | Path | None = None,
    max_steps_per_game: int = 5000,
) -> dict[str, Any]:
    """Play search-vs-search self-play games, recording every decision.

    Returns a summary dict (games, decisions, wall/search seconds, steps).
    When ``out_path`` is given the decisions are written there as an .npz.
    """

    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero=f"search-{sims}-a"[:32],
            villain=f"search-{sims}-b"[:32],
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
        FlatMCPlayer(sims, seed=seed * 2 + 1),
        FlatMCPlayer(sims, seed=seed * 2 + 2),
    ]

    obs_buffers: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
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
            actions.append(action)
            game_indices.append(game_index)
            seats.append(acting)
            num_valids.append(int(np.sum(obs["actions_valid"] > 0)))
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
    return {
        "teacher": spec_name({"kind": "search", "sims": sims}),
        "num_games": num_games,
        "decisions": len(actions),
        "wall_seconds": wall_seconds,
        "search": search_stats,
        "steps_per_game": steps_per_game,
        "winners": [w if w is not None else -1 for w in winners],
        "out_path": str(out_path) if out_path is not None else None,
    }


# -----------------------------------------------------------------------------
# Task 2 — behavior cloning
# -----------------------------------------------------------------------------


def load_shards(paths: list[str | Path]) -> dict[str, np.ndarray]:
    """Load and concatenate dataset shards."""

    shards = [np.load(Path(p)) for p in paths]
    keys = list(OBS_KEYS) + list(META_KEYS)
    return {key: np.concatenate([s[key] for s in shards]) for key in keys}


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
    log: bool = False,
) -> tuple[Agent, ObservationSpace, list[BCEpochStats]]:
    """Behavior-clone a fresh Agent on (observation, search action) pairs.

    Cross-entropy on the Agent's masked logits (invalid actions are already
    filled with -1e8 inside Agent.forward, so the distribution is over valid
    actions only). The train/val split is by game.
    """

    torch.manual_seed(seed)
    dev = torch.device(device)
    obs_space = ObservationSpace()
    agent = Agent(obs_space, agent_hypers or AgentHypers()).to(dev)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    train_idx, val_idx = split_by_game(dataset, val_fraction=val_fraction, seed=seed)
    rng = np.random.default_rng(seed)
    history: list[BCEpochStats] = []

    for epoch in range(epochs):
        agent.train()
        order = train_idx.copy()
        rng.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            obs, target = _batch_tensors(dataset, batch, dev)
            logits, _ = agent.forward(obs)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
        val_loss, val_acc, val_acc_nontrivial = evaluate_bc(
            agent, dataset, val_idx, batch_size=batch_size, device=dev
        )
        stats = BCEpochStats(
            epoch=epoch,
            train_loss=total_loss / max(1, len(order)),
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_accuracy_nontrivial=val_acc_nontrivial,
        )
        history.append(stats)
        if log:
            print(
                f"  epoch {epoch}: train_loss {stats.train_loss:.4f} "
                f"val_loss {stats.val_loss:.4f} val_acc {stats.val_accuracy:.4f} "
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
