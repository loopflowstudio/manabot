"""Whole-game validation partitions and policy targets for supervised fitting."""

import numpy as np
import torch


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
