"""Seat-aware Gaussian-MAP Bradley-Terry ratings and paired-block bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

ELO_FACTOR = 400.0 / math.log(10.0)


@dataclass(frozen=True)
class PopulationFit:
    players: tuple[str, ...]
    ratings: dict[str, float]
    seat0_elo: float
    converged: bool
    iterations: int
    gradient_norm: float
    hessian_condition: float
    log_loss: float
    deviance: float
    rows: tuple[dict[str, Any], ...]


def _design(
    rows: list[dict[str, Any]], players: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    index = {player: i for i, player in enumerate(players)}
    matrix = np.zeros((len(rows), len(players) + 1), dtype=np.float64)
    scores = np.zeros(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows):
        matrix[row_index, index[str(row["player_a"])]] = 1.0
        matrix[row_index, index[str(row["player_b"])]] = -1.0
        matrix[row_index, -1] = 1.0 if int(row["player_a_seat"]) == 0 else -1.0
        scores[row_index] = float(row["score_a"])
    return matrix, scores


def fit_population(
    rows: Iterable[dict[str, Any]],
    *,
    anchor: str = "random-v1",
    prior_elo_std: float = 400.0,
    weights: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> PopulationFit:
    retained = [
        dict(row)
        for row in rows
        if not row.get("truncated") and row.get("termination_reason") != "truncated"
    ]
    players = tuple(
        sorted(
            {str(row["player_a"]) for row in retained}
            | {str(row["player_b"]) for row in retained}
        )
    )
    if len(players) < 2 or anchor not in players:
        raise ValueError("rating requires at least two players and the random anchor")
    matrix, scores = _design(retained, players)
    sample_weights = (
        np.ones(len(retained))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if sample_weights.shape != scores.shape:
        raise ValueError("weights shape mismatch")
    theta = np.zeros(len(players) + 1, dtype=np.float64)
    prior_sd = prior_elo_std / ELO_FACTOR
    precision = 1.0 / (prior_sd * prior_sd)
    converged = False
    gradient_norm = math.inf
    hessian = np.eye(len(theta))
    for iteration in range(1, max_iterations + 1):
        eta = np.clip(matrix @ theta, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-eta))
        gradient = (
            matrix.T @ (sample_weights * (probabilities - scores)) + precision * theta
        )
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = matrix.T @ (matrix * curvature[:, None]) + precision * np.eye(
            len(theta)
        )
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= tolerance:
            converged = True
            break
        theta -= np.linalg.solve(hessian, gradient)
    skills = theta[:-1] * ELO_FACTOR
    offset = 1000.0 - skills[players.index(anchor)]
    ratings = {player: float(skills[i] + offset) for i, player in enumerate(players)}
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(matrix @ theta, -40.0, 40.0)))
    eps = 1e-15
    log_loss = -float(
        np.sum(
            sample_weights
            * (
                scores * np.log(probabilities + eps)
                + (1.0 - scores) * np.log(1.0 - probabilities + eps)
            )
        )
        / np.sum(sample_weights)
    )
    residual_rows = []
    for row, predicted in zip(retained, probabilities, strict=True):
        score = float(row["score_a"])
        variance = max(float(predicted * (1.0 - predicted)), eps)
        deviance = 2.0 * (
            (score * math.log(score / predicted) if score > 0.0 else 0.0)
            + (
                (1.0 - score) * math.log((1.0 - score) / (1.0 - predicted))
                if score < 1.0
                else 0.0
            )
        )
        residual_rows.append(
            {
                **row,
                "predicted_score_a": float(predicted),
                "raw_residual": float(score - predicted),
                "pearson_residual": float((score - predicted) / math.sqrt(variance)),
                "deviance": float(deviance),
            }
        )
    return PopulationFit(
        players,
        ratings,
        float(theta[-1] * ELO_FACTOR),
        converged,
        iteration,
        gradient_norm,
        float(np.linalg.cond(hessian)),
        log_loss,
        float(sum(row["deviance"] for row in residual_rows)),
        tuple(residual_rows),
    )


def bootstrap_population(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    anchor: str = "random-v1",
    prior_elo_std: float = 400.0,
) -> dict[str, Any]:
    blocks = sorted({int(row["deal_block"]) for row in rows})
    if not blocks:
        raise ValueError("bootstrap requires deal blocks")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    difference_samples: dict[str, list[float]] = {}
    seat_samples: list[float] = []
    failures = 0
    row_blocks = np.asarray([int(row["deal_block"]) for row in rows])
    for _ in range(replicates):
        selected = rng.choice(blocks, size=len(blocks), replace=True)
        counts = {block: int(np.count_nonzero(selected == block)) for block in blocks}
        weights = np.asarray(
            [counts[int(block)] for block in row_blocks], dtype=np.float64
        )
        try:
            fit = fit_population(
                rows, anchor=anchor, prior_elo_std=prior_elo_std, weights=weights
            )
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        if not fit.converged:
            failures += 1
            continue
        for player, rating in fit.ratings.items():
            samples.setdefault(player, []).append(rating)
        for left_index, left in enumerate(fit.players):
            for right in fit.players[left_index + 1 :]:
                key = f"{left}__minus__{right}"
                difference_samples.setdefault(key, []).append(
                    fit.ratings[left] - fit.ratings[right]
                )
        seat_samples.append(fit.seat0_elo)

    def interval(values: list[float]) -> list[float]:
        return (
            [float(value) for value in np.percentile(values, [2.5, 50.0, 97.5])]
            if values
            else [math.nan] * 3
        )

    return {
        "replicates": replicates,
        "failures": failures,
        "ratings": {player: interval(values) for player, values in samples.items()},
        "rating_differences": {
            pair: interval(values) for pair, values in difference_samples.items()
        },
        "seat0_elo": interval(seat_samples),
    }


def payoff_matrix(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    retained = [
        dict(row)
        for row in rows
        if not row.get("truncated") and row.get("termination_reason") != "truncated"
    ]
    fit = fit_population(retained)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in fit.rows:
        a, b = sorted((str(row["player_a"]), str(row["player_b"])))
        cells.setdefault((a, b), []).append(dict(row))
    result = {}
    for (a, b), cell in cells.items():
        scores = [
            float(row["score_a"])
            if row["player_a"] == a
            else 1.0 - float(row["score_a"])
            for row in cell
        ]
        predicted_scores = [
            float(row["predicted_score_a"])
            if row["player_a"] == a
            else 1.0 - float(row["predicted_score_a"])
            for row in cell
        ]
        seat_results = {}
        for seat in (0, 1):
            seat_scores = [
                score
                for row, score in zip(cell, scores, strict=True)
                if (
                    int(row["player_a_seat"])
                    if row["player_a"] == a
                    else 1 - int(row["player_a_seat"])
                )
                == seat
            ]
            seat_results[str(seat)] = {
                "games": len(seat_scores),
                "score_a": float(sum(seat_scores)),
                "mean_score_a": float(np.mean(seat_scores)) if seat_scores else None,
            }
        paired: dict[int, list[float]] = {}
        for row, score in zip(cell, scores, strict=True):
            paired.setdefault(int(row["deal_block"]), []).append(score)
        sweeps_a = sum(values == [1.0, 1.0] for values in paired.values())
        sweeps_b = sum(values == [0.0, 0.0] for values in paired.values())
        splits = sum(sorted(values) == [0.0, 1.0] for values in paired.values())
        paired_draws = len(paired) - sweeps_a - sweeps_b - splits
        observed = float(np.sum(scores))
        expected = float(np.sum(predicted_scores))
        variance = float(np.sum([value * (1.0 - value) for value in predicted_scores]))
        result[f"{a}__{b}"] = {
            "player_a": a,
            "player_b": b,
            "games": len(cell),
            "score_a": observed,
            "mean_score_a": float(np.mean(scores)),
            "wins_a": sum(score == 1.0 for score in scores),
            "draws": sum(score == 0.5 for score in scores),
            "wins_b": sum(score == 0.0 for score in scores),
            "per_seat": seat_results,
            "paired_blocks": {
                "sweeps_a": sweeps_a,
                "splits": splits,
                "sweeps_b": sweeps_b,
                "draw_or_mixed": paired_draws,
            },
            "expected_score_a": expected,
            "expected_mean_score_a": expected / len(cell),
            "raw_residual_percentage_points": 100.0 * (observed - expected) / len(cell),
            "pearson_residual": (observed - expected) / math.sqrt(variance)
            if variance > 0.0
            else 0.0,
            "cell_deviance": float(sum(float(row["deviance"]) for row in cell)),
        }
    return result
