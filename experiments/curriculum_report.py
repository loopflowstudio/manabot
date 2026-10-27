"""Paired estimates for MTG-175's fixed evaluation grids.

`compare` consumes the same GameRow contract as MTG-183's evaluator. Both
seats of a deal stay together when resampling, within each ordered deck cell;
cells and seats have equal weight. It reports no primary contrast until every
planned game of all four arms is valid. Missing/invalid games remain counted,
with worst-case bounds over the complete target population. These intervals describe
evaluation-deal uncertainty for this one seed, not training-seed uncertainty.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from experiments.elemental_grid import Game, GameRow

Arm = Literal["U", "M", "E", "ME"]
ARMS: tuple[Arm, ...] = ("U", "M", "E", "ME")
type GameKey = tuple[str, str, int, int]


@dataclass(frozen=True)
class Coverage:
    expected: int
    received: int
    valid: int
    invalid: int
    missing: int


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class Bounds:
    low: float
    high: float


@dataclass(frozen=True)
class Comparison:
    coverage: dict[Arm, Coverage]
    scores: dict[Arm, float]
    contrasts: dict[str, Estimate]
    complete: bool
    score_bounds: dict[Arm, Bounds]
    contrast_bounds: dict[str, Bounds]


def _key(row: Game | GameRow) -> GameKey:
    return row.model_deck, row.heuristic_deck, row.deal, row.model_seat


def compare(
    schedule: Sequence[Game],
    results: Mapping[Arm, Sequence[GameRow]],
    *,
    seed: int,
    replicates: int = 2000,
) -> Comparison:
    """Complete-grid scores and M−U, E−U, ME−U, ME−M−E+U intervals.

    Reject duplicates and off-schedule rows instead of choosing a preferred
    attempt. An incomplete grid returns coverage and worst-case bounds, with no point
    scores or bootstrap contrasts. Bounds assign each missing/invalid score
    any value in [0, 1]; they are not confidence intervals.
    Percentile intervals resample deals independently within each cell, using
    the same sampled deals for every arm and both native seats.
    """
    if not schedule or replicates < 2:
        raise ValueError("a nonempty schedule and at least two replicates are required")
    planned = {_key(row) for row in schedule}
    if len(planned) != len(schedule):
        raise ValueError("duplicate scheduled game")
    cells = sorted({key[:2] for key in planned})
    deals = {
        cell: sorted({key[2] for key in planned if key[:2] == cell}) for cell in cells
    }
    if any(
        (*cell, deal, seat) not in planned
        for cell in cells
        for deal in deals[cell]
        for seat in (0, 1)
    ):
        raise ValueError("each scheduled deal must include both seats")
    if set(results) != set(ARMS):
        raise ValueError("all four arms must be present, even if their rows are empty")
    coverage: dict[Arm, Coverage] = {}
    indexed: dict[Arm, dict[GameKey, float]] = {}
    for arm in ARMS:
        seen: set[GameKey] = set()
        valid: dict[GameKey, float] = {}
        for row in results[arm]:
            key = _key(row)
            if key not in planned or key in seen:
                raise ValueError(f"off-schedule or duplicate game in {arm}: {key}")
            seen.add(key)
            if row.valid:
                if (
                    row.score is None
                    or not math.isfinite(row.score)
                    or not 0 <= row.score <= 1
                ):
                    raise ValueError("a valid game requires a finite score in [0, 1]")
                valid[key] = row.score
        indexed[arm] = valid
        coverage[arm] = Coverage(
            len(planned),
            len(seen),
            len(valid),
            len(seen) - len(valid),
            len(planned) - len(seen),
        )
    weights = {
        "M-U": (-1, 1, 0, 0),
        "E-U": (-1, 0, 1, 0),
        "ME-U": (-1, 0, 0, 1),
        "ME-M-E+U": (1, -1, -1, 1),
    }
    score_bounds: dict[Arm, Bounds] = {}
    for arm in ARMS:
        lower = upper = 0.0
        for cell in cells:
            keys = [(*cell, deal, seat) for deal in deals[cell] for seat in (0, 1)]
            lower += (
                sum(indexed[arm].get(key, 0.0) for key in keys) / len(keys) / len(cells)
            )
            upper += (
                sum(indexed[arm].get(key, 1.0) for key in keys) / len(keys) / len(cells)
            )
        score_bounds[arm] = Bounds(lower, upper)
    contrast_bounds = {
        name: Bounds(
            sum(
                w * (score_bounds[a].low if w > 0 else score_bounds[a].high)
                for a, w in zip(ARMS, coefficients, strict=True)
            ),
            sum(
                w * (score_bounds[a].high if w > 0 else score_bounds[a].low)
                for a, w in zip(ARMS, coefficients, strict=True)
            ),
        )
        for name, coefficients in weights.items()
    }
    if any(row.valid != row.expected for row in coverage.values()):
        return Comparison(coverage, {}, {}, False, score_bounds, contrast_bounds)
    # Deal values average both seats; cell means then receive equal mass.
    scores = np.zeros(4)
    draws = np.zeros((replicates, 4))
    rng = np.random.default_rng(seed)
    for cell in cells:
        values = np.asarray(
            [
                [
                    sum(indexed[arm][(*cell, deal, seat)] for seat in (0, 1)) / 2
                    for arm in ARMS
                ]
                for deal in deals[cell]
            ]
        )
        scores += values.mean(axis=0) / len(cells)
        selected = rng.integers(len(values), size=(replicates, len(values)))
        draws += values[selected].mean(axis=1) / len(cells)
    contrasts: dict[str, Estimate] = {}
    for name, coefficients in weights.items():
        vector = np.asarray(coefficients)
        low, high = np.percentile(draws @ vector, [2.5, 97.5])
        contrasts[name] = Estimate(float(scores @ vector), float(low), float(high))
    return Comparison(
        coverage,
        dict(zip(ARMS, map(float, scores), strict=True)),
        contrasts,
        True,
        score_bounds,
        contrast_bounds,
    )
