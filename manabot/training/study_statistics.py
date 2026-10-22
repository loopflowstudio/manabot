"""Descriptive paired uncertainty for saved, replay-verified study cohorts.

Shared by portable report readers and historical experiment analysis. The
bootstrap resamples training seeds and common four-leg deals, not individual
games; it never substitutes for missing or incomplete evidence.
"""

from typing import TypedDict

import numpy as np


class GameRow(TypedDict):
    failure: str | None
    terminated: bool
    truncated: bool
    replay_passed: bool
    deal_seed: int
    score_a: float


class ComparisonCell(TypedDict):
    a: str
    b: str
    cutoff: int
    phase: str
    variant: str
    training_seed: int | None
    b_training_seed: int | None
    replay: dict[str, int | bool]
    rows: list[GameRow]
    scheduled_games: int


class BootstrapResult(TypedDict, total=False):
    a: str
    b: str
    cutoff: int
    phase: str
    variant: str
    status: str
    reason: str
    score_a: float
    interval_95: list[float]
    training_seeds: int
    deal_blocks: int
    method: str


def valid_game(row: GameRow) -> bool:
    """Only authoritative terminal outcomes with exact replay can be scored."""
    return (
        row["failure"] is None
        and row["terminated"]
        and not row["truncated"]
        and row["replay_passed"]
    )


def paired_uncertainty(comparisons: list[ComparisonCell]) -> list[BootstrapResult]:
    """Bootstrap training seeds and common four-leg deals, never individual games."""
    groups: dict[tuple[str, str, int, str, str], list[ComparisonCell]] = {}
    for cell in comparisons:
        groups.setdefault(
            (
                cell["a"],
                cell["b"],
                cell["cutoff"],
                cell.get("phase", "development"),
                cell.get("variant", "raw"),
            ),
            [],
        ).append(cell)
    results: list[BootstrapResult] = []
    for (a, b, cutoff, phase, variant), cells in sorted(groups.items()):
        result: BootstrapResult = dict(
            a=a, b=b, cutoff=cutoff, phase=phase, variant=variant, status="unavailable"
        )
        results.append(result)
        seeds = {c.get("training_seed") for c in cells}
        if None in seeds or len(seeds) < 3:
            result["reason"] = "requires at least three distinct training seeds"
            continue
        if any(
            not c["replay"]["passed"]
            or len(c["rows"]) != c["scheduled_games"]
            or not all(valid_game(r) for r in c["rows"])
            for c in cells
        ):
            result["reason"] = "incomplete cohort"
            continue
        blocks = {}
        for cell in cells:
            deals = {}
            for row in cell["rows"]:
                deals.setdefault(row["deal_seed"], []).append(row["score_a"])
            key = (cell["training_seed"], cell.get("b_training_seed"))
            if key in blocks:
                raise ValueError("duplicate training seed cell")
            blocks[key] = deals
        deals = sorted(next(iter(blocks.values())))
        if any(
            set(block) != set(deals) or any(len(v) != 4 for v in block.values())
            for block in blocks.values()
        ):
            result["reason"] = "unmatched four-leg deal blocks"
            continue
        a_seeds = sorted(seeds)
        b_seeds = {key[1] for key in blocks}
        crossed = None not in b_seeds and len(blocks) == len(a_seeds) * len(b_seeds)
        if crossed:
            b_seeds = sorted(b_seeds)
            scores = np.array(
                [
                    [[np.mean(blocks[(sa, sb)][d]) for d in deals] for sb in b_seeds]
                    for sa in a_seeds
                ]
            )
        elif len(blocks) == len(a_seeds) and (
            b_seeds == {None} or all(sa == sb for sa, sb in blocks)
        ):
            scores = np.array(
                [[np.mean(blocks[key][d]) for d in deals] for key in sorted(blocks)]
            )
        else:
            result["reason"] = "seed schedule is neither complete crossed nor paired"
            continue
        rng = np.random.default_rng(0)
        estimates = []
        for _ in range(2000):
            indexes = [rng.integers(n, size=n) for n in scores.shape]
            estimates.append(float(scores[np.ix_(*indexes)].mean()))
        result.update(
            status="available",
            score_a=float(scores.mean()),
            interval_95=np.quantile(estimates, [0.025, 0.975]).tolist(),
            training_seeds=len(seeds),
            deal_blocks=len(deals),
            method=("crossed" if crossed else "paired")
            + " training-seed/common-deal bootstrap; descriptive at small seed count",
        )
    return results
