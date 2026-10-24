"""MTG-183 report: the heuristic curve, the checkpoint rating and saturation.

Reads the follower's durable results (`<directory>/ledger.json` and each done
unit's `rows.json`) and writes `<out>/summary.json`, `<out>/report.md` and the
Etude metric runs/pages under `<out>/etude`. The analysis is the frozen
protocol's (experiments/protocols/elemental-183.md, "Rating and analysis"):

- `fit_rows` turns every valid fit-pair and heuristic game into a
  `manabot/arena/rating.py` row (player A = newer checkpoint or the rated
  checkpoint against the heuristic; `deal_block` = the deal, shared by every
  pair or grid that plays it). `rating.fit_population` fits the seat-aware
  Bradley-Terry model with the heuristic anchored at 1,000 and
  `rating.bootstrap_population` resamples deals.
- `pair_table` gives each played pair's score, Wilson interval and the fit's
  prediction; held-out pairs are predicted, never fitted.
- `triangles` lists significant intransitive cycles; `prefix_stability` refits
  on each prefix of rated checkpoints; `saturation` applies the declared rule.

    uv run --extra artifacts --extra reports python -m experiments.elemental_long_report \
        --directory /Users/jack/.manabot/mtg183-follow/follow --out experiments/data/mtg183
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from pydantic import BaseModel

from experiments.elemental_grid import DECKS, GameRow, wilson
from experiments.elemental_long import SEED
from experiments.elemental_long_follow import H2HRow, Ledger, Unit
from manabot.arena.rating import (
    ELO_FACTOR,
    PopulationFit,
    bootstrap_population,
    fit_population,
)

ANCHOR = "heuristic"
REPLICATES = 2000


def player(updates: int) -> str:
    return f"u{updates:06d}"


@dataclass(frozen=True)
class Results:
    ledger: Ledger
    heuristic: dict[int, list[GameRow]]
    pairs: dict[tuple[int, int], tuple[Unit, list[H2HRow]]]

    @property
    def rated(self) -> list[int]:
        return sorted(self.heuristic)


def load(directory: Path) -> Results:
    ledger = Ledger.model_validate_json((directory / "ledger.json").read_bytes())
    heuristic: dict[int, list[GameRow]] = {}
    pairs: dict[tuple[int, int], tuple[Unit, list[H2HRow]]] = {}
    for unit in ledger.units.values():
        path = directory / "units" / unit.key / "rows.json"
        if unit.phase != "done" or not path.exists():
            continue
        raw = json.loads(path.read_text())
        if unit.kind == "heuristic":
            rows = [GameRow.model_validate(r) for r in raw]
            # An endpoint on a cadence update may hold both grids; keep the full one.
            if len(rows) >= len(heuristic.get(unit.newer, [])):
                heuristic[unit.newer] = rows
        else:
            assert unit.older is not None
            pairs[(unit.newer, unit.older)] = (
                unit,
                [H2HRow.model_validate(r) for r in raw],
            )
    return Results(ledger, heuristic, pairs)


# ---------------------------------------------------------------------------
# Rating rows and fit


def heuristic_rows(updates: int, rows: Sequence[GameRow]) -> list[dict[str, Any]]:
    return [
        {
            "player_a": player(updates),
            "player_b": ANCHOR,
            "player_a_seat": r.model_seat,
            "score_a": r.score,
            "truncated": False,
            "deal_block": r.deal,
            "deck_a": r.model_deck,
            "deck_b": r.heuristic_deck,
        }
        for r in rows
        if r.valid and r.score is not None
    ]


def h2h_rows(newer: int, older: int, rows: Sequence[H2HRow]) -> list[dict[str, Any]]:
    return [
        {
            "player_a": player(newer),
            "player_b": player(older),
            "player_a_seat": r.newer_seat,
            "score_a": r.score,
            "truncated": False,
            "deal_block": r.deal,
            "deck_a": r.newer_deck,
            "deck_b": r.older_deck,
        }
        for r in rows
        if r.valid and r.score is not None
    ]


def fit_rows(results: Results, *, through: int | None = None) -> list[dict[str, Any]]:
    """Every valid game of fit pairs and heuristic grids, optionally a prefix."""
    keep = (lambda u: True) if through is None else (lambda u: u <= through)
    rows = [
        row
        for updates, games in results.heuristic.items()
        if keep(updates)
        for row in heuristic_rows(updates, games)
    ]
    for (newer, older), (unit, games) in results.pairs.items():
        if unit.role != "held-out" and keep(newer):
            rows.extend(h2h_rows(newer, older, games))
    return rows


def predict(fit: PopulationFit, a: str, b: str, seat_a: int) -> float:
    seat = fit.seat0_elo if seat_a == 0 else -fit.seat0_elo
    eta = (fit.ratings[a] - fit.ratings[b] + seat) / ELO_FACTOR
    return 1.0 / (1.0 + math.exp(-eta))


class ScoreView(BaseModel):
    score: float | None
    low: float | None
    high: float | None
    valid: int
    invalid: int


def _score(rows: Sequence[GameRow] | Sequence[H2HRow]) -> ScoreView:
    # Wilson intervals over valid rows; `wilson` reads only valid/score.
    return ScoreView.model_validate(wilson(rows).model_dump())  # type: ignore[arg-type]


class PairView(BaseModel):
    newer: int
    older: int
    role: str
    score: ScoreView
    predicted: float | None
    pearson_z: float | None
    by_newer_deck: dict[str, ScoreView]
    by_newer_seat: dict[str, ScoreView]


def pair_table(results: Results, fit: PopulationFit | None) -> list[PairView]:
    views = []
    for (newer, older), (unit, games) in sorted(results.pairs.items()):
        valid = [r for r in games if r.valid and r.score is not None]
        predicted = z = None
        if fit is not None and {player(newer), player(older)} <= set(fit.ratings):
            probabilities = [
                predict(fit, player(newer), player(older), r.newer_seat) for r in valid
            ]
            if valid:
                predicted = float(np.mean(probabilities))
                spread = math.sqrt(sum(p * (1 - p) for p in probabilities))
                observed = sum(r.score or 0.0 for r in valid)
                z = (observed - sum(probabilities)) / spread if spread else None
        views.append(
            PairView(
                newer=newer,
                older=older,
                role=unit.role or "",
                score=_score(games),
                predicted=predicted,
                pearson_z=z,
                by_newer_deck={
                    d: _score([r for r in games if r.newer_deck == d]) for d in DECKS
                },
                by_newer_seat={
                    "play": _score([r for r in games if r.newer_seat == 0]),
                    "draw": _score([r for r in games if r.newer_seat == 1]),
                },
            )
        )
    return views


def triangles(pairs: Sequence[PairView]) -> list[tuple[int, int, int]]:
    """Cycles A > B > C > A where every edge's Wilson interval excludes 0.5."""
    beats: set[tuple[int, int]] = set()
    for p in pairs:
        s = p.score
        if s.low is None or s.high is None:
            continue
        if s.low > 0.5:
            beats.add((p.newer, p.older))
        elif s.high < 0.5:
            beats.add((p.older, p.newer))
    nodes = sorted({x for edge in beats for x in edge})
    found = []
    for a, b, c in itertools.combinations(nodes, 3):
        for x, y, z in ((a, b, c), (a, c, b)):
            if {(x, y), (y, z), (z, x)} <= beats:
                found.append((x, y, z))
    return found


class HeldOut(BaseModel):
    games: int
    log_loss: float | None
    baseline_log_loss: float
    calibration: list[dict[str, float]]


def held_out(results: Results, fit: PopulationFit) -> HeldOut:
    predictions, scores = [], []
    for (newer, older), (unit, games) in results.pairs.items():
        if unit.role != "held-out":
            continue
        for r in games:
            if r.valid and r.score is not None:
                predictions.append(
                    predict(fit, player(newer), player(older), r.newer_seat)
                )
                scores.append(r.score)
    p = np.clip(np.asarray(predictions), 1e-9, 1 - 1e-9)
    y = np.asarray(scores)
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else None
    calibration = []
    for low in np.arange(0.0, 1.0, 0.1):
        mask = (p >= low) & (p < low + 0.1)
        if mask.any():
            calibration.append(
                {
                    "bin_low": float(low),
                    "games": int(mask.sum()),
                    "predicted": float(p[mask].mean()),
                    "observed": float(y[mask].mean()),
                }
            )
    return HeldOut(
        games=len(y),
        log_loss=loss,
        baseline_log_loss=math.log(2.0),
        calibration=calibration,
    )


def prefix_stability(
    results: Results, full: PopulationFit
) -> dict[int, dict[str, float]]:
    """Each prefix fit's rating minus the full fit's, per checkpoint in it."""
    drift: dict[int, dict[str, float]] = {}
    for through in results.rated[1:]:
        rows = fit_rows(results, through=through)
        try:
            fit = fit_population(rows, anchor=ANCHOR)
        except ValueError:
            continue
        drift[through] = {
            name: fit.ratings[name] - full.ratings[name]
            for name in fit.ratings
            if name in full.ratings
        }
    return drift


# ---------------------------------------------------------------------------
# Heuristic curve and saturation


def _paired_difference(
    later: Sequence[GameRow],
    earlier: Sequence[GameRow],
    *,
    seed: int,
    replicates: int = REPLICATES,
) -> tuple[float, float, float] | None:
    """later − earlier over the deals both played validly; deal bootstrap."""
    a = {r.deal: r.score for r in later if r.valid and r.score is not None}
    b = {r.deal: r.score for r in earlier if r.valid and r.score is not None}
    deals = sorted(a.keys() & b.keys())
    if not deals:
        return None
    diff = np.asarray([a[d] - b[d] for d in deals])  # type: ignore[operator]
    rng = np.random.default_rng(seed)
    samples = diff[rng.integers(0, len(diff), (replicates, len(diff)))].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


# Amendment (2026-10-24, before any evaluation result): remaining heuristic gain
# below this margin, at 95% confidence, counts as saturated. The frozen strict
# rule (gain not significantly above zero) is also reported.
MARGIN = 0.03


class Saturation(BaseModel):
    scope: str
    rule: str
    updates: int | None
    heuristic_gap: dict[int, tuple[float, float, float]]
    endpoint_rating_gain: tuple[float, float, float] | None
    endpoint_pair: ScoreView | None
    exists: bool


def saturation(
    results: Results,
    endpoint: int,
    *,
    deck: str | None = None,
    boot: dict[str, Any] | None = None,
    margin: float | None = MARGIN,
) -> Saturation:
    """The protocol's U_h and whether self-play still gains after it.

    With `margin`, U_h is the earliest rated update whose remaining gain
    (endpoint − U_h) has a 95% upper bound below the margin; with None, the
    frozen strict rule: a 95% interval that includes zero or lies below it.
    Either way no later rated checkpoint may be significantly above the endpoint.
    """

    def rows(updates: int) -> list[GameRow]:
        found = results.heuristic[updates]
        return found if deck is None else [r for r in found if r.model_deck == deck]

    gap: dict[int, tuple[float, float, float]] = {}
    for updates in results.rated:
        if updates == endpoint:
            continue
        found = _paired_difference(rows(endpoint), rows(updates), seed=SEED + updates)
        if found is not None:
            gap[updates] = found
    u_h = None
    for updates in sorted(gap):
        later = [u for u in gap if u > updates]
        close = gap[updates][2] < margin if margin is not None else gap[updates][1] <= 0
        if close and all(gap[u][2] >= 0.0 for u in later):
            u_h = updates
            break
    rating_gain = None
    pair = None
    if u_h is not None:
        if boot is not None and deck is None:
            key = f"{player(endpoint)}__minus__{player(u_h)}"
            reverse = f"{player(u_h)}__minus__{player(endpoint)}"
            differences = boot["rating_differences"]
            if key in differences:
                rating_gain = tuple(differences[key])
            elif reverse in differences:
                low, mid, high = differences[reverse]
                rating_gain = (-high, -mid, -low)
        played = results.pairs.get((endpoint, u_h))
        if played is not None:
            games = played[1]
            pair = _score(
                games if deck is None else [r for r in games if r.newer_deck == deck]
            )
    exists = (
        u_h is not None
        and pair is not None
        and pair.low is not None
        and pair.low > 0.5
        and (deck is not None or (rating_gain is not None and rating_gain[0] > 0))
    )
    return Saturation(
        scope=deck or "overall",
        rule="strict" if margin is None else f"margin {margin}",
        updates=u_h,
        heuristic_gap=gap,
        endpoint_rating_gain=rating_gain,  # type: ignore[arg-type]
        endpoint_pair=pair,
        exists=exists,
    )


# ---------------------------------------------------------------------------
# Report


class CurvePoint(BaseModel):
    updates: int
    grid: str
    learner_transitions: float
    heuristic: ScoreView
    heuristic_by_deck: dict[str, ScoreView]
    heuristic_by_seat: dict[str, ScoreView]
    rating: float | None
    rating_interval: list[float] | None


class Summary(BaseModel):
    seed: int
    endpoint: int | None
    segment_outcome: str | None
    rated: list[int]
    curve: list[CurvePoint]
    seat0_elo: float | None
    seat0_interval: list[float] | None
    fit: dict[str, float | int | bool] | None
    pairs: list[PairView]
    triangles: list[tuple[int, int, int]]
    held_out: HeldOut | None
    prefix_drift: dict[int, dict[str, float]]
    saturation: list[Saturation]
    missing_units: list[str]


def summarize(results: Results, *, replicates: int = REPLICATES) -> Summary:
    rows = fit_rows(results)
    fit = boot = None
    if rows and len(results.heuristic) >= 1:
        fit = fit_population(rows, anchor=ANCHOR)
        if replicates:
            boot = bootstrap_population(
                rows, replicates=replicates, seed=SEED, anchor=ANCHOR
            )
    endpoint = results.ledger.endpoint
    curve = []
    for updates in results.rated:
        games = results.heuristic[updates]
        checkpoint = results.ledger.checkpoints.get(updates)
        name = player(updates)
        curve.append(
            CurvePoint(
                updates=updates,
                grid="full" if len(games) > 1000 else "subset",
                learner_transitions=float(
                    checkpoint.coordinates.get("learner_transitions", 0)  # type: ignore[arg-type]
                    if checkpoint
                    else 0
                ),
                heuristic=_score(games),
                heuristic_by_deck={
                    d: _score([r for r in games if r.model_deck == d]) for d in DECKS
                },
                heuristic_by_seat={
                    "play": _score([r for r in games if r.model_seat == 0]),
                    "draw": _score([r for r in games if r.model_seat == 1]),
                },
                rating=fit.ratings.get(name) if fit else None,
                rating_interval=boot["ratings"].get(name) if boot else None,
            )
        )
    pairs = pair_table(results, fit)
    scopes: list[str | None] = [None, *DECKS]
    return Summary(
        seed=SEED,
        endpoint=endpoint,
        segment_outcome=results.ledger.segment_outcome,
        rated=results.rated,
        curve=curve,
        seat0_elo=fit.seat0_elo if fit else None,
        seat0_interval=boot["seat0_elo"] if boot else None,
        fit=None
        if fit is None
        else {
            "converged": fit.converged,
            "iterations": fit.iterations,
            "log_loss": fit.log_loss,
            "deviance": fit.deviance,
            "games": len(rows),
            "bootstrap_failures": boot["failures"] if boot else 0,
        },
        pairs=pairs,
        triangles=triangles([p for p in pairs if p.role != "held-out"]),
        held_out=held_out(results, fit) if fit else None,
        prefix_drift=prefix_stability(results, fit) if fit else {},
        saturation=[
            saturation(results, endpoint, deck=d, boot=boot, margin=margin)
            for margin in (MARGIN, None)
            for d in scopes
        ]
        if endpoint is not None and endpoint in results.heuristic
        else [],
        missing_units=sorted(
            u.key for u in results.ledger.units.values() if u.phase != "done"
        ),
    )


def _pct(view: ScoreView) -> str:
    if view.score is None:
        return "—"
    return f"{view.score:.3f} [{view.low:.3f}, {view.high:.3f}]"


def markdown(summary: Summary) -> str:
    lines = [
        f"# MTG-183 seed {summary.seed}: heuristic and checkpoint rating",
        "",
        f"Endpoint update: {summary.endpoint}; Etude segment: "
        f"{summary.segment_outcome or 'not ended'}. "
        f"Seat-0 advantage {summary.seat0_elo:.1f} Elo."
        if summary.seat0_elo is not None
        else "",
        "",
        "| Update | Grid | Heuristic score [95%] | Rating (Elo, heuristic = 1000) [95%] |",
        "|---:|---|---|---|",
    ]
    for p in summary.curve:
        interval = (
            f" [{p.rating_interval[0]:.0f}, {p.rating_interval[2]:.0f}]"
            if p.rating_interval
            else ""
        )
        rating = "—" if p.rating is None else f"{p.rating:.0f}{interval}"
        lines.append(f"| {p.updates:,} | {p.grid} | {_pct(p.heuristic)} | {rating} |")
    lines += [
        "",
        "## Pairs",
        "",
        "| Newer | Older | Role | Score [95%] | Fit prediction | z |",
        "|---:|---:|---|---|---:|---:|",
    ]
    for p in summary.pairs:
        predicted = "—" if p.predicted is None else f"{p.predicted:.3f}"
        z = "—" if p.pearson_z is None else f"{p.pearson_z:+.2f}"
        lines.append(
            f"| {p.newer:,} | {p.older:,} | {p.role} | {_pct(p.score)} | {predicted} | {z} |"
        )
    lines += ["", f"Significant intransitive triangles: {summary.triangles or 'none'}"]
    if summary.held_out is not None:
        h = summary.held_out
        lines.append(
            f"Held-out pairs: {h.games} games, log loss {h.log_loss} vs {h.baseline_log_loss:.4f} at 0.5."
        )
    lines += ["", "## Saturation", ""]
    for s in summary.saturation:
        lines.append(
            f"- {s.scope} ({s.rule}): U_h = {s.updates}; endpoint v U_h "
            f"{_pct(s.endpoint_pair) if s.endpoint_pair else '—'}; rating gain "
            f"{s.endpoint_rating_gain}; exists: {s.exists}"
        )
    if summary.missing_units:
        lines += ["", f"Missing units: {len(summary.missing_units)}"]
    return "\n".join(lines) + "\n"


def etude(summary: Summary, out: Path) -> list[str]:
    """Etude metric runs (clock: updates) and their report pages."""
    from etude.run.records import Clock
    from etude.train.report import pages, write
    from etude.train.run import track

    staging = out / "etude-track.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    run = track(
        experiment="mtg183-elemental-long",
        run=f"rating-{summary.seed}",
        seed=summary.seed,
        kind="evaluation",
        clocks=[
            Clock(name="updates", kind="step", meaning="absolute training updates"),
            Clock(
                name="learner_transitions",
                kind="step",
                meaning="collected learner decisions",
            ),
        ],
        config={"anchor": ANCHOR, "endpoint": summary.endpoint},
        directory=staging,
    )
    for point in summary.curve:
        values = {"heuristic/overall": point.heuristic.score or 0.0}
        if point.rating is not None:
            values["rating/elo"] = point.rating
        for deck, view in point.heuristic_by_deck.items():
            values[f"heuristic/model_deck/{deck}"] = view.score or 0.0
        run.log(
            values,
            updates=point.updates,
            learner_transitions=point.learner_transitions,
        )
    run.finish()
    track_dir = out / "etude-track"
    shutil.rmtree(track_dir, ignore_errors=True)
    staging.replace(track_dir)
    names = []
    for name, page in pages(track_dir).items():
        write(page, out / "etude")
        names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--no-etude", action="store_true")
    args = parser.parse_args()
    summary = summarize(load(args.directory), replicates=args.replicates)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(summary.model_dump_json(indent=2))
    (args.out / "report.md").write_text(markdown(summary))
    if not args.no_etude:
        etude(summary, args.out)
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
