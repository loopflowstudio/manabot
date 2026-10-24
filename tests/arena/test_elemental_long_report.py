"""The MTG-183 report recovers a known rating curve and its saturation point."""

import json
import math
from pathlib import Path
import random

from experiments.elemental_grid import GameRow, schedule
from experiments.elemental_long import h2h_schedule, heuristic_games, pair_design
from experiments.elemental_long_follow import Checkpoint, H2HRow, Ledger, Unit
from experiments.elemental_long_report import load, summarize, triangles

RATED = (0, 100, 200, 400, 800)
# Elo strengths: the heuristic score saturates by 400 (the heuristic is
# beaten almost always) while 800 still beats 400 clearly.
ELO = {0: 700.0, 100: 1300.0, 200: 1500.0, 400: 1900.0, 800: 2150.0}
HEURISTIC = 1000.0


def _win(rng: random.Random, a: float, b: float) -> float:
    return 1.0 if rng.random() < 1 / (1 + 10 ** ((b - a) / 400)) else 0.0


def _write(directory: Path) -> None:
    rng = random.Random(7)
    endpoint = RATED[-1]
    ledger = Ledger(
        heuristic_schedule_sha256="x",
        h2h_schedule_sha256="y",
        endpoint=endpoint,
        segments={"w3": "done: succeeded"},
        settled=True,
        checkpoints={
            u: Checkpoint(
                updates=u,
                job="j",
                training_run="r",
                path="p",
                sha256="s",
                coordinates={"learner_transitions": u * 10},
            )
            for u in RATED
        },
    )

    def save(unit: Unit, rows: list) -> None:
        unit.phase = "done"
        ledger.units[unit.key] = unit
        target = directory / "units" / unit.key
        target.mkdir(parents=True)
        (target / "rows.json").write_text(json.dumps([r.model_dump() for r in rows]))

    for u in RATED:
        grid, games = heuristic_games(u, endpoint)
        rows = [
            GameRow(
                model_deck=g.model_deck,
                heuristic_deck=g.heuristic_deck,
                model_seat=g.model_seat,
                deal=g.deal,
                score=_win(rng, ELO[u], HEURISTIC),
                valid=True,
                terminated=True,
                truncated=False,
                failure=None,
                decisions=1,
                model_decisions=1,
                seconds=1.0,
            )
            for g in games
        ]
        save(Unit(kind="heuristic", newer=u, grid=grid), rows)
    for pair in pair_design(RATED, endpoint):
        rows = [
            H2HRow(
                newer_deck=g.newer_deck,
                older_deck=g.older_deck,
                newer_seat=g.newer_seat,
                deal=g.deal,
                score=_win(rng, ELO[pair.newer], ELO[pair.older]),
                valid=True,
                terminated=True,
                truncated=False,
                failure=None,
                decisions=1,
                seconds=1.0,
            )
            for g in h2h_schedule()
        ]
        save(
            Unit(kind="pair", newer=pair.newer, older=pair.older, role=pair.role), rows
        )
    (directory / "ledger.json").write_text(ledger.model_dump_json())


def test_report_recovers_order_and_saturation(tmp_path: Path) -> None:
    _write(tmp_path)
    summary = summarize(load(tmp_path), replicates=20)
    ratings = [p.rating for p in summary.curve]
    assert ratings == sorted(ratings)
    assert summary.fit is not None and summary.fit["converged"]
    by_update = {p.updates: p.rating for p in summary.curve}
    assert math.isclose(by_update[800] - by_update[400], 250, abs_tol=80)
    assert summary.held_out is not None and summary.held_out.games > 0
    assert summary.held_out.log_loss is not None
    assert summary.held_out.log_loss < summary.held_out.baseline_log_loss
    assert summary.triangles == []
    overall = summary.saturation[0]
    assert (overall.scope, overall.rule) == ("overall", "margin 0.03")
    assert overall.updates == 400 and overall.exists
    # The strict rule never fires while the heuristic score still creeps upward.
    strict = summary.saturation[11]
    assert strict.rule == "strict" and strict.updates is None and not strict.exists
    assert len(summary.saturation) == 22
    assert len(schedule()) == 5000


def test_triangles_need_significant_edges() -> None:
    from experiments.elemental_long_report import PairView, ScoreView

    def view(newer: int, older: int, low: float, high: float) -> PairView:
        score = ScoreView(
            score=(low + high) / 2, low=low, high=high, valid=1, invalid=0
        )
        return PairView(
            newer=newer,
            older=older,
            role="fit",
            score=score,
            predicted=None,
            pearson_z=None,
            by_newer_deck={},
            by_newer_seat={},
        )

    cycle = [view(2, 1, 0.6, 0.7), view(3, 2, 0.6, 0.7), view(3, 1, 0.3, 0.4)]
    assert triangles(cycle) == [(1, 3, 2)]
    assert triangles([*cycle[:2], view(3, 1, 0.45, 0.55)]) == []
