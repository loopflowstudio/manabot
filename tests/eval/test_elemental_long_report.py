"""The MTG-183 report recovers a known rating curve from a real Etude eval run."""

import hashlib
import math
from pathlib import Path
import random

from etude.eval.match import Tables, evaluate
from etude.eval.records import Match
from etude.run.records import ArtifactRef
import pytest

from experiments.elemental_grid import GameRow
from experiments.elemental_long_eval import (
    EXPERIMENT,
    RUN,
    Checkpoint,
    Conditions,
    EvalState,
    eval_environment,
    game_rows,
    units,
)
from experiments.elemental_long_report import load, summarize, triangles

RATED = (0, 100, 200, 400, 800)
# Elo strengths: the heuristic score saturates by 400 (the heuristic is
# beaten almost always) while 800 still beats 400 clearly.
ELO = {0: 700.0, 100: 1300.0, 200: 1500.0, 400: 1900.0, 800: 2150.0}
HEURISTIC = 1000.0


def _strength(name: str) -> float:
    """A rated checkpoint `uNNNNNN`'s strength, or the heuristic's."""
    return HEURISTIC if name == "heuristic" else ELO[int(name[1:])]


def _play(match: Match) -> Tables:
    """Bradley-Terry games between the seated strengths, seeded by the match."""
    rng = random.Random(match.seed)
    conditions = Conditions.model_validate(match.conditions)
    a, b = (_strength(player.name) for player in match.players)
    rows = [
        GameRow(
            model_deck=conditions.deck,
            heuristic_deck=group.heuristic_deck,
            model_seat=group.model_seat,
            deal=deal,
            score=1.0 if rng.random() < 1 / (1 + 10 ** ((b - a) / 400)) else 0.0,
            valid=True,
            terminated=True,
            truncated=False,
            failure=None,
            decisions=1,
            model_decisions=1,
            seconds=1.0,
        )
        for group in conditions.groups
        for deal in group.deals
    ]
    return {"games": game_rows(rows, conditions.kind)}


def _state(tmp_path: Path) -> EvalState:
    checkpoints = {}
    for u in RATED:
        path = tmp_path / f"{u}.pt"
        path.write_bytes(str(u).encode())
        checkpoints[u] = Checkpoint(
            updates=u,
            run="w1",
            artifact=ArtifactRef(
                uri=path.as_uri(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            ),
            coordinates={"learner_transitions": u * 10},
        )
    return EvalState(
        checkpoints=checkpoints, endpoint=RATED[-1], segment_outcome="succeeded"
    )


def test_report_recovers_order_and_saturation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "etude"))
    state = _state(tmp_path)
    (tmp_path / "state.json").write_text(state.model_dump_json())
    design = units(state)
    track = tmp_path / "track"
    # Every unit but the last pair is played; the report names that one missing.
    evaluate(
        experiment=EXPERIMENT,
        environment=eval_environment(),
        matches=[m for u in design[:-1] for m in u.matches],
        play=_play,
        run=RUN,
        directory=track,
    )
    results = load(tmp_path, track)
    assert results.missing == [design[-1].key]
    summary = summarize(results, replicates=20)
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
