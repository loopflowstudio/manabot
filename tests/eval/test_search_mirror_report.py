"""Scientific accounting checks for the retained mirror report; no games run."""

from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

import pytest

from experiments.elemental_grid import Game, schedule
from experiments.search_mirror import ARMS, MirrorRow
from experiments.search_mirror_noise import Position, analyze
from experiments.search_mirror_report import (
    _bootstrap,
    _coverage,
    _outcomes,
    _sensitivity,
)


def _row(game: Game) -> MirrorRow:
    arm = ARMS[0]
    return MirrorRow(
        f"{arm.key}/{game.deal}", arm.key, arm.opponent, arm.depth, arm.beta,
        game.model_deck, game.heuristic_deck, game.model_seat, game.deal,
        1.0, None, 1, 1, 1, 0.1, "test",
    )


def test_missing_scores_keep_full_denominator() -> None:
    # First deal gains one; the second has only a known loss for A. The missing
    # control can make its contrast -1 or 0, not an invented zero outcome.
    assert _sensitivity({1: 1.0, 2: 0.0}, {1: 0.0}, {1, 2}) == [0.0, 0.5]
    games = schedule()[:3]
    rows = {games[0].deal: _row(games[0]), games[1].deal: replace(_row(games[1]), score=None, failure="timeout")}
    coverage = _coverage(rows, games)
    assert (coverage["planned"], coverage["valid"], coverage["invalid"], coverage["missing"]) == (3, 1, 1, 1)
    assert coverage["failures"][0]["failure"] == "timeout"


def test_joint_bootstrap_preserves_pairing_and_fixed_cells() -> None:
    games = [Game("a", "b", i, i, 0) for i in range(3)] + [Game("c", "d", 0, 3, 1)]
    scores = {"a": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0}, "b": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
              "zero": {i: 0.0 for i in range(4)}}
    intervals, common = _bootstrap(scores, games, [("a", "b"), ("a", "zero"), ("b", "zero")])
    assert common == {0, 1, 2, 3}
    assert intervals["a", "b"].low == intervals["a", "b"].high == 0
    assert intervals["a", "zero"] == intervals["b", "zero"]
    assert intervals["a", "zero"].mean == pytest.approx(1 / 3)  # cells, not 4 unequal slots
    del scores["b"][3]
    intervals, common = _bootstrap(scores, games, [("a", "b")])
    assert intervals == {}  # never silently discard a wholly missing cell
    assert common == {0, 1, 2}


def test_outcomes_reject_identity_drift_and_duplicates(tmp_path: Path) -> None:
    row = _row(schedule()[0])
    path = tmp_path / "games-test.jsonl"
    path.write_text(json.dumps(asdict(replace(row, model_seat=1 - row.model_seat))) + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        _outcomes([tmp_path], mirror=True)
    path.write_text((json.dumps(asdict(row)) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        _outcomes([tmp_path], mirror=True)


def test_noise_draws_do_not_become_independent_source_deals(tmp_path: Path) -> None:
    # Two source deals have deterministic but different value gaps. Repeating
    # positions/draws must not turn their cluster interval into a narrow SE.
    positions = [Position(str(i), "policy", i % 2, 1, 1, [0.5, 0.5],
                          [[0.0] * 64, [float(i % 2)] * 64], [0, 0], 1.0) for i in range(20)]
    with gzip.open(tmp_path / "positions-test.jsonl.gz", "wt") as stream:
        for p in positions:
            stream.write(json.dumps(asdict(p)) + "\n")
    summary, rows = analyze(tmp_path, draws=1)
    _, repeated = analyze(tmp_path, draws=5)
    assert summary["source_deals"] == 2
    assert summary["positions"] == 20
    for a, b in zip(rows, repeated, strict=True):
        assert a.gain_low == pytest.approx(b.gain_low)
        assert a.gain_high == pytest.approx(b.gain_high)
        assert a.gain_high - a.gain_low > 0.2
