"""Scientific accounting checks for the retained mirror report; no games run."""

from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

import pytest

from experiments.elemental_grid import Game, schedule
from experiments.search_mirror import ARMS, V2_ARMS, MirrorRow
from experiments.search_mirror_noise import Position, analyze
from experiments.search_mirror_report import (
    _bootstrap,
    _coverage,
    _extension_report,
    _extension_section,
    _factorial_contrasts,
    _linear_contrast,
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


def test_factorial_interaction_pairs_four_arms_and_retains_missing_slots() -> None:
    games = [Game("a", "b", i, i, 0) for i in range(2)]
    interaction = next(c for c in _factorial_contrasts()
                       if c.label == "uniform minus oracle shuffle gain, 8 minus 4")
    scores = {a.key: {0: 0.0, 1: 0.0} for a in V2_ARMS}
    for arm in V2_ARMS:
        if arm.worlds == 8:
            scores[arm.key] = {0: 1.0, 1: 0.0} if arm.sampler == "oracle" else {0: 0.0, 1: 1.0}
    result = _linear_contrast(interaction, scores, games)
    assert result["normal"]["mean"] == 0
    assert result["cell_stratified_bootstrap"]["low"] == -1
    assert result["cell_stratified_bootstrap"]["high"] == 1
    assert result["failure_sensitivity"] == [0, 0]
    # Four cell means would hide the missing oracle result and invent equivalence.
    oracle8 = next(a.key for a in V2_ARMS if a.sampler == "oracle" and a.worlds == 8)
    del scores[oracle8][1]
    result = _linear_contrast(interaction, scores, games)
    assert (result["paired"], result["planned"], result["missing_deals"]) == (1, 2, [1])
    assert result["failure_sensitivity"] == [-0.5, 0]
    empty = _linear_contrast(interaction, {}, games)
    assert empty["normal"] is None and empty["cell_stratified_bootstrap"] is None
    assert empty["failure_sensitivity"] == [-2, 2]


def test_extension_snapshot_reports_unmeasured_not_zero(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "reconciliation"
    receipt_dir.mkdir()
    receipts = {
        "remote-inventory": {"v2_directory_exists": False, "observed_utc": "2026-11-09T18:26:41Z"},
        "launcher-inventory": {"quote_error": "no free machine", "new_rental_spend_usd": 0},
        "validation": {"deadline_unix": 1794852989},
    }
    for name, receipt in receipts.items():
        (receipt_dir / f"{name}.json").write_text(json.dumps(receipt))
    extension = _extension_report(tmp_path, {})
    assert (extension["valid_games"], extension["missing_games"], extension["planned_games"]) == (0, 3000, 3000)
    assert extension["invalid_games"] == 0
    assert len(extension["arms"]) == 6
    assert len(extension["contrasts"]) == 15
    for arm in extension["arms"].values():
        assert arm["score"] is None and arm["search_cost"] is None
        assert len(arm["coverage"]["missing_deals"]) == 500
    for contrast in extension["contrasts"]:
        assert contrast["planned"] == 500 and contrast["paired"] == 0
        assert contrast["normal"] is None and contrast["cell_stratified_bootstrap"] is None
    html = _extension_section(extension)
    assert "0/3,000 valid games" in html and "unavailable" in html
    assert "remain proposals" not in html
    # Admission can progress while every scientific outcome is still missing.
    receipts["remote-inventory"]["execution"] = "native smoke complete; campaign starting"
    receipts["validation"]["health"] = "six smoke outcomes retained"
    receipts["validation"]["native_support"] = "bounded mini smoke passed"
    for name, receipt in receipts.items():
        (receipt_dir / f"{name}.json").write_text(json.dumps(receipt))
    admitted = _extension_report(tmp_path, {})
    assert admitted["valid_games"] == 0 and admitted["missing_games"] == 3000
    html = _extension_section(admitted)
    assert "bounded mini smoke passed" in html and "six smoke outcomes retained" in html
    assert "No native smoke" not in html
    # Later performance diagnostics and ledger counts cannot fill sealed game slots.
    (tmp_path / "batching").mkdir()
    (tmp_path / "batching/summary.json").write_text(json.dumps({
        "decision": "not admitted", "reason": "timings inconclusive",
        "correctness": "fixed roots match", "limits": "one pair per cell", "cells": [],
    }))
    (receipt_dir / "batching-inventory.json").write_text(json.dumps({
        "completed": 77, "failed": 0, "observed_utc": "later", "workers": ["worker"],
    }))
    diagnostics = _extension_report(tmp_path, {})
    assert diagnostics["valid_games"] == 0 and diagnostics["missing_games"] == 3000
    assert all(arm["search_cost"] is None for arm in diagnostics["arms"].values())
    html = _extension_section(diagnostics)
    assert "77/3,000 completed rows" in html and "0/3,000 valid games" in html
    assert "not admitted" in html and "earlier closed-evidence snapshot" in html
    # A typo in the input path must not manufacture a no-games receipt.
    with pytest.raises(FileNotFoundError):
        _extension_report(tmp_path / "missing", {})


def test_extension_reader_rejects_wrong_worlds_sampler_and_smoke(tmp_path: Path) -> None:
    arm = V2_ARMS[0]
    row = replace(_row(schedule()[0]), key=f"{arm.key}/{schedule()[0].deal}",
                  condition=arm.key, depth=arm.depth, beta=arm.beta,
                  worlds=arm.worlds, sampler=arm.sampler, study=arm.study)
    path = tmp_path / "games-test.jsonl"
    path.write_text(json.dumps(asdict(row)) + "\n")
    assert len(_outcomes([tmp_path], mirror=True, arms=V2_ARMS)[arm.key]) == 1
    for bad in (replace(row, worlds=16), replace(row, sampler="uniform"), replace(row, study="v2-smoke")):
        path.write_text(json.dumps(asdict(bad)) + "\n")
        with pytest.raises(ValueError, match="identity mismatch"):
            _outcomes([tmp_path], mirror=True, arms=V2_ARMS)
