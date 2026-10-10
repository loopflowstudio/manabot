"""Report arithmetic for the head-to-head runner, over synthetic retained rows."""

import json
from pathlib import Path

import pytest

from experiments.runners import head_to_head
from experiments.runners.head_to_head import GREEDY, Unit
from manabot.arena.models import ArenaKey, canonical_sha256
from manabot.training.execution import atomic_json

KEY = ArenaKey(
    world="w4",
    content_suite="suite",
    viewer_boundary="acting-viewer",
    arena_version="test",
    rating_model_version="test",
    rating_prior_sha256=canonical_sha256({}),
    anchor_cohort_sha256=canonical_sha256({}),
    evaluation_compute_envelope_id="test",
)


def _write_unit(
    out: Path,
    pairing: tuple[str, str],
    seed: int,
    scores: tuple[float, ...],
    *,
    mirrors: bool = False,
) -> None:
    """Persist four rows laid out as the arena does: legs alternate player_a's seat."""
    rows = []
    for leg, score in enumerate(scores, start=4 if mirrors else 0):
        decks = ["ur_lessons", "gw_allies"] if leg < 2 else ["gw_allies", "ur_lessons"]
        if mirrors:
            decks = ["ur_lessons"] * 2 if leg < 6 else ["gw_allies"] * 2
        rows.append(
            {
                "arena_key": KEY.model_dump(),
                "deal_seed": seed,
                "deal_block": 0,
                "leg": leg,
                "player_a": pairing[0],
                "player_b": pairing[1],
                "player_a_registration_sha256": "a",
                "player_b_registration_sha256": "b",
                "player_a_seat": leg % 2,
                "seat_decks": decks,
                "score_a": score,
                "failure": None,
                "terminated": True,
                "truncated": False,
                "replay_passed": True,
                "trace_path": "traces/x",
                "game_seconds": 1.0,
                "integrity": {"illegal_actions": 0},
            }
        )
    directory = out / "units" / Unit(*pairing, seed).directory
    directory.mkdir(parents=True)
    atomic_json(directory / "rows.json", rows)


def test_report_splits_and_orients_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "strong" is scheduled as player_b of its meeting with "weak", so the
    # contrast must flip that cell to read it from strong's side.
    pairings = [("strong", GREEDY), ("weak", GREEDY), ("weak", "strong")]
    seeds = [1, 2, 3]
    atomic_json(tmp_path / "design.json", {"pairings": pairings, "deal_seeds": seeds})
    for seed in seeds:
        # Legs: Lessons on play, Allies on draw, Allies on play, Lessons on draw.
        _write_unit(tmp_path, ("strong", GREEDY), seed, (0.0, 1.0, 1.0, 1.0))
        _write_unit(tmp_path, ("weak", GREEDY), seed, (0.0, 1.0, 1.0, 0.0))
        _write_unit(tmp_path, ("weak", "strong"), seed, (0.0, 0.0, 1.0, 0.0))
    monkeypatch.setattr(head_to_head, "CONTRASTS", (("strong", "weak"),))
    found = head_to_head.report(tmp_path)

    strong = found.pairings[0]
    assert strong.games == 12 and strong.rejected is None
    assert strong.score is not None and strong.score.mean == 0.75
    assert strong.as_lessons is not None and strong.as_lessons.mean == 0.5
    assert strong.as_allies is not None and strong.as_allies.mean == 1.0
    assert strong.on_play is not None and strong.on_play.mean == 0.5
    assert strong.on_draw is not None and strong.on_draw.mean == 1.0

    (contrast,) = found.contrasts
    assert contrast.deals == 3
    assert contrast.greedy_difference.mean == 0.25
    assert contrast.direct.mean == 0.75
    assert found.ratings["strong"][1] > found.ratings["weak"][1]
    assert abs(found.ratings[GREEDY][1]) < 1e-6


def test_invalid_game_rejects_its_pairing(tmp_path: Path) -> None:
    atomic_json(
        tmp_path / "design.json",
        {"pairings": [("weak", GREEDY)], "deal_seeds": [1, 2, 3]},
    )
    for seed in (1, 2, 3):
        _write_unit(tmp_path, ("weak", GREEDY), seed, (0.0, 1.0, 1.0, 0.0))
    path = tmp_path / "units" / Unit("weak", GREEDY, 3).directory / "rows.json"
    path.write_text(
        path.read_text().replace('"replay_passed": true', '"replay_passed": false')
    )

    (score,) = head_to_head.report(tmp_path).pairings
    assert score.rejected == "invalid games"
    assert score.score is None and score.invalid_games == 4


def test_schedule_covers_every_contrast() -> None:
    scheduled = {frozenset(pair) for pair in head_to_head.pairings()}
    for first, second in head_to_head.CONTRASTS:
        assert frozenset((first, second)) in scheduled
        assert frozenset((first, GREEDY)) in scheduled
        assert frozenset((second, GREEDY)) in scheduled


def test_mirror_plan_is_shared_disjoint_and_keeps_anchor() -> None:
    plan = head_to_head.full_plan("mirrors")
    assert plan.roster == head_to_head.ROSTER
    assert plan.pairings == head_to_head.pairings()
    assert not set(plan.deal_seeds) & set(head_to_head.DEAL_SEEDS)
    assert not set(plan.deal_seeds) & set(head_to_head.smoke_plan("mirrors").deal_seeds)
    for pair in plan.pairings:
        assert [
            u.deal_seed for u in plan.units() if (u.player_a, u.player_b) == pair
        ] == list(plan.deal_seeds)
    assert all(u.matchup_mode == "mirrors" for u in plan.units())


def test_mirror_deck_seat_contrasts_and_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = [("strong", GREEDY), ("weak", GREEDY), ("weak", "strong")]
    seeds = [1, 2, 3, 4]
    atomic_json(
        tmp_path / "design.json",
        {
            "pairings": pairs,
            "deal_seeds": seeds,
            "matchup_mode": "mirrors",
        },
    )
    for seed in seeds:
        _write_unit(tmp_path, pairs[0], seed, (1.0, 1.0, 0.0, 0.0), mirrors=True)
        _write_unit(tmp_path, pairs[1], seed, (0.0, 0.0, 1.0, 1.0), mirrors=True)
        # Strong wins Lessons, loses Allies. Aggregate hides this exactly.
        _write_unit(tmp_path, pairs[2], seed, (0.0, 0.0, 1.0, 1.0), mirrors=True)
    monkeypatch.setattr(head_to_head, "CONTRASTS", (("strong", "weak"),))
    found = head_to_head.report(tmp_path)
    for deck, expected in (("ur_lessons", 1.0), ("gw_allies", 0.0)):
        panel = found.decks[deck]
        score = panel.pairings[0]
        assert score.games == 8 and score.invalid_games == 0
        assert score.score.mean == expected
        assert score.on_play.mean == expected
        assert score.on_draw.mean == expected
        assert panel.contrasts[0].direct.mean == expected
        assert panel.contrasts[0].greedy_difference.mean == 2 * expected - 1
        assert (panel.ratings["strong"][1] > panel.ratings["weak"][1]) == bool(expected)
        assert abs(panel.ratings[GREEDY][1]) < 1e-6
    text = head_to_head.render(found)
    assert "Lessons vs Lessons" in text and "Allies vs Allies" in text
    assert head_to_head.TRAINING_CAVEAT in text


def test_mirror_intervals_resample_deals_not_games(tmp_path: Path) -> None:
    pair = ("strong", GREEDY)
    seeds = list(range(12))
    atomic_json(
        tmp_path / "design.json",
        {
            "pairings": [pair],
            "deal_seeds": seeds,
            "matchup_mode": "mirrors",
        },
    )
    for seed in seeds:
        # Every deal averages exactly half; independent-game resampling would
        # fabricate uncertainty even though these paired observations agree.
        _write_unit(tmp_path, pair, seed, (1.0, 0.0, 1.0, 0.0), mirrors=True)
    panel = head_to_head.report(tmp_path).decks["ur_lessons"]
    score = panel.pairings[0]
    assert score.score == head_to_head.Interval(0.5, 0.5, 0.5)
    assert score.on_play == head_to_head.Interval(1.0, 1.0, 1.0)
    assert score.on_draw == head_to_head.Interval(0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "corruption", ["duplicate", "deck", "seat", "seed", "pair", "replay", "unfinished"]
)
def test_mirror_corruption_is_not_dropped(tmp_path: Path, corruption: str) -> None:
    pair = ("weak", GREEDY)
    atomic_json(
        tmp_path / "design.json",
        {
            "pairings": [pair],
            "deal_seeds": [1, 2, 3],
            "matchup_mode": "mirrors",
        },
    )
    for seed in (1, 2, 3):
        _write_unit(tmp_path, pair, seed, (0.0, 1.0, 1.0, 0.0), mirrors=True)
    path = tmp_path / "units" / Unit(*pair, 3).directory / "rows.json"
    rows = json.loads(path.read_text())
    if corruption == "unfinished":
        path.unlink()
    else:
        name, value = {
            "duplicate": ("leg", 5),
            "deck": ("seat_decks", ["gw_allies", "ur_lessons"]),
            "seat": ("player_a_seat", 1),
            "seed": ("deal_seed", 99),
            "pair": ("player_a", "other"),
            "replay": ("replay_passed", False),
        }[corruption]
        rows[0][name] = value
        atomic_json(path, rows)
    found = head_to_head.report(tmp_path)
    for panel in found.decks.values():
        assert panel.pairings[0].rejected
        assert panel.pairings[0].score is None
        assert panel.pairings[0].invalid_games or panel.pairings[0].failed_units
        assert not panel.ratings and not panel.contrasts


def test_mirror_mode_cannot_overwrite_cross_run(tmp_path: Path) -> None:
    atomic_json(tmp_path / "design.json", {"matchup_mode": "cross"})
    atomic_json(tmp_path / "status.json", {"state": "running"})
    before = (tmp_path / "status.json").read_bytes()
    with pytest.raises(ValueError, match="different matchup mode"):
        head_to_head.run(
            tmp_path,
            tmp_path / "missing-models",
            plan=head_to_head.smoke_plan("mirrors"),
        )
    assert (tmp_path / "status.json").read_bytes() == before
