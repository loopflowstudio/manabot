"""Report arithmetic for the head-to-head runner, over synthetic retained rows."""

from pathlib import Path

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
    out: Path, pairing: tuple[str, str], seed: int, scores: tuple[float, ...]
) -> None:
    """Persist four rows laid out as the arena does: legs alternate player_a's seat."""
    rows = []
    for leg, score in enumerate(scores):
        decks = ["ur_lessons", "gw_allies"] if leg < 2 else ["gw_allies", "ur_lessons"]
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


def test_report_splits_and_orients_scores(tmp_path: Path) -> None:
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
    original = head_to_head.CONTRASTS
    head_to_head.CONTRASTS = (("strong", "weak"),)
    try:
        found = head_to_head.report(tmp_path)
    finally:
        head_to_head.CONTRASTS = original

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
