"""Independent native counters agree with retained facts and preserve play."""

from pathlib import Path

import pytest

import managym
from managym.card_capture import CapturedCards
from managym.recording import ScalarCapture
from managym.statistics import reduce_cards


def players() -> list[managym.PlayerConfig]:
    return [managym.PlayerConfig(n, {"Mountain": 20}) for n in ("a", "b")]


def test_statistics_and_recording_are_independent(tmp_path: Path) -> None:
    frames: list[list[str]] = []
    summaries = []
    for stats, logs in [(False, False), (True, False), (False, True), (True, True)]:
        game = managym.Env(seed=42)
        if stats:
            game.enable_statistics()
        capture = (
            ScalarCapture(42, journal_path=tmp_path / f"{stats}.jsonl")
            if logs
            else None
        )
        if capture:
            capture.reset(game, players())
        else:
            game.reset(players())
        track = []
        for _ in range(8):
            track.append(game.semantic_decision_frame_json())
            if capture:
                capture.step(game, 0)
            else:
                game.step(0)
        frames.append(track)
        summary = game.statistics()
        assert (summary is not None) == stats
        if summary:
            summaries.append(summary.cards)
            with pytest.raises(managym.AgentError, match="capture unavailable"):
                game.card_capture_json()
    assert all(f == frames[0] for f in frames)
    assert summaries[0] == summaries[1]
    assert len(list(tmp_path.iterdir())) == 2


def test_statistics_reduce_retained_facts_and_drain_generations() -> None:
    game = managym.Env(seed=17)
    assert game.statistics() is None
    game.enable_statistics()
    assert game.statistics() is None
    game.enable_card_capture()
    game.reset(players())
    for _ in range(8):
        game.step(0)
    live = game.statistics()
    assert live is not None and not live.terminal and live.seed == 17
    retained = reduce_cards(CapturedCards.model_validate_json(game.card_capture_json()))
    assert [
        (r.seat, r.definition, r.deck_copies, r.opening_copies, r.drawn_copies)
        for r in live.cards
    ] == [
        (r.seat, r.definition, r.deck_copies, r.opening_copies, r.drawn_copies)
        for r in retained
    ]
    assert all(r.opportunities is None for r in retained)
    assert game.take_statistics() == []
    game.reset(players())
    assert game.take_statistics() == [live]
    assert game.take_statistics() == []
    assert game.statistics().generation != live.generation
    with pytest.raises(managym.AgentError, match="before reset"):
        game.enable_statistics()


def test_vector_statistics_retain_streams() -> None:
    vector = managym.VectorEnv(num_envs=2, seed=17)
    assert vector.statistics() == [None, None]
    vector.enable_statistics()
    vector.reset_all(players())
    first = vector.statistics()
    assert [s.stream for s in first] == [0, 1]
    assert [s.seed for s in first] == [17, 18]
    vector.reset_all(players())
    assert vector.take_statistics() == first
    assert vector.take_statistics() == []


def test_completed_generation_drains_once() -> None:
    game = managym.Env(seed=17)
    game.enable_statistics()
    game.reset([managym.PlayerConfig(n, {"Mountain": 8}) for n in ("a", "b")])
    for _ in range(40):
        _, _, done, _, _ = game.step(0)
        if done:
            break
    else:
        pytest.fail("controlled game did not terminate")
    summary = game.statistics()
    assert summary is not None and summary.terminal
    assert game.take_statistics() == [summary]
    assert game.take_statistics() == []
    game.reset(players())
    assert game.take_statistics() == []
