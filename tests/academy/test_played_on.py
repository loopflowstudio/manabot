"""Playing on from a finished game: a new game whose first moves are the record's."""

import json
from pathlib import Path
import random

import pytest

from managym.academy import server
from managym.academy.recording import ReplayedViews
from tests.academy.test_played_replay import BOLT, play, played


def decision(events: list[dict], nth: int) -> int:
    """The index of the person's `nth` deliberate decision."""
    return [
        index
        for index, event in enumerate(events)
        if event["actor"] == "hero" and not event.get("auto")
    ][nth]


def test_playing_on_starts_at_your_decision_with_the_games_moves_before_it(
    tmp_path: Path,
) -> None:
    # Priority stops: the source's automated passes replay as recorded.
    game, source = played(tmp_path, auto_pass=True)
    events = source.trace["events"]
    event = decision(events, 4)
    source_id = game.match_id

    message = game.new_game({"played_from": {"trace_id": source_id, "event": event}})

    frame = message["frame"]
    assert frame["status"] != "game_over" and frame["prompt"] is not None
    # The game's own stops, though the prefix surfaced every window.
    assert frame["stops"]["auto_pass"] is True
    assert [o["label"] for o in frame["offers"]] == [
        action["description"] for action in events[event]["actions"]
    ]
    play(game, random.Random(5))
    attempt = game.attempt_store.load_attempt(game.match_id)
    assert attempt is not None
    trace = attempt.trace
    assert trace["played_from"] == {"trace_id": source_id, "event": event}
    assert trace["end_reason"] == "game_over"
    # The same moves, and the person saw the same boards, up to the decision.
    for mine, theirs in zip(trace["events"][: event + 1], events[: event + 1]):
        assert mine["actor"] == theirs["actor"]
        assert mine["action_description"] == theirs["action_description"]
        assert mine["observation"] == theirs["observation"]
    # It is an ordinary game: it re-executes to its own result.
    views = ReplayedViews(attempt, 0)
    assert views.board(len(trace["events"])) is not None
    assert views.stop is not None and views.stop.status == "complete"


def test_only_your_own_finished_games_can_be_played_on(tmp_path: Path) -> None:
    game, source = played(tmp_path)
    event = decision(source.trace["events"], 0)
    stranger = server.GameSession(tmp_path)
    with pytest.raises(ValueError, match="played_from_unavailable"):
        stranger.new_game({"played_from": {"trace_id": game.match_id, "event": event}})
    opponent_step = next(
        i for i, e in enumerate(source.trace["events"]) if e["actor"] == "villain"
    )
    with pytest.raises(ValueError, match="played_from_decision"):
        game.new_game(
            {"played_from": {"trace_id": game.match_id, "event": opponent_step}}
        )
    live = server.GameSession(tmp_path)
    live.new_game({"hero_deck": BOLT, "villain_deck": BOLT, "seed": 2})
    with pytest.raises(ValueError, match="played_from_active"):
        live.new_game({"played_from": {"trace_id": live.match_id, "event": 0}})


def test_a_game_played_on_is_left_out_of_my_record(tmp_path: Path) -> None:
    game, source = played(tmp_path)
    owners = {game.attempt_owner}
    (before,) = game.attempt_store.record(owners)
    event = decision(source.trace["events"], 0)
    game.new_game({"played_from": {"trace_id": game.match_id, "event": event}})
    play(game, random.Random(6))
    numbers = game.attempt_store.numbers(game.match_id)
    assert numbers is not None and numbers[1] is not None
    assert json.loads(numbers[1])["played_from"] == event
    assert game.attempt_store.record(owners) == [before]
