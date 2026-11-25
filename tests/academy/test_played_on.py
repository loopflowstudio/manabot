"""Playing on from a finished game: a new game whose first moves are the record's."""

import json
from pathlib import Path
import random

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace
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


def test_only_your_own_games_can_be_played_on_at_your_decisions(
    tmp_path: Path,
) -> None:
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


def test_an_unfinished_game_can_be_played_on(tmp_path: Path) -> None:
    live = server.GameSession(tmp_path)
    live.new_game(
        {"hero_deck": BOLT, "villain_deck": BOLT, "seed": 2, "auto_pass": False}
    )
    play(live, random.Random(3), commands=6)
    source_id = live.match_id
    source = live.attempt_store.load_attempt(source_id)
    assert source is not None and source.trace["end_reason"] == "active"
    event = decision(source.trace["events"], 3)

    frame = live.new_game({"played_from": {"trace_id": source_id, "event": event}})[
        "frame"
    ]

    assert frame["match_id"] != source_id
    assert [offer["label"] for offer in frame["offers"]] == [
        action["description"] for action in source.trace["events"][event]["actions"]
    ]
    # The game played on from is untouched and still unfinished.
    assert live.attempt_store.unfinished(source_id)
    assert live.attempt_store.load_attempt(source_id).trace == source.trace


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


def test_recording_index_handles_played_on_game_before_first_new_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace, "TRACES_DIR", tmp_path)
    token = "p" * 36
    game, source = played(tmp_path, player_token=token)
    event = decision(source.trace["events"], 2)
    source_id = game.match_id
    game.new_game(
        {
            "played_from": {"trace_id": source_id, "event": event},
            "player_token": token,
        }
    )
    # Stop before committing a move at `event`: len(events) == event.
    game.close(end_reason="new_game")
    client = TestClient(server.app)
    response = client.get(
        f"/api/traces/{game.match_id}/recording",
        headers={"x-etude-player-token": token},
    )
    assert response.status_code == 200, response.text
    played_from = response.json()["played_from"]
    assert played_from["trace_id"] == source_id
    assert played_from["event"] == event
    assert (
        played_from["turn"]
        == source.trace["events"][event]["observation"]["turn"]["turn_number"]
    )


def test_watch_game_cannot_be_played_on(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": BOLT,
            "villain_type": "random",
            "seed": 11,
        }
    )
    game.close(end_reason="new_game")
    # Mark the stored trace as a Watch recording.
    store = game.attempt_store
    with store.connect() as db:
        (raw,) = db.execute(
            "SELECT trace_json FROM attempts WHERE id=?", (game.match_id,)
        ).fetchone()
        payload = json.loads(raw)
        payload["config"]["watch_players"] = ["random", "random"]
        db.execute(
            "UPDATE attempts SET trace_json=? WHERE id=?",
            (json.dumps(payload), game.match_id),
        )
    with pytest.raises(ValueError, match="played_from_unavailable"):
        game.new_game({"played_from": {"trace_id": game.match_id, "event": 0}})
