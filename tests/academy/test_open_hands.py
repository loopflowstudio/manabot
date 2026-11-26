"""Open hands: the person playing sees the bot's hand, and nothing else changes."""

import json
from pathlib import Path
import random
from typing import Any

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace as trace_store
from managym.academy.attempts import AttemptStore
from managym.academy.numbers import game_numbers
from managym.experience_projection import serialize_observation
from tests.academy.socket_messages import make_command, settled_message
from tests.academy.test_played_replay import BOLT, play

SETUP: dict[str, Any] = {
    "hero_deck": BOLT,
    "villain_deck": BOLT,
    "villain_type": "random",
    "seed": 11,
    "auto_pass": False,
}


def started(tmp_path: Path, **config: Any) -> server.GameSession:
    game = server.GameSession(tmp_path)
    game.new_game({**SETUP, **config})
    return game


def bot_hand(game: server.GameSession) -> list[dict[str, Any]]:
    assert game.env is not None
    seat = 1 - game.human_seat
    return serialize_observation(game.env.observation_for_player(seat))["agent"]["hand"]


def stored_config(tmp_path: Path, game: server.GameSession) -> dict[str, Any]:
    attempt = AttemptStore(tmp_path / "play.sqlite").load_attempt(game.match_id)
    assert attempt is not None
    return attempt.trace["config"]


def test_open_hands_shows_the_bots_hand_and_changes_nothing_else(
    tmp_path: Path,
) -> None:
    blind = started(tmp_path / "blind")
    game = started(tmp_path / "open")
    assert game.open_hand() is None
    dealt = game.current_message()["frame"]

    assert game.set_open_hands(True) == {"type": "open_hands_state"}
    # The frame the person decides from is the same frame, hash included.
    assert game.current_message()["frame"] == dealt
    shown = game.open_hand()
    assert shown is not None and shown["revision"] == game.revision
    assert shown["hand"] == bot_hand(game) and len(shown["hand"]) == 7
    assert all(card["name"] for card in shown["hand"])

    play(blind, random.Random(3), commands=12)
    play(game, random.Random(3), commands=12)
    assert blind.trace is not None and game.trace is not None

    # The same game as one played blind: every board the person decided from,
    # and every observation the person and the bot chose from.
    frame, same = game.current_message()["frame"], blind.current_message()["frame"]
    assert frame["projection"] == same["projection"]
    assert frame["projection"]["opponent"]["hand"] == []
    assert [d.frame.projection for d in game.canonical_decisions] == [
        d.frame.projection for d in blind.canonical_decisions
    ]
    assert [(e.actor, e.observation, e.action) for e in game.trace.events] == [
        (e.actor, e.observation, e.action) for e in blind.trace.events
    ]

    # Only the person's deliberate decisions are marked.
    marked = [e for e in game.trace.events if e.open_hands]
    assert marked and all(e.actor == "hero" and not e.auto for e in marked)
    assert len(marked) == sum(
        1 for e in game.trace.events if e.actor == "hero" and not e.auto
    )
    assert not any(e.open_hands for e in blind.trace.events)
    assert stored_config(tmp_path / "open", game)["open_hands"] is True
    assert "open_hands" not in stored_config(tmp_path / "blind", blind)

    # Off again: no hand, and the decisions from here on are unmarked.
    count = len(game.trace.events)
    game.set_open_hands(False)
    assert game.open_hand() is None
    play(game, random.Random(4), commands=4, tag="off")
    assert len(game.trace.events) > count
    assert not any(e.open_hands for e in game.trace.events[count:])
    assert "open_hands" not in stored_config(tmp_path / "open", game)

    attempt = AttemptStore(tmp_path / "open" / "play.sqlite").load_attempt(
        game.match_id
    )
    assert attempt is not None
    numbers = game_numbers(attempt)
    assert numbers is not None and numbers.open_hands == len(marked)


def test_a_resumed_game_and_its_rematch_keep_open_hands(tmp_path: Path) -> None:
    game = started(tmp_path, open_hands=True)
    play(game, random.Random(3), commands=6)
    assert game.trace is not None and game.trace.events[0].open_hands

    resumed = server.GameSession(tmp_path)
    resumed.resume(game.match_id)
    shown = resumed.open_hand()
    assert shown is not None and shown["hand"] == bot_hand(resumed)
    assert resumed.trace is not None
    assert [e.open_hands for e in resumed.trace.events] == [
        e.open_hands for e in game.trace.events
    ]
    assert server.shown_setup(resumed.trace.config, False)["open_hands"] is True


def test_open_hands_after_game_over_peeks_without_changing_the_saved_game(
    tmp_path: Path,
) -> None:
    game = started(tmp_path)
    play(game, random.Random(3))
    assert game.obs is not None and game.obs.game_over
    before = stored_config(tmp_path, game)

    game.set_open_hands(True)
    shown = game.open_hand()
    assert shown is not None and shown["hand"] == bot_hand(game)
    assert game.trace is not None and not game.trace.config.open_hands
    assert stored_config(tmp_path, game) == before


def test_a_watched_game_refuses_open_hands(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path, watch_mode=True)
    game.new_game({**SETUP, "open_hands": True})
    assert game.open_hand() is None
    with pytest.raises(server.PlayRequestError, match="no seat of yours"):
        game.set_open_hands(True)


def test_the_play_socket_carries_the_hand_beside_its_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closed decklists allow it: only `open_hand` holds the bot's hand."""
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    config = {
        **SETUP,
        "hero_deck": {"Mountain": 20},
        "villain_deck": {"Forest": 20},
        "decklists": "closed",
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            first = settled_message(websocket)
            assert first["type"] == "observation" and first["open_hand"] is None
            assert first["setup"]["open_hands"] is False
            assert "Forest" not in json.dumps(first)
            game = first["game"]

            websocket.send_json({"type": "set_open_hands", "open": True})
            state = settled_message(websocket)
            assert state["type"] == "open_hands_state" and "frame" not in state
            hand = state["open_hand"]
            assert hand["revision"] == first["frame"]["revision"]
            assert {card["name"] for card in hand["hand"]} == {"Forest"}

            # Each reply that advances the frame carries the hand at its revision.
            frame = first["frame"]
            websocket.send_json(
                {
                    "type": "command",
                    "command": make_command(frame, frame["offers"][0], "open.0"),
                }
            )
            outcome = settled_message(websocket)
            assert outcome["type"] == "command_outcome"
            advanced = outcome["update"]["frame"]
            assert outcome["open_hand"]["revision"] == advanced["revision"]
            theirs = advanced["projection"]["opponent"]
            assert theirs["hand"] == []
            assert theirs["hand_hidden_count"] == len(outcome["open_hand"]["hand"])
            assert outcome["setup"]["open_hands"] is True

        # A reconnect's recovery carries it with the frame.
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "resume", "game": game})
            recovered = settled_message(websocket)
            assert recovered["type"] == "observation"
            assert recovered["open_hand"]["revision"] == recovered["frame"]["revision"]
            assert {card["name"] for card in recovered["open_hand"]["hand"]} == {
                "Forest"
            }

            websocket.send_json({"type": "set_open_hands", "open": False})
            assert settled_message(websocket)["open_hand"] is None

        # The stored game, as a viewer reads it, holds no hand of the bot's.
        export = client.get(f"/api/traces/{game}")
        assert export.status_code == 200
        stored = export.json()
        assert '"open_hand"' not in json.dumps(stored)
        boards = [event["observation"] for event in stored["events"]]
        for board in (*boards, stored["final_observation"]):
            bot = board["opponent" if board["agent"]["player_index"] == 0 else "agent"]
            assert bot["hand"] == [] and bot["hand_hidden_count"] > 0
    server.LIVE_GAMES.clear()


def test_playing_on_keeps_each_prefix_mark_and_starts_blind(tmp_path: Path) -> None:
    game = started(tmp_path)
    play(game, random.Random(3), commands=3)
    game.set_open_hands(True)
    play(game, random.Random(4), commands=3, tag="open")
    source_id = game.match_id
    source = game.attempt_store.load_attempt(source_id)
    assert source is not None
    events = source.trace["events"]
    marks = [bool(event.get("open_hands")) for event in events]
    decisions = [
        index
        for index, event in enumerate(events)
        if event["actor"] == "hero" and not event.get("auto")
    ]
    # Blind decisions, then decisions made seeing the hand, both in the prefix.
    upto = decisions[-1]
    assert True in marks[:upto] and marks[decisions[0]] is False

    game.new_game({"played_from": {"trace_id": source_id, "event": upto}})

    assert game.trace is not None and not game.open_hands
    assert game.open_hand() is None
    assert [event.open_hands for event in game.trace.events] == marks[:upto]
    assert "open_hands" not in stored_config(tmp_path, game)
    stored = game.attempt_store.load_attempt(game.match_id)
    assert stored is not None
    assert [
        bool(event.get("open_hands")) for event in stored.trace["events"]
    ] == marks[:upto]
    play(game, random.Random(5), commands=1)
    assert not game.trace.events[upto].open_hands
