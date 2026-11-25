"""An unfinished played game is taken up again from its stored record."""

import json
from pathlib import Path
import random
import sqlite3
from typing import Any

import pytest

from managym.academy import server
from managym.academy.recording import Recording, ReplayedViews
from tests.academy.test_played_replay import BOLT, play


def start(tmp_path: Path, commands: int = 8, **config: Any) -> server.GameSession:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": BOLT,
            "villain_type": "random",
            "seed": 11,
            **config,
        }
    )
    play(game, random.Random(3), commands=commands)
    return game


def row(tmp_path: Path, game: str) -> sqlite3.Row:
    """The stored row as another process would read it."""
    db = sqlite3.connect(tmp_path / "play.sqlite")
    db.row_factory = sqlite3.Row
    try:
        return db.execute("SELECT * FROM attempts WHERE id=?", (game,)).fetchone()
    finally:
        db.close()


def rewrite(tmp_path: Path, game: str, sql: str, *args: Any) -> None:
    db = sqlite3.connect(tmp_path / "play.sqlite")
    with db:
        db.execute(sql, (*args, game))
    db.close()


@pytest.mark.parametrize(
    "config",
    [
        {"villain_type": "search", "villain_sims": 4, "auto_pass": True},
        {"human_seat": 1, "decklists": "closed", "auto_pass": False},
    ],
    ids=["search-with-stops", "second-seat-closed"],
)
def test_a_game_resumes_as_the_same_game_at_its_pending_decision(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    game = start(tmp_path, **config)
    identifier = game.match_id
    pending = game.current_message()["frame"]
    digest = game.env.state_digest()
    stored = row(tmp_path, identifier)
    before = json.loads(stored["trace_json"])
    decisions = before["canonical_replay"]["decisions"]
    assert stored["status"] == "active" and len(before["events"]) > 8

    resumed = server.GameSession(tmp_path)
    frame = resumed.resume(identifier)["frame"]

    assert resumed.match_id == frame["match_id"] == identifier
    assert resumed.env.state_digest() == digest
    assert frame["revision"] == pending["revision"]
    assert frame["offers"] == pending["offers"]
    assert frame["projection"] == pending["projection"]
    assert frame["prompt"]["id"] > max(row["prompt_id"] for row in decisions)
    # Nothing was played, so nothing was written.
    assert row(tmp_path, identifier)["trace_json"] == stored["trace_json"]
    # A Study root for every recorded decision.
    assert sorted(resumed._study_roots) == list(range(len(decisions)))

    play(resumed, random.Random(5), commands=1, tag="next")
    after = row(tmp_path, identifier)
    # The stored steps are kept byte for byte; the new ones follow them.
    assert json.dumps(before["events"])[:-1] in after["trace_json"]
    assert json.dumps(decisions)[:-1] in after["trace_json"]
    assert after["num_events"] > stored["num_events"]

    play(resumed, random.Random(5), tag="rest")
    attempt = resumed.attempt_store.load_attempt(identifier)
    assert attempt is not None and attempt.trace["end_reason"] == "game_over"
    trace = attempt.trace
    if config.get("villain_type") == "search":
        assert all(
            event["bot_decision"]
            for event in trace["events"]
            if event["actor"] == "villain"
        )
    for track in trace["canonical_replay"]["presentation_tracks"]:
        assert [event["seq"] for event in track["events"]] == list(range(track["head"]))
    assert [d["ordinal"] for d in trace["canonical_replay"]["decisions"]] == list(
        range(len(trace["canonical_replay"]["decisions"]))
    )
    views = ReplayedViews(attempt, trace["config"].get("human_seat", 0))
    assert views.board(len(trace["events"])) is not None
    assert views.stop is not None and views.stop.status == "complete"
    assert json.loads(row(tmp_path, identifier)["numbers_json"])["complete"] is True


def test_the_opponent_replies_when_the_game_stopped_on_its_turn(
    tmp_path: Path,
) -> None:
    game = start(tmp_path, commands=2, auto_pass=False)
    identifier = game.match_id

    class Killed(Exception):
        pass

    def killed(_: dict[str, Any]) -> None:
        if not server._is_hero_turn(game.obs, game.human_seat):
            raise Killed

    # The person's step is saved before the opponent is asked; stop there.
    for index in range(40):
        frame = game.current_message()["frame"]
        offer = next(o for o in frame["offers"] if "Pass" in o["label"])
        try:
            game.hero_command(
                {
                    "command_id": f"k.{index}",
                    "match_id": identifier,
                    "expected_revision": frame["revision"],
                    "prompt_id": frame["prompt"]["id"],
                    "offer_id": offer["id"],
                    "answers": [],
                },
                committed=killed,
            )
        except Killed:
            break
    else:
        pytest.fail("The opponent never had to act.")
    stored = row(tmp_path, identifier)
    assert json.loads(stored["trace_json"])["events"][-1]["actor"] == "hero"

    resumed = server.GameSession(tmp_path)
    frame = resumed.resume(identifier)["frame"]

    assert frame["status"] == "ready" and frame["prompt"] is not None
    events = resumed.trace.events
    assert len(events) > stored["num_events"]
    assert events[stored["num_events"]].actor == "villain"
    play(resumed, random.Random(5), tag="rest")
    assert row(tmp_path, identifier)["status"] == "completed"


def test_stops_changed_during_the_game_are_the_stops_it_resumes_with(
    tmp_path: Path,
) -> None:
    game = start(tmp_path, commands=4, auto_pass=True)
    stops = {"my": ["main1"], "opponent": []}
    live = game.set_stops(stops, False, True)["frame"]
    stored = row(tmp_path, game.match_id)
    assert stored["revision"] > stored["num_events"]

    frame = server.GameSession(tmp_path).resume(game.match_id)["frame"]

    assert frame["revision"] == live["revision"]
    assert frame["stops"] == live["stops"]
    assert frame["stops"]["my"] == ["main1"] and not frame["stops"]["stop_on_stack"]
    assert frame["offers"] == live["offers"]


def test_a_record_the_engine_no_longer_plays_ends_and_stays_replayable(
    tmp_path: Path,
) -> None:
    game = start(tmp_path)
    identifier = game.match_id
    trace = json.loads(row(tmp_path, identifier)["trace_json"])
    trace["events"][3]["action_description"] = "Something else"
    rewrite(
        tmp_path,
        identifier,
        "UPDATE attempts SET trace_json=? WHERE id=?",
        json.dumps(trace),
    )

    resumed = server.GameSession(tmp_path)
    with pytest.raises(ValueError, match="record_diverged.*step 4"):
        resumed.resume(identifier)

    stored = row(tmp_path, identifier)
    assert (stored["status"], stored["ending_reason"]) == ("stopped", "record_diverged")
    assert stored["winner"] is None and resumed.env is None
    attempt = resumed.attempt_store.load_attempt(identifier)
    assert attempt is not None and attempt.trace["end_reason"] == "record_diverged"
    recording = Recording("game", attempt.trace, attempt)
    assert recording.read("latest")["observation"] == trace["final_observation"]
    with pytest.raises(ValueError, match="resume_unavailable"):
        resumed.resume(identifier)


def test_a_missing_opponent_refuses_and_leaves_the_game_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = start(tmp_path)
    identifier = game.match_id
    stored = row(tmp_path, identifier)

    def missing(*_: Any, **__: Any) -> None:
        raise ValueError("policy_unavailable: That policy is not registered.")

    resumed = server.GameSession(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(server, "build_villain_policy", missing)
        with pytest.raises(ValueError, match="policy_unavailable"):
            resumed.resume(identifier)
    assert dict(row(tmp_path, identifier)) == dict(stored)
    assert resumed.attempt_store.unfinished(identifier)

    assert resumed.resume(identifier)["frame"]["match_id"] == identifier


@pytest.mark.parametrize(
    ("status", "ending"),
    [
        ("interrupted", "server_restart"),
        ("interrupted", "server_shutdown"),
        ("stopped", "session_expired"),
        ("stopped", "new_game"),
    ],
)
def test_a_game_an_earlier_build_closed_is_reopened_when_resumed(
    tmp_path: Path, status: str, ending: str
) -> None:
    game = start(tmp_path)
    identifier = game.match_id
    if status == "stopped":
        game.close(ending)
    rewrite(
        tmp_path,
        identifier,
        "UPDATE attempts SET status=?, ending_reason=?, ended_at='2026-01-01' WHERE id=?",
        status,
        ending,
    )

    resumed = server.GameSession(tmp_path)
    assert resumed.attempt_store.unfinished(identifier)
    frame = resumed.resume(identifier)["frame"]

    assert frame["status"] == "ready"
    stored = row(tmp_path, identifier)
    assert (stored["status"], stored["ending_reason"], stored["ended_at"]) == (
        "active",
        None,
        None,
    )
    assert json.loads(stored["trace_json"])["end_reason"] == "active"
    play(resumed, random.Random(5), tag="rest")
    assert row(tmp_path, identifier)["status"] == "completed"


def test_only_an_unfinished_played_game_resumes(tmp_path: Path) -> None:
    game = start(tmp_path, commands=400)
    assert row(tmp_path, game.match_id)["status"] == "completed"
    other = server.GameSession(tmp_path)
    for identifier in (game.match_id, "no-such-game"):
        with pytest.raises(ValueError, match="resume_unavailable"):
            other.resume(identifier)
    stopped = start(tmp_path)
    stopped.close("error")
    with pytest.raises(ValueError, match="resume_unavailable"):
        other.resume(stopped.match_id)


def test_replacing_or_releasing_a_game_ends_nothing(tmp_path: Path) -> None:
    game = start(tmp_path)
    first = game.match_id
    game.new_game({"hero_deck": BOLT, "villain_deck": BOLT, "seed": 2})
    second = game.match_id
    game.release()

    assert game.env is None
    for identifier in (first, second):
        assert row(tmp_path, identifier)["status"] == "active"
        assert game.attempt_store.unfinished(identifier)
    # Another process opening the store ends neither.
    rewrite(tmp_path, first, "UPDATE attempts SET process_id='other' WHERE id=?")
    assert server.AttemptStore(tmp_path / "play.sqlite").unfinished(first)
    assert row(tmp_path, first)["status"] == "active"


def test_numbers_of_an_unfinished_game_follow_its_stored_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    game = start(tmp_path)
    identifier = game.match_id

    numbers = server.get_game_numbers(identifier)["numbers"]

    assert numbers is not None and numbers["complete"] is False
    assert json.loads(row(tmp_path, identifier)["numbers_json"]) == numbers
    play(game, random.Random(5), commands=1, tag="next")
    assert row(tmp_path, identifier)["numbers_json"] is None
    assert server.get_game_numbers(identifier)["numbers"]["complete"] is False
