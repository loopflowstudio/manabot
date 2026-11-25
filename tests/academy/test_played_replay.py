"""A played game plays again: Review reads each seat's view by re-executing it."""

import json
from pathlib import Path
import random
from typing import Any

import pytest

from managym.academy import server
from managym.academy.attempts import RetainedAttempt
from managym.academy.recording import Recording, person_board

BOLT = {"Mountain": 10, "Lightning Bolt": 10}


def play(game: server.GameSession, rng: random.Random, commands: int = 400) -> None:
    for index in range(commands):
        frame = game.current_message()["frame"]
        if frame["status"] == "game_over":
            return
        offer = rng.choice(frame["offers"])
        result = game.hero_command(
            {
                "command_id": f"c.{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"


def played(tmp_path: Path, **config: Any) -> tuple[server.GameSession, RetainedAttempt]:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": BOLT,
            "villain_deck": BOLT,
            "villain_type": "random",
            "seed": 11,
            "auto_pass": False,
            **config,
        }
    )
    play(game, random.Random(3))
    attempt = game.attempt_store.load_attempt(game.match_id)
    assert attempt is not None
    return game, attempt


def hand_names(board: dict[str, Any]) -> list[str | None]:
    return [card.get("name") for card in board["agent"]["hand"]]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"villain_type": "search", "villain_sims": 4},
        {"human_seat": 1, "decklists": "closed"},
        {"auto_pass": True},
    ],
    ids=["random", "search", "second-seat-closed", "priority-stops"],
)
def test_every_position_of_a_finished_game_replays_to_your_own_view(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    _, attempt = played(tmp_path, **config)
    trace = attempt.trace
    assert trace["end_reason"] == "game_over"
    recording = Recording("game", trace, attempt)
    seat = recording.human_seat
    events = trace["events"]
    for position in recording.positions:
        read = recording.read(position.key)
        if position.kind == "choice":
            continue
        board = read["observation"]
        assert read["views_stopped"] is None
        assert board["agent"]["player_index"] == seat
        # Your whole hand, by name, whoever acts next.
        assert all(hand_names(board)) and not board["agent"].get("hand_hidden_count")
        index = position.event_index + (position.kind == "consequences")
        if index < len(events) and events[index]["actor"] == "hero":
            assert board == person_board(
                events[index]["observation"], seat, winner=trace["winner"]
            )
    assert recording.views is not None and recording.views.stop is not None
    assert recording.views.stop.status == "complete"
    assert recording.views.stop.winner == trace["winner"]
    assert recording.read("latest")["observation"] == trace["final_observation"]


def test_your_hand_stays_when_the_opponent_acts_next(tmp_path: Path) -> None:
    _, attempt = played(tmp_path)
    events = attempt.trace["events"]
    recording = Recording("game", attempt.trace, attempt)
    handover = next(
        i
        for i, event in enumerate(events[:-1])
        if event["actor"] == "hero" and events[i + 1]["actor"] == "villain"
    )
    position = recording.by_key[f"after:{handover}"]
    stored = recording._board(position)
    assert stored["agent"]["hand_hidden_count"] and not stored["agent"]["hand"]
    replayed = recording.read(position.key)["observation"]["agent"]
    assert len(replayed["hand"]) == stored["agent"]["hand_hidden_count"]
    assert all(card["name"] for card in replayed["hand"])


def test_show_both_hands_after_a_complete_game(tmp_path: Path) -> None:
    _, attempt = played(tmp_path)
    recording = Recording("game", attempt.trace, attempt)
    for key in ("after:0", "latest"):
        read = recording.read(key, reveal=True)
        assert read["can_reveal"] and read["revealed"]
        opponent = read["observation"]["opponent"]
        assert opponent["hand_hidden_count"] == 0
        assert all(card["name"] for card in opponent["hand"])
    # The ordinary read still hides it.
    assert not recording.read("after:0")["observation"]["opponent"]["hand"]


def test_an_unfinished_game_never_shows_both_hands(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(
        {"hero_deck": BOLT, "villain_deck": BOLT, "villain_type": "random", "seed": 11}
    )
    play(game, random.Random(3), commands=6)
    game.close(end_reason="new_game")
    attempt = game.attempt_store.load_attempt(game.match_id)
    assert attempt is not None and attempt.trace["end_reason"] == "new_game"
    recording = Recording("game", attempt.trace, attempt)
    assert recording.read("latest")["observation"] == attempt.trace["final_observation"]
    assert recording.views is not None and recording.views.stop is not None
    assert recording.views.stop.status == "interrupted"
    assert not recording.read("after:0")["can_reveal"]
    with pytest.raises(ValueError, match="spectator_view_unavailable"):
        recording.read("after:0", reveal=True)


def test_replay_stops_where_the_stored_track_differs(tmp_path: Path) -> None:
    _, attempt = played(tmp_path)
    trace = json.loads(json.dumps(attempt.trace))
    events = trace["events"]
    broken = next(
        i for i, event in enumerate(events) if i > 5 and event["actor"] == "villain"
    )
    events[broken]["action_description"] = "Something else"
    recording = Recording("game", trace, RetainedAttempt(trace, attempt.world_json))
    assert recording.read(f"after:{broken - 1}")["views_stopped"] is None
    after = recording.by_key[f"after:{broken}"]
    read = recording.read(after.key)
    assert (
        read["views_stopped"] and f"through step {broken + 1} " in read["views_stopped"]
    )
    assert read["observation"] == recording._board(after)
    assert not read["can_reveal"]


def test_a_game_from_another_card_set_opens_from_its_stored_track(
    tmp_path: Path,
) -> None:
    _, attempt = played(tmp_path)
    world = {
        **json.loads(attempt.world_json),
        "content_digest": "another-card-set",
        "world_version": "another-world",
    }
    recording = Recording(
        "game", attempt.trace, RetainedAttempt(attempt.trace, json.dumps(world))
    )
    read = recording.read("after:0")
    assert read["observation"] == recording._board(recording.by_key["after:0"])
    assert read["views_stopped"] and "another-world" in read["views_stopped"]


def test_every_step_records_when_it_was_committed(tmp_path: Path) -> None:
    _, attempt = played(tmp_path)
    times = [event["committed_at"] for event in attempt.trace["events"]]
    assert all(times) and times == sorted(times)
