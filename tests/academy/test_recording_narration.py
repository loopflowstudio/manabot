"""A recorded Play game names the opponent's moves as the person saw them."""

from pathlib import Path
from typing import Any

from managym.academy import server
from managym.academy.recording import Recording

CONFIG: dict[str, Any] = {
    "hero_deck": {"Island": 20},
    "villain_deck": {"Mountain": 10, "Lightning Bolt": 10},
    "villain_type": "random",
    "seed": 5,
    "auto_pass": False,
}


def play_passing(game: server.GameSession, commands: int) -> None:
    for index in range(commands):
        frame = game.current_message()["frame"]
        if frame["status"] == "game_over":
            return
        offer = next(
            (
                o
                for o in frame["offers"]
                if o["action_type"] == "PRIORITY_PASS_PRIORITY"
            ),
            frame["offers"][0],
        )
        result = game.hero_command(
            {
                "command_id": f"pass.{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"


def test_opponent_moves_are_narrated_from_the_person_s_view(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    play_passing(game, 120)
    payload = game.attempt_store.load(game.match_id)
    assert payload is not None
    recording = Recording(game.match_id, payload)
    theirs = [
        (position, recording.metadata(position)["label"])
        for position in recording.positions
        if position.kind == "consequences"
        and payload["events"][position.event_index]["actor"] == "villain"
    ]
    labels = [label for _, label in theirs]
    assert "Play Mountain" in labels
    casts = [label for label in labels if label and label.startswith("Cast ")]
    assert casts and all(
        label.startswith("Cast Lightning Bolt, targeting ") for label in casts
    )
    # Steps nobody else could tell apart (choosing a target, paying) are
    # named once, where the spell reaches the stack; the private action
    # never stands in for what was seen.
    assert not any(
        label and label.startswith(("Target ", "Pay ", "Opponent decision"))
        for label in labels
    )
    assert None in labels
    # The read names the same move, and its history names the spell.
    position, label = next(item for item in theirs if item[1] in casts)
    read = recording.read(position.key)
    assert read["description"] == label
    assert any(
        "Lightning Bolt" in names["objects"].values()
        or "Lightning Bolt" in names["stacks"].values()
        for names in read["history_labels"]
    )
