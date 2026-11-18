"""Land plays and discards reach the person's presentation track as the public
objects they became, so the table can tell and point at every move."""

from pathlib import Path
from typing import Any

from managym.academy import server
from test_recording_narration import play_passing

CONFIG: dict[str, Any] = {
    # The person never plays a land, so their hand passes seven and they
    # discard at cleanup; the opponent plays Mountains.
    "hero_deck": {"Island": 20},
    "villain_deck": {"Mountain": 10, "Lightning Bolt": 10},
    "villain_type": "random",
    "seed": 5,
    "auto_pass": False,
}


def test_land_plays_and_discards_name_the_objects_on_the_table(
    tmp_path: Path,
) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    play_passing(game, 60)
    kinds = [event["kind"] for event in game.presentation_events]
    projection = game.current_message()["frame"]["projection"]
    human = projection["agent"]["player_index"]

    lands = [kind for kind in kinds if kind["kind"] == "land_played"]
    battlefield = {p["id"]: p["name"] for p in projection["opponent"]["battlefield"]}
    assert lands and all(kind["player"] == 1 - human for kind in lands)
    assert {battlefield.get(kind["object"]["entity"]) for kind in lands} == {
        "Mountain"
    }

    discards = [kind for kind in kinds if kind["kind"] == "discarded"]
    graveyard = {card["id"]: card["name"] for card in projection["agent"]["graveyard"]}
    assert discards and all(kind["player"] == human for kind in discards)
    assert {
        graveyard.get(obj["entity"]) for kind in discards for obj in kind["objects"]
    } == {"Island"}
