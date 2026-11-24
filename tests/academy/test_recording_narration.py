"""A recorded Play game names the opponent's moves as the person saw them."""

from pathlib import Path
from typing import Any

from managym.academy import server
from managym.academy.recording import OwnStep, Recording, narrate_move, own_steps

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


def test_live_log_narrates_the_opponent_as_review_does(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    live: list[str] = []
    for index in range(120):
        frame = game.current_message()["frame"]
        if frame["status"] == "game_over":
            break
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
        live += [
            line["text"]
            for line in result["update"].get("log", [])
            if line["actor"] == "villain"
        ]
    payload = game.attempt_store.load(game.match_id)
    assert payload is not None
    recording = Recording(game.match_id, payload)
    review = [
        label
        for position in recording.positions
        if position.kind == "consequences"
        and payload["events"][position.event_index]["actor"] == "villain"
        and (label := recording.metadata(position)["label"]) is not None
    ]
    # Live play prints the same moves Review does, never the private action.
    assert live == review
    assert any(text.startswith("Cast Lightning Bolt, targeting Seat ") for text in live)
    assert not any(text.startswith(("Target ", "Pay ")) for text in live)


def test_your_own_moves_read_as_the_opponent_s_do(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            "hero_deck": {"Mountain": 10, "Lightning Bolt": 10},
            "villain_deck": {"Island": 20},
            "villain_type": "passive",
            "seed": 5,
            "auto_pass": False,
        }
    )
    live: list[str] = []
    for index in range(150):
        frame = game.current_message()["frame"]
        if frame["status"] == "game_over":
            break
        # Act whenever there is something to do: play lands, cast and aim Bolt.
        offer = next(
            (
                o
                for o in frame["offers"]
                if o["action_type"] != "PRIORITY_PASS_PRIORITY"
            ),
            frame["offers"][0],
        )
        result = game.hero_command(
            {
                "command_id": f"act.{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"
        live += [
            line["text"]
            for line in result["update"].get("log", [])
            if line["actor"] == "hero"
        ]
    payload = game.attempt_store.load(game.match_id)
    assert payload is not None
    recording = Recording(game.match_id, payload)
    review = [
        label
        for position in recording.positions
        if position.kind == "consequences"
        and payload["events"][position.event_index]["actor"] == "hero"
        and not payload["events"][position.event_index].get("auto")
        and (label := recording.metadata(position)["label"]) is not None
    ]
    # One line per move, the same live and in Review: a cast names its target
    # once it is on the stack, never the steps that announced and paid for it.
    assert live == review
    assert "Play Mountain" in live
    assert any(text.startswith("Cast Lightning Bolt, targeting ") for text in live)
    assert not any(text.startswith(("Target ", "Pay ", "Tap ")) for text in live)


def board(stage: str, **context: Any) -> dict[str, Any]:
    return {"decision_context": {"actor": 0, "stage": stage, **context}}


def own(stage: str, chosen: int, *offers: str) -> OwnStep:
    return OwnStep(stage, offers[chosen], "X", offers, chosen)


def test_your_own_unseen_moves_are_named_as_you_made_them() -> None:
    order = ("Put on stack next — Ally — When…", "Put on stack next — Tiger-Seal — At…")
    three = ("Put on stack next — Kyoshi — When…", *order)
    # Trigger ordering is named once the last trigger is placed, in order.
    assert (
        narrate_move(
            board("order_triggers"),
            board("order_triggers"),
            [],
            [own("order_triggers", 0, *three)],
        )
        is None
    )
    assert (
        narrate_move(
            board("order_triggers"),
            board("priority"),
            [],
            [own("order_triggers", 0, *three), own("order_triggers", 1, *order)],
        )
        == "Ordered triggers: Kyoshi, then Tiger-Seal, then Ally"
    )
    assert (
        narrate_move(
            board("learn"),
            board("priority"),
            [],
            [own("learn", 0, "Take Accumulate Wisdom")],
        )
        == "Learn: take Accumulate Wisdom"
    )
    # Declining reads once, not as "Learn: decline Learn".
    assert (
        narrate_move(
            board("learn"),
            board("priority"),
            [],
            [own("learn", 1, "Take Accumulate Wisdom", "Decline Learn")],
        )
        == "Declined to learn"
    )
    paid = board(
        "payment",
        announcing="activated_ability",
        source={"label": "Fire Nation Cadets"},
    )
    assert (
        narrate_move(paid, board("priority"), [], [own("payment", 0, "Pay the cost")])
        == "Activate Fire Nation Cadets"
    )
    # Another seat's hidden choices stay unnamed.
    assert narrate_move(board("learn"), board("priority"), []) is None


def test_a_resolution_s_choices_are_one_line_for_its_source() -> None:
    wisdom = {"label": "Accumulate Wisdom"}
    look = OwnStep(
        "private_inspection",
        "Put Island into your hand",
        "SELECT_CARD",
        ("Put Pop Quiz into your hand", "Put Island into your hand"),
        1,
        "Accumulate Wisdom",
    )
    order = OwnStep(
        "order_library_bottom",
        "Put Pop Quiz next on the bottom",
        "SELECT_CARD",
        (
            "Put Pop Quiz next on the bottom",
            "Put Igneous Inspiration next on the bottom",
        ),
        0,
        "Accumulate Wisdom",
    )
    # The first choice goes straight on to the next one for the same spell.
    assert (
        narrate_move(
            board("private_inspection", source=wisdom),
            board("order_library_bottom", source=wisdom),
            [],
            [look],
        )
        is None
    )
    # The last one names them all, the card placed for the person included.
    assert (
        narrate_move(
            board("order_library_bottom", source=wisdom),
            board("priority"),
            [],
            [look, order],
        )
        == "Accumulate Wisdom: kept Island, put Pop Quiz, then Igneous Inspiration"
        " on the bottom"
    )


def test_repeated_cards_placed_together_read_once_with_their_count() -> None:
    offers = (
        "Put Island next on the bottom",
        "Put Island next on the bottom",
        "Put Pop Quiz next on the bottom",
    )
    first = OwnStep(
        "order_library_bottom", offers[0], "SELECT_CARD", offers, 0, "Wisdom"
    )
    second = OwnStep(
        "order_library_bottom", offers[1], "SELECT_CARD", offers[1:], 0, "Wisdom"
    )
    assert (
        narrate_move(
            board("order_library_bottom", source={"label": "Wisdom"}),
            board("priority"),
            [],
            [first, second],
        )
        == "Wisdom: put Island ×2, then Pop Quiz on the bottom"
    )


def test_a_reveal_is_the_move_another_seat_made() -> None:
    revealed = [{"kind": "revealed", "cards": ["Kyoshi Warriors"], "player": 0}]
    assert (
        narrate_move(board("resolution_choice"), board("priority"), revealed)
        == "Reveal Kyoshi Warriors"
    )
    # The person's own move names the card in its own words.
    assert (
        narrate_move(
            board("learn"),
            board("priority"),
            revealed,
            [own("learn", 0, "Take Kyoshi Warriors")],
        )
        == "Learn: take Kyoshi Warriors"
    )


def assignment(actor: int, chosen: int, *offers: str) -> dict[str, Any]:
    return {
        "actor": actor,
        "action": chosen,
        "action_description": offers[chosen],
        "actions": [{"type": "CHOOSE_MODE", "description": o} for o in offers],
        "observation": {"decision_context": {"stage": "assign_combat_damage"}},
    }


def test_an_attacker_s_damage_among_its_blockers_is_one_move() -> None:
    elves = (
        "Hill Giant assigns 1 to Llanowar Elves, 2 left to assign",
        "Hill Giant assigns 2 to Llanowar Elves, 1 left to assign",
    )
    bears = ("Hill Giant assigns 1 to Grizzly Bears, 0 to defending player",)
    events = [
        {
            "actor": 1,
            "action": 0,
            "action_description": "Pass priority",
            "actions": [{"type": "PRIORITY_PASS_PRIORITY"}],
            "observation": {"decision_context": {"stage": "priority"}},
        },
        assignment(0, 1, *elves),
        assignment(0, 0, *bears),
    ]
    steps = own_steps(events, 2)
    assert [step.chosen for step in steps] == [elves[1], bears[0]]
    assert narrate_move(
        board("assign_combat_damage"), board("priority"), [], steps
    ) == (
        "Hill Giant assigns 2 to Llanowar Elves, 1 left to assign, "
        "Hill Giant assigns 1 to Grizzly Bears, 0 to defending player"
    )
