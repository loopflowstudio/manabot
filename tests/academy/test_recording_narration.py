"""A recorded Play game names the opponent's moves as the person saw them."""

from collections.abc import Callable
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


def play_passing(
    game: server.GameSession,
    commands: int,
    *,
    stop_when: Callable[[], bool] | None = None,
) -> None:
    for index in range(commands):
        if stop_when is not None and stop_when():
            return
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
    elves = tuple(
        f"Colossal Dreadmaw assigns {n} to Llanowar Elves, {6 - n} left to assign"
        for n in range(7)
    )
    bears = tuple(
        f"Colossal Dreadmaw assigns {n} to Grizzly Bears, {4 - n} to defending player"
        for n in range(2, 5)
    )
    events = [
        {
            "actor": 1,
            "action": 0,
            "action_description": "Pass priority",
            "actions": [{"type": "PRIORITY_PASS_PRIORITY"}],
            "observation": {"decision_context": {"stage": "priority"}},
        },
        assignment(0, 2, *elves),
        assignment(0, 0, *bears),
    ]
    assigning = board("assign_combat_damage")
    assert narrate_move(assigning, assigning, [], own_steps(events, 1)) is None
    steps = own_steps(events, 2)
    assert [step.chosen for step in steps] == [elves[2], bears[0]]
    assert narrate_move(assigning, board("priority"), [], steps) == (
        "Colossal Dreadmaw assigns 2 to Llanowar Elves, 4 left to assign, "
        "Colossal Dreadmaw assigns 2 to Grizzly Bears, 2 to defending player"
    )


def _trace_event(
    stage: str,
    chosen: int,
    *offers: str,
    action_type: str,
    source: str | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {"stage": stage}
    if source is not None:
        context["source"] = {"label": source}
    return {
        "actor": 0,
        "action": chosen,
        "action_description": offers[chosen],
        "actions": [{"type": action_type, "description": o} for o in offers],
        "observation": {"decision_context": context},
    }


def test_choice_of_color_mana_tap_records_resolved_color_not_raw_offer() -> None:
    events = [
        _trace_event(
            "priority",
            0,
            "Tap Kyoshi Village for {G}/{W}",
            action_type="PRIORITY_ACTIVATE_ABILITY",
        ),
        _trace_event(
            "mana_color",
            1,
            "Add {G}",
            "Add {W}",
            action_type="CHOOSE_MODE",
            source="Kyoshi Village",
        ),
    ]
    # Opening the color modal waits for the chosen color instead of logging the offer.
    assert (
        narrate_move(
            board("priority"),
            board("mana_color", source={"label": "Kyoshi Village"}),
            [],
            own_steps(events, 0),
        )
        is None
    )
    # Completing the color modal logs the resolved mana produced.
    assert (
        narrate_move(
            board("mana_color", source={"label": "Kyoshi Village"}),
            board("priority"),
            [],
            own_steps(events, 1),
        )
        == "Tap Kyoshi Village for {W}"
    )

    # During a payment, opening the color modal does not leak the raw offer label,
    # and returning to payment stays quiet until the spell is cast.
    assert (
        narrate_move(
            board("payment"),
            board("mana_color", source={"label": "Gruul Guildgate"}),
            [],
            [own("payment", 0, "Tap Gruul Guildgate for {R}/{G}")],
        )
        is None
    )
    assert (
        narrate_move(
            board("mana_color", source={"label": "Gruul Guildgate"}),
            board("payment"),
            [],
            [
                own("payment", 0, "Tap Gruul Guildgate for {R}/{G}"),
                own("mana_color", 0, "Add {R}", "Add {G}"),
            ],
        )
        is None
    )

    # A resolution payment (e.g. Ward) folds the tap + color choice into one phrase.
    ward_events = [
        _trace_event(
            "resolution_payment",
            0,
            "Tap Kyoshi Village for {G}/{W}",
            action_type="PRIORITY_ACTIVATE_ABILITY",
            source="Waterfall Aerialist",
        ),
        _trace_event(
            "mana_color",
            1,
            "Add {G}",
            "Add {W}",
            action_type="CHOOSE_MODE",
            source="Kyoshi Village",
        ),
        _trace_event(
            "resolution_payment",
            0,
            "Pay the cost",
            action_type="PAY_COST",
            source="Waterfall Aerialist",
        ),
    ]
    assert (
        narrate_move(
            board("resolution_payment", source={"label": "Waterfall Aerialist"}),
            board("priority"),
            [],
            own_steps(ward_events, 2),
        )
        == "Waterfall Aerialist: tap Kyoshi Village for {W}, pay the cost"
    )

    # Even without a modal (auto-picked color), pool delta resolves the mana symbol.
    before_pool = {
        **board("priority"),
        "agent": {"player_index": 0, "mana": [0, 0, 0, 0, 0, 0]},
    }
    after_pool = {
        **board("priority"),
        "agent": {"player_index": 0, "mana": [0, 0, 0, 0, 1, 0]},
    }
    assert (
        narrate_move(
            before_pool,
            after_pool,
            [],
            [own("priority", 0, "Tap Kyoshi Village for {G}/{W}")],
        )
        == "Tap Kyoshi Village for {G}"
    )
