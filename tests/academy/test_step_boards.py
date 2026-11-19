"""The person's board after each step of a batch travels with the frame as patches."""

from __future__ import annotations

from typing import Any

from managym.academy import server
from managym.academy.step_boards import StepBoard, patch, wire_boards


def apply(base: Any, node: dict[str, Any]) -> Any:
    """The patch format as `academy/ui/src/lib/board/boards.ts` applies it."""
    if "=" in node:
        return node["="]
    if "a" in node:
        by_id = {item["id"]: item for item in base}
        return [
            entry["="]
            if "=" in entry
            else apply(by_id[entry["k"]], entry["p"])
            if "p" in entry
            else by_id[entry["k"]]
            for entry in node["a"]
        ]
    out = {key: value for key, value in base.items() if key not in node.get("x", [])}
    for key, child in node.get("o", {}).items():
        out[key] = apply(base.get(key), child)
    return out


def rebuild(projection: dict[str, Any], sent: list[dict[str, Any]]) -> list[Any]:
    boards: list[Any] = []
    after: Any = projection
    for entry in reversed(sent):
        after = apply(after, entry["patch"])
        boards.append(after)
    return boards[::-1]


def test_a_patch_turns_one_board_into_the_next() -> None:
    before = {
        "turn": {"step": "MAIN", "number": 3},
        "battlefield": [
            {"id": 1, "name": "Ogre", "tapped": False},
            {"id": 2, "name": "Elf", "tapped": False},
        ],
        "life": [20, 20],
        "gone": True,
    }
    after = {
        "turn": {"step": "COMBAT", "number": 3},
        "battlefield": [
            {"id": 2, "name": "Elf", "tapped": True},
            {"id": 7, "name": "Ally", "tapped": False},
        ],
        "life": [20, 17],
        "new": {"a": 1},
    }
    node = patch(before, after)
    assert node is not None
    assert apply(before, node) == after
    assert patch(after, after) is None
    # The unchanged parts are named, not sent: Ogre left, Elf is patched in place.
    assert node["o"]["battlefield"]["a"][0] == {"k": 2, "p": {"o": {"tapped": {"=": True}}}}


def test_equal_boards_merge_and_a_board_equal_to_the_frame_is_not_sent() -> None:
    frame = {"battlefield": [{"id": 1, "tapped": True}]}
    attacking = {"battlefield": [{"id": 1, "tapped": True}, {"id": 2, "tapped": True}]}
    boards = [
        StepBoard(first=0, last=1, projection=attacking),
        StepBoard(first=2, last=2, projection=attacking),
        StepBoard(first=3, last=4, projection=frame),
    ]
    sent = wire_boards(boards, frame)
    assert [(entry["first"], entry["last"]) for entry in sent] == [(0, 2)]
    assert rebuild(frame, sent) == [attacking]


def test_each_opponent_step_reaches_the_table_on_its_own_board() -> None:
    game = server.GameSession()
    game.new_game({"villain_type": "random", "seed": 11})
    kept: list[StepBoard] = []
    keep = game._keep_step_board

    def recording(events: list[dict[str, Any]]) -> None:
        keep(events)
        kept.append(game._step_boards[-1])

    game._keep_step_board = recording  # type: ignore[method-assign]
    rebuilt_any = differed = 0
    for _ in range(80):
        frame = game.current_recovery("explicit_resync")["frame"]
        if frame["status"] != "ready":
            break
        kept.clear()
        offer = next(
            (o for o in frame["offers"] if o["verb"] in ("play", "cast")),
            next(
                (o for o in frame["offers"] if o["verb"] == "pass_priority"),
                frame["offers"][0],
            ),
        )
        update = game.hero_command(
            {
                "command_id": f"boards-{frame['prompt']['id']}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )["update"]
        sent = update.get("boards", [])
        boards = rebuild(update["frame"]["projection"], sent)
        seqs = {event["seq"] for event in update["presentation"]}
        for entry, board in zip(sent, boards, strict=True):
            # The board after the step that made the range's last event, exactly.
            assert board == next(k.projection for k in kept if k.last == entry["last"])
            assert entry["first"] in seqs and entry["last"] in seqs
            # The person's view: the opponent's hand is never named.
            assert all(card.get("name") is None for card in board["opponent"]["hand"])
            differed += board != update["frame"]["projection"]
        rebuilt_any += bool(sent)
    assert rebuilt_any >= 5
    assert differed >= 5
