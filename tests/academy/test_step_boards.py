"""The person's board after each told line travels with the frame as patches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from managym.academy import server
from managym.academy.step_boards import StepBoard, patch, wire_boards
from tests.academy.test_explained_changes import _command, _reaper_scenario


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
    assert node["o"]["battlefield"]["a"][0] == {
        "k": 2,
        "p": {"o": {"tapped": {"=": True}}},
    }


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


def test_each_opponent_step_reaches_the_table_on_its_own_board(tmp_path) -> None:
    game = server.GameSession(tmp_path)
    game.new_game({"villain_type": "random", "seed": 11})
    kept: list[StepBoard] = []
    keep = game._keep_step_board

    def recording(events: list[dict[str, Any]], positions: list[Any]) -> None:
        # The list is capped (oldest dropped), so new boards are found by identity.
        before = {id(board) for board in game._step_boards}
        keep(events, positions)
        kept.extend(board for board in game._step_boards if id(board) not in before)

    game._keep_step_board = recording  # type: ignore[method-assign]
    rebuilt_any = differed = 0
    for _ in range(80):
        if rebuilt_any >= 5 and differed >= 5:
            break
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
            # The board after the line that made the range's last event, exactly.
            assert board == next(k.projection for k in kept if k.last == entry["last"])
            assert entry["first"] in seqs and entry["last"] in seqs
            # The person's view: the opponent's hand is never named.
            assert all(card.get("name") is None for card in board["opponent"]["hand"])
            differed += board != update["frame"]["projection"]
        rebuilt_any += bool(sent)
    assert rebuilt_any >= 5
    assert differed >= 5


def _opponent_creature(board: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next(
        (card for card in board["opponent"]["battlefield"] if card["name"] == name),
        None,
    )


def test_a_burn_spell_is_seen_on_the_stack_then_as_damage_then_as_a_death(
    tmp_path: Path,
) -> None:
    """One step resolves the Bolt, kills the Ogre and stacks the Reaper's trigger;
    each of those lines has the board as it was just after it."""
    session = _reaper_scenario(tmp_path)
    _command(session, "Cast Lightning Bolt", "cast")
    update = _command(session, "Target Gray Ogre", "target")["update"]
    sent = update["boards"]
    boards = rebuild(update["frame"]["projection"], sent)
    kinds = {event["seq"]: event["kind"]["kind"] for event in update["presentation"]}

    def board_of(kind: str) -> dict[str, Any]:
        seq = next(seq for seq, told in kinds.items() if told == kind)
        return next(
            board
            for entry, board in zip(sent, boards, strict=True)
            if entry["first"] <= seq <= entry["last"]
        )

    cast = board_of("cast")
    assert board_of("targeted") is cast
    assert [item["label"] for item in cast["stack_objects"]] == ["Lightning Bolt"]
    ogre = _opponent_creature(cast, "Gray Ogre")
    assert ogre is not None and ogre["damage"] == 0

    damaged = board_of("damage")
    assert board_of("resolved") is damaged
    assert damaged["stack_objects"] == []
    ogre = _opponent_creature(damaged, "Gray Ogre")
    assert ogre is not None and ogre["damage"] == 3

    died = board_of("died")
    assert _opponent_creature(died, "Gray Ogre") is None
    assert died["stack_objects"] == []

    triggered = board_of("triggered")
    assert [item["kind"] for item in triggered["stack_objects"]] == [
        "triggered_ability"
    ]
    # A historical board holds no decision: none was published there.
    assert "decision_context" not in cast
    assert "decision_context" not in died


def test_trace_retains_per_line_boards_and_recording_read_returns_them(
    tmp_path: Path,
) -> None:
    """Each transition's `TraceEvent.boards` survives persistence and `Recording.read`
    rebuilds the cast, damage and death boards back from the consequences board."""
    from managym.academy import trace as trace_store
    from managym.academy.recording import Recording

    session = _reaper_scenario(tmp_path)
    _command(session, "Cast Lightning Bolt", "cast")
    update = _command(session, "Target Gray Ogre", "target")["update"]
    kinds = {event["seq"]: event["kind"]["kind"] for event in update["presentation"]}
    trace_id = session.trace_id
    assert trace_id is not None
    session.close(end_reason="disconnect")

    payload = trace_store.load_trace(trace_id, tmp_path)
    recording = Recording(trace_id, payload)
    # Find the consequences position whose transition emitted the Bolt cast/damage/died events.
    cast_seq = next(seq for seq, kind in kinds.items() if kind == "cast")
    damage_seq = next(seq for seq, kind in kinds.items() if kind == "damage")
    died_seq = next(seq for seq, kind in kinds.items() if kind == "died")
    consequences_pos = next(
        pos
        for pos in recording.positions
        if pos.kind == "consequences"
        and any(
            int(ev["seq"]) == cast_seq
            for ev in payload["events"][pos.event_index].get("presentation", [])
        )
    )
    read = recording.read(consequences_pos.key)
    assert read["boards"], "expected mid-step boards on the consequences transition"
    rebuilt = rebuild(read["observation"], read["boards"])

    def board_of(seq: int) -> dict[str, Any]:
        for entry, board in zip(read["boards"], rebuilt, strict=True):
            if entry["first"] <= seq <= entry["last"]:
                return board
        return read["observation"]

    cast = board_of(cast_seq)
    assert [item["label"] for item in cast["stack_objects"]] == ["Lightning Bolt"]
    ogre_before = _opponent_creature(cast, "Gray Ogre")
    assert ogre_before is not None and ogre_before["damage"] == 0

    damaged = board_of(damage_seq)
    assert damaged["stack_objects"] == []
    ogre_damaged = _opponent_creature(damaged, "Gray Ogre")
    assert ogre_damaged is not None and ogre_damaged["damage"] == 3

    died = board_of(died_seq)
    assert _opponent_creature(died, "Gray Ogre") is None


def test_auto_passes_emit_null_caused_by() -> None:
    """Priority-stop automatic passes are automation (`caused_by: None`), never human moves."""
    game = server.GameSession()
    game.new_game(
        {
            "villain_type": "random",
            "seed": 0,
            "hero_deck": "interactive",
            "villain_deck": "interactive",
            "auto_pass": True,
            "stops": {"my": ["main1"], "opponent": []},
        }
    )
    for _ in range(30):
        frame = game.current_recovery("explicit_resync")["frame"]
        if frame["status"] != "ready":
            break
        offer = next(
            (o for o in frame["offers"] if o["verb"] == "play_land"),
            next(
                (o for o in frame["offers"] if o["verb"] == "pass_priority"),
                frame["offers"][0],
            ),
        )
        game.hero_command(
            {
                "command_id": f"auto-check-{frame['prompt']['id']}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
    assert game.trace is not None
    auto_passes = [
        ev
        for trace_ev in game.trace.events
        if trace_ev.auto
        for ev in trace_ev.presentation
        if ev["kind"]["kind"] == "passed"
    ]
    assert auto_passes, "expected at least one priority-stop auto-pass"
    assert all(ev["caused_by"] is None for ev in auto_passes)


def test_watch_reads_never_carry_person_seat_boards(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Watch mode never records or returns person-seat `boards` patches."""
    from managym.academy import trace as trace_store
    from managym.academy.recording import Recording
    from managym.academy.watch import WatchRequest, create_watch, step_watch

    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    created = create_watch(WatchRequest(players=["random", "random"], seed=5))
    trace_id = created["id"]
    try:
        revision = 0
        for _ in range(12):
            result = step_watch(trace_id, revision)
            revision = result["revision"]
            if result.get("game_over"):
                break
    finally:
        server.end_game(trace_id, "test")

    payload = trace_store.load_trace(trace_id, tmp_path)
    assert all("boards" not in ev for ev in payload["events"])
    recording = Recording(trace_id, payload)
    for pos in recording.positions:
        if pos.kind == "consequences":
            assert recording.read(pos.key)["boards"] == []
