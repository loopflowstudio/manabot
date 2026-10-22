"""Authoritative semantic presentation for the Lightning Bolt vertical slice."""

from __future__ import annotations

from typing import Any

from academy import trace as trace_store
from academy.server import GameSession


def _command_payload(
    session: GameSession, label: str, command_id: str
) -> dict[str, Any]:
    prompt = session._publish_current_prompt()
    assert prompt is not None
    offer_id = next(
        index
        for index, action in enumerate(prompt.actions)
        if action["description"] == label
    )
    return {
        "command_id": command_id,
        "match_id": session.match_id,
        "expected_revision": prompt.revision,
        "prompt_id": prompt.prompt_id,
        "offer_id": offer_id,
        "answers": [],
    }


def _command(session: GameSession, label: str, command_id: str) -> dict[str, Any]:
    return session.hero_command(_command_payload(session, label, command_id))


def _bolt_scenario(trace_dir) -> GameSession:
    session = GameSession(trace_dir=trace_dir)
    session.new_game(
        {
            "villain_type": "passive",
            "hero_deck": "interactive",
            "villain_deck": "interactive",
            "auto_pass": False,
            "seed": 7,
        }
    )
    assert session.env is not None
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    session.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    session.env.scenario_force_battlefield(0, "Mountain", True)
    session.env.scenario_force_battlefield(1, "Gray Ogre", True)
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None
    session.presentation.reset()
    return session


<<<<<<< HEAD
=======
def _resolve_bolt(session: GameSession) -> dict[str, Any]:
    for index in range(12):
        result = _command(session, "Pass priority", f"response-pass.{index}")
        if any(
            event["kind"]["kind"] == "resolved" for event in session.presentation_events
        ):
            return result
    raise AssertionError("Bolt did not resolve after explicit priority responses")


>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
def test_bolt_facts_are_identical_in_live_update_and_persisted_replay(tmp_path):
    session = _bolt_scenario(tmp_path)

    cast = _command(session, "Cast Lightning Bolt", "command-cast")
    assert cast["update"]["presentation"] == []

    recovery_cursor = session.presentation.next_seq
    target_command = _command_payload(session, "Target Gray Ogre", "command-target")
    target = session.hero_command(target_command)
<<<<<<< HEAD
    live_events = target["update"]["presentation"]
=======
    assert [event["kind"]["kind"] for event in target["update"]["presentation"]] == [
        "cast",
        "targeted",
    ]
    latest = _resolve_bolt(session)
    live_events = list(session.presentation_events)
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f

    bolt_events = live_events[:5]
    assert [event["kind"]["kind"] for event in bolt_events] == [
        "cast",
        "targeted",
        "resolved",
        "damage",
        "died",
    ]
    assert [event["seq"] for event in live_events] == list(range(len(live_events)))
<<<<<<< HEAD
    assert all(event["from_revision"] == 1 for event in bolt_events)
    assert all(event["to_revision"] == 2 for event in bolt_events)
    assert all(event["caused_by"] == "command-target" for event in bolt_events)
=======
    assert all(event["from_revision"] == 1 for event in bolt_events[:2])
    assert all(event["to_revision"] == 2 for event in bolt_events[:2])
    assert all(event["caused_by"] == "command-target" for event in bolt_events[:2])
    assert all(event["from_revision"] >= 2 for event in bolt_events[2:])
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    assert all(
        1 <= event["from_revision"] < event["to_revision"] <= session.revision
        for event in live_events
    )
    assert bolt_events[3]["kind"]["amount"] == 3

    target_id = bolt_events[1]["kind"]["target"]["id"]
    assert bolt_events[4]["kind"]["objects"] == [target_id]
<<<<<<< HEAD
    battlefield = target["update"]["frame"]["projection"]["opponent"]["battlefield"]
=======
    battlefield = latest["update"]["frame"]["projection"]["opponent"]["battlefield"]
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    assert all(permanent["id"] != target_id["entity"] for permanent in battlefield)

    # Simulate an authority commit whose WebSocket response was lost. Reconnect
    # from the last received cursor reconstructs the accepted frame and exact
    # semantic tail without applying the gameplay choice again.
    trace_length = len(session.trace.events)
    accepted_revision = session.revision
    recovered = session.current_message(recovery_cursor)["recovery"]
    assert recovered["presentation_cursor"] == recovery_cursor
    assert recovered["presentation_tail"] == live_events
<<<<<<< HEAD
    assert recovered["frame"]["frame_hash"] == target["update"]["frame"]["frame_hash"]
=======
    assert recovered["frame"]["frame_hash"] == latest["update"]["frame"]["frame_hash"]
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f

    duplicate = session.hero_command(target_command)
    assert duplicate["status"] == "duplicate"
    assert duplicate["receipt"] == target["update"]["receipt"]
    assert duplicate["recovery"]["presentation_tail"] == live_events
    assert session.revision == accepted_revision
    assert len(session.trace.events) == trace_length
    assert session.presentation_events == live_events

    refreshed = session.current_message(recovery_cursor)["recovery"]
    assert refreshed["frame"] == recovered["frame"]
    assert refreshed["presentation_tail"] == recovered["presentation_tail"]

    session.close(end_reason="disconnect")
    assert not list(tmp_path.glob("*.json"))
    persisted = trace_store.load_trace(session.trace_id, tmp_path)
    replay_events = [
        event
        for trace_event in persisted["events"]
        for event in trace_event.get("presentation", [])
    ]
    assert replay_events == live_events


def test_recovery_cursor_slices_tail_and_recovers_from_gaps(tmp_path):
    session = _bolt_scenario(tmp_path)
    _command(session, "Cast Lightning Bolt", "command-cast")
<<<<<<< HEAD
    target = _command(session, "Target Gray Ogre", "command-target")
    events = target["update"]["presentation"]
=======
    _command(session, "Target Gray Ogre", "command-target")
    latest = _resolve_bolt(session)
    events = list(session.presentation_events)
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    assert session.presentation_events == events
    head = session.presentation.next_seq

    partial = session.current_recovery("reconnect", presentation_cursor=3)
    assert partial["presentation_cursor"] == 3
    assert partial["presentation_tail"] == [
        event for event in events if event["seq"] >= 3
    ]

    at_head = session.current_recovery("reconnect", presentation_cursor=head)
    assert at_head["presentation_cursor"] == head
    assert at_head["presentation_tail"] == []

    # A cursor beyond the authority head is a gap. The complete frame remains
    # canonical and the semantic tail restarts at the oldest retained address.
    gap = session.current_recovery("reconnect", presentation_cursor=99)
<<<<<<< HEAD
    assert gap["frame"]["frame_hash"] == target["update"]["frame"]["frame_hash"]
=======
    assert gap["frame"]["frame_hash"] == latest["update"]["frame"]["frame_hash"]
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    assert gap["presentation_cursor"] == 0
    assert gap["presentation_tail"] == events

    session.presentation_events = events[2:]
    expired = session.current_recovery("reconnect", presentation_cursor=0)
    assert expired["presentation_cursor"] == 2
    assert expired["presentation_tail"] == events[2:]


def test_scenario_snapshot_changes_do_not_invent_presentation(tmp_path):
    session = _bolt_scenario(tmp_path)

    # Scenario setup changed several zones, but no spell domain event has
    # committed. Snapshot state alone must never become semantic theater.
    assert (
        session.presentation.drain(
            from_revision=session.revision,
            to_revision=session.revision,
            caused_by=None,
        )
        == []
    )
