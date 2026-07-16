"""Authoritative semantic presentation for the Lightning Bolt vertical slice."""

from __future__ import annotations

from typing import Any

from gui import trace as trace_store
from gui.server import GameSession


def _command(session: GameSession, label: str, command_id: str) -> dict[str, Any]:
    prompt = session._publish_current_prompt()
    assert prompt is not None
    offer_id = next(
        index
        for index, action in enumerate(prompt.actions)
        if action["description"] == label
    )
    return session.hero_command(
        {
            "command_id": command_id,
            "match_id": session.match_id,
            "expected_revision": prompt.revision,
            "prompt_id": prompt.prompt_id,
            "offer_id": offer_id,
            "answers": [],
        }
    )


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


def test_bolt_facts_are_identical_in_live_update_and_persisted_replay(tmp_path):
    session = _bolt_scenario(tmp_path)

    cast = _command(session, "Cast Lightning Bolt", "command-cast")
    assert cast["update"]["presentation"] == []

    target = _command(session, "Target Gray Ogre", "command-target")
    live_events = target["update"]["presentation"]

    assert [event["kind"]["kind"] for event in live_events] == [
        "cast",
        "targeted",
        "resolved",
        "damage",
        "died",
    ]
    assert [event["seq"] for event in live_events] == list(range(5))
    assert all(event["from_revision"] == 1 for event in live_events)
    assert all(event["to_revision"] == 2 for event in live_events)
    assert all(event["caused_by"] == "command-target" for event in live_events)
    assert live_events[3]["kind"]["amount"] == 3

    target_id = live_events[1]["kind"]["target"]["id"]
    assert live_events[4]["kind"]["objects"] == [target_id]
    battlefield = target["update"]["frame"]["projection"]["opponent"]["battlefield"]
    assert all(permanent["id"] != target_id["entity"] for permanent in battlefield)

    session.close(end_reason="disconnect")
    trace_files = list(tmp_path.glob("*.json"))
    assert len(trace_files) == 1
    persisted = trace_store.load_trace(trace_files[0].stem, tmp_path)
    replay_events = [
        event
        for trace_event in persisted["events"]
        for event in trace_event.get("presentation", [])
    ]
    assert replay_events == live_events


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
