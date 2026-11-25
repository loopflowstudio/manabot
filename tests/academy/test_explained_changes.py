"""Why things happened: causes and rules on the presentation track.

A real game, played through the session: the hero's Lightning Bolt kills a
creature, the rules say why it died, and the opponent's death trigger follows
from the death and its effects from the trigger's resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from managym.academy.server import INTERACTIVE_DECK, GameSession

# The opponent's Midnight Reaper draws and deals 1 damage to it when Gray
# Ogre dies.
REAPER_DECK = {
    **{
        card: count
        for card, count in INTERACTIVE_DECK.items()
        if card != "Raging Goblin"
    },
    "Midnight Reaper": 4,
}


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


def _reaper_scenario(trace_dir: Path) -> GameSession:
    session = GameSession(trace_dir=trace_dir)
    session.new_game(
        {
            "villain_type": "passive",
            "hero_deck": "interactive",
            "villain_deck": REAPER_DECK,
            "auto_pass": False,
            "seed": 7,
        }
    )
    assert session.env is not None
    _command(session, "Keep hand", "keep")
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    session.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    session.env.scenario_force_battlefield(0, "Mountain", True)
    session.env.scenario_force_battlefield(1, "Gray Ogre", True)
    session.env.scenario_force_battlefield(1, "Midnight Reaper", True)
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None
    return session


def _bolt_the_ogre(session: GameSession) -> list[dict[str, Any]]:
    """The track's events from the cast on."""
    start = len(session.presentation_events)
    _command(session, "Cast Lightning Bolt", "cast")
    _command(session, "Target Gray Ogre", "target")
    # The passive opponent leaves its trigger for the hero to pass over.
    for index in range(4):
        told = {event["kind"]["kind"] for event in session.presentation_events[start:]}
        prompt = session._publish_current_prompt()
        if "ability_resolved" in told or prompt is None:
            break
        if not any(
            action["description"] == "Pass priority" for action in prompt.actions
        ):
            break
        _command(session, "Pass priority", f"pass-{index}")
    return list(session.presentation_events[start:])


def _first(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    return next(event for event in events if event["kind"]["kind"] == kind)


def test_a_death_names_its_rule_and_its_trigger_follows_from_it(tmp_path: Path) -> None:
    session = _reaper_scenario(tmp_path)
    events = _bolt_the_ogre(session)
    kinds = [event["kind"]["kind"] for event in events]
    assert kinds[:7] == [
        "cast",
        "targeted",
        "resolved",
        "damage",
        "died",
        "triggered",
        "ability_resolved",
    ], kinds

    cast, targeted, resolved, damage, died, triggered, ability = events[:7]
    # A move is a root; its target is part of it.
    assert "cause" not in cast and cast["group"] == cast["seq"]
    assert targeted["cause"] == cast["seq"]
    # The resolution is its own root, and what it did follows from it.
    assert "cause" not in resolved
    assert damage["cause"] == resolved["seq"]
    assert died["cause"] == resolved["seq"]
    assert died["rule"] == "lethal_damage"
    assert died["group"] == resolved["seq"]
    # The trigger follows from the death, inside the same move.
    assert triggered["cause"] == died["seq"]
    assert triggered["group"] == resolved["seq"]
    assert triggered["kind"]["controller"] == 1
    # The trigger's resolution is a root, and its effects follow from it.
    assert "cause" not in ability
    assert ability["kind"]["stack"] == triggered["kind"]["stack"]
    reaper_damage = _first(events[7:], "damage")
    assert reaper_damage["cause"] == ability["seq"]
    assert reaper_damage["kind"]["target"] == {"kind": "player", "id": 1}
    assert reaper_damage["kind"]["source"]["kind"] == "object"


def test_rules_and_causes_reach_the_persisted_replay(tmp_path: Path) -> None:
    from managym.academy import trace as trace_store

    session = _reaper_scenario(tmp_path)
    _bolt_the_ogre(session)
    live = list(session.presentation_events)
    session.close(end_reason="disconnect")
    persisted = trace_store.load_trace(session.trace_id, tmp_path)
    replay = [
        event
        for trace_event in persisted["events"]
        for event in trace_event.get("presentation", [])
    ]
    assert replay == live
    assert _first(replay, "died")["rule"] == "lethal_damage"
