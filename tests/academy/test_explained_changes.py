"""Why things happened: causes and rules on the presentation track.

A real game, played through the session: the hero's Lightning Bolt kills a
creature, the rules say why it died, and the opponent's death trigger follows
from the death and its effects from the trigger's resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from managym.academy.server import INTERACTIVE_DECK, GameSession
from managym.experience_projection import decision_surface_view

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


def test_each_seat_gets_its_own_labels_for_the_reapers_draw(tmp_path: Path) -> None:
    session = _reaper_scenario(tmp_path)
    live = _bolt_the_ogre(session)
    for viewer, track in session.canonical_presentation.items():
        cast = _first(track, "cast")
        targeted = _first(track, "targeted")
        died = _first(track, "died")
        resolved = _first(track, "resolved")
        assert resolved["labels"] == {
            "stacks": {str(cast["kind"]["stack"]): "Lightning Bolt"}
        }
        target = targeted["kind"]["target"]["id"]
        key = f"{target['entity']}:{target['incarnation']}"
        assert targeted["labels"] == {
            "objects": {key: "Gray Ogre"},
            "stacks": {str(cast["kind"]["stack"]): "Lightning Bolt"},
        }
        assert died["labels"] == {"objects": {key: "Gray Ogre"}}

        ability = _first(track, "ability_resolved")
        draw = next(
            e
            for e in track
            if e["kind"]["kind"] == "drew" and e.get("cause") == ability["seq"]
        )
        assert draw["kind"]["player"] == 1
        if viewer == 0:
            assert draw["kind"] == {"kind": "drew", "player": 1}
            assert "labels" not in draw
            assert draw in live
            assert (
                draw
                in session.current_recovery(
                    "reconnect", presentation_cursor=draw["seq"]
                )["presentation_tail"]
            )
        else:
            card = draw["kind"]["object"]
            assert session.env is not None
            hand = decision_surface_view(session.env, 1)["agent"]["hand"]
            card_name = next(c["name"] for c in hand if c["id"] == card["entity"])
            assert draw["labels"] == {
                "objects": {f"{card['entity']}:{card['incarnation']}": card_name}
            }


def test_real_game_tracks_stay_private_per_seat(tmp_path: Path) -> None:
    session = GameSession(trace_dir=tmp_path)
    session.new_game({"villain_type": "random", "seed": 11, "auto_pass": False})
    for index in range(40):
        frame = session.current_message()["frame"]
        if frame["status"] != "ready":
            break
        offer = next(
            (o for o in frame["offers"] if o["verb"] in ("play", "cast")),
            next(
                (o for o in frame["offers"] if o["verb"] == "pass_priority"),
                frame["offers"][0],
            ),
        )
        session.hero_command(
            {
                "command_id": f"privacy-{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )

    seat0 = session.canonical_presentation[0]
    seat1 = session.canonical_presentation[1]
    assert len(seat0) == len(seat1) and len(seat0) > 0
    draws_by_player = {0: 0, 1: 0}
    for viewer, track in ((0, seat0), (1, seat1)):
        for event in track:
            kind = event["kind"]
            if kind["kind"] != "drew":
                continue
            drawer = kind["player"]
            if viewer == 0:
                draws_by_player[drawer] += 1
            if drawer == viewer:
                card = kind["object"]
                key = f"{card['entity']}:{card['incarnation']}"
                assert event["labels"]["objects"][key]
            else:
                assert "object" not in kind
                assert "labels" not in event
    assert draws_by_player[0] >= 1 and draws_by_player[1] >= 1
    assert seat0 != seat1


def test_turn_6_track_passes_activation_counters_and_review_index(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Real game covering cast, target, pass, resolve, damage, lethal death,
    death trigger, trigger resolution, draw, activation and +1/+1 counter, with
    no live log and Play Review receiving the full person track."""
    from fastapi.testclient import TestClient

    from managym.academy import server

    hero_deck = {
        **{
            card: count
            for card, count in INTERACTIVE_DECK.items()
            if card != "Raging Goblin"
        },
        "Fire Sages": 4,
    }
    owner_token = "turn6-owner-token-" + "a" * 24
    session = GameSession(trace_dir=tmp_path)
    session.new_game(
        {
            "villain_type": "passive",
            "hero_deck": hero_deck,
            "villain_deck": REAPER_DECK,
            "auto_pass": False,
            "seed": 7,
            "player_token": owner_token,
        }
    )
    assert session.env is not None
    _command(session, "Keep hand", "keep")
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    session.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    for _ in range(4):
        session.env.scenario_force_battlefield(0, "Mountain", True)
    session.env.scenario_force_battlefield(0, "Fire Sages", True)
    session.env.scenario_force_battlefield(1, "Gray Ogre", True)
    session.env.scenario_force_battlefield(1, "Midnight Reaper", True)
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None

    start = len(session.presentation_events)
    r_cast = _command(session, "Cast Lightning Bolt", "cast")
    assert "log" not in r_cast["update"]
    r_target = _command(session, "Target Gray Ogre", "target")
    assert "log" not in r_target["update"]
    # With 4 untapped Mountains + Fire Sages, hero pays for Bolt, then holds
    # priority over Bolt and Midnight Reaper's trigger; explicit passes emit `passed`.
    for index in range(6):
        told = [e["kind"]["kind"] for e in session.presentation_events[start:]]
        if "ability_resolved" in told:
            break
        prompt = session._publish_current_prompt()
        assert prompt is not None
        label = next(
            (a["description"] for a in prompt.actions if a["description"].startswith("Pay")),
            "Pass priority",
        )
        res = _command(session, label, f"step-{index}")
        assert "log" not in res["update"]

    prompt = session._publish_current_prompt()
    assert prompt is not None
    activate_label = next(
        action["description"]
        for action in prompt.actions
        if "Fire Sages" in action["description"]
    )
    r_act = _command(session, activate_label, "activate")
    assert "log" not in r_act["update"]
    # Pay mana if prompted, then let the activation resolve.
    for index in range(6):
        told = [e["kind"]["kind"] for e in session.presentation_events[start:]]
        if "counters" in told:
            break
        prompt = session._publish_current_prompt()
        assert prompt is not None
        pay = next(
            (a["description"] for a in prompt.actions if a["description"].startswith("Pay")),
            "Pass priority",
        )
        _command(session, pay, f"finish-{index}")

    events = list(session.presentation_events[start:])
    kinds = [event["kind"]["kind"] for event in events]
    for required in (
        "cast",
        "targeted",
        "passed",
        "resolved",
        "damage",
        "died",
        "triggered",
        "ability_resolved",
        "drew",
        "activated",
        "counters",
    ):
        assert required in kinds, kinds

    cast = _first(events, "cast")
    targeted = _first(events, "targeted")
    resolved = _first(events, "resolved")
    died = _first(events, "died")
    triggered = _first(events, "triggered")
    passed = _first(events, "passed")
    activated = _first(events, "activated")
    counters = _first(events, "counters")
    ability_resolutions = [
        e for e in events if e["kind"]["kind"] == "ability_resolved"
    ]
    assert len(ability_resolutions) == 2
    trigger_resolved, activation_resolved = ability_resolutions

    assert targeted["cause"] == cast["seq"]
    assert passed["kind"] == {"kind": "passed", "player": 0}
    assert passed["caused_by"] is not None
    assert died["cause"] == resolved["seq"] and died["rule"] == "lethal_damage"
    assert triggered["cause"] == died["seq"]
    assert trigger_resolved["kind"]["stack"] == triggered["kind"]["stack"]
    assert activation_resolved["kind"]["stack"] == activated["kind"]["stack"]
    assert counters["cause"] == activation_resolved["seq"]
    assert counters["kind"]["counter"] == "plus_one_plus_one"
    assert counters["kind"]["count"] == 1 and counters["kind"]["total"] == 1
    counter_obj = counters["kind"]["object"]
    assert (
        counters["labels"]["objects"][
            f"{counter_obj['entity']}:{counter_obj['incarnation']}"
        ]
        == "Fire Sages"
    )

    # Play Review's recording index exposes `track_history` and the full person track.
    from managym.academy import trace as trace_store

    session.close(end_reason="disconnect")
    assert session.trace_id is not None
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    client = TestClient(server.app)
    resp = client.get(
        f"/api/traces/{session.trace_id}/recording",
        headers={"x-etude-player-token": owner_token},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["track_history"] is True
    assert body["history"] == session.canonical_presentation[0]

