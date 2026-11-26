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
            (
                a["description"]
                for a in prompt.actions
                if a["description"].startswith("Pay")
            ),
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
            (
                a["description"]
                for a in prompt.actions
                if a["description"].startswith("Pay")
            ),
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
    ability_resolutions = [e for e in events if e["kind"]["kind"] == "ability_resolved"]
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


def test_remaining_kinds_searched_looked_sacrificed_and_lost_through_session(
    tmp_path: Path,
) -> None:
    """Real GameSession games emit `sacrificed`, `searched`, `looked` (scry and
    surveil) and `lost` (`zero_life`), validated against protocol v1."""
    deck = {
        "Island": 12,
        "Forest": 12,
        "Mountain": 12,
        "Evolving Wilds": 6,
        "Opt": 6,
        "Wary Thespian": 6,
        "Lightning Bolt": 6,
    }
    session = GameSession(trace_dir=tmp_path)
    session.new_game(
        {
            "villain_type": "passive",
            "hero_deck": deck,
            "villain_deck": {"Mountain": 60},
            "auto_pass": False,
            "seed": 19,
        }
    )
    assert session.env is not None
    _command(session, "Keep hand", "keep")
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    session.env.scenario_force_battlefield(0, "Island", True)
    session.env.scenario_force_battlefield(0, "Forest", True)
    session.env.scenario_force_battlefield(0, "Forest", True)
    session.env.scenario_force_battlefield(0, "Evolving Wilds", True)
    session.env.scenario_force_card_in_hand(0, "Opt")
    session.env.scenario_force_card_in_hand(0, "Wary Thespian")
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None

    start = len(session.presentation_events)

    def advance_until(predicate: Any, prefix: str, limit: int = 16) -> None:
        for step in range(limit):
            if predicate(session.presentation_events[start:]):
                return
            prompt = session._publish_current_prompt()
            if prompt is None:
                return
            descriptions = [a["description"] for a in prompt.actions]
            pick = next(
                (
                    d
                    for d in descriptions
                    if d.startswith("Pay")
                    or "Select" in d
                    or "top" in d
                    or "graveyard" in d
                    or d == "Pass priority"
                ),
                descriptions[0],
            )
            _command(session, pick, f"{prefix}-{step}")

    prompt = session._publish_current_prompt()
    assert prompt is not None
    ew_label = next(
        a["description"] for a in prompt.actions if "Evolving Wilds" in a["description"]
    )
    _command(session, ew_label, "ew-activate")
    advance_until(lambda evs: any(e["kind"]["kind"] == "searched" for e in evs), "ew")

    _command(session, "Cast Opt", "cast-opt")
    advance_until(
        lambda evs: any(
            e["kind"]["kind"] == "looked" and e["kind"]["action"] == "scry" for e in evs
        ),
        "opt",
    )

    _command(session, "Cast Wary Thespian", "cast-thespian")
    advance_until(
        lambda evs: any(
            e["kind"]["kind"] == "looked" and e["kind"]["action"] == "surveil"
            for e in evs
        ),
        "thespian",
    )

    events = list(session.presentation_events[start:])
    sacrificed = _first(events, "sacrificed")
    searched = _first(events, "searched")
    scry = next(
        e
        for e in events
        if e["kind"]["kind"] == "looked" and e["kind"]["action"] == "scry"
    )
    surveil = next(
        e
        for e in events
        if e["kind"]["kind"] == "looked" and e["kind"]["action"] == "surveil"
    )

    assert sacrificed["kind"]["player"] == 0
    assert "Evolving Wilds" in sacrificed["labels"]["objects"].values()
    assert searched["kind"] == {"kind": "searched", "player": 0}
    assert "labels" not in searched
    assert scry["kind"] == {"kind": "looked", "player": 0, "action": "scry"}
    assert "labels" not in scry
    assert surveil["kind"] == {"kind": "looked", "player": 0, "action": "surveil"}
    assert "labels" not in surveil

    # Burn lethal finishes the game and emits `lost` with rule `zero_life`.
    burn = GameSession(trace_dir=tmp_path)
    burn.new_game(
        {
            "villain_type": "passive",
            "hero_deck": {"Mountain": 24, "Lightning Bolt": 36},
            "villain_deck": {"Mountain": 60},
            "auto_pass": False,
            "seed": 3,
        }
    )
    assert burn.env is not None
    _command(burn, "Keep hand", "burn-keep")
    burn.env.scenario_clear_hand(0)
    burn.env.scenario_clear_hand(1)
    for _ in range(7):
        burn.env.scenario_force_battlefield(0, "Mountain", True)
        burn.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    burn.obs = burn.env.scenario_refresh()
    burn.published_prompt = None
    for idx in range(40):
        prompt = burn._publish_current_prompt()
        if prompt is None:
            break
        descriptions = [a["description"] for a in prompt.actions]
        pick = next(
            (
                d
                for d in descriptions
                if d == "Cast Lightning Bolt"
                or d.startswith("Target Villain")
                or d.startswith("Target Seat 2")
                or d.startswith("Pay")
                or d == "Pass priority"
            ),
            descriptions[0],
        )
        _command(burn, pick, f"burn-{idx}")
    lost = _first(burn.presentation_events, "lost")
    assert lost["kind"] == {"kind": "lost", "player": 1}
    assert lost["rule"] == "zero_life"
    assert "cause" in lost


def test_spell_and_ability_results_exile_destroy_tokens_pump_and_control_change(
    tmp_path: Path,
) -> None:
    """Real GameSession games tell exile, destroy, token creation, P/T pump,
    control change + cleanup return, and token death labels."""

    def advance_until(
        session: GameSession, start: int, predicate: Any, prefix: str, limit: int = 24
    ) -> None:
        for step in range(limit):
            if predicate(session.presentation_events[start:]):
                return
            prompt = session._publish_current_prompt()
            if prompt is None:
                return
            descriptions = [a["description"] for a in prompt.actions]
            pick = next(
                (
                    d
                    for d in descriptions
                    if d.startswith("Pay") or d == "Pass priority"
                ),
                descriptions[0],
            )
            _command(session, pick, f"{prefix}-{step}")

    # 1. Exile (Airbending Lesson), token creation (Treetop Freedom Fighters),
    #    token death with label (Lightning Bolt on Ally), P/T pump (Wandering Musicians).
    hero_deck = {
        "Plains": 15,
        "Mountain": 15,
        "Airbending Lesson": 6,
        "Treetop Freedom Fighters": 6,
        "Wandering Musicians": 6,
        "Lightning Bolt": 6,
        "Airbender's Reversal": 6,
    }
    session = GameSession(trace_dir=tmp_path)
    session.new_game(
        {
            "villain_type": "passive",
            "hero_deck": hero_deck,
            "villain_deck": {"Mountain": 30, "Gray Ogre": 30},
            "auto_pass": False,
            "seed": 23,
        }
    )
    assert session.env is not None
    _command(session, "Keep hand", "keep-1")
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    for _ in range(5):
        session.env.scenario_force_battlefield(0, "Plains", True)
        session.env.scenario_force_battlefield(0, "Mountain", True)
    session.env.scenario_force_battlefield(0, "Wandering Musicians", True)
    session.env.scenario_force_battlefield(1, "Gray Ogre", True)
    session.env.scenario_force_card_in_hand(0, "Airbending Lesson")
    session.env.scenario_force_card_in_hand(0, "Treetop Freedom Fighters")
    session.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    session.env.scenario_force_card_in_hand(0, "Airbender's Reversal")
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None

    start = len(session.presentation_events)
    _command(session, "Cast Airbending Lesson", "cast-airbend")
    _command(session, "Target Gray Ogre", "target-ogre")
    advance_until(
        session,
        start,
        lambda evs: any(e["kind"]["kind"] == "exiled" for e in evs),
        "airbend",
    )

    _command(session, "Cast Treetop Freedom Fighters", "cast-tff")
    advance_until(
        session,
        start,
        lambda evs: any(e["kind"]["kind"] == "token_created" for e in evs),
        "tff",
    )

    _command(session, "Cast Lightning Bolt", "cast-bolt")
    _command(session, "Target Ally", "target-ally")
    advance_until(
        session,
        start,
        lambda evs: any(e["kind"]["kind"] == "died" for e in evs),
        "bolt-ally",
    )

    # Pass to Declare Attackers, decline Treetop Freedom Fighters (has haste) if asked first,
    # attack with Wandering Musicians -> pump trigger, then destroy it with Airbender's Reversal.
    for step in range(8):
        prompt = session._publish_current_prompt()
        assert prompt is not None
        descriptions = [a["description"] for a in prompt.actions]
        if any("Attack with Wandering Musicians" in d for d in descriptions):
            break
        skip = next(
            (
                d
                for d in descriptions
                if d.startswith("Do not attack") or d == "Pass priority"
            ),
            descriptions[0],
        )
        _command(session, skip, f"to-combat-{step}")
    _command(session, "Attack with Wandering Musicians", "atk-musicians")
    for step in range(4):
        prompt = session._publish_current_prompt()
        assert prompt is not None
        descriptions = [a["description"] for a in prompt.actions]
        decline = next((d for d in descriptions if d.startswith("Do not attack")), None)
        if decline is None:
            break
        _command(session, decline, f"no-other-atk-{step}")
    advance_until(
        session,
        start,
        lambda evs: any(e["kind"]["kind"] == "pt_changed" for e in evs),
        "pump",
    )

    _command(session, "Cast Airbender's Reversal", "cast-reversal")
    prompt = session._publish_current_prompt()
    assert prompt is not None
    destroy_mode = next(
        a["description"] for a in prompt.actions if "Destroy" in a["description"]
    )
    _command(session, destroy_mode, "reversal-mode")
    prompt = session._publish_current_prompt()
    if prompt is not None and any(
        a["description"] == "Target Wandering Musicians" for a in prompt.actions
    ):
        _command(session, "Target Wandering Musicians", "reversal-target")
    advance_until(
        session,
        start,
        lambda evs: any(e["kind"]["kind"] == "destroyed" for e in evs),
        "destroy",
    )

    events = list(session.presentation_events[start:])
    exiled = _first(events, "exiled")
    assert "Gray Ogre" in exiled["labels"]["objects"].values()
    assert "cause" in exiled

    token = _first(events, "token_created")
    assert token["kind"]["player"] == 0
    assert "Ally" in token["labels"]["objects"].values()
    assert "cause" in token

    died = _first(events, "died")
    assert died["rule"] == "lethal_damage"
    assert "Ally" in died["labels"]["objects"].values()

    pump = _first(events, "pt_changed")
    assert pump["kind"]["power"] == 1 and pump["kind"]["toughness"] == 0
    assert "Wandering Musicians" in pump["labels"]["objects"].values()
    assert "cause" in pump

    destroyed = _first(events, "destroyed")
    assert "Wandering Musicians" in destroyed["labels"]["objects"].values()
    assert "cause" in destroyed

    # 2. Control change + Clue token + cleanup return (Jet's Brainwashing kicked).
    ctrl = GameSession(trace_dir=tmp_path)
    ctrl.new_game(
        {
            "villain_type": "passive",
            "hero_deck": {"Mountain": 40, "Jet's Brainwashing": 20},
            "villain_deck": {"Mountain": 30, "Gray Ogre": 30},
            "auto_pass": False,
            "seed": 31,
        }
    )
    assert ctrl.env is not None
    _command(ctrl, "Keep hand", "keep-ctrl")
    ctrl.env.scenario_clear_hand(0)
    ctrl.env.scenario_clear_hand(1)
    for _ in range(5):
        ctrl.env.scenario_force_battlefield(0, "Mountain", True)
    ctrl.env.scenario_force_battlefield(1, "Gray Ogre", True)
    ctrl.env.scenario_force_card_in_hand(0, "Jet's Brainwashing")
    ctrl.obs = ctrl.env.scenario_refresh()
    ctrl.published_prompt = None

    cstart = len(ctrl.presentation_events)
    _command(ctrl, "Cast Jet's Brainwashing", "cast-jet")
    for step in range(24):
        evs = ctrl.presentation_events[cstart:]
        if any(
            e["kind"]["kind"] == "control_changed" and e["kind"]["returned"] is True
            for e in evs
        ):
            break
        prompt = ctrl._publish_current_prompt()
        assert prompt is not None
        descriptions = [a["description"] for a in prompt.actions]
        pick = next(
            (
                d
                for d in descriptions
                if d.startswith("Pay kicker")
                or d == "Yes"
                or "Pay {" in d
                or d == "Target Gray Ogre"
                or d.startswith("Pay")
                or d.startswith("Do not attack")
                or d == "Pass priority"
            ),
            descriptions[0],
        )
        _command(ctrl, pick, f"jet-{step}")

    cevents = list(ctrl.presentation_events[cstart:])
    gain = next(
        e
        for e in cevents
        if e["kind"]["kind"] == "control_changed" and e["kind"]["returned"] is False
    )
    assert gain["kind"]["controller"] == 0
    assert "Gray Ogre" in gain["labels"]["objects"].values()
    assert "cause" in gain

    clue = _first(cevents, "token_created")
    assert "Clue" in clue["labels"]["objects"].values()

    ret = next(
        e
        for e in cevents
        if e["kind"]["kind"] == "control_changed" and e["kind"]["returned"] is True
    )
    assert ret["kind"]["controller"] == 1
    assert ret["rule"] == "cleanup"
    assert "Gray Ogre" in ret["labels"]["objects"].values()


def test_illegal_target_fizzles_for_spells_and_abilities_through_session(
    tmp_path: Path,
) -> None:
    """Real GameSession games distinguish CR 608.2b illegal-target fizzles
    (both spells and triggered abilities) from counterspells."""

    def advance_until(
        session: GameSession, start: int, predicate: Any, prefix: str, limit: int = 24
    ) -> None:
        for step in range(limit):
            if predicate(session.presentation_events[start:]):
                return
            prompt = session._publish_current_prompt()
            if prompt is None:
                return
            descriptions = [a["description"] for a in prompt.actions]
            pick = next(
                (
                    d
                    for d in descriptions
                    if d.startswith("Pay") or d == "Pass priority"
                ),
                descriptions[0],
            )
            _command(session, pick, f"{prefix}-{step}")

    deck = {
        "Plains": 15,
        "Mountain": 15,
        "Lightning Bolt": 15,
        "Earth Kingdom Jailer": 15,
    }
    session = GameSession(trace_dir=tmp_path)
    session.new_game(
        {
            "villain_type": "passive",
            "hero_deck": deck,
            "villain_deck": {"Mountain": 30, "Gray Ogre": 30},
            "auto_pass": False,
            "seed": 41,
        }
    )
    assert session.env is not None
    _command(session, "Keep hand", "keep-fizzle")
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    for _ in range(5):
        session.env.scenario_force_battlefield(0, "Plains", True)
        session.env.scenario_force_battlefield(0, "Mountain", True)
    session.env.scenario_force_battlefield(1, "Gray Ogre", True)
    session.env.scenario_force_battlefield(1, "Gray Ogre", True)
    session.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    session.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    session.env.scenario_force_card_in_hand(0, "Earth Kingdom Jailer")
    session.env.scenario_force_card_in_hand(0, "Lightning Bolt")
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None

    # 1. Double-Bolt the same Gray Ogre so the first Bolt fizzles with `illegal_targets`.
    start = len(session.presentation_events)
    _command(session, "Cast Lightning Bolt", "bolt1-cast")
    prompt = session._publish_current_prompt()
    assert prompt is not None
    ogre_target_idx = next(
        i
        for i, a in enumerate(prompt.actions)
        if a["description"] == "Target Gray Ogre"
    )
    session.hero_command(
        {
            "command_id": "bolt1-target",
            "match_id": session.match_id,
            "expected_revision": prompt.revision,
            "prompt_id": prompt.prompt_id,
            "offer_id": ogre_target_idx,
            "answers": [],
        }
    )
    advance_until(
        session,
        start,
        lambda _evs: any(
            a["description"] == "Cast Lightning Bolt"
            for a in (
                session._publish_current_prompt().actions
                if session._publish_current_prompt()
                else []
            )
        ),
        "bolt1-pay",
    )
    _command(session, "Cast Lightning Bolt", "bolt2-cast")
    prompt = session._publish_current_prompt()
    assert prompt is not None
    ogre_target_idx2 = next(
        i
        for i, a in enumerate(prompt.actions)
        if a["description"] == "Target Gray Ogre"
    )
    session.hero_command(
        {
            "command_id": "bolt2-target",
            "match_id": session.match_id,
            "expected_revision": prompt.revision,
            "prompt_id": prompt.prompt_id,
            "offer_id": ogre_target_idx2,
            "answers": [],
        }
    )
    advance_until(
        session,
        start,
        lambda evs: any(e["kind"]["kind"] == "countered" for e in evs),
        "bolt-fizzle",
    )
    bolt_events = list(session.presentation_events[start:])
    bolt_fizzle = _first(bolt_events, "countered")
    assert bolt_fizzle["rule"] == "illegal_targets"
    assert "cause" not in bolt_fizzle
    assert (
        bolt_fizzle["labels"]["stacks"][str(bolt_fizzle["kind"]["stack"])]
        == "Lightning Bolt"
    )

    # 2. Earth Kingdom Jailer ETB trigger targets the remaining Gray Ogre; Bolt kills it in response.
    tstart = len(session.presentation_events)
    _command(session, "Cast Earth Kingdom Jailer", "cast-jailer")
    for step in range(16):
        prompt = session._publish_current_prompt()
        assert prompt is not None
        descriptions = [a["description"] for a in prompt.actions]
        if "Target Gray Ogre" in descriptions:
            _command(session, "Target Gray Ogre", "jailer-target")
            break
        pick = next(
            (d for d in descriptions if d.startswith("Pay") or d == "Pass priority"),
            descriptions[0],
        )
        _command(session, pick, f"jailer-{step}")
    _command(session, "Cast Lightning Bolt", "bolt3-cast")
    _command(session, "Target Gray Ogre", "bolt3-target")
    advance_until(
        session,
        tstart,
        lambda evs: any(e["kind"]["kind"] == "countered" for e in evs),
        "trigger-fizzle",
    )
    trigger_events = list(session.presentation_events[tstart:])
    trig = _first(trigger_events, "triggered")
    trig_fizzle = _first(trigger_events, "countered")
    assert trig_fizzle["kind"]["stack"] == trig["kind"]["stack"]
    assert trig_fizzle["rule"] == "illegal_targets"
    assert "cause" not in trig_fizzle
    assert (
        trig_fizzle["labels"]["stacks"][str(trig["kind"]["stack"])]
        == "Earth Kingdom Jailer"
    )
