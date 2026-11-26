"""
test_stops.py
Arena-style priority stops: server-side auto-pass, active-combat stops,
non-mana play gating, stop configuration (new_game + set_stops),
stop-on-stack, F6 pass-turn, and trace auto marks.
"""

from collections.abc import Callable
import json
import sqlite3
from typing import Any

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace as trace_store
from managym.academy.server import app

# Local imports
from tests.academy.socket_messages import settled_message

MAX_HERO_MOVES = 3000

# Decks that reproduce the live complaint: the hero holds instants, so the
# engine offers a priority window at every step of both players' turns.
BOLT_DECK = {"Mountain": 8, "Lightning Bolt": 8}
COUNTER_DECK = {"Island": 10, "Counterspell": 6}
OGRE_DECK = {"Mountain": 10, "Gray Ogre": 6}


def _recorded_traces(directory):
    with sqlite3.connect(directory / "play.sqlite") as db:
        return [
            (row[0], json.loads(row[1]))
            for row in db.execute(
                "SELECT id, trace_json FROM attempts ORDER BY started_at"
            )
        ]


@pytest.fixture()
def isolated_traces(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    return tmp_path


def _action_types(payload: dict) -> set[str]:
    return {action["type"] for action in payload["actions"]}


def _is_priority_window(payload: dict) -> bool:
    # Only priority action spaces contain a pass-priority action.
    return "PRIORITY_PASS_PRIORITY" in _action_types(payload)


def _turn_side(payload: dict) -> str:
    data = payload["data"]
    is_my_turn = data["turn"]["active_player_id"] == data["agent"]["id"]
    return "my" if is_my_turn else "opponent"


def _stack_cards(payload: dict) -> list[str]:
    data = payload["data"]
    return [card["name"] for card in data["agent"]["stack"] + data["opponent"]["stack"]]


def _stop_key(payload: dict) -> str | None:
    return server.ENGINE_STEP_TO_STOP_STEP.get(payload["data"]["turn"]["step"])


def _pass_index(payload: dict) -> int:
    for action in payload["actions"]:
        if action["type"] == "PRIORITY_PASS_PRIORITY":
            return action["index"]
    raise AssertionError("Expected a pass-priority action")


def _lands_then_pass_policy(payload: dict) -> int:
    """Scripted human: develop mana, otherwise pass. Deterministic."""
    for action in payload["actions"]:
        if action["type"] == "PRIORITY_PLAY_LAND":
            return action["index"]
    for action in payload["actions"]:
        if action["type"] == "PRIORITY_PASS_PRIORITY":
            return action["index"]
    return payload["actions"][0]["index"]


def _play_game(websocket, config: dict, policy, on_payload=None) -> dict:
    """Drive a full game; returns the terminal payload."""
    websocket.send_json({"type": "new_game", "config": config})
    payload = settled_message(websocket)
    for _ in range(MAX_HERO_MOVES):
        assert payload["type"] in {"observation", "game_over"}, payload
        if on_payload is not None:
            on_payload(payload)
        if payload["type"] == "game_over":
            return payload
        websocket.send_json({"type": "action", "index": policy(payload)})
        payload = settled_message(websocket)
    pytest.fail("Game did not reach a terminal state within the move budget")


def _load_single_trace(trace_dir) -> dict:
    trace_files = _recorded_traces(trace_dir)
    assert len(trace_files) == 1
    return trace_files[0][1]


def test_default_stops_surface_only_configured_windows(isolated_traces):
    """With default stops, every surfaced pure-priority window sits on a
    configured stop; everything else was auto-passed and marked in the
    trace."""
    surfaced: list[tuple[str, str | None]] = []
    surfaced_decisions = []

    def observe(payload):
        if payload["type"] == "observation":
            surfaced_decisions.append(payload["action_space"])
        if payload["type"] == "observation" and _is_priority_window(payload):
            surfaced.append((_turn_side(payload), _stop_key(payload)))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            payload = _play_game(
                websocket,
                {
                    "villain_type": "passive",
                    "seed": 3,
                    "hero_deck": BOLT_DECK,
                    "villain_deck": {"Island": 12},
                },
                _lands_then_pass_policy,
                on_payload=observe,
            )
    assert payload["type"] == "game_over"

    assert surfaced, "Expected surfaced priority windows"
    # BOLT_DECK has no creatures, so active-combat stops auto-pass on empty boards.
    allowed = {("my", "main1"), ("my", "main2"), ("opponent", "end_step")}
    assert set(surfaced) <= allowed, sorted(set(surfaced) - allowed)
    # The hero held castable instants all game (BOLT_DECK), so un-stopped
    # windows existed and were skipped.
    trace = _load_single_trace(isolated_traces)
    hero_events = [e for e in trace["events"] if e["actor"] == "hero"]
    auto_events = [e for e in hero_events if e["auto"]]
    clicked_events = [e for e in hero_events if not e["auto"]]
    assert auto_events, "Expected auto-passed hero windows in the trace"
    assert all(e["action_description"] == "Pass priority" for e in auto_events), (
        "Auto events must all be passes"
    )
    assert len(clicked_events) == len(surfaced_decisions), (
        "Every clicked decision corresponds to a surfaced window"
    )
    assert all(
        len(e["actions"]) == 1
        for e in trace["events"]
        if e["actor"] == "villain" and e["auto"]
    )


def test_disabling_auto_pass_surfaces_every_window(isolated_traces):
    """auto_pass=false restores the pre-stops behavior: same game trajectory
    (hero passes are identical either way), but every window surfaces and
    nothing is marked auto."""
    config = {
        "villain_type": "passive",
        "seed": 3,
        "hero_deck": BOLT_DECK,
        "villain_deck": {"Island": 12},
    }

    def run(extra: dict) -> tuple[int, dict]:
        server.LIVE_GAMES.clear()
        surfaced = 0

        def observe(payload):
            nonlocal surfaced
            if payload["type"] == "observation":
                surfaced += 1

        with TestClient(app) as client:
            with client.websocket_connect("/ws/play") as websocket:
                _play_game(
                    websocket,
                    {**config, **extra},
                    _lands_then_pass_policy,
                    on_payload=observe,
                )
        return surfaced, _load_single_trace(isolated_traces)

    surfaced_default, trace_default = run({})
    (isolated_traces / "play.sqlite").unlink()
    surfaced_manual, trace_manual = run({"auto_pass": False})

    # Identical engine trajectory: the auto-passes in the default run are the
    # same passes the scripted hero clicks in the manual run.
    assert len(trace_default["events"]) == len(trace_manual["events"])
    assert trace_default["winner"] == trace_manual["winner"]
    assert not any(e["auto"] for e in trace_manual["events"])
    assert surfaced_manual > 2 * surfaced_default, (
        f"auto-pass should remove most windows: {surfaced_default} vs {surfaced_manual}"
    )


def test_stack_nonempty_forces_surfacing(isolated_traces):
    """A villain spell on the stack surfaces a hero window even at an
    un-stopped step (this is how counterspells happen); with
    stop_on_stack=false the same window auto-passes."""
    config = {
        "villain_type": "random",
        "seed": 0,
        "hero_deck": COUNTER_DECK,
        "villain_deck": OGRE_DECK,
    }

    def run(extra: dict) -> tuple[list[dict], dict]:
        server.LIVE_GAMES.clear()
        stack_windows: list[dict] = []

        def observe(payload):
            if payload["type"] != "observation":
                return
            if not _is_priority_window(payload):
                return
            side, stop = _turn_side(payload), _stop_key(payload)
            on_stop = stop is not None and stop in (
                server.DEFAULT_STOPS[side] if side in server.DEFAULT_STOPS else []
            )
            if not on_stop:
                stack_windows.append(
                    {
                        "stack": _stack_cards(payload),
                        "can_counter": any(
                            a["description"] == "Cast Counterspell"
                            for a in payload["actions"]
                        ),
                    }
                )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/play") as websocket:
                _play_game(
                    websocket,
                    {**config, **extra},
                    _lands_then_pass_policy,
                    on_payload=observe,
                )
        return stack_windows, _load_single_trace(isolated_traces)

    stack_windows, trace_on = run({})
    assert stack_windows, "Expected surfaced windows outside stops"
    # Only the stack rule can surface outside a stop, and it produced real
    # counter opportunities.
    assert all(w["stack"] for w in stack_windows)
    assert any(w["can_counter"] for w in stack_windows)
    assert any("Gray Ogre" in w["stack"] for w in stack_windows)

    (isolated_traces / "play.sqlite").unlink()
    no_stack_windows, trace_off = run({"stop_on_stack": False})
    assert no_stack_windows == [], "stop_on_stack=false must not surface stack windows"
    # Same trajectory either way (the scripted hero passed at those windows).
    assert len(trace_on["events"]) == len(trace_off["events"])
    assert trace_on["winner"] == trace_off["winner"]


def test_set_stops_mid_game_takes_effect(isolated_traces):
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "villain_type": "passive",
                        "seed": 3,
                        "hero_deck": BOLT_DECK,
                        "villain_deck": {"Island": 12},
                    },
                }
            )
            payload = settled_message(websocket)
            assert payload["type"] == "observation"
            if payload["data"]["turn"]["turn_number"] == 0:
                websocket.send_json({"type": "action", "index": 0})
                payload = settled_message(websocket)
            # Default stops: game opens at my main1, with the config echoed.
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"
            assert payload["stops"] == {
                "my": ["main1", "declare_attackers", "declare_blockers", "main2"],
                "opponent": ["declare_attackers", "declare_blockers", "end_step"],
                "stop_on_stack": True,
                "auto_pass": True,
            }

            # Develop a land so the hero holds castable bolts: the engine only
            # offers priority windows when there is something to do, and a
            # stop without a window never surfaces.
            websocket.send_json(
                {"type": "action", "index": _lands_then_pass_policy(payload)}
            )
            payload = settled_message(websocket)
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"

            # Drop the main stops mid-window: the server fast-forwards off the
            # now-unstopped main1 window straight to the end step.
            websocket.send_json(
                {
                    "type": "set_stops",
                    "stops": {"my": ["end_step"], "opponent": []},
                }
            )
            payload = settled_message(websocket)
            assert payload["type"] == "observation"
            assert payload["stops"]["my"] == ["end_step"]
            assert payload["stops"]["opponent"] == []
            assert payload["stops"]["stop_on_stack"] is True
            assert payload["data"]["turn"]["turn_number"] == 1
            assert payload["data"]["turn"]["step"] == "ENDING_END"
            assert _turn_side(payload) == "my"
            assert payload.get("auto_passed", 0) > 0

            # Partial update: only stop_on_stack; the stop set is preserved.
            websocket.send_json({"type": "set_stops", "stop_on_stack": False})
            payload = settled_message(websocket)
            assert payload["stops"]["my"] == ["end_step"]
            assert payload["stops"]["stop_on_stack"] is False

            # Validation errors leave the session playable.
            websocket.send_json({"type": "set_stops", "stops": {"my": ["untap"]}})
            payload = settled_message(websocket)
            assert payload["type"] == "error"
            assert "Unknown stop step" in payload["message"]

            websocket.send_json({"type": "set_stops", "stops": {"theirs": []}})
            payload = settled_message(websocket)
            assert payload["type"] == "error"
            assert "my" in payload["message"]


def test_pass_turn_yields_until_end_of_turn(isolated_traces):
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "villain_type": "passive",
                        "seed": 3,
                        "hero_deck": BOLT_DECK,
                        "villain_deck": {"Island": 12},
                    },
                }
            )
            payload = settled_message(websocket)
            if payload["data"]["turn"]["turn_number"] == 0:
                websocket.send_json({"type": "action", "index": 0})
                payload = settled_message(websocket)
            assert payload["data"]["turn"]["turn_number"] == 1
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"

            # Develop a land so the hero holds castable bolts (the engine only
            # offers windows when there is something to do).
            websocket.send_json(
                {"type": "action", "index": _lands_then_pass_policy(payload)}
            )
            payload = settled_message(websocket)
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"

            # F6: skips the rest of MY turn — through the my-main2 stop — and
            # clears at the turn boundary, so the next surfaced window is the
            # opponent-end-step stop of turn 2.
            websocket.send_json({"type": "pass_turn"})
            payload = settled_message(websocket)
            assert payload["type"] == "observation"
            assert payload["data"]["turn"]["turn_number"] == 2
            assert payload["data"]["turn"]["step"] == "ENDING_END"
            assert _turn_side(payload) == "opponent"
            assert payload["auto_passed"] > 0

            # F6 cleared: normal stops apply again on turn 3.
            websocket.send_json({"type": "action", "index": _pass_index(payload)})
            payload = settled_message(websocket)
            assert payload["data"]["turn"]["turn_number"] == 3
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"
            assert _turn_side(payload) == "my"

    # The game is still live (no trace file yet) — inspect the session trace.
    record = next(iter(server.LIVE_GAMES.values()))
    hero_events = [e for e in record.game.trace.events if e.actor == "hero"]
    assert any(e.auto for e in hero_events)
    assert all(
        not e.auto for e in hero_events if e.action_description != "Pass priority"
    )


def test_non_priority_spaces_always_surface(isolated_traces):
    """Attack/target decisions surface even when their steps carry no stop:
    stops govern priority windows only."""
    surfaced_types: set[str] = set()

    def observe(payload):
        if payload["type"] == "observation" and not _is_priority_window(payload):
            surfaced_types.update(_action_types(payload))

    def policy(payload):
        for action in payload["actions"]:
            if action["type"] in {"PRIORITY_PLAY_LAND", "PRIORITY_CAST_SPELL"}:
                return action["index"]
        return payload["actions"][0]["index"]

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            payload = _play_game(
                websocket,
                {
                    "villain_type": "passive",
                    "seed": 9,
                    "hero_deck": {
                        "Mountain": 8,
                        "Raging Goblin": 8,
                        "Lightning Bolt": 4,
                    },
                    "villain_deck": {"Island": 20},
                    # No combat stops at all — the attack declaration must
                    # still surface.
                    "stops": {"my": ["main1"], "opponent": []},
                },
                policy,
                on_payload=observe,
            )
    assert payload["type"] == "game_over"
    assert "DECLARE_ATTACKER" in surfaced_types
    assert "CHOOSE_TARGET" in surfaced_types


def test_new_game_rejects_bad_stops_configs(isolated_traces):
    cases = [
        ({"stops": ["main1"]}, "stops must be an object"),
        ({"stops": {"my": "main1"}}, "must be a list"),
        ({"stops": {"my": ["untap"]}}, "unknown stop step"),
        ({"stops": {"villain": []}}, "my"),
        ({"stop_on_stack": "yes"}, "boolean"),
        ({"auto_pass": 1}, "boolean"),
    ]
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            for config, expected_fragment in cases:
                websocket.send_json({"type": "new_game", "config": config})
                payload = settled_message(websocket)
                assert payload["type"] == "error", (config, payload)
                assert expected_fragment.lower() in payload["message"].lower()


def test_pass_turn_requires_active_game(isolated_traces):
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "pass_turn"})
            payload = settled_message(websocket)
            assert payload["type"] == "error"
            assert "No active game session" in payload["message"]


RAZOR_RINGS_DECK = {"Plains": 10, "Savannah Lions": 6, "Razor Rings": 6}
ACTIVE_COMBAT_ENGINE_STEPS = {"COMBAT_DECLARE_ATTACKERS", "COMBAT_DECLARE_BLOCKERS"}


def _play_ws_game(
    config: dict[str, Any],
    policy: Callable[[dict[str, Any]], int],
    *,
    on_payload: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            return _play_game(websocket, config, policy, on_payload=on_payload)


def _make_razor_rings_policy(*, attack: bool) -> Callable[[dict[str, Any]], int]:
    """Develop Plains and Savannah Lions, hold Razor Rings, and attack or hold back."""
    attacker_phrase = "Attack" if attack else "Don't"

    def policy(payload: dict[str, Any]) -> int:
        for action in payload["actions"]:
            if action["type"] == "PRIORITY_PLAY_LAND":
                return action["index"]
        for action in payload["actions"]:
            if action["type"] == "PRIORITY_CAST_SPELL" and "Savannah Lions" in (
                action.get("description") or ""
            ):
                return action["index"]
        for action in payload["actions"]:
            if action["type"] == "DECLARE_ATTACKER" and attacker_phrase in (
                action.get("description") or ""
            ):
                return action["index"]
        if not attack:
            for action in payload["actions"]:
                if action["type"] == "DECLARE_BLOCKER" and "Don't" in (
                    action.get("description") or ""
                ):
                    return action["index"]
        for action in payload["actions"]:
            if action["type"] == "PRIORITY_PASS_PRIORITY":
                return action["index"]
        return payload["actions"][0]["index"]

    return policy


def _combat_window_recorder(
    side: str | None = None,
) -> tuple[list[tuple[str, list[str]]], Callable[[dict[str, Any]], None]]:
    windows: list[tuple[str, list[str]]] = []

    def observe(payload: dict[str, Any]) -> None:
        if payload["type"] != "observation" or not _is_priority_window(payload):
            return
        step = payload["data"]["turn"]["step"]
        if step in ACTIVE_COMBAT_ENGINE_STEPS and (
            side is None or _turn_side(payload) == side
        ):
            windows.append(
                (step, [a.get("description") or "" for a in payload["actions"]])
            )

    return windows, observe


def test_active_combat_stop_surfaces_razor_rings_for_attacker(isolated_traces):
    """Default stops pause in declare_attackers when attacking while holding Razor Rings."""
    combat_windows, observe = _combat_window_recorder(side="my")
    payload = _play_ws_game(
        {
            "villain_type": "passive",
            "seed": 3,
            "hero_deck": RAZOR_RINGS_DECK,
            "villain_deck": {"Island": 16},
        },
        _make_razor_rings_policy(attack=True),
        on_payload=observe,
    )
    assert payload["type"] == "game_over"
    assert any(
        step == "COMBAT_DECLARE_ATTACKERS"
        and any("Cast Razor Rings" in desc for desc in descs)
        for step, descs in combat_windows
    ), combat_windows


def test_active_combat_stops_surface_razor_rings_for_defender(isolated_traces):
    """Default stops pause in both declare_attackers and declare_blockers while defending."""
    combat_windows, observe = _combat_window_recorder(side="opponent")
    payload = _play_ws_game(
        {
            "villain_type": "random",
            "seed": 3,
            "hero_deck": RAZOR_RINGS_DECK,
            "villain_deck": {"Mountain": 10, "Raging Goblin": 10},
        },
        _make_razor_rings_policy(attack=False),
        on_payload=observe,
    )
    assert payload["type"] == "game_over"
    defender_steps = {
        step
        for step, descs in combat_windows
        if any("Cast Razor Rings" in desc for desc in descs)
    }
    assert defender_steps == ACTIVE_COMBAT_ENGINE_STEPS, combat_windows


def test_active_combat_with_no_instant_surfaces_no_combat_priority_window(
    isolated_traces,
):
    """Attacking with no playable instant never pauses at combat priority."""
    declared_attack = False
    combat_windows, record_combat = _combat_window_recorder()

    def observe(payload: dict) -> None:
        nonlocal declared_attack
        if payload["type"] == "observation" and "DECLARE_ATTACKER" in _action_types(
            payload
        ):
            declared_attack = True
        record_combat(payload)

    payload = _play_ws_game(
        {
            "villain_type": "passive",
            "seed": 3,
            "hero_deck": {"Plains": 10, "Savannah Lions": 10},
            "villain_deck": {"Island": 16},
        },
        _make_razor_rings_policy(attack=True),
        on_payload=observe,
    )
    assert payload["type"] == "game_over"
    assert declared_attack, "Expected combat with attacking creatures"
    assert combat_windows == []


def test_mana_abilities_alone_auto_pass_under_auto_pass_and_surface_in_full_control(
    isolated_traces,
):
    """Under mana_floats='all', untapped lands and mana-producing activated
    abilities alone do not hold priority when auto_pass=True, but still surface
    in Full control (auto_pass=False)."""
    # Plains gives TapForMana; Kyoshi Village gives a mana-choice
    # PRIORITY_ACTIVATE_ABILITY. No instants exist in the deck, so combat
    # windows only ever offer pass + mana abilities.
    base_config = {
        "villain_type": "passive",
        "seed": 3,
        "mana_floats": "all",
        "hero_deck": {"Plains": 6, "Kyoshi Village": 6, "Savannah Lions": 8},
        "villain_deck": {"Island": 16},
    }

    def run(auto_pass: bool) -> tuple[list[dict], list[dict]]:
        server.LIVE_GAMES.clear()
        combat_windows: list[dict] = []
        mana_only_windows: list[dict] = []

        def observe(payload: dict) -> None:
            if payload["type"] != "observation" or not _is_priority_window(payload):
                return
            types = _action_types(payload)
            if payload["data"]["turn"]["step"] in ACTIVE_COMBAT_ENGINE_STEPS:
                combat_windows.append(payload)
            # Exclude Kyoshi Village's "{4}, {T}, Sacrifice: Draw a card" non-mana ability.
            if types <= {
                "PRIORITY_PASS_PRIORITY",
                "TAP_FOR_COST",
                "PRIORITY_ACTIVATE_ABILITY",
            } and not any(
                "Draw" in (a.get("description") or "") for a in payload["actions"]
            ):
                mana_only_windows.append(payload)

        payload = _play_ws_game(
            {**base_config, "auto_pass": auto_pass},
            _make_razor_rings_policy(attack=True),
            on_payload=observe,
        )
        assert payload["type"] == "game_over"
        return combat_windows, mana_only_windows

    auto_combat, auto_mana_only = run(auto_pass=True)
    assert auto_combat == [], "Mana abilities alone must not hold active combat"
    assert auto_mana_only == [], (
        "Mana abilities alone must not hold any stop under auto_pass=True"
    )

    (isolated_traces / "play.sqlite").unlink()
    full_combat, full_mana_only = run(auto_pass=False)
    assert full_combat, "Full control must still surface combat mana-float windows"
    assert any("TAP_FOR_COST" in _action_types(w) for w in full_mana_only), (
        "Expected fixed-source mana taps in Full control"
    )
    assert any(
        "PRIORITY_ACTIVATE_ABILITY" in _action_types(w) for w in full_mana_only
    ), "Expected mana-producing activated abilities in Full control"


def test_turning_off_combat_stops_skips_razor_rings_window(isolated_traces):
    """Turning combat stops off skips the Razor Rings combat window."""
    combat_windows, observe = _combat_window_recorder()
    payload = _play_ws_game(
        {
            "villain_type": "passive",
            "seed": 3,
            "hero_deck": RAZOR_RINGS_DECK,
            "villain_deck": {"Island": 16},
            "stops": {"my": ["main1", "main2"], "opponent": ["end_step"]},
        },
        _make_razor_rings_policy(attack=True),
        on_payload=observe,
    )
    assert payload["type"] == "game_over"
    assert combat_windows == []
    trace = _load_single_trace(isolated_traces)
    assert any(e["actor"] == "hero" and e["auto"] for e in trace["events"])


def test_f6_yields_through_active_combat_stops(isolated_traces):
    """F6 at an active-combat Razor Rings window yields through the rest of the turn."""
    policy = _make_razor_rings_policy(attack=True)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "villain_type": "passive",
                        "seed": 3,
                        "hero_deck": RAZOR_RINGS_DECK,
                        "villain_deck": {"Island": 16},
                    },
                }
            )
            payload = settled_message(websocket)
            combat_turn: int | None = None
            for _ in range(200):
                assert payload["type"] == "observation"
                if (
                    _is_priority_window(payload)
                    and payload["data"]["turn"]["step"] == "COMBAT_DECLARE_ATTACKERS"
                    and any(
                        "Cast Razor Rings" in (a.get("description") or "")
                        for a in payload["actions"]
                    )
                ):
                    combat_turn = int(payload["data"]["turn"]["turn_number"])
                    break
                websocket.send_json({"type": "action", "index": policy(payload)})
                payload = settled_message(websocket)
            assert combat_turn is not None, "Expected Razor Rings window in combat"

            websocket.send_json({"type": "pass_turn"})
            after_f6 = settled_message(websocket)
            assert after_f6["type"] == "observation"
            assert int(after_f6["data"]["turn"]["turn_number"]) > combat_turn
            assert after_f6.get("auto_passed", 0) > 0
