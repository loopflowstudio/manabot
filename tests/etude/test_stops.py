"""
test_stops.py
MTGO-style priority stops: server-side auto-pass, stop configuration
(new_game + set_stops), stop-on-stack, F6 pass-turn, and trace auto marks.
"""

import json

from fastapi.testclient import TestClient
import pytest

# Local imports
from etude import server, trace as trace_store
from etude.server import app

MAX_HERO_MOVES = 3000

# Decks that reproduce the live complaint: the hero holds instants, so the
# engine offers a priority window at every step of both players' turns.
BOLT_DECK = {"Mountain": 12, "Lightning Bolt": 12}
COUNTER_DECK = {"Island": 14, "Counterspell": 8}
OGRE_DECK = {"Mountain": 14, "Gray Ogre": 8}


@pytest.fixture()
def isolated_traces(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
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
    return [
        card["name"] for card in data["agent"]["stack"] + data["opponent"]["stack"]
    ]


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
    payload = websocket.receive_json()
    for _ in range(MAX_HERO_MOVES):
        assert payload["type"] in {"observation", "game_over"}, payload
        if on_payload is not None:
            on_payload(payload)
        if payload["type"] == "game_over":
            return payload
        websocket.send_json({"type": "action", "index": policy(payload)})
        payload = websocket.receive_json()
    pytest.fail("Game did not reach a terminal state within the move budget")


def _load_single_trace(trace_dir) -> dict:
    trace_files = sorted(trace_dir.glob("*.json"))
    assert len(trace_files) == 1
    return json.loads(trace_files[0].read_text(encoding="utf-8"))


def test_default_stops_surface_only_configured_windows(isolated_traces):
    """With default stops, every surfaced pure-priority window sits on a
    configured stop; everything else was auto-passed and marked in the
    trace."""
    surfaced: list[tuple[str, str | None]] = []

    def observe(payload):
        if payload["type"] == "observation" and _is_priority_window(payload):
            surfaced.append((_turn_side(payload), _stop_key(payload)))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            payload = _play_game(
                websocket,
                {"villain_type": "passive", "seed": 3, "hero_deck": BOLT_DECK,
                 "villain_deck": {"Island": 20}},
                _lands_then_pass_policy,
                on_payload=observe,
            )
    assert payload["type"] == "game_over"

    assert surfaced, "Expected surfaced priority windows"
    allowed = {("my", "main1"), ("my", "main2"), ("opponent", "end_step")}
    assert set(surfaced) <= allowed, sorted(set(surfaced) - allowed)
    # The hero held castable instants all game (BOLT_DECK), so un-stopped
    # windows existed and were skipped.
    trace = _load_single_trace(isolated_traces)
    hero_events = [e for e in trace["events"] if e["actor"] == "hero"]
    auto_events = [e for e in hero_events if e["auto"]]
    clicked_events = [e for e in hero_events if not e["auto"]]
    assert auto_events, "Expected auto-passed hero windows in the trace"
    assert all(
        e["action_description"] == "Pass priority" for e in auto_events
    ), "Auto events must all be passes"
    assert len(clicked_events) == len(surfaced) + 0, (
        "Every clicked decision corresponds to a surfaced window"
    )
    assert all(not e["auto"] for e in trace["events"] if e["actor"] == "villain")


def test_disabling_auto_pass_surfaces_every_window(isolated_traces):
    """auto_pass=false restores the pre-stops behavior: same game trajectory
    (hero passes are identical either way), but every window surfaces and
    nothing is marked auto."""
    config = {
        "villain_type": "passive",
        "seed": 3,
        "hero_deck": BOLT_DECK,
        "villain_deck": {"Island": 20},
    }

    def run(extra: dict) -> tuple[int, dict]:
        server.SESSION_REGISTRY.clear()
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
    for trace_file in isolated_traces.glob("*.json"):
        trace_file.unlink()
    surfaced_manual, trace_manual = run({"auto_pass": False})

    # Identical engine trajectory: the auto-passes in the default run are the
    # same passes the scripted hero clicks in the manual run.
    assert len(trace_default["events"]) == len(trace_manual["events"])
    assert trace_default["winner"] == trace_manual["winner"]
    assert not any(e["auto"] for e in trace_manual["events"])
    assert surfaced_manual > 2 * surfaced_default, (
        f"auto-pass should remove most windows: {surfaced_default} vs "
        f"{surfaced_manual}"
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
        server.SESSION_REGISTRY.clear()
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
                    websocket, {**config, **extra},
                    _lands_then_pass_policy, on_payload=observe,
                )
        return stack_windows, _load_single_trace(isolated_traces)

    stack_windows, trace_on = run({})
    assert stack_windows, "Expected surfaced windows outside stops"
    # Only the stack rule can surface outside a stop, and it produced real
    # counter opportunities.
    assert all(w["stack"] for w in stack_windows)
    assert any(w["can_counter"] for w in stack_windows)
    assert any("Gray Ogre" in w["stack"] for w in stack_windows)

    for trace_file in isolated_traces.glob("*.json"):
        trace_file.unlink()
    no_stack_windows, trace_off = run({"stop_on_stack": False})
    assert no_stack_windows == [], (
        "stop_on_stack=false must not surface stack windows"
    )
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
                        "villain_deck": {"Island": 20},
                    },
                }
            )
            payload = websocket.receive_json()
            assert payload["type"] == "observation"
            # Default stops: game opens at my main1, with the config echoed.
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"
            assert payload["stops"] == {
                "my": ["main1", "main2"],
                "opponent": ["end_step"],
                "stop_on_stack": True,
                "auto_pass": True,
            }

            # Develop a land so the hero holds castable bolts: the engine only
            # offers priority windows when there is something to do, and a
            # stop without a window never surfaces.
            websocket.send_json(
                {"type": "action", "index": _lands_then_pass_policy(payload)}
            )
            payload = websocket.receive_json()
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"

            # Drop the main stops mid-window: the server fast-forwards off the
            # now-unstopped main1 window straight to the end step.
            websocket.send_json(
                {
                    "type": "set_stops",
                    "stops": {"my": ["end_step"], "opponent": []},
                }
            )
            payload = websocket.receive_json()
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
            payload = websocket.receive_json()
            assert payload["stops"]["my"] == ["end_step"]
            assert payload["stops"]["stop_on_stack"] is False

            # Validation errors leave the session playable.
            websocket.send_json(
                {"type": "set_stops", "stops": {"my": ["untap"]}}
            )
            payload = websocket.receive_json()
            assert payload["type"] == "error"
            assert "Unknown stop step" in payload["message"]

            websocket.send_json(
                {"type": "set_stops", "stops": {"theirs": []}}
            )
            payload = websocket.receive_json()
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
                        "villain_deck": {"Island": 20},
                    },
                }
            )
            payload = websocket.receive_json()
            assert payload["data"]["turn"]["turn_number"] == 1
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"

            # Develop a land so the hero holds castable bolts (the engine only
            # offers windows when there is something to do).
            websocket.send_json(
                {"type": "action", "index": _lands_then_pass_policy(payload)}
            )
            payload = websocket.receive_json()
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"

            # F6: skips the rest of MY turn — through the my-main2 stop — and
            # clears at the turn boundary, so the next surfaced window is the
            # opponent-end-step stop of turn 2.
            websocket.send_json({"type": "pass_turn"})
            payload = websocket.receive_json()
            assert payload["type"] == "observation"
            assert payload["data"]["turn"]["turn_number"] == 2
            assert payload["data"]["turn"]["step"] == "ENDING_END"
            assert _turn_side(payload) == "opponent"
            assert payload["auto_passed"] > 0

            # F6 cleared: normal stops apply again on turn 3.
            websocket.send_json({"type": "action", "index": _pass_index(payload)})
            payload = websocket.receive_json()
            assert payload["data"]["turn"]["turn_number"] == 3
            assert payload["data"]["turn"]["step"] == "PRECOMBAT_MAIN_STEP"
            assert _turn_side(payload) == "my"

    # The game is still live (no trace file yet) — inspect the session trace.
    record = next(iter(server.SESSION_REGISTRY.values()))
    hero_events = [e for e in record.game.trace.events if e.actor == "hero"]
    assert any(e.auto for e in hero_events)
    assert all(
        not e.auto
        for e in hero_events
        if e.action_description != "Pass priority"
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
                    "hero_deck": {"Mountain": 8, "Raging Goblin": 8,
                                  "Lightning Bolt": 4},
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
                payload = websocket.receive_json()
                assert payload["type"] == "error", (config, payload)
                assert expected_fragment.lower() in payload["message"].lower()


def test_pass_turn_requires_active_game(isolated_traces):
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "pass_turn"})
            payload = websocket.receive_json()
            assert payload["type"] == "error"
            assert "No active game session" in payload["message"]
