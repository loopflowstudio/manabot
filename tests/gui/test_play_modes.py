"""
test_play_modes.py
End-to-end play-mode tests: pluggable opponents, full games driven through the
HTTP/WebSocket interface as if a human were clicking, and hidden-information
integrity of every human-facing payload.
"""

import json
import random
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

# Local imports
from gui import server, trace as trace_store
from gui.server import app

MAX_HERO_MOVES = 3000


def _assert_hero_payload_is_clean(payload: dict) -> None:
    """The human-facing payload must never leak the villain's hidden info."""
    data = payload["data"]

    # The payload is always from the hero's perspective.
    assert data["agent"]["player_index"] == 0
    assert data["opponent"]["player_index"] == 1

    # The villain's hand must be absent (count only).
    assert data["opponent"]["hand"] == []
    assert "hand_hidden_count" in data["opponent"]
    assert data["opponent"]["hand_hidden_count"] == (
        data["opponent"]["zone_counts"].get("HAND", 0)
    )

    # Libraries are never serialized as card lists for either player.
    for side in ("agent", "opponent"):
        assert "library" not in data[side]
        assert "LIBRARY" not in {c.get("zone") for c in data[side]["graveyard"]}


def _play_full_game(client: TestClient, config: dict, hero_seed: int) -> dict:
    """Drive one full game through the WebSocket as a scripted human.

    Picks uniformly random legal actions from the server-provided action list,
    asserting on every payload that no illegal action is ever produced and no
    hidden information leaks. Returns the terminal payload.
    """
    rng = random.Random(hero_seed)
    with client.websocket_connect("/ws/play") as websocket:
        websocket.send_json({"type": "new_game", "config": config})
        payload = websocket.receive_json()

        for _ in range(MAX_HERO_MOVES):
            assert payload["type"] in {"observation", "game_over"}, payload
            _assert_hero_payload_is_clean(payload)

            if payload["type"] == "game_over":
                assert payload["winner"] in {0, 1, None}
                assert payload["data"]["game_over"] is True
                return payload

            actions = payload["actions"]
            assert actions, "Observation must come with legal actions"
            for action in actions:
                assert isinstance(action["description"], str)
                assert action["description"], "Actions must be labeled"

            choice = actions[rng.randrange(len(actions))]
            websocket.send_json({"type": "action", "index": choice["index"]})
            payload = websocket.receive_json()

        pytest.fail("Game did not reach a terminal state within the move budget")


@pytest.fixture()
def isolated_traces(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    return tmp_path


def test_full_game_vs_search_villain(isolated_traces):
    """Full game vs flat-MC search opponent; trace records the opponent config."""
    with TestClient(app) as client:
        payload = _play_full_game(
            client,
            {"villain_type": "search", "villain_sims": 8, "seed": 3},
            hero_seed=1,
        )
    assert payload["type"] == "game_over"

    trace_files = sorted(isolated_traces.glob("*.json"))
    assert len(trace_files) == 1
    trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_payload["config"]["villain_type"] == "search"
    assert trace_payload["config"]["villain_sims"] == 8
    assert trace_payload["end_reason"] == "game_over"
    assert trace_payload["winner"] in {0, 1, None}

    # The stored game must be loadable through the replay API: always
    # hero-perspective, villain hand hidden by default, visible on reveal.
    trace_id = trace_files[0].stem
    with TestClient(app) as client:
        loaded = client.get(f"/api/traces/{trace_id}").json()
        observations = [event["observation"] for event in loaded["events"]]
        observations.append(loaded["final_observation"])
        for observation in observations:
            assert observation["agent"]["player_index"] == 0
            assert observation["opponent"]["hand"] == []

        revealed = client.get(f"/api/traces/{trace_id}?reveal_hidden=true").json()
        villain_hands = [
            event["observation"]["opponent"]["hand"]
            for event in revealed["events"]
            if event["actor"] == "villain"
        ]
        assert villain_hands, "Expected villain decision points in the trace"
        assert any(hand for hand in villain_hands), (
            "reveal_hidden must expose the villain's hand"
        )


def test_full_game_vs_random_villain(isolated_traces):
    with TestClient(app) as client:
        payload = _play_full_game(
            client,
            {"villain_type": "random", "seed": 11},
            hero_seed=2,
        )
    assert payload["type"] == "game_over"

    trace_files = sorted(isolated_traces.glob("*.json"))
    assert len(trace_files) == 1
    trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_payload["config"]["villain_type"] == "random"


def _write_tiny_checkpoint(path) -> None:
    """Write a minimal (untrained) Agent checkpoint in the training format."""
    import torch

    from manabot.env import ObservationSpace
    from manabot.infra.hypers import AgentHypers, ObservationSpaceHypers
    from manabot.model.agent import Agent

    obs_hypers = ObservationSpaceHypers()
    agent_hypers = AgentHypers()
    agent = Agent(ObservationSpace(obs_hypers), agent_hypers)
    torch.save(
        {
            "hypers": {
                "observation_hypers": obs_hypers.model_dump(),
                "agent_hypers": agent_hypers.model_dump(),
            },
            "model_state_dict": agent.state_dict(),
        },
        path,
    )


def test_full_game_vs_checkpoint_villain(isolated_traces, tmp_path):
    checkpoint_path = tmp_path / "tiny_agent.pt"
    _write_tiny_checkpoint(checkpoint_path)

    with TestClient(app) as client:
        payload = _play_full_game(
            client,
            {
                "villain_type": "checkpoint",
                "villain_checkpoint": str(checkpoint_path),
                "villain_deterministic": False,
                "seed": 5,
            },
            hero_seed=3,
        )
    assert payload["type"] == "game_over"

    trace_files = sorted(isolated_traces.glob("*.json"))
    assert len(trace_files) == 1
    trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_payload["config"]["villain_type"] == "checkpoint"
    assert trace_payload["config"]["villain_checkpoint"] == str(checkpoint_path)


def test_new_game_rejects_bad_villain_configs(isolated_traces):
    cases = [
        ({"villain_type": "nonsense"}, "villain_type"),
        ({"villain_type": "checkpoint"}, "villain_checkpoint"),
        (
            {"villain_type": "checkpoint", "villain_checkpoint": "/no/such/file.pt"},
            "not found",
        ),
        ({"villain_type": "search", "villain_sims": 0}, "villain_sims"),
        ({"villain_type": "search", "villain_sims": "many"}, "villain_sims"),
    ]

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            for config, expected_fragment in cases:
                websocket.send_json({"type": "new_game", "config": config})
                payload = websocket.receive_json()
                assert payload["type"] == "error", (config, payload)
                assert expected_fragment.lower() in payload["message"].lower()


def test_named_decks_mirror_verify_util_constants():
    from manabot.verify.util import (
        GW_ALLIES_DECK,
        INTERACTIVE_DECK,
        UR_LESSONS_DECK,
    )

    assert server.NAMED_DECKS["interactive"] == INTERACTIVE_DECK
    assert server.NAMED_DECKS["ur_lessons"] == UR_LESSONS_DECK
    assert server.NAMED_DECKS["gw_allies"] == GW_ALLIES_DECK
    assert server.DEFAULT_DECK == INTERACTIVE_DECK


def test_default_villain_is_search_64(isolated_traces):
    config = server._parse_game_config({})
    assert config.villain_type == "search"
    assert config.villain_sims == 64
    # Default matchup is the Milestone-1 two-deck slice: UR hero vs GW.
    assert config.hero_deck == server.UR_LESSONS_DECK
    assert config.villain_deck == server.GW_ALLIES_DECK
    assert config.hero_deck_name == "ur_lessons"
    assert config.villain_deck_name == "gw_allies"


def test_deck_names_echoed_and_recorded_in_trace(isolated_traces):
    """Named decks: payloads echo display names, traces record identifiers."""
    with TestClient(app) as client:
        payload = _play_full_game(
            client,
            {
                "villain_type": "random",
                "seed": 11,
                "hero_deck": "ur_lessons",
                "villain_deck": "gw_allies",
                "auto_pass": False,
            },
            hero_seed=11,
        )
        assert payload["deck_names"] == {
            "hero": "UR Lessons",
            "villain": "GW Allies",
        }

    trace_files = sorted(isolated_traces.glob("*.json"))
    assert trace_files
    trace = json.loads(trace_files[-1].read_text())
    assert trace["config"]["hero_deck_name"] == "ur_lessons"
    assert trace["config"]["villain_deck_name"] == "gw_allies"
    assert trace["config"]["hero_deck"] == server.UR_LESSONS_DECK
    assert trace["config"]["villain_deck"] == server.GW_ALLIES_DECK


def test_named_deck_selection_and_custom_deck_names(isolated_traces):
    config = server._parse_game_config(
        {"hero_deck": "gw_allies", "villain_deck": {"Mountain": 40}}
    )
    assert config.hero_deck == server.GW_ALLIES_DECK
    assert config.hero_deck_name == "gw_allies"
    assert config.villain_deck == {"Mountain": 40}
    assert config.villain_deck_name == "custom"

    with pytest.raises(ValueError, match="Unknown deck name"):
        server._parse_game_config({"hero_deck": "not_a_deck"})


def _stub_action(action_type: int, focus: list[int]) -> SimpleNamespace:
    return SimpleNamespace(action_type=action_type, focus=focus)


def test_format_action_uses_magic_terms():
    from manabot.env.observation import ActionEnum

    names = {
        1: "Hero",
        2: "Villain",
        10: "Lightning Bolt",
        11: "Mountain",
        20: "Gray Ogre",
        21: "Wind Drake",
    }
    cases = [
        (_stub_action(int(ActionEnum.PRIORITY_PASS_PRIORITY), []), "Pass priority"),
        (_stub_action(int(ActionEnum.PRIORITY_PLAY_LAND), [11]), "Play Mountain"),
        (
            _stub_action(int(ActionEnum.PRIORITY_CAST_SPELL), [10]),
            "Cast Lightning Bolt",
        ),
        (
            _stub_action(int(ActionEnum.DECLARE_ATTACKER), [20]),
            "Attack with Gray Ogre",
        ),
        (
            _stub_action(int(ActionEnum.DECLARE_BLOCKER), [21, 20]),
            "Block Gray Ogre with Wind Drake",
        ),
        (
            _stub_action(int(ActionEnum.DECLARE_BLOCKER), [21]),
            "Wind Drake: do not block",
        ),
        (_stub_action(int(ActionEnum.CHOOSE_TARGET), [2]), "Target Villain"),
        (_stub_action(int(ActionEnum.CHOOSE_TARGET), [21]), "Target Wind Drake"),
    ]
    for action, expected in cases:
        assert server._format_action(action, names) == expected


def test_hero_view_swaps_terminal_villain_perspective_and_redacts(monkeypatch):
    """At game over the engine observation may be villain-perspective; the
    wire payload must still present the hero as `agent` with villain hand
    hidden."""

    villain_side = {
        "player_index": 1,
        "hand": [{"id": 50, "name": "Counterspell"}],
        "zone_counts": {"HAND": 1},
    }
    hero_side = {
        "player_index": 0,
        "hand": [{"id": 51, "name": "Lightning Bolt"}],
        "zone_counts": {"HAND": 1},
    }
    serialized = {
        "game_over": True,
        "won": True,  # villain-perspective: villain won
        "agent": villain_side,
        "opponent": hero_side,
    }
    monkeypatch.setattr(server, "serialize_observation", lambda obs: serialized)

    obs = SimpleNamespace(game_over=True, won=True)
    data = server.hero_view(obs)

    assert data["agent"]["player_index"] == 0
    assert data["opponent"]["player_index"] == 1
    assert data["won"] is False  # hero lost
    assert data["opponent"]["hand"] == []
    assert data["opponent"]["hand_hidden_count"] == 1
    # The hero still sees their own hand.
    assert data["agent"]["hand"][0]["name"] == "Lightning Bolt"
