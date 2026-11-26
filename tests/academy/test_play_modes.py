"""
test_play_modes.py
End-to-end play-mode tests: pluggable opponents, full games driven through the
HTTP/WebSocket interface as if a human were clicking, and hidden-information
integrity of every human-facing payload.
"""

import json
from pathlib import Path
import random
import sqlite3
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

import managym
from managym import experience_projection

# Local imports
from managym.academy import server, trace as trace_store
from managym.academy.attempts import AttemptStore
from managym.academy.opponent import set_policy_registry
from managym.academy.recording import Recording
from managym.academy.server import app
from managym.academy.watch import WatchRequest, create_watch
from managym.experience import UNVERSIONED_CONTENT_HASH
from managym.policy import PolicyRegistry
from tests.academy.socket_messages import settled_message
from tests.academy.test_policy_runtime import IndependentFactory
from tests.academy.test_server import _command, _offer

MAX_HERO_MOVES = 3000
CURATED_COMBAT_FIXTURE = json.loads(
    (
        Path(__file__).parents[2]
        / "academy/ui/src/lib/fixtures/curated-combat-to-turn-w4.json"
    ).read_text(encoding="utf-8")
)


def _recorded_traces(directory):
    with sqlite3.connect(directory / "play.sqlite") as db:
        return [
            (row[0], json.loads(row[1]))
            for row in db.execute(
                "SELECT id, trace_json FROM attempts ORDER BY started_at"
            )
        ]


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
        payload = settled_message(websocket)

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
            payload = settled_message(websocket)

        pytest.fail("Game did not reach a terminal state within the move budget")


@pytest.fixture()
def isolated_traces(monkeypatch, tmp_path):
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    return tmp_path


def test_full_game_vs_search_villain(isolated_traces):
    """Full game vs flat-MC search opponent; trace records the opponent config."""
    with TestClient(app) as client:
        payload = _play_full_game(
            client,
            {
                "villain_type": "search",
                "villain_sims": 8,
                "hero_deck": {"Mountain": 8, "Lightning Bolt": 4, "Gray Ogre": 4},
                "villain_deck": {"Mountain": 8, "Gray Ogre": 8},
                "seed": 3,
            },
            hero_seed=1,
        )
    assert payload["type"] == "game_over"

    trace_files = _recorded_traces(isolated_traces)
    assert len(trace_files) == 1
    trace_payload = trace_files[0][1]
    assert trace_payload["config"]["villain_type"] == "search"
    assert trace_payload["config"]["villain_sims"] == 8
    assert trace_payload["end_reason"] == "game_over"
    assert trace_payload["winner"] in {0, 1, None}

    # The stored game must be loadable through the replay API: always
    # hero-perspective, villain hand hidden by default, visible on reveal.
    trace_id = trace_files[0][0]
    with TestClient(app) as client:
        loaded = client.get(
            f"/api/traces/{trace_id}",
        ).json()
        observations = [event["observation"] for event in loaded["events"]]
        observations.append(loaded["final_observation"])
        for observation in observations:
            assert observation["agent"]["player_index"] == 0
            assert observation["opponent"]["hand"] == []

        revealed = client.get(
            f"/api/traces/{trace_id}?reveal_hidden=true",
        ).json()
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
            {
                "villain_type": "random",
                "hero_deck": {"Mountain": 8, "Lightning Bolt": 4, "Gray Ogre": 4},
                "villain_deck": {"Mountain": 8, "Gray Ogre": 8},
                "seed": 11,
            },
            hero_seed=2,
        )
    assert payload["type"] == "game_over"

    trace_files = _recorded_traces(isolated_traces)
    assert len(trace_files) == 1
    trace_payload = trace_files[0][1]
    assert trace_payload["config"]["villain_type"] == "random"


def _write_tiny_checkpoint(path) -> None:
    """Write a minimal (untrained) ActorCritic checkpoint in the training format."""
    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.infra.config import ModelConfig, ObservationConfig

    obs_config = ObservationConfig()
    agent_config = ModelConfig(world="ur-lessons-vs-gw-allies")
    agent = ActorCritic(ObservationSpace(obs_config), agent_config)
    from manabot.backends.torch.checkpoint import save_checkpoint
    import managym

    save_checkpoint(
        agent,
        agent.observation_space,
        path,
        player_configs=[
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ],
    )


def test_new_game_rejects_bad_villain_configs(isolated_traces):
    cases = [
        ({"villain_type": "nonsense"}, "villain_type"),
        ({"villain_type": "checkpoint"}, "opponent_sha256"),
        (
            {"villain_type": "checkpoint", "villain_checkpoint": "/no/such/file.pt"},
            "opponent_sha256",
        ),
        ({"villain_type": "search", "villain_sims": 0}, "villain_sims"),
        ({"villain_type": "search", "villain_sims": "many"}, "villain_sims"),
    ]

    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            for config, expected_fragment in cases:
                websocket.send_json({"type": "new_game", "config": config})
                payload = settled_message(websocket)
                assert payload["type"] == "error", (config, payload)
                assert expected_fragment.lower() in payload["message"].lower()


def test_named_decks_mirror_verify_util_constants():
    from manabot.verify.games import GW_ALLIES_DECK, INTERACTIVE_DECK, UR_LESSONS_DECK

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


def test_closed_game_plays_through_and_keeps_its_setup_in_the_stored_trace(
    isolated_traces,
):
    """Stack labels and public cards keep their names through a whole closed game,
    and the stored trace can still rebuild it."""
    with TestClient(app) as client:
        payload = _play_full_game(
            client,
            {"villain_type": "random", "seed": 11, "decklists": "closed"},
            hero_seed=11,
        )
    assert payload["deck_names"]["villain"] == "Closed decklist"
    assert "asset_pack" not in payload
    assert "main_counts" not in payload["frame"]["projection"]["opponent"]
    trace = _recorded_traces(isolated_traces)[-1][1]
    assert trace["config"]["decklists"] == "closed"
    assert trace["config"]["villain_deck"] == server.GW_ALLIES_DECK
    assert trace["config"]["asset_pack"] == server.CURATED_PACK.reference


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
        assert payload["asset_pack"] == server.CURATED_PACK.reference
        assert payload["frame"]["asset_pack"] == server.CURATED_PACK.reference
        assert (
            payload["frame"]["asset_manifest_hash"]
            == server.CURATED_PACK.manifest_sha256
        )
        assert (
            payload["recovery"]["asset_manifest_hash"]
            == server.CURATED_PACK.manifest_sha256
        )

    trace_files = _recorded_traces(isolated_traces)
    assert trace_files
    trace = trace_files[-1][1]
    assert trace["config"]["hero_deck_name"] == "ur_lessons"
    assert trace["config"]["villain_deck_name"] == "gw_allies"
    assert trace["config"]["hero_deck"] == server.UR_LESSONS_DECK
    assert trace["config"]["villain_deck"] == server.GW_ALLIES_DECK
    assert trace["config"]["asset_pack"] == server.CURATED_PACK.reference

    presentation = [
        beat for event in trace["events"] for beat in event.get("presentation", [])
    ]
    kinds = [beat["kind"]["kind"] for beat in presentation]
    assert {"attack_group", "damage", "died", "turn_started"} <= set(kinds)
    assert [beat["seq"] for beat in presentation] == sorted(
        beat["seq"] for beat in presentation
    )
    assert all(beat["to_revision"] >= beat["from_revision"] for beat in presentation)


def test_curated_combat_tape_is_identical_live_trace_and_inspector_fixture(
    isolated_traces: Path,
) -> None:
    """One exact UR-vs-GW position crosses every semantic event consumer."""

    from manabot.env.observation import ActionEnum

    def attacking_villain(_env: object, obs: managym.Observation) -> int:
        for index, action in enumerate(obs.action_space.actions):
            if (
                int(action.action_type) == int(ActionEnum.DECLARE_ATTACKER)
                and action.declared is True
            ):
                return index
        for index, action in enumerate(obs.action_space.actions):
            if int(action.action_type) == int(ActionEnum.PRIORITY_PASS_PRIORITY):
                return index
        return 0

    session = server.GameSession(trace_dir=isolated_traces)
    session.new_game(
        {
            "villain_type": "passive",
            "seed": 203,
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
            "auto_pass": False,
        }
    )
    assert session.env is not None and session.obs is not None
    while session.obs.turn.turn_number == 0:
        session.obs, *_ = session.env.step(0)
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    session.env.scenario_force_battlefield(0, "Otter-Penguin", True)
    session.env.scenario_force_battlefield(1, "Badgermole Cub", True)
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None
    session.villain_policy = attacking_villain

    live_presentation = []
    for step in range(500):
        frame = session._experience_frame()
        assert frame["prompt"] is not None
        prompt = session.published_prompt
        assert prompt is not None

        chosen = 0
        if prompt.action_space == "DECLARE_BLOCKER":
            chosen = next(
                index
                for index, action in enumerate(prompt.actions)
                if action["declared"] is True
            )
        elif prompt.action_space == "DECLARE_ATTACKER":
            chosen = next(
                index
                for index, action in enumerate(prompt.actions)
                if action["declared"] is False
            )
        else:
            chosen = next(
                index
                for index, action in enumerate(prompt.actions)
                if action["type"] == "PRIORITY_PASS_PRIORITY"
            )

        outcome = session.hero_command(
            {
                "command_id": f"curated-combat-{step}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": frame["offers"][chosen]["id"],
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
        live_presentation.extend(outcome["update"]["presentation"])
        kinds = [beat["kind"]["kind"] for beat in live_presentation]
        if "died" in kinds and "turn_started" in kinds[kinds.index("died") + 1 :]:
            break
    else:
        pytest.fail("curated combat tape did not reach the next turn")

    assert session.trace is not None
    trace_presentation = [
        beat for event in session.trace.events for beat in event.presentation
    ]
    assert trace_presentation == live_presentation
    live_kinds = [beat["kind"]["kind"] for beat in live_presentation]
    combat_start = live_kinds.index("attack_group")
    curated_tape = live_presentation[combat_start:]
    assert live_kinds[combat_start:] == [
        "attack_group",
        "blocked",
        "damage",
        "damage",
        "died",
        "turn_started",
        "drew",
    ]
    assert curated_tape == CURATED_COMBAT_FIXTURE["events"]
    recovery = session.current_recovery("explicit_resync")
    assert recovery["presentation_cursor"] == live_presentation[0]["seq"]
    assert recovery["presentation_tail"] == live_presentation
    # Player-0's track preserves the opponent's semantic attack but does not
    # expose the private policy command that caused it.
    assert curated_tape[0]["caused_by"] is None
    assert all(beat["caused_by"] for beat in curated_tape[1:])


def test_projection_states_combat_assignments_only_while_they_hold(
    isolated_traces: Path,
) -> None:
    """The live projection carries an attack and a block, omitted otherwise."""

    from manabot.env.observation import ActionEnum

    def attacking_villain(_env: object, obs: managym.Observation) -> int:
        for index, action in enumerate(obs.action_space.actions):
            if (
                int(action.action_type) == int(ActionEnum.DECLARE_ATTACKER)
                and action.declared is True
            ):
                return index
        for index, action in enumerate(obs.action_space.actions):
            if int(action.action_type) == int(ActionEnum.PRIORITY_PASS_PRIORITY):
                return index
        return 0

    session = server.GameSession(trace_dir=isolated_traces)
    session.new_game(
        {
            "villain_type": "passive",
            "seed": 203,
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
            "auto_pass": False,
        }
    )
    assert session.env is not None and session.obs is not None
    while session.obs.turn.turn_number == 0:
        session.obs, *_ = session.env.step(0)
    session.env.scenario_clear_hand(0)
    session.env.scenario_clear_hand(1)
    session.env.scenario_force_battlefield(0, "Otter-Penguin", True)
    session.env.scenario_force_battlefield(1, "Badgermole Cub", True)
    # A castable instant keeps the hero's priority after blocks a real
    # decision, so forced-decision skipping stops before combat damage.
    session.env.scenario_force_battlefield(0, "Mountain", True)
    session.env.scenario_force_card_in_hand(0, "Firebending Lesson")
    session.obs = session.env.scenario_refresh()
    session.published_prompt = None
    session.villain_policy = attacking_villain

    def battlefield(side: str) -> dict[str, dict[str, object]]:
        data = experience_projection.serialize_observation(session.obs)
        return {str(entry["name"]): entry for entry in data[side]["battlefield"]}

    assert "attacking" not in battlefield("opponent")["Badgermole Cub"]
    blocked = False
    for step in range(500):
        frame = session._experience_frame()
        prompt = session.published_prompt
        assert prompt is not None and frame["prompt"] is not None
        if blocked:
            attacker = battlefield("opponent")["Badgermole Cub"]
            blocker = battlefield("agent")["Otter-Penguin"]
            assert attacker["attacking"] is True
            assert blocker["blocking"] == attacker["id"]
            assert "attacking" not in blocker
            break
        if prompt.action_space == "DECLARE_BLOCKER":
            assert battlefield("opponent")["Badgermole Cub"]["attacking"] is True
            assert "blocking" not in battlefield("agent")["Otter-Penguin"]
            chosen = next(
                index
                for index, action in enumerate(prompt.actions)
                if action["declared"] is True
            )
            blocked = True
        elif prompt.action_space == "DECLARE_ATTACKER":
            chosen = next(
                index
                for index, action in enumerate(prompt.actions)
                if action["declared"] is False
            )
        else:
            chosen = next(
                index
                for index, action in enumerate(prompt.actions)
                if action["type"] == "PRIORITY_PASS_PRIORITY"
            )
        outcome = session.hero_command(
            {
                "command_id": f"combat-projection-{step}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": frame["offers"][chosen]["id"],
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
    else:
        pytest.fail("the villain never attacked into a block")


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


def test_jeong_matchup_uses_session_scoped_pack_identity(isolated_traces):
    session = server.GameSession(trace_dir=isolated_traces)
    payload = session.new_game(
        {
            "villain_type": "random",
            "seed": 23,
            "hero_deck": "gw_allies_jeong",
            "villain_deck": "ur_lessons",
            "auto_pass": False,
        }
    )
    assert session.env is not None
    manifest = session.env.content_pack_manifest()
    assert manifest["content_digest"] == managym.world_identity()["card_digest"]
    assert session.content_hash == manifest["content_digest"]
    assert session.content_hash != UNVERSIONED_CONTENT_HASH
    assert session.asset_manifest_hash == server.JEONG_INCREMENT_PACK.manifest_sha256
    assert payload["frame"]["content_hash"] == session.content_hash
    assert payload["frame"]["asset_manifest_hash"] == session.asset_manifest_hash
    assert payload["asset_pack"] == server.JEONG_INCREMENT_PACK.reference

    replay = session.canonical_replay()
    assert replay.content_hash == session.content_hash
    assert replay.asset_manifest_hash == session.asset_manifest_hash


def _stub_action(
    action_type: int, focus: list[int], declared: bool | None = None
) -> SimpleNamespace:
    return SimpleNamespace(action_type=action_type, focus=focus, declared=declared)


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
            _stub_action(int(ActionEnum.DECLARE_ATTACKER), [20], False),
            "Do not attack with Gray Ogre",
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
        assert experience_projection.format_action(action, names) == expected


def test_viewer_view_swaps_terminal_villain_perspective_and_redacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At game over the engine observation may be villain-perspective; the
    wire payload must still present the hero as `agent` with villain hand
    hidden, and read the hero's result from the winner: a draw leaves the
    villain's `won: false` and the hero's alike."""

    def serialized(won: bool) -> dict[str, object]:
        return {
            "game_over": True,
            "won": won,
            "agent": {
                "player_index": 1,
                "hand": [{"id": 50, "name": "Counterspell"}],
                "zone_counts": {"HAND": 1},
            },
            "opponent": {
                "player_index": 0,
                "hand": [{"id": 51, "name": "Lightning Bolt"}],
                "zone_counts": {"HAND": 1},
            },
        }

    for villain_won, winner, hero_won in [
        (True, 1, False),
        (False, 0, True),
        (False, None, False),
    ]:
        monkeypatch.setattr(
            server,
            "serialize_observation",
            lambda _obs, *, include_details=False: serialized(villain_won),
        )
        obs = SimpleNamespace(game_over=True, won=villain_won)
        data = server.viewer_view(obs, 0, winner=winner)

        assert data["agent"]["player_index"] == 0
        assert data["opponent"]["player_index"] == 1
        assert data["won"] is hero_won
        assert data["opponent"]["hand"] == []
        assert data["opponent"]["hand_hidden_count"] == 1
        # The hero still sees their own hand.
        assert data["agent"]["hand"][0]["name"] == "Lightning Bolt"


def test_trace_replay_keeps_a_draw_unwon_for_the_hero() -> None:
    """A drawn game whose final observation is the villain's (`won: false`)
    is no win for the hero after the sides swap back."""
    for winner, hero_won in [(None, False), (0, True), (1, False)]:
        observation: dict[str, object] = {
            "game_over": True,
            "won": winner == 1,
            "agent": {"player_index": 1},
            "opponent": {"player_index": 0},
        }
        trace_store.normalize_observation_to_hero(observation, 0, winner=winner)
        assert observation["agent"] == {"player_index": 0}
        assert observation["won"] is hero_won


def test_redaction_hides_only_the_opponents_face_down_exile() -> None:
    """A foretold card is face down: the owner's view lists it, and a view
    swapped onto the other player removes it while keeping face-up exile."""

    def player(index: int) -> dict[str, object]:
        return {
            "player_index": index,
            "hand": [],
            "zone_counts": {"HAND": 0, "EXILE": 2},
            "exile": [
                {"id": 60 + index, "name": "Sozin's Comet", "face_down": True},
                {"id": 70 + index, "name": "Gray Ogre"},
            ],
        }

    observation: dict[str, object] = {"agent": player(0), "opponent": player(1)}
    experience_projection.redact_observation(observation)

    assert [card["id"] for card in observation["agent"]["exile"]] == [60, 70]
    assert [card["id"] for card in observation["opponent"]["exile"]] == [71]


# ———————————————— Setup: native seats, mana options, rematch and resume

CONFIG: dict[str, Any] = {
    "hero_deck": {"Island": 8},
    "villain_deck": {"Mountain": 8},
    "villain_type": "passive",
    "seed": "18446744073709551615",
    "auto_pass": False,
}


def finish(game: server.GameSession) -> None:
    for index in range(100):
        frame = game.current_message()["frame"]
        if frame["status"] == "game_over":
            return
        offer = next(
            (
                o
                for o in frame["offers"]
                if o["action_type"] == "PRIORITY_PASS_PRIORITY"
            ),
            frame["offers"][0],
        )
        result = game.hero_command(
            {
                "command_id": f"finish.{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"
    raise AssertionError("short deck-out fixture did not finish")


@pytest.mark.parametrize("seat", [0, 1])
def test_native_seats_complete_record_reconstruct_and_study(
    tmp_path: Path, seat: int
) -> None:
    game = server.GameSession(tmp_path)
    frame = game.new_game({**CONFIG, "human_seat": seat})["frame"]
    assert frame["projection"]["agent"]["player_index"] == seat
    assert frame["prompt"]["actor"] == seat
    assert game.trace is not None
    configs = game.trace.config.to_rust()
    assert configs.mana_floats == ("off", "off")
    assert configs[seat].decklist == CONFIG["hero_deck"]
    finish(game)
    payload = game.attempt_store.load(game.match_id)
    assert payload is not None
    assert payload["winner"] == 0  # Native seat 0 skips the first draw.
    assert payload["final_observation"]["agent"]["player_index"] == seat
    human = next(
        p
        for p in game.attempt_store.metadata(game.match_id)["players"]
        if p["kind"] == "human"
    )
    assert human["seat"] == seat
    recording = Recording(game.match_id, payload)
    choices = [p for p in recording.positions if p.kind == "choice"]
    assert choices and all(p.decision.viewer == seat for p in choices if p.decision)
    source = recording.read(choices[0].key)
    assert source["observation"]["agent"]["player_index"] == seat
    restored = game.fork_study(choices[0].key)
    assert restored is not None
    prepared = trace_store.prepare_trace_payload(payload)
    assert prepared["events"][0]["observation"]["agent"]["player_index"] == seat


def test_mana_tap_setup_records_default_off_identically_and_rejects_flags() -> None:
    omitted = server._parse_game_config(CONFIG).to_record()
    assert "mana_floats" not in omitted and "mana_payment" not in omitted
    assert (
        server._parse_game_config({**CONFIG, "mana_floats": "off"}).to_record()
        == omitted
    )
    for flag in (True, False):
        with pytest.raises(ValueError):
            server._parse_game_config({**CONFIG, "mana_floats": flag})


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("mode", ["all", "usable"])
def test_mana_tap_survives_reconnect_and_records_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seat: int, mode: managym.ManaFloats
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    config = {
        **CONFIG,
        "human_seat": seat,
        "mana_floats": mode,
        "mana_payment": "auto",
        "hero_deck": {"Island": 8, "Consider": 4}
        if mode == "usable"
        else {"Island": 16},
        "villain_deck": {"Mountain": 16},
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as ws:
            ws.send_json({"type": "new_game", "config": config})
            start = settled_message(ws)
            assert start["setup"]["mana_floats"] == mode
            assert start["setup"]["mana_payment"] == "auto"
            frame = start["recovery"]["frame"]
            keep = _offer(frame, action_type="CHOOSE_MODE", label="Keep hand")
            ws.send_json({"type": "command", "command": _command(frame, keep, "keep")})
            opening = settled_message(ws)
            assert opening["status"] == "accepted"
            frame = opening["update"]["frame"]
            land = _offer(frame, action_type="PRIORITY_PLAY_LAND")
            ws.send_json({"type": "command", "command": _command(frame, land, "land")})
            pending = settled_message(ws)["update"]["frame"]
            source = pending["projection"]["agent"]["battlefield"][0]
            tap = _offer(
                pending, action_type="TAP_FOR_COST", label="Tap Island for mana"
            )
            assert tap["actor"] == seat
            assert tap["focus"] == [source["id"]]
            assert tap["subjects"][0]["object"]["display_id"] == source["id"]
            command = _command(pending, tap, "float")

        with client.websocket_connect("/ws/play") as ws:
            ws.send_json({"type": "resume", "game": start["game"]})
            assert settled_message(ws)["recovery"]["frame"] == pending
            ws.send_json({"type": "command", "command": command})
            accepted = settled_message(ws)
            assert accepted["status"] == "accepted"
            ws.send_json(
                {"type": "command", "command": {**command, "command_id": "stale"}}
            )
            rejected = settled_message(ws)
            assert rejected["rejection"]["code"] == "stale_revision"
            assert rejected["recovery"]["frame"] == accepted["update"]["frame"]

            game = server.LIVE_GAMES[start["game"]].game
            payload = game.attempt_store.load(game.match_id)
            assert payload is not None
            recorded = trace_store.GameConfig(**payload["config"])
            assert payload["mana_payment"] == list(recorded.to_rust().mana_payment)
            assert payload["mana_floats"] == list(recorded.to_rust().mana_floats)
            assert recorded.to_rust().mana_payment[seat] == "auto"
            assert recorded.to_rust().mana_payment[1 - seat] == "manual"
            assert recorded.to_rust().mana_floats == (
                mode if seat == 0 else "off",
                mode if seat == 1 else "off",
            )
            recording = Recording(game.match_id, payload)
            position = next(
                p
                for p in recording.positions
                if p.decision
                and p.decision.command_id == "float"
                and p.kind == "choice"
            )
            decision = recording.read(position.key)["decision"]
            assert decision["frame"] == pending
            assert decision["offer"] == tap
            assert decision["command"] == command


def _rematch(ws: Any, game: str, **fields: Any) -> dict[str, Any]:
    ws.send_json({"type": "rematch", "game": game, **fields})
    return settled_message(ws)


def test_rematch_starts_from_the_played_setup_and_leaves_the_game_unfinished(
    isolated_traces: Path,
) -> None:
    server.LIVE_GAMES.clear()
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new_game", "config": CONFIG})
        first = settled_message(ws)
        assert first["setup"]["human_seat"] == 0
        # Stops changed during play are part of the setup the next game starts from.
        ws.send_json(
            {
                "type": "set_stops",
                "stops": {"my": [], "opponent": []},
                "auto_pass": False,
            }
        )
        assert settled_message(ws)["setup"]["stops"] == {"my": [], "opponent": []}

        swapped = _rematch(ws, first["game"], intent="swap_sides")
        assert swapped["type"] == "observation" and swapped["game"] != first["game"]
        assert swapped["setup"]["human_seat"] == 1
        assert swapped["setup"]["stops"] == {"my": [], "opponent": []}
        played = server.LIVE_GAMES[swapped["game"]].game.trace
        assert played is not None
        assert played.config.hero_deck == CONFIG["hero_deck"]
        assert played.config.seed == 2**64 - 1
        # The game left behind is dropped from memory and stays resumable.
        store = AttemptStore(isolated_traces / "play.sqlite")
        assert list(server.LIVE_GAMES) == [swapped["game"]]
        assert store.unfinished(first["game"]) and store.unfinished(swapped["game"])

        # A rematch reads the setup of the game it names, in memory or not.
        again = _rematch(ws, first["game"])
        assert again["setup"]["human_seat"] == 0 and again["game"] != first["game"]

        refused = _rematch(
            ws,
            again["game"],
            config={"villain_type": "policy", "opponent_sha256": "f" * 64},
        )
        assert refused["type"] == "error"
        for reserved in ("launch", "demo", "played_from"):
            assert (
                _rematch(ws, again["game"], config={reserved: "x"})["type"] == "error"
            )
        assert _rematch(ws, "no-such-game")["message"].startswith("rematch_unavailable")
        # A refused rematch starts nothing: the game in play still takes Commands.
        assert list(server.LIVE_GAMES) == [again["game"]]
        frame = again["recovery"]["frame"]
        keep = _offer(frame, action_type="CHOOSE_MODE", label="Keep hand")
        ws.send_json({"type": "command", "command": _command(frame, keep, "keep")})
        assert settled_message(ws)["status"] == "accepted"
        assert len(store.summaries()) == 3


def test_resume_takes_the_game_from_another_connection(isolated_traces: Path) -> None:
    server.LIVE_GAMES.clear()
    with TestClient(server.app) as client, client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new_game", "config": CONFIG})
        start = settled_message(ws)
        with client.websocket_connect("/ws/play") as other:
            other.send_json({"type": "resume", "game": start["game"]})
            taken = settled_message(other)
            assert taken["recovery"]["frame"] == start["recovery"]["frame"]
            # The first connection is closed: one connection plays a game.
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == 4000
            assert server.LIVE_GAMES[start["game"]].websocket is not None


@pytest.mark.parametrize("seat", [0, 1])
def test_rematch_keeps_imported_sideboards_and_takes_a_named_opponent(
    isolated_traces: Path, seat: int
) -> None:
    registry = PolicyRegistry(providers={})
    set_policy_registry(registry)
    first = IndependentFactory("First", runner="first-runner")
    second = IndependentFactory("Second", runner="second-runner")
    first_id, second_id = registry.register(first), registry.register(second)
    server.LIVE_GAMES.clear()
    config = {
        "villain_type": "policy",
        "opponent_sha256": first_id,
        "human_seat": seat,
        "decklists": "closed",
        "seed": 3,
        "deck_import": {
            "hero": {"source": "8 Island\nSideboard\n1 Swamp", "filename": "human.txt"},
            "villain": {
                "source": "8 Mountain\nSideboard\n1 Forest",
                "filename": "bot.txt",
            },
        },
    }
    try:
        with (
            TestClient(server.app) as client,
            client.websocket_connect("/ws/play") as ws,
        ):
            ws.send_json({"type": "new_game", "config": config})
            start = settled_message(ws)
            assert first.sessions[-1].seat == 1 - seat
            source = server.LIVE_GAMES[start["game"]].game.trace.config.hero_source
            ws.send_json(
                {
                    "type": "rematch",
                    "game": start["game"],
                    "config": {"opponent_sha256": second_id},
                }
            )
            next_game = settled_message(ws)
            assert next_game["opponent"]["sha256"] == second_id
            played = server.LIVE_GAMES[next_game["game"]].game.trace.config
            assert played.hero_source == source
            assert played.to_rust()[seat].sideboard == {"Swamp": 1}
            assert played.to_rust()[1 - seat].sideboard == {"Forest": 1}
            assert played.to_rust().decklists == "closed"
            assert next_game["frame"]["projection"]["opponent"]["hand"] == []
    finally:
        for game_id in list(server.LIVE_GAMES):
            server.end_game(game_id, "test")
        set_policy_registry(None)


def test_watch_setup_admits_imports(
    isolated_traces: Path,
) -> None:
    server.LIVE_GAMES.clear()
    created = create_watch(
        WatchRequest(
            players=["passive", "passive"],
            seed=None,
            decklists="closed",
            deck_import={
                "hero": {"source": "8 Island", "filename": "first.txt"},
                "villain": {"source": "8 Mountain", "filename": "second.txt"},
            },
        )
    )
    assert set(created) == {"id", "finished"}
    with TestClient(server.app) as client:
        index = client.get(f"/api/traces/{created['id']}/recording")
        assert index.status_code == 200, index.text
        assert index.json()["watch"]
        # A watched game has no seat to take up again.
        assert not index.json()["resumable"]
    server.end_game(created["id"], "new_game")


def test_native_draw_is_recorded_as_a_draw(tmp_path: Path) -> None:
    game = server.GameSession(tmp_path)
    frame = game.new_game(CONFIG)["frame"]
    keep = _offer(frame, action_type="CHOOSE_MODE", label="Keep hand")
    assert game.hero_command(_command(frame, keep, "keep"))["status"] == "accepted"
    assert game.env is not None
    # A constructed simultaneous-loss position, never a game for a policy or person.
    game.env.scenario_set_life(0, 0)
    game.env.scenario_set_life(1, 0)
    game.obs = game.env.scenario_refresh()
    game.obs, *_ = game.env.step(0)  # Run the native state-based checks.
    assert game.obs.game_over
    assert game.env.winner_index() is None
    game._persist_attempt(end_reason="game_over")
    (saved,) = game.attempt_store.summaries(status="completed")
    assert saved["winner"] is None and not saved["resumable"]


@pytest.mark.parametrize("seed", [True, 1.5, "1.5", "-1", str(2**64)])
def test_invalid_seed_never_retires_current_game(tmp_path: Path, seed: object) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(CONFIG)
    identifier = game.match_id
    with pytest.raises(ValueError, match="seed"):
        game.new_game({**CONFIG, "seed": seed})
    assert game.match_id == identifier
    assert game.attempt_store.load(identifier)["end_reason"] == "active"
    game.close("new_game")


@pytest.mark.parametrize("seat", [0, 1])
def test_player_targets_name_roles_at_either_native_seat(
    tmp_path: Path, seat: int
) -> None:
    game = server.GameSession(tmp_path)
    game.new_game(
        {
            **CONFIG,
            "hero_deck": {"Mountain": 10, "Lightning Bolt": 10},
            "villain_deck": {"Plains": 20},
            "human_seat": seat,
        }
    )
    for index in range(100):
        frame = game.current_message()["frame"]
        targets = {
            subject["object"]["subject"]["id"]: offer["label"]
            for offer in frame["offers"]
            for subject in offer.get("subjects", [])
            if subject["role"] == "target"
            and subject["object"]["subject"]["kind"] == "player"
        }
        if targets:
            assert targets == {seat: "Target Hero", 1 - seat: "Target Villain"}
            game.close("new_game")
            return
        offer = next(
            (
                o
                for prefix in ("Cast Lightning Bolt", "Play Mountain", "Pass")
                for o in frame["offers"]
                if o["label"].startswith(prefix)
            ),
            frame["offers"][0],
        )
        result = game.hero_command(
            {
                "command_id": f"target.{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"
    raise AssertionError("no player target was offered")


def test_recording_names_each_deck_by_its_display_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.LIVE_GAMES.clear()
    created = create_watch(WatchRequest(players=["passive", "passive"], seed=3))
    with TestClient(server.app) as client:
        index = client.get(f"/api/traces/{created['id']}/recording").json()
    names = {player["deck"]: player["deck_name"] for player in index["players"]}
    assert names == {
        server.DEFAULT_HERO_DECK_NAME: "UR Lessons",
        server.DEFAULT_VILLAIN_DECK_NAME: "GW Allies",
    }
    server.end_game(created["id"], "new_game")
