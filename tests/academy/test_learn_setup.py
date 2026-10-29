"""Full authored setup survives real session startup, seat reversal and storage."""

from dataclasses import asdict
from pathlib import Path

from managym.experience import ExperienceFrame
import pytest

from academy import server
from academy.curated_pack import CURATED_PACK, JEONG_INCREMENT_PACK
from academy.play_demo import load_play_demo
from academy.server import viewer_view
from academy.trace import GameConfig, load_trace
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, RewardConfig
import managym


def test_session_reveal_evidence_uses_definition_identity(tmp_path: Path) -> None:
    session = server.GameSession(trace_dir=tmp_path, capture_authority_evidence=True)
    frame = session.new_game({"demo": "learn"})["frame"]
    offer = next(o for o in frame["offers"] if o["action_type"] == "LEARN_TAKE_LESSON")
    expected_name = next(
        card.name
        for card in session.env.observation_for_player(0).agent_sideboard
        if card.candidate_id in offer["focus"]
    )
    result = session.hero_command(
        {
            "command_id": "reveal-lesson",
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"],
            "prompt_id": frame["prompt"]["id"],
            "offer_id": offer["id"],
            "answers": [],
        }
    )
    assert result["status"] == "accepted"
    reveals = [
        event
        for transition in session.authority_transitions
        for event in transition.semantic_events
        if event["source_kind"] == int(managym.EventEntityKindEnum.DEFINITION)
    ]
    definitions = {
        definition["card_def_id"]: definition["registry_name"]
        for definition in session.env.content_pack_manifest()["definitions"]
    }
    assert len(reveals) == 1
    assert {definitions[event["source_id"]] for event in reveals} == {expected_name}
    for event in reveals:
        assert event["event_type"] == "CARD_REVEALED"
        assert event["definition_ids"] == [event["source_id"]]


@pytest.mark.parametrize("reverse", [False, True])
def test_session_manabot_and_stored_root_agree(tmp_path: Path, reverse: bool) -> None:
    decks = ["ur_lessons", "gw_allies"]
    if reverse:
        decks.reverse()
    session = server.GameSession(trace_dir=tmp_path)
    session.new_game(
        {
            "hero_deck": decks[0],
            "villain_deck": decks[1],
            "seed": 0,
            "villain_type": "random",
            "auto_pass": False,
        }
    )
    config = session.trace.config
    assert (
        session.env.content_pack_manifest()["compiled_semantics"]["pack_key"]
        == CURATED_PACK.semantic_pack_key
    )
    assert sum(config.hero_deck.values()) == (40 if reverse else 41)
    assert sum(config.villain_deck.values()) == (41 if reverse else 40)
    assert sum(config.hero_sideboard.values()) == (2 if reverse else 3)
    assert sum(config.villain_sideboard.values()) == (3 if reverse else 2)

    match = Match(
        MatchConfig.authored(
            CURATED_PACK.semantic_pack_key, *decks, hero="Hero", villain="Villain"
        )
    )
    env = Env(match, ObservationSpace(), Reward(RewardConfig()), seed=0)
    env.reset(seed=0)
    witness = session.env.search_witness_json()
    training_witness = env._engine.search_witness_json()
    env.reset(seed=0)
    assert env._engine.search_witness_json() == training_witness

    recovery = session.current_recovery("reconnect")
    assert recovery["frame"]["match_id"] == session.match_id
    assert session.env.search_witness_json() == witness

    payload = load_trace(session.match_id, tmp_path)
    restored = GameConfig(**payload["config"])
    assert asdict(restored) == asdict(config)
    # Compare each consumer at its own native decision boundary. Academy
    # surfaces sole choices; training collapses them using the same setup.
    rebuilt = managym.Env(seed=restored.seed, skip_trivial=False)
    rebuilt.reset(restored.to_rust())
    assert rebuilt.search_witness_json() == witness
    training_root = managym.Env(seed=restored.seed, skip_trivial=True)
    training_root.reset(restored.to_rust())
    assert training_root.search_witness_json() == training_witness


@pytest.mark.parametrize("reverse", [False, True])
def test_jeong_uses_its_explicit_empty_setup(tmp_path, reverse):
    decks = ["gw_allies_jeong", "ur_lessons"]
    if reverse:
        decks.reverse()
    session = server.GameSession(trace_dir=tmp_path)
    session.new_game(
        {
            "hero_deck": decks[0],
            "villain_deck": decks[1],
            "seed": 0,
            "villain_type": "random",
            "auto_pass": False,
        }
    )
    assert session.trace.config.hero_sideboard == {}
    assert session.trace.config.villain_sideboard == {}
    assert session.asset_pack == JEONG_INCREMENT_PACK.reference
    assert (
        session.env.content_pack_manifest()["compiled_semantics"]["pack_key"]
        == JEONG_INCREMENT_PACK.semantic_pack_key
    )


def test_named_setup_cannot_silently_drop_sideboard():
    with pytest.raises(ValueError, match="hero_sideboard must match"):
        server._parse_game_config({"hero_sideboard": {}})
    custom = server._parse_game_config({"hero_deck": {"Island": 20}})
    assert custom.hero_sideboard == {}
    assert custom.asset_pack is None


def test_recorded_named_root_rejects_missing_sideboards_and_old_asset_identity():
    config = server._parse_game_config({"seed": 0, "villain_type": "random"})
    missing = asdict(config)
    missing.pop("hero_sideboard")
    missing.pop("villain_sideboard")
    with pytest.raises(ValueError, match="differs from compiled deck/sideboard setup"):
        GameConfig(**missing).to_rust()
    stale = asdict(config)
    stale["asset_pack"]["version"] = "1.0.0"
    with pytest.raises(ValueError, match="Recorded asset pack differs"):
        GameConfig(**stale).to_rust()
    malformed = asdict(config)
    malformed["hero_sideboard"]["Firebending Lesson"] = True
    with pytest.raises(ValueError, match="positive integer counts"):
        GameConfig(**malformed).to_rust()


@pytest.mark.parametrize(
    "kind", ["LEARN_TAKE_LESSON", "LEARN_DISCARD", "DECLINE_CHOICE"]
)
def test_checked_demo_uses_normal_commands_and_viewer_safe_projection(
    tmp_path: Path, kind: str
) -> None:
    session = server.GameSession(trace_dir=tmp_path)
    frame = session.new_game({"demo": "learn"})["frame"]
    assert frame["action_space"] == "LEARN"
    assert frame["revision"] == load_play_demo("learn").decision_revision
    ExperienceFrame.model_validate(frame)
    assert len(session.canonical_decisions) >= len(load_play_demo("learn").prefix)
    assert len(frame["projection"]["agent"]["sideboard"]) == 3
    assert frame["projection"]["opponent"].get("sideboard", []) == []
    root = session.env.clone_env()
    witness = root.search_witness_json()
    offer = next(o for o in frame["offers"] if o["action_type"] == kind)
    command = {
        "command_id": kind,
        "match_id": frame["match_id"],
        "expected_revision": frame["revision"],
        "prompt_id": frame["prompt"]["id"],
        "offer_id": offer["id"],
        "answers": [],
    }
    result = session.hero_command(command)
    assert result["status"] == "accepted"
    assert root.search_witness_json() == witness
    for viewer in (0, 1):
        obs = session.env.observation_for_player(viewer)
        product = viewer_view(obs, viewer)
        assert product["opponent"]["hand"] == []
        assert "sideboard" not in product["opponent"]
        assert product["opponent"]["known_hand"] == {
            str(k): v for k, v in obs.opponent.known_hand.items()
        }
        # Even the transitional swapped-view adapter must redact owner IDs.
        swapped = viewer_view(obs, 1 - viewer)
        assert "sideboard" not in swapped["opponent"]
    recovery = session.current_recovery("reconnect")["frame"]
    assert recovery["projection"] == result["update"]["frame"]["projection"]
    assert len(recovery["projection"]["agent"]["sideboard"]) == (
        2 if kind == "LEARN_TAKE_LESSON" else 3
    )


def test_checked_demo_rejects_wrong_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from academy import play_demo

    demo = play_demo.load_play_demo("learn").model_copy(
        update={"decision_digest": "wrong"}
    )
    monkeypatch.setattr(play_demo, "load_play_demo", lambda name: demo)
    with pytest.raises(ValueError, match="checked decision"):
        server.GameSession(trace_dir=tmp_path).new_game({"demo": "learn"})
