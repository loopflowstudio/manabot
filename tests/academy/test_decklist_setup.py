"""Imported novel setups use the ordinary session, Command and replay boundary."""

from dataclasses import asdict
import json
from pathlib import Path

from fastapi.testclient import TestClient
from managym.decision import SemanticContractError
import pytest

from academy import server, trace as trace_store
from academy.trace import GameConfig, prepare_trace_payload
from manabot.env.match import Match
from manabot.infra.config import MatchConfig
from manabot.semantic.decision_contract import SemanticDecisionContract
import managym


def _request() -> dict[str, object]:
    return {
        "villain_type": "random",
        "seed": 7,
        "auto_pass": False,
        "deck_import": {
            "hero": {
                "filename": "novel.txt",
                "source": "# private note\n10 Forest\n10 Mountain\n2 Avatar Enthusiasts\nSideboard\n1 Island",
            },
            "villain": {"source": "20 Island"},
        },
    }


def test_imported_setup_matches_manabot_reconnect_and_rebuilt_root(
    tmp_path: Path,
) -> None:
    session = server.GameSession(trace_dir=tmp_path)
    first = session.new_game(_request())
    assert session.trace is not None
    config = session.trace.config
    assert config.hero_source is not None and config.villain_source is not None
    match_config = MatchConfig(
        world=managym.WORLD_SETUP,
        hero_source=managym.Decklist.from_dict(config.hero_source),
        villain_source=managym.Decklist.from_dict(config.villain_source),
        hero="Hero",
        villain="Villain",
        hero_deck=config.hero_deck,
        villain_deck=config.villain_deck,
        hero_sideboard=config.hero_sideboard,
        villain_sideboard=config.villain_sideboard,
    )
    bot = managym.Env(seed=7)
    bot.reset(Match(match_config).to_rust())
    assert bot.content_pack_manifest() == session.env.content_pack_manifest()
    assert bot.search_witness_json() == session.env.search_witness_json()
    human_contract = SemanticDecisionContract.from_env(session.env)
    bot_contract = SemanticDecisionContract.from_env(bot)
    assert human_contract.frame == bot_contract.frame
    recovery = session.current_recovery("reconnect")
    assert recovery["frame"]["prompt"] == first["frame"]["prompt"]
    assert recovery["frame"]["offers"] == first["frame"]["offers"]
    assert "private note" not in json.dumps(recovery)
    restored = GameConfig(**asdict(config))
    rebuilt = managym.Env(seed=7)
    rebuilt.reset(restored.to_rust())
    assert rebuilt.search_witness_json() == bot.search_witness_json()
    restored.hero_sideboard = {"Mountain": 1}
    with pytest.raises(ValueError, match="retained source"):
        restored.to_rust()
    restored.hero_sideboard = {"Island": True}
    with pytest.raises(ValueError, match="positive integer counts"):
        restored.to_rust()
    public = prepare_trace_payload(
        {"config": asdict(config), "events": [], "final_observation": {}}
    )
    assert "hero_source" not in public["config"]
    assert "villain_source" not in public["config"]
    assert public["config"]["hero_deck"] == {}
    assert public["config"]["villain_sideboard"] == {}
    offer = bot_contract.frame.find_verb("pass_priority")
    original = bot.search_witness_json()
    fork = bot.clone_env()
    SemanticDecisionContract.from_env(fork).apply(
        fork, offer["id"], command_id="import-pass"
    )
    assert bot.search_witness_json() == original
    bot_contract.apply(bot, offer["id"], command_id="import-pass")
    assert bot.search_witness_json() == fork.search_witness_json()
    with pytest.raises(SemanticContractError, match="stale"):
        bot_contract.apply(bot, offer["id"], command_id="import-stale")
    SemanticDecisionContract.from_env(rebuilt).apply(
        rebuilt, offer["id"], command_id="import-pass"
    )
    assert rebuilt.search_witness_json() == bot.search_witness_json()


def test_import_validation_returns_line_issues_with_source() -> None:
    request = _request()["deck_import"]
    assert isinstance(request, dict)
    request["hero"] = {"source": "10 Forest\n0 Island\n2 Unrecognized Card"}
    with TestClient(server.app) as client:
        response = client.post("/api/decklists/validate", json=request)
    assert response.status_code == 200
    payload = response.json()
    assert not payload["admitted"]
    document = payload["documents"]["hero"]
    assert document["source"] == request["hero"]["source"]
    assert {(issue["line"], issue["code"]) for issue in document["issues"]} >= {
        (2, "count"),
        (3, "unknown"),
    }


def test_imported_invalid_setup_does_not_replace_live_game(tmp_path: Path) -> None:
    session = server.GameSession(trace_dir=tmp_path)
    session.new_game(_request())
    witness = session.env.search_witness_json()
    request = _request()
    request["deck_import"] = {
        "hero": {"source": "bad line"},
        "villain": {"source": "20 Island"},
    }
    with pytest.raises(ValueError):
        session.new_game(request)
    assert session.env.search_witness_json() == witness


def test_rematch_re_admits_retained_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": _request()})
            assert websocket.receive_json()["type"] == "observation"
            (record,) = server.SESSION_REGISTRY.values()
            assert record.game.trace is not None
            first = record.game.trace.config
            websocket.send_json({"type": "rematch", "grant_revision": 1, "config": {}})
            assert websocket.receive_json()["type"] == "observation"
            assert record.game.trace is not None
            second = record.game.trace.config
    assert second is not first
    assert second.hero_source == first.hero_source
    assert second.villain_source == first.villain_source
    assert first.hero_source is not None
    assert first.hero_source["source"].startswith("# private note")
    assert second.hero_deck == first.hero_deck
