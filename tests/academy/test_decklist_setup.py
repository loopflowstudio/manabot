"""Imported novel setups use the ordinary session, Command and replay boundary."""

from dataclasses import asdict
import json
from pathlib import Path

from fastapi.testclient import TestClient
from managym.decision import SemanticContractError
import pytest

from academy import server
from academy.trace import GameConfig, prepare_trace_payload
from manabot.decklist import ImportedMatch
from manabot.env.match import Match
from manabot.semantic.decision_contract import SemanticDecisionContract
import managym


def _request() -> dict[str, object]:
    return {
        "villain_type": "random",
        "seed": 7,
        "auto_pass": False,
        "deck_import": {
            "pack_key": "tla-pool-v1",
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
    imported = ImportedMatch.model_validate(config.imported_decks)
    match_config = imported.match_config()
    match_config.hero, match_config.villain = "Hero", "Villain"
    bot = managym.Env(seed=7, skip_trivial=False)
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
    rebuilt = managym.Env(seed=7, skip_trivial=False)
    rebuilt.reset(restored.to_rust())
    assert rebuilt.search_witness_json() == bot.search_witness_json()
    public = prepare_trace_payload(
        {"config": asdict(config), "events": [], "final_observation": {}}
    )
    assert "imported_decks" not in public["config"]
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
