"""Contract tests for the installed UR Lessons versus GW Allies pack."""

from __future__ import annotations

import hashlib
import json

import pytest

from gui import server
from gui.curated_pack import CURATED_PACK, PACK_MANIFEST_PATH, load_curated_pack


def _contains_remote_value(value) -> bool:
    if isinstance(value, str):
        return value.startswith(("http://", "https://", "//"))
    if isinstance(value, list):
        return any(_contains_remote_value(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_remote_value(item) for item in value.values())
    return False


def test_pack_freezes_current_matchup_and_reachable_inventory():
    assert CURATED_PACK.pack_id == "tla-ur-lessons-vs-gw-allies"
    assert CURATED_PACK.version == "1.0.0"
    assert CURATED_PACK.hero_deck_id == "ur_lessons"
    assert CURATED_PACK.villain_deck_id == "gw_allies"
    assert sum(CURATED_PACK.hero_deck.values()) == 41
    assert sum(CURATED_PACK.villain_deck.values()) == 40
    assert len(CURATED_PACK.identities) == 31
    assert CURATED_PACK.identities["Ally"]["kind"] == "token"
    assert CURATED_PACK.identities["Clue"]["kind"] == "token"

    deck_names = set(CURATED_PACK.hero_deck) | set(CURATED_PACK.villain_deck)
    assert set(CURATED_PACK.identities) == deck_names | {"Ally", "Clue"}


def test_pack_treatments_are_local_and_rights_are_explicit():
    manifest = CURATED_PACK.manifest
    required_rights = {
        "asset_kind",
        "contains_third_party_art",
        "creator",
        "copyright_notice",
        "license",
    }
    for record in manifest["rights"].values():
        assert required_rights <= set(record)
        assert record["contains_third_party_art"] is False
        assert record["license"] == "NOASSERTION"

    for identity in CURATED_PACK.identities.values():
        assert not _contains_remote_value(identity["treatment"])


def test_pack_hash_and_backend_deck_views_derive_from_manifest():
    expected_hash = hashlib.sha256(PACK_MANIFEST_PATH.read_bytes()).hexdigest()
    assert CURATED_PACK.manifest_sha256 == expected_hash
    assert len(expected_hash) == 64
    assert server.ASSET_MANIFEST_HASH == expected_hash
    assert server.UR_LESSONS_DECK == CURATED_PACK.hero_deck
    assert server.GW_ALLIES_DECK == CURATED_PACK.villain_deck
    assert server.DECK_DISPLAY_NAMES["ur_lessons"] == "UR Lessons"
    assert server.DECK_DISPLAY_NAMES["gw_allies"] == "GW Allies"

    from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK

    assert UR_LESSONS_DECK == CURATED_PACK.hero_deck
    assert GW_ALLIES_DECK == CURATED_PACK.villain_deck


def test_exact_oriented_matchup_receives_pack_reference():
    config = server._parse_game_config(
        {"hero_deck": "ur_lessons", "villain_deck": "gw_allies"}
    )
    assert config.asset_pack == CURATED_PACK.reference

    reversed_config = server._parse_game_config(
        {"hero_deck": "gw_allies", "villain_deck": "ur_lessons"}
    )
    assert reversed_config.asset_pack is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest["matchup"]["hero"].__setitem__("card_count", 40),
            "declares 40 cards",
        ),
        (
            lambda manifest: manifest["identities"]["Island"]["treatment"].__setitem__(
                "image", "https://example.test/island.png"
            ),
            "must not contain remote URLs",
        ),
        (
            lambda manifest: manifest["identities"].pop("Clue"),
            "identity inventory mismatch",
        ),
    ],
)
def test_invalid_manifest_fails_closed(tmp_path, mutate, message):
    manifest = json.loads(PACK_MANIFEST_PATH.read_text(encoding="utf-8"))
    mutate(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_curated_pack(path)
