"""Load and select immutable Etude curated asset packs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import managym

PACK_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "lib"
    / "packs"
    / "tla-ur-lessons-vs-gw-allies"
    / "v2"
    / "manifest.json"
)
JEONG_INCREMENT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "lib"
    / "packs"
    / "tla-gw-allies-jeong-vs-ur-lessons"
    / "v2"
    / "manifest.json"
)
HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
REMOTE_PREFIXES = ("http://", "https://", "//")
RIGHTS_FIELDS = {
    "asset_kind",
    "contains_third_party_art",
    "creator",
    "copyright_notice",
    "license",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key!r}")
        result[key] = value
    return result


def _record(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _rights_ref(value: Any, rights: dict[str, Any], path: str) -> str:
    reference = _string(value, path)
    if reference not in rights:
        raise ValueError(f"{path} references unknown rights record {reference!r}")
    return reference


def _contains_remote_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith(REMOTE_PREFIXES)
    if isinstance(value, list):
        return any(_contains_remote_value(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_remote_value(item) for item in value.values())
    return False


def _validate_rights(raw_rights: Any) -> dict[str, Any]:
    rights = _record(raw_rights, "rights")
    if not rights:
        raise ValueError("rights must contain at least one record")
    for rights_id, raw_record in rights.items():
        record = _record(raw_record, f"rights.{rights_id}")
        missing = RIGHTS_FIELDS - set(record)
        if missing:
            raise ValueError(
                f"rights.{rights_id} is missing: {', '.join(sorted(missing))}"
            )
        _string(record["asset_kind"], f"rights.{rights_id}.asset_kind")
        if not isinstance(record["contains_third_party_art"], bool):
            raise ValueError(
                f"rights.{rights_id}.contains_third_party_art must be a boolean"
            )
        _string(record["creator"], f"rights.{rights_id}.creator")
        _string(
            record["copyright_notice"],
            f"rights.{rights_id}.copyright_notice",
        )
        _string(record["license"], f"rights.{rights_id}.license")
    return rights


def _validate_card_counts(value: Any, path: str) -> dict[str, int]:
    cards = _record(value, path)
    for name, count in cards.items():
        _string(name, f"{path} key")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{path}.{name} must be a positive integer")
    return dict(cards)


def _validate_deck(
    raw_seat: Any, seat: str, pack_key: str
) -> tuple[str, str, dict[str, int], dict[str, int]]:
    record = _record(raw_seat, f"matchup.{seat}")
    deck_id = _string(record.get("deck_id"), f"matchup.{seat}.deck_id")
    display_name = _string(record.get("display_name"), f"matchup.{seat}.display_name")
    card_count = record.get("card_count")
    if (
        not isinstance(card_count, int)
        or isinstance(card_count, bool)
        or card_count <= 0
    ):
        raise ValueError(f"matchup.{seat}.card_count must be a positive integer")
    cards = _validate_card_counts(record.get("cards"), f"matchup.{seat}.cards")
    if sum(cards.values()) != card_count:
        raise ValueError(
            f"matchup.{seat} declares {card_count} cards but contains "
            f"{sum(cards.values())}"
        )
    sideboard = _validate_card_counts(
        record.get("sideboard"), f"matchup.{seat}.sideboard"
    )
    setup = managym.authored_deck_setup(pack_key, deck_id)
    if cards != setup.decklist or sideboard != setup.sideboard:
        raise ValueError(f"matchup.{seat} differs from compiled deck/sideboard setup")
    return deck_id, display_name, cards, sideboard


def _validate_palette(raw_palette: Any, path: str) -> tuple[str, str, str]:
    if not isinstance(raw_palette, list) or len(raw_palette) != 3:
        raise ValueError(f"{path} must contain exactly three colors")
    colors = tuple(raw_palette)
    if not all(
        isinstance(color, str) and HEX_COLOR_PATTERN.fullmatch(color)
        for color in colors
    ):
        raise ValueError(f"{path} colors must use #RRGGBB")
    return colors  # type: ignore[return-value]


def _validate_fallback(raw_fallback: Any, rights: dict[str, Any]) -> None:
    fallback = _record(raw_fallback, "fallback")
    if fallback.get("version") != "fallback-v1":
        raise ValueError("fallback.version must be 'fallback-v1'")
    if fallback.get("algorithm") != "fnv1a-32-utf8":
        raise ValueError("fallback.algorithm must be 'fnv1a-32-utf8'")
    _rights_ref(fallback.get("rights_ref"), rights, "fallback.rights_ref")

    palettes = fallback.get("palettes")
    if not isinstance(palettes, list) or not palettes:
        raise ValueError("fallback.palettes must be a non-empty list")
    for index, palette in enumerate(palettes):
        _validate_palette(palette, f"fallback.palettes[{index}]")

    motifs = fallback.get("motifs")
    if not isinstance(motifs, list) or not motifs:
        raise ValueError("fallback.motifs must be a non-empty list")
    for index, motif in enumerate(motifs):
        _string(motif, f"fallback.motifs[{index}]")


def _validate_identities(
    raw_identities: Any,
    expected_names: set[str],
    token_names: set[str],
    rights: dict[str, Any],
) -> dict[str, Any]:
    identities = _record(raw_identities, "identities")
    actual_names = set(identities)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"identity inventory mismatch; missing={missing}, extra={extra}"
        )

    for name, raw_identity in identities.items():
        path = f"identities.{name}"
        identity = _record(raw_identity, path)
        expected_kind = "token" if name in token_names else "card"
        if identity.get("kind") != expected_kind:
            raise ValueError(f"{path}.kind must be {expected_kind!r}")

        provenance = _record(identity.get("provenance"), f"{path}.provenance")
        _string(provenance.get("provider"), f"{path}.provenance.provider")
        _string(provenance.get("source_uri"), f"{path}.provenance.source_uri")
        _string(provenance.get("retrieved_at"), f"{path}.provenance.retrieved_at")
        _rights_ref(
            provenance.get("rights_ref"), rights, f"{path}.provenance.rights_ref"
        )
        identity_field = "local_identity" if expected_kind == "token" else "oracle_id"
        _string(provenance.get(identity_field), f"{path}.provenance.{identity_field}")

        treatment = _record(identity.get("treatment"), f"{path}.treatment")
        if _contains_remote_value(treatment):
            raise ValueError(f"{path}.treatment must not contain remote URLs")
        _validate_palette(treatment.get("palette"), f"{path}.treatment.palette")
        _string(treatment.get("motif"), f"{path}.treatment.motif")
        seed = treatment.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"{path}.treatment.seed must be an integer")
        _rights_ref(treatment.get("rights_ref"), rights, f"{path}.treatment.rights_ref")
    return identities


@dataclass(frozen=True)
class CuratedPack:
    manifest_path: Path
    manifest_sha256: str
    pack_id: str
    version: str
    title: str
    semantic_pack_key: str
    hero_deck_id: str
    hero_display_name: str
    hero_deck: dict[str, int]
    villain_deck_id: str
    villain_display_name: str
    villain_deck: dict[str, int]
    identities: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def reference(self) -> dict[str, str]:
        return {
            "id": self.pack_id,
            "version": self.version,
            "manifest_sha256": self.manifest_sha256,
        }

    def player_config(self, deck_id: str, name: str) -> managym.PlayerConfig:
        """Resolve a complete setup from this pack's compiled authority."""
        if deck_id not in (self.hero_deck_id, self.villain_deck_id):
            raise ValueError(f"Deck {deck_id!r} is not in pack {self.pack_id!r}")
        setup = managym.authored_deck_setup(self.semantic_pack_key, deck_id)
        return managym.PlayerConfig(name, setup.decklist, setup.sideboard)


def load_curated_pack(path: Path = PACK_MANIFEST_PATH) -> CuratedPack:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Curated pack manifest is unavailable: {path}") from exc
    try:
        manifest = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Curated pack manifest is invalid: {path}: {exc}") from exc

    try:
        root = _record(manifest, "manifest")
        if root.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        pack = _record(root.get("pack"), "pack")
        pack_id = _string(pack.get("id"), "pack.id")
        version = _string(pack.get("version"), "pack.version")
        title = _string(pack.get("title"), "pack.title")
        if (
            type(root.get("setup_schema_version")) is not int
            or root["setup_schema_version"] != 1
        ):
            raise ValueError("setup_schema_version must be 1")
        semantic_pack_key = _string(root.get("semantic_pack_key"), "semantic_pack_key")
        rights = _validate_rights(root.get("rights"))

        matchup = _record(root.get("matchup"), "matchup")
        hero_deck_id, hero_display_name, hero_deck, hero_sideboard = _validate_deck(
            matchup.get("hero"), "hero", semantic_pack_key
        )
        villain_deck_id, villain_display_name, villain_deck, villain_sideboard = (
            _validate_deck(matchup.get("villain"), "villain", semantic_pack_key)
        )
        if hero_deck_id == villain_deck_id:
            raise ValueError("hero and villain deck IDs must differ")

        raw_tokens = matchup.get("reachable_tokens")
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ValueError("matchup.reachable_tokens must be a non-empty list")
        token_names = {
            _string(name, f"matchup.reachable_tokens[{index}]")
            for index, name in enumerate(raw_tokens)
        }
        if len(token_names) != len(raw_tokens):
            raise ValueError("matchup.reachable_tokens must not contain duplicates")

        _validate_fallback(root.get("fallback"), rights)
        expected_names = (
            set(hero_deck)
            | set(villain_deck)
            | set(hero_sideboard)
            | set(villain_sideboard)
            | token_names
        )
        identities = _validate_identities(
            root.get("identities"), expected_names, token_names, rights
        )
    except ValueError as exc:
        raise RuntimeError(f"Curated pack manifest is invalid: {path}: {exc}") from exc

    return CuratedPack(
        manifest_path=path,
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        pack_id=pack_id,
        version=version,
        title=title,
        semantic_pack_key=semantic_pack_key,
        hero_deck_id=hero_deck_id,
        hero_display_name=hero_display_name,
        hero_deck=hero_deck,
        villain_deck_id=villain_deck_id,
        villain_display_name=villain_display_name,
        villain_deck=villain_deck,
        identities=identities,
        manifest=root,
    )


CURATED_PACK = load_curated_pack()
JEONG_INCREMENT_PACK = load_curated_pack(JEONG_INCREMENT_MANIFEST_PATH)
CURATED_PACK_CATALOG = (CURATED_PACK, JEONG_INCREMENT_PACK)


def curated_pack_for_matchup(
    hero_deck_name: str,
    villain_deck_name: str,
    catalog: tuple[CuratedPack, ...] = CURATED_PACK_CATALOG,
) -> CuratedPack | None:
    """Select the same immutable pack in either seat, failing on ambiguity."""
    matches = [
        pack
        for pack in catalog
        if {hero_deck_name, villain_deck_name}
        == {pack.hero_deck_id, pack.villain_deck_id}
    ]
    if len(matches) > 1:
        raise RuntimeError(
            "Curated pack catalog is ambiguous: "
            + ", ".join(pack.pack_id for pack in matches)
        )
    return matches[0] if matches else None
