"""Explicit native/semantic bindings admitted for ordinary policy encoders.

A policy catalog is not inferred from card names or tensor dimensions. Each
entry names immutable compiled IR. The native pool descriptor supplies its
transport IDs. The native compilation receipt must match the IR before
any IDs reach the encoder. Research pools must register their own exact entry;
this module does not relax subset, checkpoint or setup admission.
"""

from pathlib import Path

from manabot.semantic.learning import (
    DEFAULT_IR_PATH,
    DEFAULT_SCHEMA_PATH,
    BoundSemanticPack,
    ContentPackBindingError,
)
import managym

ROOT = DEFAULT_IR_PATH.parent


_CATALOGS = {
    "tla-pool-v1": "pool.ir.json",
    "tla-baseline-v1": "baseline.ir.json",
    "ur-lessons-vs-gw-allies": "two_deck.ir.json",
    "elemental-combat-v1": "elemental.ir.json",
    "elemental-sacrifice-v1": "sacrifice.ir.json",
    "elemental-temur-v1": "temur.ir.json",
    "elemental-world-v2": "optional_payments.ir.json",
    "elemental-world-v1": "elemental_world.ir.json",
    "mtg122-complexity-v1": "complexity.ir.json",
}


def bind_policy_catalog(pack_key: str) -> BoundSemanticPack:
    """Admit checked programs against the actual native compiled manifest."""
    try:
        filename = _CATALOGS[pack_key]
    except KeyError as error:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}") from error
    pool = managym.content_pool(pack_key)
    manifest = {
        "schema_version": 1,
        "content_digest": pool["content_digest"],
        "compiled_semantics": pool["compiled_semantics"],
        "definitions": [
            {
                "card_def_id": row["card_def_id"],
                "registry_name": row["registry_name"],
            }
            for row in pool["definitions"]
            if row["card_def_id"] is not None
        ],
    }
    pack = BoundSemanticPack.bind(
        manifest,
        ir_path=ROOT / filename,
        schema_path=learning_schema_path(pack_key),
    )
    compiled = manifest.get("compiled_semantics")
    expected = {
        "pack_key": pack_key,
        "ir_hash": pack.ir.ir_hash,
        "source_hash": pack.ir.raw["source_hash"],
    }
    if not isinstance(compiled, dict) or any(
        compiled.get(key) != value for key, value in expected.items()
    ):
        raise ContentPackBindingError("native compiled policy catalog differs from IR")
    if len(pack.definition_row_by_card_def_id) != len(manifest["definitions"]):
        raise ContentPackBindingError(
            "policy catalog does not cover native definitions"
        )
    return pack


def learning_schema_path(pack_key: str | None) -> Path:
    """Current catalogs share one schema; reject unregistered policy packs."""
    if pack_key is not None and pack_key not in _CATALOGS:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}")
    return DEFAULT_SCHEMA_PATH
