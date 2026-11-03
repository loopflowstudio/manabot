"""Explicit native/semantic bindings admitted for ordinary policy encoders.

A policy catalog is not inferred from card names or tensor dimensions. A
policy names its vocabulary (``ModelConfig.world``) from the registered names;
every name binds the world's one catalog, whose pool descriptor supplies the
transport IDs and content digest the engine's learning catalog must match
before any IDs reach the encoder.
"""

from pathlib import Path

from manabot.semantic.learning import (
    CURRENT_PACK_KEYS,
    DEFAULT_SCHEMA_PATH,
    BoundSemanticPack,
    ContentPackBindingError,
)
import managym


def bind_policy_catalog(pack_key: str) -> BoundSemanticPack:
    """Admit the engine's catalog against the world pool's native descriptor."""
    if pack_key not in CURRENT_PACK_KEYS:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}")
    pool = managym.content_pool(managym.WORLD_SETUP)
    manifest = {
        "schema_version": 1,
        "content_digest": pool["content_digest"],
        "definitions": [
            {
                "card_def_id": row["card_def_id"],
                "registry_name": row["registry_name"],
            }
            for row in pool["definitions"]
        ],
    }
    pack = BoundSemanticPack.bind(manifest)
    if len(pack.definition_row_by_card_def_id) != len(manifest["definitions"]):
        raise ContentPackBindingError(
            "policy catalog does not cover native definitions"
        )
    return pack


def learning_schema_path(pack_key: str | None) -> Path:
    """Current catalogs share one schema; reject unregistered policy packs."""
    if pack_key is not None and pack_key not in CURRENT_PACK_KEYS:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}")
    return DEFAULT_SCHEMA_PATH
