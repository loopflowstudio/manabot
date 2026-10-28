"""Explicit native/semantic bindings admitted for ordinary policy encoders.

A policy catalog is not inferred from card names or tensor dimensions. Each
entry names immutable compiled IR. The native pool descriptor supplies its
transport IDs. The native compilation receipt must match the IR before
any IDs reach the encoder. Research pools must register their own exact entry;
this module does not relax subset, checkpoint or setup admission.
"""

from collections.abc import Mapping
from pathlib import Path

from manabot.semantic.learning import (
    CURRENT_IR_PATHS,
    DEFAULT_SCHEMA_PATH,
    BoundSemanticPack,
    ContentPackBindingError,
)
import managym


def bind_policy_catalog(pack_key: str) -> BoundSemanticPack:
    """Admit checked programs against the actual native compiled manifest."""
    try:
        ir_path = CURRENT_IR_PATHS[pack_key]
    except KeyError as error:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}") from error
    pool = managym.content_pool(pack_key)
    compiled = pool.get("compiled_semantics")
    # Generic observation binding permits uncompiled manifests; policy catalogs
    # require native provenance before the shared binder compares its hashes.
    if not isinstance(compiled, Mapping):
        raise ContentPackBindingError("policy catalog requires a native compiled receipt")
    manifest = {
        "schema_version": 1,
        "content_digest": pool["content_digest"],
        "compiled_semantics": compiled,
        "definitions": [
            {
                "card_def_id": row["card_def_id"],
                "registry_name": row["registry_name"],
            }
            for row in pool["definitions"]
            if row["card_def_id"] is not None
        ],
    }
    pack = BoundSemanticPack.bind(manifest, ir_path=ir_path)
    if len(pack.definition_row_by_card_def_id) != len(manifest["definitions"]):
        raise ContentPackBindingError(
            "policy catalog does not cover native definitions"
        )
    return pack


def learning_schema_path(pack_key: str | None) -> Path:
    """Current catalogs share one schema; reject unregistered policy packs."""
    if pack_key is not None and pack_key not in CURRENT_IR_PATHS:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}")
    return DEFAULT_SCHEMA_PATH
