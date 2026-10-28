"""Explicit native/semantic bindings admitted for ordinary policy encoders.

A policy catalog is not inferred from card names or tensor dimensions. Each
entry names immutable compiled IR. Authored setups or the native pool descriptor
supply its transport IDs. The native compilation receipt must match the IR before
any IDs reach the encoder. Research pools must register their own exact entry;
this module does not relax subset, checkpoint or setup admission.
"""

from dataclasses import dataclass
from pathlib import Path

from manabot.semantic.learning import BoundSemanticPack, ContentPackBindingError
import managym

ROOT = Path(__file__).resolve().parents[2] / "content/semantic/current/generated"


@dataclass(frozen=True)
class PolicyCatalogSpec:
    ir_file: str
    setup_keys: tuple[str, str] | None
    schema_file: str = "learning_schema.json"


_CATALOGS = {
    "tla-pool-v1": PolicyCatalogSpec("pool.ir.json", None),
    "tla-baseline-v1": PolicyCatalogSpec("baseline.ir.json", ("tla_white", "tla_red")),
    "ur-lessons-vs-gw-allies": PolicyCatalogSpec(
        "two_deck.ir.json", ("ur_lessons", "gw_allies")
    ),
    "elemental-combat-v1": PolicyCatalogSpec(
        "elemental.ir.json",
        ("elemental_left", "elemental_right"),
        "learning_schema.json",
    ),
    "elemental-sacrifice-v1": PolicyCatalogSpec(
        "sacrifice.ir.json",
        ("sacrifice_left", "sacrifice_right"),
        "learning_schema.json",
    ),
    "elemental-temur-v1": PolicyCatalogSpec(
        "temur.ir.json",
        ("temur_left", "temur_right"),
        "learning_schema.json",
    ),
    "elemental-world-v2": PolicyCatalogSpec(
        "optional_payments.ir.json",
        ("elemental_bg", "elemental_gr"),
        "learning_schema.json",
    ),
    "elemental-world-v1": PolicyCatalogSpec(
        "elemental_world.ir.json",
        ("elemental_bg", "elemental_gr"),
        "learning_schema.json",
    ),
    "mtg122-complexity-v1": PolicyCatalogSpec(
        "complexity.ir.json",
        ("l1_swift", "l1_heavy"),
        "learning_schema.json",
    ),
}


def bind_policy_catalog(pack_key: str) -> BoundSemanticPack:
    """Admit checked programs against the actual native compiled manifest."""
    try:
        spec = _CATALOGS[pack_key]
    except KeyError as error:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}") from error
    if spec.setup_keys is None:
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
    else:
        engine = managym.Env(seed=0)
        engine.reset(
            [managym.authored_deck_setup(pack_key, key) for key in spec.setup_keys]
        )
        manifest = engine.content_pack_manifest()
    pack = BoundSemanticPack.bind(
        manifest,
        ir_path=ROOT / spec.ir_file,
        schema_path=ROOT.parent / spec.schema_file,
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
    """Select schema by registered pack, retaining the historical general default."""
    if pack_key is None:
        return ROOT.parent / "learning_schema.json"
    try:
        return ROOT.parent / _CATALOGS[pack_key].schema_file
    except KeyError as error:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}") from error
