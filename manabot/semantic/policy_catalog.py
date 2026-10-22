"""Explicit native/semantic bindings admitted for ordinary policy encoders.

A policy catalog is not inferred from card names or tensor dimensions. Each
entry names immutable compiled IR and two authored setups used to obtain its
native transport IDs. The native compilation receipt must match the IR before
any IDs reach the encoder. Research pools must register their own exact entry;
this module does not relax subset, checkpoint or setup admission.
"""

from dataclasses import dataclass
from pathlib import Path

from manabot.semantic.learning import BoundSemanticPack, ContentPackBindingError
import managym

ROOT = Path(__file__).resolve().parents[2] / "content/semantic/v1/generated"


@dataclass(frozen=True)
class PolicyCatalogSpec:
    ir_file: str
    setup_keys: tuple[str, str]
    schema_file: str = "learning_schema.json"


_CATALOGS = {
    "ur-lessons-vs-gw-allies": PolicyCatalogSpec(
        "two_deck.ir.json", ("ur_lessons", "gw_allies")
    ),
    "elemental-combat-v1": PolicyCatalogSpec(
        "elemental.ir.json",
        ("elemental_left", "elemental_right"),
        "../w7/learning_schema.json",
    ),
    "elemental-sacrifice-v1": PolicyCatalogSpec(
        "sacrifice.ir.json",
        ("sacrifice_left", "sacrifice_right"),
        "../w8/learning_schema.json",
    ),
    "elemental-temur-v1": PolicyCatalogSpec(
        "temur.ir.json",
        ("temur_left", "temur_right"),
        "../w9/learning_schema.json",
    ),
<<<<<<< HEAD
=======
    "elemental-world-v2": PolicyCatalogSpec(
        "../../w11/generated/elemental_world.ir.json",
        ("elemental_bg", "elemental_gr"),
        "../w11/learning_schema.json",
    ),
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    "elemental-world-v1": PolicyCatalogSpec(
        "elemental_world.ir.json",
        ("elemental_bg", "elemental_gr"),
        "../w10/learning_schema.json",
    ),
    "mtg122-complexity-v1": PolicyCatalogSpec(
        "complexity.ir.json",
        ("l1_swift", "l1_heavy"),
        "complexity.learning_schema.json",
    ),
}


def bind_policy_catalog(pack_key: str) -> BoundSemanticPack:
    """Admit checked programs against the actual native compiled manifest."""
    try:
        spec = _CATALOGS[pack_key]
    except KeyError as error:
        raise ValueError(f"unsupported policy semantic pack: {pack_key}") from error
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
