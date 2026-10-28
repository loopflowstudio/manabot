"""Current authoring has one owner; qualifications cannot survive clause edits."""

import json
from pathlib import Path
import shutil

import pytest

from manabot.semantic.compiler import SemanticCompileError, compile_source
from manabot.semantic.content import CONTENT_ROOT, ContentSource, generate
from manabot.semantic.learning import BoundSemanticPack


def test_current_outputs_and_all_learning_projections() -> None:
    generate(check=True)
    source = ContentSource.load()
    for preset in [*source.setups, "pool"]:
        ir, _ = source.project(preset)
        manifest = {
            "schema_version": 1,
            "content_digest": "projection-test",
            "definitions": [
                {"card_def_id": i, "registry_name": d["content_pack_binding"]["value"]}
                for i, d in enumerate(ir["definitions"])
            ],
        }
        bound = BoundSemanticPack.bind(
            manifest,
            ir_path=CONTENT_ROOT / "generated" / f"{preset}.ir.json",
            schema_path=CONTENT_ROOT / "learning_schema.json",
        )
        assert len(bound.definition_row_by_card_def_id) == len(ir["definitions"])


def test_missing_token_and_changed_qualified_clause_fail_closed(tmp_path: Path) -> None:
    source = ContentSource.load()
    ally = next(k for k, d in source.definitions.items() if d["registry_name"] == "Ally")
    del source.definitions[ally]
    with pytest.raises(SemanticCompileError, match="missing content definition"):
        source.project("two_deck")

    shutil.copytree(CONTENT_ROOT, tmp_path / "content")
    path = tmp_path / "content" / "definitions.json"
    rows = json.loads(path.read_text())
    enthusiast = next(d for d in rows if d["registry_name"] == "Avatar Enthusiasts")
    enthusiast["programs"] = []
    path.write_text(json.dumps(rows))
    with pytest.raises(SemanticCompileError, match="stale qualification"):
        generate(root=path.parent)


def test_new_definition_compiles_without_a_native_registration() -> None:
    ir, _ = compile_source({
        "schema_version": 1, "pack_key": "authoring-proof", "decks": [],
        "pool": ["proof.draw"],
        "definitions": [{
            "key": "proof.draw", "registry_name": "Unregistered authoring witness",
            "text_box": "Draw a card.",
            "characteristics": {"types": ["sorcery"], "mana_cost": "U"},
            "programs": [{"key": "draw", "kind": "spell", "cost": {"mana": "U"},
                          "ops": [{"op": "draw_cards", "count": 1}]}],
        }],
    })
    assert ir["definitions"][0]["text_box"] == "Draw a card."
    assert ir["programs"][0]["instructions"][0]["op_name"] == "draw_cards"
