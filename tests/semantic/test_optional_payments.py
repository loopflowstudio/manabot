"""Optional payment authoring checks both continuations through the shared compiler."""

import json
from pathlib import Path
from typing import Any

import pytest

from manabot.semantic.compiler import SemanticCompileError, compile_source, pretty_json
from manabot.semantic.learning import BoundSemanticPack
from manabot.semantic.policy_catalog import bind_policy_catalog

SOURCE = Path("content/semantic/w11/elemental_world.source.json")


def _lesson(source: dict[str, Any]) -> dict[str, Any]:
    # JSON is deliberately kept at the compiler boundary in these authoring tests.
    definition = next(
        row
        for row in source["definitions"]
        if row["registry_name"] == "Waterbending Lesson"
    )
    return definition["programs"][0]["ops"][1]


def test_both_payment_methods_compile_and_both_branches_reach_policy_input(
    tmp_path: Path,
) -> None:
    source = json.loads(SOURCE.read_text())
    payment = _lesson(source)
    payment["cost"] = {"mana": "{2}{U}"}
    payment["if_paid"] = [{"op": "draw_cards", "count": 1}]
    payment["if_declined"] = [{"op": "deal_damage_to_controller", "amount": 2}]
    ir, coverage = compile_source(source)
    assert coverage["opcode_counts"]["optional_payment"] == 1
    program = next(
        row
        for row in ir["programs"]
        if row["semantic_key"] == "temur.waterbending_lesson.p0"
    )
    optional = program["instructions"][1]
    assert optional["cost"] == {"mana": "{2}{U}"}
    assert optional["if_paid"][0]["op_name"] == "draw_cards"
    assert optional["if_declined"][0]["op_name"] == "deal_damage_to_controller"
    pack = bind_policy_catalog("elemental-world-v2")
    assert pack.schema.opcodes["optional_payment"] == 65
    assert pack.schema.raw["schema_version"] == 8
    # Project the authored variant too: admitting the checked pack alone does
    # not prove that the encoder retains a nonempty paid continuation.
    ir_path = tmp_path / "optional-payment.ir.json"
    ir_path.write_text(pretty_json(ir))
    projected = BoundSemanticPack.bind(
        {
            "schema_version": 1,
            "content_digest": ir["ir_hash"],
            "definitions": [
                {
                    "card_def_id": index,
                    "registry_name": definition["content_pack_binding"]["value"],
                }
                for index, definition in enumerate(ir["definitions"])
            ],
        },
        schema_path=SOURCE.parent / "learning_schema.json",
        ir_path=ir_path,
    )
    row = ir["programs"].index(program)
    start, end = projected.catalog.program_offsets[row : row + 2]
    tokens = list(
        zip(
            projected.catalog.token_kind[start:end],
            projected.catalog.token_value[start:end],
        )
    )
    for branch, opcode in (
        ("if_paid", "draw_cards"),
        ("if_declined", "deal_damage_to_controller"),
    ):
        begin = tokens.index(
            (
                pack.schema.token_kinds["structure"],
                pack.schema.structures[f"{branch}_begin"],
            )
        )
        end = tokens.index(
            (
                pack.schema.token_kinds["structure"],
                pack.schema.structures[f"{branch}_end"],
            )
        )
        assert (
            pack.schema.token_kinds["opcode"],
            pack.schema.opcodes[opcode],
        ) in tokens[begin:end]


@pytest.mark.parametrize("branch", ["if_paid", "if_declined"])
def test_payment_branches_reject_unbound_targets(branch: str) -> None:
    source = json.loads(SOURCE.read_text())
    _lesson(source)[branch] = [{"op": "deal_damage", "target": "unknown", "amount": 1}]
    with pytest.raises(SemanticCompileError, match="unknown target role"):
        compile_source(source)


@pytest.mark.parametrize(
    "cost",
    [
        {},
        {"waterbend": -1},
        {"waterbend": 256},
        {"waterbend": True},
        {"mana": "{2}", "waterbend": 2},
    ],
)
def test_payment_cost_rejects_ambiguous_or_invalid_obligations(
    cost: dict[str, object],
) -> None:
    source = json.loads(SOURCE.read_text())
    _lesson(source)["cost"] = cost
    with pytest.raises(SemanticCompileError):
        compile_source(source)
