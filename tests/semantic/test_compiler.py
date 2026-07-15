from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from manabot.semantic.compiler import (
    Opcode,
    SemanticCompileError,
    compile_paths,
    compile_source,
    default_paths,
    pretty_json,
)

ROOT = Path(__file__).resolve().parents[2]


def _source() -> dict:
    source_path, _, _, _ = default_paths()
    return json.loads(source_path.read_text(encoding="utf-8"))


def _named_decks_from_server() -> dict[str, dict[str, int]]:
    tree = ast.parse((ROOT / "gui" / "server.py").read_text(encoding="utf-8"))
    wanted = {"UR_LESSONS_DECK", "GW_ALLIES_DECK"}
    found: dict[str, dict[str, int]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            found[target.id] = ast.literal_eval(node.value)
    assert found.keys() == wanted
    return found


def _deck_registry_counts(source: dict, deck_key: str) -> dict[str, int]:
    adapters = {
        definition["key"]: definition["registry_name"]
        for definition in source["definitions"]
    }
    deck = next(deck for deck in source["decks"] if deck["key"] == deck_key)
    return {adapters[card["definition"]]: card["count"] for card in deck["cards"]}


def _walk_instructions(instructions: list[dict]) -> list[dict]:
    walked: list[dict] = []
    for instruction in instructions:
        walked.append(instruction)
        for field in ("then", "otherwise", "body"):
            if isinstance(instruction.get(field), list):
                walked.extend(_walk_instructions(instruction[field]))
    return walked


def test_checked_in_ir_and_coverage_are_current_and_deterministic():
    source_path, fixtures_path, ir_path, coverage_path = default_paths()
    first_ir, first_coverage = compile_paths(source_path, fixtures_path)
    second_ir, second_coverage = compile_paths(source_path, fixtures_path)

    assert pretty_json(first_ir) == pretty_json(second_ir)
    assert pretty_json(first_coverage) == pretty_json(second_coverage)
    assert ir_path.read_text(encoding="utf-8") == pretty_json(first_ir)
    assert coverage_path.read_text(encoding="utf-8") == pretty_json(first_coverage)


def test_admission_manifest_matches_the_current_product_decks_exactly():
    source = _source()
    named_decks = _named_decks_from_server()

    assert _deck_registry_counts(source, "ur_lessons") == named_decks["UR_LESSONS_DECK"]
    assert _deck_registry_counts(source, "gw_allies") == named_decks["GW_ALLIES_DECK"]


def test_coverage_is_complete_for_decks_and_referenced_tokens():
    source_path, fixtures_path, _, _ = default_paths()
    _, coverage = compile_paths(source_path, fixtures_path)

    assert coverage["admission_closure_complete"] is True
    assert coverage["no_card_name_dispatch"] is True
    assert coverage["definition_count"] == 31
    assert coverage["deck_definition_count"] == 29
    assert coverage["referenced_definition_count"] == 2
    assert coverage["fixture_count"] == 5
    assert {deck["key"]: deck["card_count"] for deck in coverage["decks"]} == {
        "gw_allies": 40,
        "ur_lessons": 41,
    }


def test_content_pack_bindings_all_name_current_rust_definitions():
    source = _source()
    cardset_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "managym" / "src" / "cardsets").glob("*.rs"))
    )
    card_source = (ROOT / "managym" / "src" / "state" / "card.rs").read_text(
        encoding="utf-8"
    )
    registered_names = set(re.findall(r'name:\s*"([^"]+)"', cardset_source))
    registered_names.update(
        re.findall(r'basic_land\("([^"]+)"', cardset_source + card_source)
    )

    adapters = {definition["registry_name"] for definition in source["definitions"]}
    assert adapters <= registered_names


def test_instructions_dispatch_only_on_typed_opcodes_and_resolved_references():
    source_path, fixtures_path, _, _ = default_paths()
    ir, _ = compile_paths(source_path, fixtures_path)

    instructions = [
        instruction
        for program in ir["programs"]
        for instruction in _walk_instructions(program["instructions"])
    ]
    assert instructions
    assert all(isinstance(instruction["opcode"], int) for instruction in instructions)
    assert all("card_name" not in instruction for instruction in instructions)
    assert all("registry_name" not in instruction for instruction in instructions)
    assert all("definition_ref" not in instruction for instruction in instructions)
    assert any("definition_index" in instruction for instruction in instructions)

    compiler_source = (ROOT / "manabot" / "semantic" / "compiler.py").read_text(
        encoding="utf-8"
    )
    selected_names = {
        definition["registry_name"]
        for definition in _source()["definitions"]
        if not definition["characteristics"].get("token")
    }
    assert not any(json.dumps(name) in compiler_source for name in selected_names)


def test_opcode_numbers_are_an_explicit_compatibility_surface():
    assert {opcode.name: int(opcode) for opcode in Opcode} == {
        "ADD_MANA": 1,
        "DRAW_CARDS": 2,
        "PUT_TOP_CARDS_IN_HAND": 3,
        "CREATE_TOKEN": 4,
        "PUT_COUNTERS": 5,
        "GAIN_LIFE": 6,
        "SCRY": 7,
        "TAP": 8,
        "UNTAP": 9,
        "MODIFY_PT": 10,
        "GRANT_KEYWORDS": 11,
        "RESTRICT_BLOCKING": 12,
        "DEAL_DAMAGE": 13,
        "RETURN_TO_HAND": 14,
        "LEARN": 15,
        "COUNTER_UNLESS_PAYS": 16,
        "LOOK_AND_SELECT": 17,
        "BRANCH": 18,
        "FOR_EACH_TARGET": 19,
        "EARTHBEND": 20,
        "EXILE_UNTIL_SOURCE_LEAVES": 21,
        "DEAL_POWER_DAMAGE": 22,
        "SET_POWER_FROM_COUNT": 23,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source: source["definitions"][0]["programs"][0]["ops"][0].update(
                {"op": "invoke_card_by_name"}
            ),
            "unknown opcode",
        ),
        (
            lambda source: source["definitions"][0]["programs"][0]["ops"][0].update(
                {"card_name": "special case"}
            ),
            "unknown fields: card_name",
        ),
        (
            lambda source: source["definitions"][0]["programs"][0]["ops"].append(
                {"op": "create_token", "definition_ref": "missing.token", "count": 1}
            ),
            "unknown definition reference",
        ),
    ],
)
def test_compiler_rejects_untyped_or_name_dispatched_semantics(mutation, message):
    source = deepcopy(_source())
    mutation(source)

    with pytest.raises(SemanticCompileError, match=message):
        compile_source(source)
