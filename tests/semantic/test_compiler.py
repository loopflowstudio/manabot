from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from manabot.semantic.compiler import (
    Opcode,
    SemanticCompileError,
    compile_paths,
    compile_source,
    default_paths,
    pretty_json,
)
import managym

ROOT = Path(__file__).resolve().parents[2]


def _source() -> dict:
    source_path, _, _, _ = default_paths()
    return json.loads(source_path.read_text(encoding="utf-8"))


def _named_product_decks() -> dict[str, dict[str, int]]:
    manifest_path = (
        ROOT
        / "academy"
        / "ui"
        / "src"
        / "lib"
        / "packs"
        / "tla-ur-lessons-vs-gw-allies"
        / "v1"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "UR_LESSONS_DECK": manifest["matchup"]["hero"]["cards"],
        "GW_ALLIES_DECK": manifest["matchup"]["villain"]["cards"],
    }


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
    named_decks = _named_product_decks()

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


def test_content_pack_bindings_all_name_current_rust_definitions() -> None:
    source = _source()
    registered_names = {
        row["registry_name"]
        for row in managym.content_pool("tla-pool-v1")["definitions"]
        if row["card_def_id"] is not None
    }

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


def test_opcode_numbers_are_an_explicit_compatibility_surface() -> None:
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
        "SEARCH_BASIC_LAND": 24,
        "OPTIONAL_DISCARD_DRAW": 25,
        "DISCARD_TARGET_PLAYER": 26,
        "DESTROY": 27,
        "EXILE": 28,
        "BUFF_TARGET_VALUE": 29,
        "RETURN_TO_HAND_DRAW_IF_CONTROLLED": 30,
        "MODIFY_TRIGGERING_PERMANENT": 31,
        "DEAL_DAMAGE_VALUE": 32,
        "PUT_COUNTERS_VALUE": 33,
        "CREATE_TOKENS_VALUE": 34,
        "EXILE_WITH_CONTROLLER_TOKEN": 35,
        "DAMAGE_OPPONENT_CREATURES": 36,
        "MILL_CARDS": 37,
        "MILL_AND_SELECT": 38,
        "DRAIN_LIFE_VALUE": 39,
        "COUNTER_SPELL": 40,
        "CANT_BE_BLOCKED_TARGET": 41,
        "RETURN_GRAVEYARD_TARGET": 42,
        "MODAL": 43,
        "TARGET_CLAUSE": 44,
        "PREVENT_LIFE_GAIN": 45,
        "COMBAT_REQUIRES_OTHER": 46,
        "BLOCKING_RESTRICTION": 47,
        "POWER_FROM_LAND_COUNTERS": 48,
        "SPELL_COST_REDUCTION": 49,
        "STATIC_PT": 50,
        "SEARCH_BASIC_LANDS": 51,
        "SACRIFICE": 52,
        "DEAL_DAMAGE_TO_CONTROLLER": 53,
        "DRAIN_LIFE": 54,
        "UNBLOCKABLE_IF": 55,
        "ADD_LESSON_MANA": 56,
        "GRANT_FLASHBACK": 57,
        "GRANT_MANA": 58,
        "CREATE_NONLEGENDARY_COPY": 59,
        "MODIFY_CONTROLLED": 60,
        "WATERBEND_OR_DISCARD": 61,
        "DISCARD": 62,
        "EXILE_IF_DIES": 63,
        "ADD_SUBTYPE": 64,
        "OPTIONAL_PAYMENT": 65,
        "AIRBEND": 66,
        "AIRBEND_ALL": 67,
        "OPPONENTS_CAST_ONLY_FROM_HAND": 68,
        "EXILE_TOP_CARD_PLAY_THIS_TURN": 69,
        "SACRIFICE_SOURCE": 70,
        "PUT_INTO_LIBRARY_SECOND_OR_BOTTOM": 71,
        "SEARCH_LANDS_OPPONENT_SPLITS": 72,
        "REVEAL_HAND_DISCARD_CHOSEN": 73,
        "SHUFFLE_GRAVEYARD_INTO_LIBRARY": 74,
        "NO_MAXIMUM_HAND_SIZE": 75,
        "CAST_FROM_HAND_FREE": 76,
        "EXILE_UNTIL_NONLAND_CAST_FREE": 77,
        "EXILE_LINKED": 78,
        "GRANT_FIREBENDING_UNTIL_EOT": 79,
        "REMOVE_COUNTERS": 80,
        "SET_BASE_PT": 81,
        "ATTACH": 82,
        "CANT_ATTACK_OR_BLOCK": 83,
        "DOESNT_UNTAP": 84,
        "SHUFFLE_INTO_LIBRARY": 85,
        "BECOME_ARTIFACT_CREATURE": 86,
        "ADD_MANA_OF_ONE_COLOR": 87,
        "CANT_BE_BLOCKED_BY_MORE_THAN": 88,
        "ATTACKS_DESPITE_DEFENDER": 89,
        "ADD_CARD_TYPES": 90,
        "ADD_SUBTYPES_CONTROLLED": 91,
        "UNSPENT_MANA_BECOMES_RED": 92,
        "DESTROY_CREATURES_LOSING_ABILITIES": 93,
        "ENTERS_WITH_X_COUNTERS": 94,
        "OPPONENT_DAMAGE_PLUS_COUNTERS": 95,
        "SET_BASE_PT_VALUE": 96,
        "FIGHT": 97,
        "GAIN_LIFE_VALUE": 98,
        "ADD_MANA_VALUE": 99,
        "KEEP_UNSPENT_RED_MANA": 100,
        "SEARCH_LIBRARY_TO_HAND": 101,
        "LOOK_AT_LIBRARY_TOP": 102,
        "CAST_FROM_LIBRARY_TOP": 103,
        "SET_CHARACTERISTICS": 104,
        "REMOVE_ALL_COUNTERS": 105,
        "DISTRIBUTE_COUNTERS": 106,
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


def test_sideboard_is_compiled_as_owned_setup_without_changing_main_decks():
    source = _source()
    ir, _ = compile_source(source)
    assert ir["setup_schema_version"] == 1
    for source_deck in source["decks"]:
        compiled = next(
            deck for deck in ir["decks"] if deck["key"] == source_deck["key"]
        )
        assert compiled["card_count"] == source_deck["card_count"]
        assert (
            sum(card["count"] for card in compiled["cards"])
            == source_deck["card_count"]
        )
        assert sum(card["count"] for card in compiled["sideboard"]) == len(
            source_deck["sideboard"]
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_sideboard_rejects_invalid_counts(count):
    source = _source()
    source["decks"][0]["sideboard"][0]["count"] = count
    with pytest.raises(SemanticCompileError, match="sideboard"):
        compile_source(source)


def test_sideboard_rejects_unknown_token_and_duplicate_definitions():
    source = _source()
    token = next(
        definition["key"]
        for definition in source["definitions"]
        if definition["characteristics"].get("token")
    )
    for entries in [
        [{"definition": "missing.lesson", "count": 1}],
        [{"definition": token, "count": 1}],
        [{"definition": "tla.firebending_lesson", "count": 1}] * 2,
    ]:
        changed = deepcopy(source)
        changed["decks"][0]["sideboard"] = entries
        with pytest.raises(SemanticCompileError, match="sideboard"):
            compile_source(changed)


def test_sideboard_absent_empty_and_nonlesson_are_valid_setup():
    for entries in [None, [], [{"definition": "basic.island", "count": 1}]]:
        source = _source()
        if entries is None:
            source["decks"][0].pop("sideboard")
        else:
            source["decks"][0]["sideboard"] = entries
        ir, _ = compile_source(source)
        deck = next(
            deck for deck in ir["decks"] if deck["key"] == source["decks"][0]["key"]
        )
        assert len(deck["sideboard"]) == (1 if entries else 0)
