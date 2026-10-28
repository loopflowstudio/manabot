"""Compile reviewed curated-card semantics into deterministic typed IR.

This is deliberately an offline boundary.  The source document is authored and
reviewed as data; compilation rejects unknown shapes, resolves definition
references, assigns stable numeric opcodes, and emits canonical JSON.  Runtime
integration crosses the shared ``ContentPack``/``CardDefId`` boundary through
a load-time binding adapter rather than baking pack-local IDs into this file.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn

SOURCE_SCHEMA_VERSION = 1
IR_SCHEMA_VERSION = 1


class SemanticCompileError(ValueError):
    """A checked semantic source document is invalid."""


class ProgramKind(IntEnum):
    MANA = 1
    SPELL = 2
    TRIGGERED = 3
    ACTIVATED = 4
    STATIC = 5
    TRIGGERED_MANA = 6


class Opcode(IntEnum):
    ADD_MANA = 1
    DRAW_CARDS = 2
    PUT_TOP_CARDS_IN_HAND = 3
    CREATE_TOKEN = 4
    PUT_COUNTERS = 5
    GAIN_LIFE = 6
    SCRY = 7
    TAP = 8
    UNTAP = 9
    MODIFY_PT = 10
    GRANT_KEYWORDS = 11
    RESTRICT_BLOCKING = 12
    DEAL_DAMAGE = 13
    RETURN_TO_HAND = 14
    LEARN = 15
    COUNTER_UNLESS_PAYS = 16
    LOOK_AND_SELECT = 17
    BRANCH = 18
    FOR_EACH_TARGET = 19
    EARTHBEND = 20
    EXILE_UNTIL_SOURCE_LEAVES = 21
    DEAL_POWER_DAMAGE = 22
    SET_POWER_FROM_COUNT = 23
    SEARCH_BASIC_LAND = 24
    OPTIONAL_DISCARD_DRAW = 25
    DISCARD_TARGET_PLAYER = 26
    DESTROY = 27
    EXILE = 28
    BUFF_TARGET_VALUE = 29
    RETURN_TO_HAND_DRAW_IF_CONTROLLED = 30
    MODIFY_TRIGGERING_PERMANENT = 31
    DEAL_DAMAGE_VALUE = 32
    PUT_COUNTERS_VALUE = 33
    CREATE_TOKENS_VALUE = 34
    EXILE_WITH_CONTROLLER_TOKEN = 35
    DAMAGE_OPPONENT_CREATURES = 36
    MILL_CARDS = 37
    MILL_AND_SELECT = 38
    DRAIN_LIFE_VALUE = 39
    COUNTER_SPELL = 40
    CANT_BE_BLOCKED_TARGET = 41
    RETURN_GRAVEYARD_TARGET = 42
    MODAL = 43
    TARGET_CLAUSE = 44
    PREVENT_LIFE_GAIN = 45
    COMBAT_REQUIRES_OTHER = 46
    BLOCKING_RESTRICTION = 47
    POWER_FROM_LAND_COUNTERS = 48
    SPELL_COST_REDUCTION = 49
    STATIC_PT = 50
    SEARCH_BASIC_LANDS = 51
    SACRIFICE = 52
    DEAL_DAMAGE_TO_CONTROLLER = 53
    DRAIN_LIFE = 54
    UNBLOCKABLE_IF = 55
    ADD_LESSON_MANA = 56
    GRANT_FLASHBACK = 57
    GRANT_MANA = 58
    CREATE_NONLEGENDARY_COPY = 59
    MODIFY_CONTROLLED = 60
    WATERBEND_OR_DISCARD = 61
    DISCARD = 62
    EXILE_IF_DIES = 63
    ADD_SUBTYPE = 64
    OPTIONAL_PAYMENT = 65


PROGRAM_KINDS = {kind.name.lower(): kind for kind in ProgramKind}
OPCODES = {opcode.name.lower(): opcode for opcode in Opcode}

CARD_TYPES = {
    "artifact",
    "battle",
    "creature",
    "enchantment",
    "instant",
    "kindred",
    "land",
    "planeswalker",
    "sorcery",
}
KEYWORDS = {
    "deathtouch",
    "defender",
    "double_strike",
    "first_strike",
    "flash",
    "flying",
    "haste",
    "hexproof",
    "indestructible",
    "lifelink",
    "menace",
    "reach",
    "trample",
    "vigilance",
}
TRIGGER_EVENTS = {
    "cast_enters",
    "sacrificed",
    "attacks",
    "becomes_tapped",
    "becomes_targeted",
    "dies",
    "draw_nth_card",
    "enters_battlefield",
    "tapped_for_mana",
    "upkeep_begins",
    "combat_begins",
    "cast_spell",
}
SUBJECT_KINDS = {"another_you_control", "any_you_control", "this"}
SELECTOR_KINDS = {
    "creature",
    "creature_opponent_controls",
    "creature_or_player",
    "creature_you_control",
    "land_you_control",
    "permanent_opponent_controls",
    "spell",
    "spell_or_permanent",
    "permanent",
    "permanent_you_control",
    "graveyard_card_you_own",
    "player",
    "opponent",
}
PREDICATE_KINDS = {
    "all",
    "any",
    "card_type",
    "not_card_types",
    "power_at_most",
    "power_at_least",
    "mana_value_at_least",
    "flying",
    "types_any",
    "subtypes_any",
    "not_subtypes",
    "subtype",
}
CONDITION_KINDS = {
    "graveyard_at_least",
    "kicked",
    "always",
    "cards_drawn_at_least",
    "nth_resolution",
    "target_matches",
}

OP_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "discard": (set(), set()),
    "exile_if_dies": ({"target"}, set()),
    "add_subtype": ({"target", "subtype"}, set()),
    "add_lesson_mana": ({"mana"}, set()),
    "grant_flashback": (set(), set()),
    "grant_mana": ({"predicates"}, set()),
    "create_nonlegendary_copy": (set(), set()),
    "modify_controlled": ({"predicate", "power", "toughness"}, set()),
    "waterbend_or_discard": ({"generic"}, set()),
    "optional_payment": ({"cost", "if_paid", "if_declined"}, set()),
    "search_basic_lands": (
        {"count", "tapped", "untap_if_lands", "bottom_source"},
        set(),
    ),
    "sacrifice": ({"predicate"}, set()),
    "deal_damage_to_controller": ({"amount"}, set()),
    "drain_life": ({"amount"}, set()),
    "unblockable_if": ({"condition"}, set()),
    "add_mana": ({"mana"}, {"until"}),
    "draw_cards": ({"count"}, set()),
    "put_top_cards_in_hand": ({"count"}, set()),
    "create_token": ({"definition_ref", "count"}, {"tapped_and_attacking"}),
    "put_counters": ({"target", "count"}, set()),
    "gain_life": ({"amount"}, set()),
    "scry": ({"count"}, set()),
    "tap": ({"target"}, set()),
    "untap": ({"target"}, set()),
    "modify_pt": ({"target", "power", "toughness", "duration"}, set()),
    "grant_keywords": ({"target", "keywords", "duration"}, set()),
    "restrict_blocking": ({"target", "duration", "predicate"}, set()),
    "deal_damage": ({"amount", "target"}, set()),
    "return_to_hand": ({"target"}, set()),
    "learn": (set(), set()),
    "counter_unless_pays": ({"target", "cost"}, set()),
    "look_and_select": (
        {"look", "min_select", "max_select", "predicate", "destination"},
        {"remainder", "remainder_order", "reveal"},
    ),
    "branch": ({"condition", "then", "otherwise"}, set()),
    "for_each_target": ({"role", "body"}, set()),
    "earthbend": ({"target", "count"}, set()),
    "exile_until_source_leaves": ({"target"}, set()),
    "deal_power_damage": ({"sources", "target"}, set()),
    "set_power_from_count": ({"predicate", "zone", "controller"}, set()),
    "search_basic_land": ({"subtype"}, set()),
    "optional_discard_draw": ({"count"}, set()),
    "discard_target_player": ({"count", "target"}, set()),
    "destroy": ({"target"}, set()),
    "exile": ({"target"}, set()),
    "buff_target_value": ({"power", "toughness", "target"}, set()),
    "return_to_hand_draw_if_controlled": ({"target"}, set()),
    "modify_triggering_permanent": ({"counters", "keywords"}, set()),
    "deal_damage_value": ({"amount", "target"}, set()),
    "put_counters_value": ({"count", "target"}, set()),
    "create_tokens_value": ({"definition_ref", "count"}, set()),
    "exile_with_controller_token": ({"target", "definition_ref"}, set()),
    "damage_opponent_creatures": ({"amount"}, set()),
    "mill_cards": ({"count"}, set()),
    "mill_and_select": ({"count", "predicate", "if_declined"}, set()),
    "drain_life_value": ({"amount"}, set()),
    "counter_spell": ({"target"}, set()),
    "cant_be_blocked_target": ({"target"}, set()),
    "return_graveyard_target": ({"target"}, set()),
    "modal": ({"modes"}, set()),
    "target_clause": ({"role", "body"}, set()),
    "prevent_life_gain": (set(), set()),
    "combat_requires_other": ({"predicate"}, set()),
    "blocking_restriction": ({"predicate", "direction"}, set()),
    "power_from_land_counters": (set(), set()),
    "spell_cost_reduction": (
        {"predicate", "condition", "generic", "first_on_own_turn"},
        set(),
    ),
    "static_pt": ({"power", "toughness", "condition"}, set()),
}


def _fail(context: str, message: str) -> NoReturn:
    raise SemanticCompileError(f"{context}: {message}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(context, "must be an object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(context, "must be an array")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(context, "must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(context, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(context, f"must be at least {minimum}")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(context, "must be a boolean")
    return value


def _keys(
    value: Mapping[str, Any],
    context: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        _fail(context, f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        _fail(context, f"unknown fields: {', '.join(sorted(unknown))}")


def _unique(values: Sequence[str], context: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        _fail(context, f"duplicate values: {', '.join(duplicates)}")


def canonical_json(value: Any) -> str:
    """Canonical compact encoding used for hashes and reproducibility."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _validate_predicate(value: Any, context: str) -> dict[str, Any]:
    predicate = _object(value, context)
    kind = _string(predicate.get("kind"), f"{context}.kind")
    if kind not in PREDICATE_KINDS:
        _fail(context, f"unknown predicate kind {kind!r}")
    if kind in {"any", "flying"}:
        _keys(predicate, context, required={"kind"})
    elif kind in {"card_type", "subtype"}:
        _keys(predicate, context, required={"kind", "value"})
        value_string = _string(predicate["value"], f"{context}.value")
        if kind == "card_type" and value_string not in CARD_TYPES:
            _fail(context, f"unknown card type {value_string!r}")
    elif kind in {"power_at_most", "power_at_least", "mana_value_at_least"}:
        _keys(predicate, context, required={"kind", "value"})
        _integer(predicate["value"], f"{context}.value")
    elif kind in {"not_card_types", "types_any", "subtypes_any", "not_subtypes"}:
        _keys(predicate, context, required={"kind", "values"})
        values = [
            _string(item, f"{context}.values")
            for item in _array(predicate["values"], context)
        ]
        _unique(values, f"{context}.values")
        unknown = (
            set(values) - CARD_TYPES
            if kind in {"not_card_types", "types_any"}
            else set()
        )
        if unknown:
            _fail(context, f"unknown card types: {', '.join(sorted(unknown))}")
    else:
        _keys(predicate, context, required={"kind", "predicates"})
        predicates = _array(predicate["predicates"], f"{context}.predicates")
        if not predicates:
            _fail(context, "all predicate must contain at least one predicate")
        for index, child in enumerate(predicates):
            _validate_predicate(child, f"{context}.predicates[{index}]")
    return deepcopy(predicate)


def _validate_condition(value: Any, context: str) -> dict[str, Any]:
    condition = _object(value, context)
    kind = _string(condition.get("kind"), f"{context}.kind")
    if kind not in CONDITION_KINDS:
        _fail(context, f"unknown condition kind {kind!r}")
    if kind in {"kicked", "always"}:
        _keys(condition, context, required={"kind"})
    elif kind == "cards_drawn_at_least":
        _keys(condition, context, required={"kind", "count"})
        _integer(condition["count"], context, minimum=0)
    elif kind == "graveyard_at_least":
        _keys(condition, context, required={"kind", "count", "predicate"})
        _integer(condition["count"], f"{context}.count", minimum=1)
        _validate_predicate(condition["predicate"], f"{context}.predicate")
    elif kind == "nth_resolution":
        _keys(condition, context, required={"kind", "n"})
        _integer(condition["n"], f"{context}.n", minimum=1)
    else:
        _keys(condition, context, required={"kind", "role", "predicate"})
        _string(condition["role"], f"{context}.role")
        _validate_predicate(condition["predicate"], f"{context}.predicate")
    return deepcopy(condition)


def _validate_selector(value: Any, context: str) -> dict[str, Any]:
    selector = _object(value, context)
    _keys(
        selector,
        context,
        required={"kind"},
        optional={"min_mana_value", "types_any", "predicate"},
    )
    kind = _string(selector["kind"], f"{context}.kind")
    if kind not in SELECTOR_KINDS:
        _fail(context, f"unknown selector kind {kind!r}")
    predicate_kinds = {"permanent", "permanent_you_control", "graveyard_card_you_own"}
    if kind in predicate_kinds and "predicate" not in selector:
        _fail(context, f"{kind} requires a predicate")
    if "predicate" in selector:
        if kind not in predicate_kinds:
            _fail(context, f"{kind} does not accept a predicate")
        _validate_predicate(selector["predicate"], f"{context}.predicate")
    if "min_mana_value" in selector:
        _integer(selector["min_mana_value"], f"{context}.min_mana_value", minimum=0)
    if "types_any" in selector:
        types = [
            _string(item, f"{context}.types_any")
            for item in _array(selector["types_any"], context)
        ]
        _unique(types, f"{context}.types_any")
        unknown = set(types) - CARD_TYPES
        if unknown:
            _fail(context, f"unknown card types: {', '.join(sorted(unknown))}")
    return deepcopy(selector)


def _validate_trigger(value: Any, context: str) -> dict[str, Any]:
    trigger = _object(value, context)
    _keys(
        trigger,
        context,
        required={"event", "subject"},
        optional={
            "n",
            "if",
            "predicate",
            "controller_only",
            "once_each_turn",
            "own_turn_only",
        },
    )
    event = _string(trigger["event"], f"{context}.event")
    if event not in TRIGGER_EVENTS:
        _fail(context, f"unknown trigger event {event!r}")
    subject = _object(trigger["subject"], f"{context}.subject")
    _keys(subject, f"{context}.subject", required={"kind"}, optional={"predicate"})
    subject_kind = _string(subject["kind"], f"{context}.subject.kind")
    if subject_kind not in SUBJECT_KINDS:
        _fail(context, f"unknown trigger subject {subject_kind!r}")
    if subject_kind == "this" and "predicate" in subject:
        _fail(context, "the 'this' subject cannot carry a predicate")
    if subject_kind != "this":
        if "predicate" not in subject:
            _fail(context, f"{subject_kind!r} requires a predicate")
        _validate_predicate(subject["predicate"], f"{context}.subject.predicate")
    for flag in ("controller_only", "once_each_turn", "own_turn_only"):
        if flag in trigger:
            _boolean(trigger[flag], f"{context}.{flag}")
    if event in {"cast_spell", "sacrificed"}:
        _validate_predicate(trigger.get("predicate"), f"{context}.predicate")
    elif "predicate" in trigger:
        _fail(context, "only cast_spell and sacrificed accept predicate")
    if event == "draw_nth_card":
        _integer(trigger.get("n"), f"{context}.n", minimum=1)
    elif "n" in trigger:
        _fail(context, "only draw_nth_card accepts n")
    if "if" in trigger:
        _validate_condition(trigger["if"], f"{context}.if")
    return deepcopy(trigger)


def _validate_target(value: Any, context: str) -> dict[str, Any]:
    target = _object(value, context)
    _keys(target, context, required={"role", "selector", "min", "max"})
    _string(target["role"], f"{context}.role")
    _validate_selector(target["selector"], f"{context}.selector")
    minimum = _integer(target["min"], f"{context}.min", minimum=0)
    maximum = _integer(target["max"], f"{context}.max", minimum=1)
    if minimum > maximum:
        _fail(context, "min cannot exceed max")
    return deepcopy(target)


def _validate_cost(value: Any, context: str) -> dict[str, Any]:
    cost = _object(value, context)
    _keys(
        cost,
        context,
        required={"mana"},
        optional={
            "kicker",
            "sacrifice_source",
            "waterbend",
            "affinity",
            "tap_source",
            "sacrifice",
            "own_turn_only",
            "life",
        },
    )
    _string(cost["mana"], f"{context}.mana")
    if "sacrifice" in cost:
        sacrifice = _object(cost["sacrifice"], context)
        _keys(sacrifice, context, required={"predicate", "another", "or_mana"})
        _validate_predicate(sacrifice["predicate"], context)
        _boolean(sacrifice["another"], context)
        if sacrifice["or_mana"] is not None:
            _string(sacrifice["or_mana"], context)
    if "life" in cost:
        _integer(cost["life"], context, minimum=0)
    if "own_turn_only" in cost:
        _boolean(cost["own_turn_only"], context)
    if "kicker" in cost:
        _string(cost["kicker"], f"{context}.kicker")
    for field in ("sacrifice_source", "tap_source", "waterbend"):
        if field in cost:
            _boolean(cost[field], f"{context}.{field}")
    if "affinity" in cost:
        _validate_predicate(cost["affinity"], f"{context}.affinity")
    return deepcopy(cost)


def _validate_value_expression(value: Any, context: str) -> None:
    expression = _object(value, context)
    kind = _string(expression.get("kind"), context)
    fields = {
        "creatures_with_counters": set(),
        "battlefield_matching": {"predicate"},
        "graveyard_matching": {"predicate", "base"},
        "cards_drawn_minus": {"subtract"},
    }
    if kind not in fields:
        _fail(context, f"unknown quantity {kind!r}")
    _keys(expression, context, required={"kind", *fields[kind]})
    if "predicate" in expression:
        _validate_predicate(expression["predicate"], context)
    for field in {"base", "subtract"} & expression.keys():
        _integer(expression[field], context, minimum=0 if field == "subtract" else None)


def _validate_instruction_scalars(
    op: str, instruction: Mapping[str, Any], context: str
) -> None:
    for field in {
        "amount",
        "count",
        "look",
        "max_select",
        "min_select",
        "power",
        "toughness",
    }:
        if field in instruction:
            minimum = (
                0 if field in {"count", "look", "max_select", "min_select"} else None
            )
            if (op, field) in {
                ("buff_target_value", "power"),
                ("deal_damage_value", "amount"),
                ("put_counters_value", "count"),
                ("create_tokens_value", "count"),
                ("drain_life_value", "amount"),
            }:
                _validate_value_expression(instruction[field], f"{context}.{field}")
            else:
                _integer(instruction[field], f"{context}.{field}", minimum=minimum)
    if "cost" in instruction and op != "optional_payment":
        _string(instruction["cost"], f"{context}.cost")
    elif op == "optional_payment":
        cost = _object(instruction["cost"], f"{context}.cost")
        if set(cost) == {"mana"}:
            _string(cost["mana"], f"{context}.cost.mana")
        elif set(cost) == {"waterbend"}:
            generic = _integer(
                cost["waterbend"], f"{context}.cost.waterbend", minimum=0
            )
            if generic > 255:
                _fail(context, "waterbend cost exceeds u8")
        else:
            _fail(context, "payment cost requires exactly one of mana or waterbend")
    for field in {
        "target",
        "sources",
        "role",
        "mana",
        "until",
        "duration",
        "zone",
        "controller",
        "destination",
        "remainder",
        "remainder_order",
    }:
        if field in instruction:
            _string(instruction[field], f"{context}.{field}")
    for field in {"generic", "counters"}:
        if field in instruction:
            _integer(instruction[field], f"{context}.{field}", minimum=0)
    for flag in ("tapped", "bottom_source"):
        if flag in instruction:
            _boolean(instruction[flag], context)
    if "untap_if_lands" in instruction and instruction["untap_if_lands"] is not None:
        _integer(instruction["untap_if_lands"], context, minimum=0)
    if "first_on_own_turn" in instruction:
        _boolean(instruction["first_on_own_turn"], context)
    if "condition" in instruction:
        _validate_condition(instruction["condition"], context)
    if "direction" in instruction and instruction["direction"] not in {
        "blocking",
        "blocked_by",
    }:
        _fail(context, "unknown blocking direction")
    if "subtype" in instruction and instruction["subtype"] is not None:
        _string(instruction["subtype"], context)
    if "tapped_and_attacking" in instruction:
        _boolean(instruction["tapped_and_attacking"], f"{context}.tapped_and_attacking")
    if "reveal" in instruction:
        _boolean(instruction["reveal"], context)
    if "keywords" in instruction:
        keywords = [
            _string(item, f"{context}.keywords")
            for item in _array(instruction["keywords"], context)
        ]
        _unique(keywords, f"{context}.keywords")
        unknown = set(keywords) - KEYWORDS
        if unknown:
            _fail(context, f"unknown keywords: {', '.join(sorted(unknown))}")
    if "predicates" in instruction:
        for predicate in _array(instruction["predicates"], context):
            _validate_predicate(predicate, context)
    if "predicate" in instruction:
        _validate_predicate(instruction["predicate"], f"{context}.predicate")


def _compile_instructions(
    values: Any,
    context: str,
    definition_indexes: Mapping[str, int],
) -> list[dict[str, Any]]:
    instructions = _array(values, context)
    compiled: list[dict[str, Any]] = []
    for index, raw in enumerate(instructions):
        item_context = f"{context}[{index}]"
        instruction = _object(raw, item_context)
        op = _string(instruction.get("op"), f"{item_context}.op")
        if op not in OPCODES:
            _fail(item_context, f"unknown opcode {op!r}")
        required, optional = OP_FIELDS[op]
        _keys(instruction, item_context, required={"op", *required}, optional=optional)
        _validate_instruction_scalars(op, instruction, item_context)

        lowered = {
            key: deepcopy(value) for key, value in instruction.items() if key != "op"
        }
        if "definition_ref" in lowered:
            reference = _string(
                lowered.pop("definition_ref"), f"{item_context}.definition_ref"
            )
            if reference not in definition_indexes:
                _fail(item_context, f"unknown definition reference {reference!r}")
            lowered["definition_index"] = definition_indexes[reference]
        # Every nested instruction list crosses the same compiler boundary.
        for field in ("then", "otherwise", "body", "if_paid", "if_declined"):
            if field in instruction:
                lowered[field] = _compile_instructions(
                    instruction[field], f"{item_context}.{field}", definition_indexes
                )
        if op == "modal":
            modes = _array(instruction["modes"], f"{item_context}.modes")
            if not modes:
                _fail(item_context, "modal requires modes")
            lowered["modes"] = [
                _compile_instructions(
                    mode, f"{item_context}.modes[{index}]", definition_indexes
                )
                for index, mode in enumerate(modes)
            ]
        lowered["opcode"] = int(OPCODES[op])
        lowered["op_name"] = op
        compiled.append(lowered)
    return compiled


def _validate_characteristics(value: Any, context: str) -> dict[str, Any]:
    characteristics = _object(value, context)
    _keys(
        characteristics,
        context,
        required={"types"},
        optional={
            "colors",
            "keywords",
            "mana_cost",
            "power",
            "subtypes",
            "supertypes",
            "token",
            "toughness",
            "enters_tapped",
        },
    )
    types = [
        _string(item, f"{context}.types")
        for item in _array(characteristics["types"], context)
    ]
    _unique(types, f"{context}.types")
    unknown_types = set(types) - CARD_TYPES
    if unknown_types:
        _fail(context, f"unknown card types: {', '.join(sorted(unknown_types))}")
    for field in {"colors", "subtypes", "supertypes"}:
        if field in characteristics:
            values = [
                _string(item, f"{context}.{field}")
                for item in _array(characteristics[field], context)
            ]
            _unique(values, f"{context}.{field}")
    if "keywords" in characteristics:
        keywords = [
            _string(item, f"{context}.keywords")
            for item in _array(characteristics["keywords"], context)
        ]
        _unique(keywords, f"{context}.keywords")
        unknown = set(keywords) - KEYWORDS
        if unknown:
            _fail(context, f"unknown keywords: {', '.join(sorted(unknown))}")
    if "mana_cost" in characteristics and characteristics["mana_cost"] is not None:
        _string(characteristics["mana_cost"], f"{context}.mana_cost")
    for field in {"power", "toughness"}:
        if field in characteristics and characteristics[field] is not None:
            _integer(characteristics[field], f"{context}.{field}")
    if "token" in characteristics:
        _boolean(characteristics["token"], f"{context}.token")
    if "enters_tapped" in characteristics:
        _boolean(characteristics["enters_tapped"], f"{context}.enters_tapped")
    return deepcopy(characteristics)


def _walk_instructions(
    instructions: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    walked: list[Mapping[str, Any]] = []
    for instruction in instructions:
        walked.append(instruction)
        for mode in instruction.get("modes", []):
            walked.extend(_walk_instructions(mode))
        for field in ("then", "otherwise", "body", "if_paid", "if_declined"):
            nested = instruction.get(field)
            if isinstance(nested, list):
                walked.extend(_walk_instructions(nested))
    return walked


def _validate_instruction_references(
    instructions: Sequence[Mapping[str, Any]],
    target_roles: set[str],
    context: str,
    *,
    current_target_available: bool = False,
) -> None:
    builtins = {"source", "controller", "triggering_spell"}
    if current_target_available:
        builtins.add("current_target")
    for index, instruction in enumerate(instructions):
        item_context = f"{context}[{index}]"
        for field in ("target", "sources"):
            reference = instruction.get(field)
            if reference is None or str(reference).startswith("each:"):
                continue
            if reference not in target_roles and reference not in builtins:
                _fail(
                    item_context,
                    f"{field} references unknown target role {reference!r}",
                )
        op_name = instruction["op_name"]
        for mode in instruction.get("modes", []):
            _validate_instruction_references(mode, target_roles, item_context)
        if op_name in {"for_each_target", "target_clause"}:
            role = instruction["role"]
            if role not in target_roles:
                _fail(item_context, f"role references unknown target role {role!r}")
            _validate_instruction_references(
                instruction["body"],
                target_roles,
                f"{item_context}.body",
                current_target_available=True,
            )
        for field in ("then", "otherwise", "if_paid", "if_declined"):
            nested = instruction.get(field)
            if isinstance(nested, list):
                _validate_instruction_references(
                    nested,
                    target_roles,
                    f"{item_context}.{field}",
                    current_target_available=current_target_available,
                )


def compile_source(source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and lower one semantic source document.

    Returns ``(ir, coverage)``.  Both values are deterministic for equivalent
    source data regardless of object-key order.
    """

    source = deepcopy(_object(source, "source"))
    _keys(
        source,
        "source",
        required={"schema_version", "pack_key", "decks", "definitions"},
        optional={"pool", "token_templates"},
    )
    if source["schema_version"] != SOURCE_SCHEMA_VERSION:
        _fail("source.schema_version", f"expected {SOURCE_SCHEMA_VERSION}")
    pack_key = _string(source["pack_key"], "source.pack_key")

    raw_definitions = _array(source["definitions"], "source.definitions")
    definition_keys = [
        _string(_object(item, "source.definitions[]").get("key"), "definition.key")
        for item in raw_definitions
    ]
    _unique(definition_keys, "source.definitions")
    definition_indexes = {
        key: index for index, key in enumerate(sorted(definition_keys))
    }

    programs: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    program_keys: list[str] = []
    source_by_key = {
        _string(item["key"], "definition.key"): _object(item, "definition")
        for item in raw_definitions
    }
    for definition_key in sorted(source_by_key):
        raw_definition = source_by_key[definition_key]
        context = f"definition[{definition_key}]"
        _keys(
            raw_definition,
            context,
            required={"key", "registry_name", "characteristics", "programs"},
            optional={"text_box"},
        )
        registry_name = _string(
            raw_definition["registry_name"], f"{context}.registry_name"
        )
        characteristics = _validate_characteristics(
            raw_definition["characteristics"], f"{context}.characteristics"
        )
        definition_program_indexes: list[int] = []
        raw_programs = _array(raw_definition["programs"], f"{context}.programs")
        local_keys = [
            _string(_object(program, f"{context}.programs[]").get("key"), "program.key")
            for program in raw_programs
        ]
        _unique(local_keys, f"{context}.programs")
        for raw_program in sorted(raw_programs, key=lambda item: item["key"]):
            local_key = _string(raw_program["key"], f"{context}.program.key")
            program_key = f"{definition_key}.{local_key}"
            program_context = f"program[{program_key}]"
            _keys(
                raw_program,
                program_context,
                required={"key", "kind", "ops"},
                optional={"cost", "targets", "trigger", "target_binding"},
            )
            if raw_program.get("target_binding", "explicit") not in {
                "explicit",
                "inferred",
            }:
                _fail(program_context, "unknown target binding")
            kind_name = _string(raw_program["kind"], f"{program_context}.kind")
            if kind_name not in PROGRAM_KINDS:
                _fail(program_context, f"unknown program kind {kind_name!r}")
            triggered_kinds = {"triggered", "triggered_mana"}
            if kind_name in triggered_kinds and "trigger" not in raw_program:
                _fail(program_context, f"{kind_name} programs require trigger")
            if kind_name not in triggered_kinds and "trigger" in raw_program:
                _fail(program_context, "only triggered programs accept trigger")
            if kind_name in {"activated", "spell"} and "cost" not in raw_program:
                _fail(program_context, f"{kind_name} programs require cost")
            if kind_name not in {"activated", "spell"} and "cost" in raw_program:
                _fail(program_context, f"{kind_name} programs cannot carry cost")

            targets = [
                _validate_target(target, f"{program_context}.targets[{index}]")
                for index, target in enumerate(raw_program.get("targets", []))
            ]
            target_roles = [target["role"] for target in targets]
            _unique(target_roles, f"{program_context}.targets")
            instructions = _compile_instructions(
                raw_program["ops"], f"{program_context}.ops", definition_indexes
            )
            _validate_instruction_references(
                instructions,
                set(target_roles),
                f"{program_context}.instructions",
            )
            program = {
                "definition_index": definition_indexes[definition_key],
                "instructions": instructions,
                "kind": int(PROGRAM_KINDS[kind_name]),
                "kind_name": kind_name,
                "program_index": len(programs),
                "semantic_key": program_key,
                "targets": targets,
            }
            if "target_binding" in raw_program:
                program["target_binding"] = raw_program["target_binding"]
            if "cost" in raw_program:
                program["cost"] = _validate_cost(
                    raw_program["cost"], f"{program_context}.cost"
                )
            if "trigger" in raw_program:
                program["trigger"] = _validate_trigger(
                    raw_program["trigger"], f"{program_context}.trigger"
                )
            definition_program_indexes.append(program["program_index"])
            program_keys.append(program_key)
            programs.append(program)
        definitions.append(
            {
                "characteristics": characteristics,
                "content_pack_binding": {
                    "kind": "legacy_registry_name",
                    "value": registry_name,
                },
                "program_indexes": definition_program_indexes,
                "semantic_index": definition_indexes[definition_key],
                "semantic_key": definition_key,
            }
        )
        if "text_box" in raw_definition:
            text_box = raw_definition["text_box"]
            if not isinstance(text_box, str):
                _fail(context, "text_box must be a string")
            definitions[-1]["text_box"] = text_box
    _unique(program_keys, "programs")

    registry_names = [
        definition["content_pack_binding"]["value"] for definition in definitions
    ]
    _unique(registry_names, "legacy registry adapters")

    decks: list[dict[str, Any]] = []
    deck_definition_keys: set[str] = set()
    for deck_index, raw_deck in enumerate(_array(source["decks"], "source.decks")):
        deck_context = f"source.decks[{deck_index}]"
        deck = _object(raw_deck, deck_context)
        _keys(
            deck,
            deck_context,
            required={"key", "card_count", "cards"},
            optional={"sideboard"},
        )
        deck_key = _string(deck["key"], f"{deck_context}.key")
        expected_card_count = _integer(
            deck["card_count"], f"{deck_context}.card_count", minimum=1
        )
        compiled_deck: dict[str, Any] = {
            "card_count": expected_card_count,
            "key": deck_key,
        }
        for field in ("cards", "sideboard"):
            cards: list[dict[str, Any]] = []
            for card_index, raw_card in enumerate(
                _array(deck.get(field, []), f"{deck_context}.{field}")
            ):
                card_context = f"{deck_context}.{field}[{card_index}]"
                card = _object(raw_card, card_context)
                _keys(card, card_context, required={"definition", "count"})
                definition_key = _string(
                    card["definition"], f"{card_context}.definition"
                )
                if definition_key not in definition_indexes:
                    _fail(card_context, f"unknown definition {definition_key!r}")
                index = definition_indexes[definition_key]
                if field == "sideboard" and definitions[index]["characteristics"].get(
                    "token", False
                ):
                    _fail(card_context, "token cannot be a sideboard card")
                count = _integer(card["count"], f"{card_context}.count", minimum=1)
                deck_definition_keys.add(definition_key)
                cards.append({"count": count, "definition_index": index})
            if field == "sideboard":
                _unique(
                    [str(card["definition_index"]) for card in cards],
                    f"{deck_context}.sideboard",
                )
            else:
                card_count = sum(card["count"] for card in cards)
                if card_count != expected_card_count:
                    _fail(
                        deck_context,
                        f"declares {expected_card_count} cards but entries sum to {card_count}",
                    )
            cards.sort(key=lambda card: card["definition_index"])
            compiled_deck[field] = cards
        decks.append(compiled_deck)
    _unique([deck["key"] for deck in decks], "source.decks")
    decks.sort(key=lambda deck: deck["key"])

    pool_keys = [
        _string(key, "source.pool[]")
        for key in _array(source.get("pool", []), "source.pool")
    ]
    _unique(pool_keys, "source.pool")
    for key in pool_keys:
        if key not in definition_indexes:
            _fail("source.pool", f"unknown definition {key!r}")
        if source_by_key[key]["characteristics"].get("token", False):
            _fail("source.pool", "token cannot be a deck pool entry")
    token_keys = [
        _string(key, "source.token_templates[]")
        for key in _array(source.get("token_templates", []), "source.token_templates")
    ]
    _unique(token_keys, "source.token_templates")
    for key in token_keys:
        if key not in definition_indexes or not source_by_key[key][
            "characteristics"
        ].get("token", False):
            _fail("source.token_templates", f"not a token definition: {key!r}")

    source_hash = _sha256(source)
    ir: dict[str, Any] = {
        "definitions": definitions,
        "decks": decks,
        "pack_key": pack_key,
        "programs": programs,
        "schema_version": IR_SCHEMA_VERSION,
        "source_hash": source_hash,
        "setup_schema_version": 1,
    }
    ir["ir_hash"] = _sha256(ir)

    referenced_indexes = {
        instruction["definition_index"]
        for program in programs
        for instruction in _walk_instructions(program["instructions"])
        if "definition_index" in instruction
    }
    closure_indexes = (
        {definition_indexes[key] for key in deck_definition_keys}
        | referenced_indexes
        | {definition_indexes[key] for key in [*pool_keys, *token_keys]}
    )
    opcode_counts = Counter(
        instruction["op_name"]
        for program in programs
        for instruction in _walk_instructions(program["instructions"])
    )
    coverage_definitions = []
    for definition in definitions:
        definition_programs = [
            programs[index] for index in definition["program_indexes"]
        ]
        op_names = sorted(
            {
                instruction["op_name"]
                for program in definition_programs
                for instruction in _walk_instructions(program["instructions"])
            }
        )
        coverage_definitions.append(
            {
                "in_admission_closure": definition["semantic_index"] in closure_indexes,
                "program_count": len(definition_programs),
                "program_kinds": sorted(
                    {program["kind_name"] for program in definition_programs}
                ),
                "semantic_key": definition["semantic_key"],
                "opcodes": op_names,
            }
        )
    coverage = {
        "admission_closure_complete": len(closure_indexes) == len(definitions),
        "definition_count": len(definitions),
        "definitions": coverage_definitions,
        "deck_definition_count": len(deck_definition_keys),
        "decks": [
            {"card_count": deck["card_count"], "key": deck["key"]} for deck in decks
        ],
        "ir_hash": ir["ir_hash"],
        "no_card_name_dispatch": all(
            "registry_name" not in instruction and "card_name" not in instruction
            for program in programs
            for instruction in _walk_instructions(program["instructions"])
        ),
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "pack_key": pack_key,
        "program_count": len(programs),
        "referenced_definition_count": len(referenced_indexes),
        "schema_version": IR_SCHEMA_VERSION,
        "source_hash": source_hash,
        "setup_schema_version": 1,
    }
    if not coverage["admission_closure_complete"]:
        absent = [
            definition["semantic_key"]
            for definition in definitions
            if definition["semantic_index"] not in closure_indexes
        ]
        _fail("coverage", f"definitions outside admission closure: {', '.join(absent)}")
    return ir, coverage


def _opcode_trace(instructions: Sequence[Mapping[str, Any]]) -> list[str]:
    trace: list[str] = []
    for instruction in instructions:
        trace.append(instruction["op_name"])
        for mode in instruction.get("modes", []):
            trace.extend(f"mode:{item}" for item in _opcode_trace(mode))
        for field in ("then", "otherwise", "body", "if_paid", "if_declined"):
            nested = instruction.get(field)
            if isinstance(nested, list):
                trace.extend(f"{field}:{item}" for item in _opcode_trace(nested))
    return trace


def validate_fixtures(ir: Mapping[str, Any], fixtures: Mapping[str, Any]) -> None:
    fixtures = _object(fixtures, "fixtures")
    _keys(fixtures, "fixtures", required={"schema_version", "cases"})
    if fixtures["schema_version"] != IR_SCHEMA_VERSION:
        _fail("fixtures.schema_version", f"expected {IR_SCHEMA_VERSION}")
    programs = {program["semantic_key"]: program for program in ir["programs"]}
    names: list[str] = []
    for index, raw_case in enumerate(_array(fixtures["cases"], "fixtures.cases")):
        context = f"fixtures.cases[{index}]"
        case = _object(raw_case, context)
        _keys(case, context, required={"name", "program", "opcode_trace"})
        names.append(_string(case["name"], f"{context}.name"))
        program_key = _string(case["program"], f"{context}.program")
        if program_key not in programs:
            _fail(context, f"unknown program {program_key!r}")
        expected = [
            _string(item, f"{context}.opcode_trace")
            for item in _array(case["opcode_trace"], f"{context}.opcode_trace")
        ]
        actual = _opcode_trace(programs[program_key]["instructions"])
        if actual != expected:
            _fail(
                context, f"opcode trace mismatch: expected {expected!r}, got {actual!r}"
            )
    _unique(names, "fixtures.cases")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as error:
        raise SemanticCompileError(f"{path}: invalid JSON: {error}") from error


def compile_paths(
    source_path: Path,
    fixtures_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ir, coverage = compile_source(load_json(source_path))
    fixtures = load_json(fixtures_path)
    validate_fixtures(ir, fixtures)
    coverage["fixture_count"] = len(fixtures["cases"])
    return ir, coverage


def write_or_check(path: Path, value: Mapping[str, Any], *, check: bool) -> None:
    expected = pretty_json(value)
    if check:
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            raise SemanticCompileError(
                f"{path} is stale; run `uv run scripts/compile_semantic_content.py`"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    base = root / "content" / "semantic" / "v1"
    return (
        base / "two_deck.source.json",
        base / "two_deck.fixtures.json",
        base / "generated" / "two_deck.ir.json",
        base / "generated" / "two_deck.coverage.json",
    )
