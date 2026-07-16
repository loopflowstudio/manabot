"""Deterministic equal-token semantic-program katas and probe utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from manabot.semantic.compiler import canonical_json, compile_source, pretty_json
from manabot.semantic.learning import BoundSemanticPack, LearningSchema
from manabot.semantic.structural import (
    FAMILY_INDEX,
    KATA_FAMILIES,
    RELATION_INDEX,
    RELATION_NAMES,
    KataBatch,
    KataProbeModel,
    project_static_relations,
    trainable_parameter_bytes,
    trainable_parameter_count,
)
from manabot.semantic.transfer import SymbolicProgramBinder, build_spec

ROOT = Path(__file__).resolve().parents[2]
COMPILER_PATH = ROOT / "manabot/semantic/compiler.py"
SCHEMA_PATH = ROOT / "content/semantic/v1/learning_schema.json"
SOURCE_PATH = ROOT / "experiments/katas/structural-semantic-katas-v1.source.json"
ORACLE_PATH = ROOT / "experiments/katas/structural-semantic-katas-v1.oracles.json"
SUITE_PATH = ROOT / "experiments/katas/structural-semantic-katas-v1.json"
CONTRACT_PATH = ROOT / "experiments/workloads/structural-semantic-katas-v1.json"

SUITE_ID = "structural-semantic-katas-v1"
DATASET_SEED = 21_400
PAIRS_PER_FAMILY = 80
PAIR_SPLITS = {"train": 48, "validation": 16, "test": 16}
EXPECTED_COMPILER_SHA256 = (
    "f10f34d7b2e458a0ea01b124261a0c0fbbda4648e4452f277254aa6cc3250367"
)
EXPECTED_SCHEMA_SHA256 = (
    "a156c592414d0d4838c5423e2cb471fc49a4450f21d62e5e5c198ba948ae7034"
)

QUERY_NAMES = {
    "order": "probe_effect_a_executes_before_b",
    "hierarchy": "probe_effect_a_owned_by_outer_then_arm",
    "field_role": "power_delta_greater_than_toughness_delta",
    "argument_binding": "tap_binds_to_creature_you_control",
    "target_choice_role": "iterated_role_permits_multiple_targets",
}


class StructuralKataError(ValueError):
    """A checked kata artifact or experiment invariant is invalid."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _selector(kind: str, variant: int) -> dict[str, Any]:
    selector: dict[str, Any] = {"kind": kind}
    if variant % 2:
        selector["min_mana_value"] = 1 + (variant % 4)
    return selector


def _filler(index: int, *, excluded: frozenset[str] = frozenset()) -> dict[str, Any]:
    choices = [
        ("scry", "count"),
        ("put_top_cards_in_hand", "count"),
        ("gain_life", "amount"),
        ("draw_cards", "count"),
    ]
    op, field = next(
        choice
        for choice in (
            choices[index % len(choices) :] + choices[: index % len(choices)]
        )
        if choice[0] not in excluded
    )
    return {"op": op, field: 100 + index}


def _condition(index: int, offset: int) -> dict[str, Any]:
    if (index + offset) % 2:
        return {"kind": "nth_resolution", "n": 1 + ((index + offset) % 5)}
    return {"kind": "kicked"}


def _program_pair(family: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    base: dict[str, Any] = {"key": "main", "kind": "static"}
    filler = _filler(index, excluded=frozenset({"draw_cards", "gain_life"}))

    if family == "order":
        effect_a = {"op": "draw_cards", "count": 1 + (index % 3)}
        effect_b = {"op": "gain_life", "amount": 2 + (index % 5)}
        return (
            {**base, "ops": [effect_a, effect_b, filler]},
            {**base, "ops": [effect_b, effect_a, filler]},
        )

    if family == "hierarchy":
        effect_a = {"op": "draw_cards", "count": 1 + (index % 3)}
        effect_b = {"op": "gain_life", "amount": 2 + (index % 5)}
        inner_a = {
            "op": "branch",
            "condition": _condition(index, 1),
            "then": [effect_a],
            "otherwise": [],
        }
        inner_b = {
            "op": "branch",
            "condition": _condition(index, 1),
            "then": [effect_b],
            "otherwise": [],
        }
        return (
            {
                **base,
                "ops": [
                    {
                        "op": "branch",
                        "condition": _condition(index, 0),
                        "then": [inner_a],
                        "otherwise": [effect_b],
                    },
                    filler,
                ],
            },
            {
                **base,
                "ops": [
                    {
                        "op": "branch",
                        "condition": _condition(index, 0),
                        "then": [inner_b],
                        "otherwise": [effect_a],
                    },
                    filler,
                ],
            },
        )

    if family == "field_role":
        low = 1 + (index % 4)
        high = low + 1 + (index % 3)
        common = {"op": "modify_pt", "target": "source", "duration": "end_of_turn"}
        return (
            {**base, "ops": [{**common, "power": high, "toughness": low}, filler]},
            {**base, "ops": [{**common, "power": low, "toughness": high}, filler]},
        )

    if family == "argument_binding":
        targets = [
            {
                "role": "alpha",
                "selector": _selector("creature_you_control", index),
                "min": 1,
                "max": 1,
            },
            {
                "role": "beta",
                "selector": _selector("creature_opponent_controls", index),
                "min": 1,
                "max": 1,
            },
        ]
        return (
            {
                **base,
                "targets": targets,
                "ops": [
                    {"op": "tap", "target": "alpha"},
                    {"op": "untap", "target": "beta"},
                    filler,
                ],
            },
            {
                **base,
                "targets": targets,
                "ops": [
                    {"op": "tap", "target": "beta"},
                    {"op": "untap", "target": "alpha"},
                    filler,
                ],
            },
        )

    if family == "target_choice_role":
        selector_kind = ("creature", "creature_or_player")[index % 2]
        targets = [
            {
                "role": "alpha",
                "selector": _selector(selector_kind, index),
                "min": 1,
                "max": 2 + (index % 3),
            },
            {
                "role": "beta",
                "selector": _selector(selector_kind, index),
                "min": 1,
                "max": 1,
            },
        ]

        def iterate(role: str) -> dict[str, Any]:
            return {
                "op": "for_each_target",
                "role": role,
                "body": [{"op": "tap", "target": "current_target"}],
            }

        return (
            {
                **base,
                "targets": targets,
                "ops": [iterate("alpha"), {"op": "untap", "target": "beta"}, filler],
            },
            {
                **base,
                "targets": targets,
                "ops": [iterate("beta"), {"op": "untap", "target": "alpha"}, filler],
            },
        )

    raise StructuralKataError(f"unknown kata family {family!r}")


def _definition(program: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    digest = sha256_json(program)
    key = f"kata.{digest[:24]}"
    return key, {
        "key": key,
        "registry_name": f"Kata {digest[:24]}",
        "characteristics": {"types": ["sorcery"]},
        "programs": [deepcopy(dict(program))],
    }


def build_source() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    definitions: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for family in KATA_FAMILIES:
        for index in range(PAIRS_PER_FAMILY):
            programs = _program_pair(family, index)
            for program in programs:
                key, definition = _definition(program)
                if key in metadata:
                    raise StructuralKataError(f"duplicate generated program {key}")
                definitions.append(definition)
                metadata[key] = {
                    "family": family,
                    "nuisance": {
                        "condition_variant": index % 2 if family == "hierarchy" else -1,
                        "filler_op": program["ops"][-1]["op"],
                        "nonce": 100 + index,
                        "selector_variant": index % 8
                        if family in {"argument_binding", "target_choice_role"}
                        else -1,
                    },
                    "pair_index": index,
                }
    definitions.sort(key=lambda row: row["key"])
    cards = [
        {"definition": definition["key"], "count": 1} for definition in definitions
    ]
    return (
        {
            "schema_version": 1,
            "pack_key": SUITE_ID,
            "decks": [
                {
                    "key": "diagnostic_catalog",
                    "card_count": len(cards),
                    "cards": cards,
                }
            ],
            "definitions": definitions,
        },
        metadata,
    )


def _pack_from_ir(ir: Mapping[str, Any]) -> BoundSemanticPack:
    with tempfile.TemporaryDirectory(prefix="manabot-structural-katas-") as directory:
        path = Path(directory) / "suite.ir.json"
        path.write_text(pretty_json(ir), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "content_digest": sha256_json(ir),
            "definitions": [
                {
                    "card_def_id": index,
                    "registry_name": definition["content_pack_binding"]["value"],
                }
                for index, definition in enumerate(ir["definitions"])
            ],
        }
        return BoundSemanticPack.bind(manifest, ir_path=path)


def _normalized_program(program: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": program["kind_name"],
        "cost": deepcopy(program.get("cost")),
        "targets": deepcopy(program["targets"]),
        "trigger": deepcopy(program.get("trigger")),
        "instructions": deepcopy(program["instructions"]),
    }


def _walk_instructions(
    instructions: Sequence[Mapping[str, Any]],
    path: tuple[str, ...] = (),
) -> Iterable[tuple[Mapping[str, Any], tuple[str, ...]]]:
    for instruction in instructions:
        yield instruction, path
        for field in ("then", "otherwise", "body"):
            nested = instruction.get(field)
            if isinstance(nested, list):
                yield from _walk_instructions(nested, (*path, field))


def oracle_label(family: str, program: Mapping[str, Any]) -> int:
    """Answer one exact predicate directly over compiler-lowered typed IR."""

    instructions = list(_walk_instructions(program["instructions"]))
    if family == "order":
        positions = {
            row["op_name"]: index for index, (row, _) in enumerate(instructions)
        }
        return int(positions["draw_cards"] < positions["gain_life"])
    if family == "hierarchy":
        draw_path = next(
            path for row, path in instructions if row["op_name"] == "draw_cards"
        )
        return int(bool(draw_path) and draw_path[0] == "then")
    if family == "field_role":
        row = next(row for row, _ in instructions if row["op_name"] == "modify_pt")
        return int(row["power"] > row["toughness"])
    if family == "argument_binding":
        selectors = {
            target["role"]: target["selector"]["kind"] for target in program["targets"]
        }
        row = next(row for row, _ in instructions if row["op_name"] == "tap")
        return int(selectors[row["target"]] == "creature_you_control")
    if family == "target_choice_role":
        maxima = {target["role"]: target["max"] for target in program["targets"]}
        row = next(
            row for row, _ in instructions if row["op_name"] == "for_each_target"
        )
        return int(maxima[row["role"]] > 1)
    raise StructuralKataError(f"unknown oracle family {family!r}")


def _compile_oracle_program(program: Mapping[str, Any]) -> Mapping[str, Any]:
    key, definition = _definition(program)
    source = {
        "schema_version": 1,
        "pack_key": "structural-kata-oracle",
        "decks": [
            {
                "key": "oracle",
                "card_count": 1,
                "cards": [{"definition": key, "count": 1}],
            }
        ],
        "definitions": [definition],
    }
    ir, _ = compile_source(source)
    return ir["programs"][0]


def validate_oracle_fixtures(path: Path = ORACLE_PATH) -> dict[str, Any]:
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    if fixtures.get("schema_version") != 1:
        raise StructuralKataError("oracle fixture schema_version must be 1")
    rows = fixtures.get("families")
    if not isinstance(rows, list) or {row.get("family") for row in rows} != set(
        KATA_FAMILIES
    ):
        raise StructuralKataError("oracle fixtures must cover exactly five families")
    checked = []
    for row in rows:
        family = row["family"]
        for case in ("negative", "positive"):
            expected = int(case == "positive")
            program = _compile_oracle_program(row[case])
            actual = oracle_label(family, program)
            if actual != expected:
                raise StructuralKataError(
                    f"oracle fixture {family}.{case}: expected {expected}, got {actual}"
                )
            checked.append(f"{family}.{case}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_bytes(path.read_bytes()),
        "checked": checked,
    }


def _histogram(values: Sequence[Any]) -> str:
    return canonical_json(
        dict(sorted(Counter(values).items(), key=lambda item: str(item[0])))
    )


def _selector_histogram(program: Mapping[str, Any]) -> str:
    return _histogram(target["selector"]["kind"] for target in program["targets"])


def _split_for_pairs(pairs: Sequence[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_family[pair["family"]].append(pair)
    for family in KATA_FAMILIES:
        ranked = sorted(
            by_family[family],
            key=lambda pair: hashlib.sha256(
                f"{SUITE_ID}{pair['pair_id']}".encode("utf-8")
            ).digest(),
        )
        if len(ranked) != PAIRS_PER_FAMILY:
            raise StructuralKataError(f"{family}: expected {PAIRS_PER_FAMILY} pairs")
        cursor = 0
        for split, count in PAIR_SPLITS.items():
            for pair in ranked[cursor : cursor + count]:
                out[pair["pair_id"]] = split
            cursor += count
    return out


def _overlap_matrix(values: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        "train_validation": len(values["train"] & values["validation"]),
        "train_test": len(values["train"] & values["test"]),
        "validation_test": len(values["validation"] & values["test"]),
    }


def _audit(
    programs: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    split_values: dict[str, dict[str, set[str]]] = {
        name: {split: set() for split in PAIR_SPLITS}
        for name in (
            "normalized_program_hash",
            "nuisance_signature",
            "pair_template_signature",
            "skeleton_id",
        )
    }
    for program in programs:
        split = program["split"]
        split_values["normalized_program_hash"][split].add(program["program_hash"])
        split_values["nuisance_signature"][split].add(program["nuisance_signature"])
        split_values["pair_template_signature"][split].add(
            program["pair_template_signature"]
        )
        split_values["skeleton_id"][split].add(program["skeleton_id"])

    overlap = {name: _overlap_matrix(values) for name, values in split_values.items()}
    for name in (
        "normalized_program_hash",
        "nuisance_signature",
        "pair_template_signature",
    ):
        if any(overlap[name].values()):
            raise StructuralKataError(f"unexpected {name} split overlap")
    if any(value != len(KATA_FAMILIES) for value in overlap["skeleton_id"].values()):
        raise StructuralKataError(
            "intentional skeleton overlap must contain five families"
        )

    fields = (
        "condition_variant",
        "filler_op",
        "nonce",
        "selector_variant",
        "token_length",
        "token_multiset_hash",
        "opcode_histogram",
        "value_histogram",
        "selector_histogram",
        "skeleton_id",
    )
    contingencies: dict[str, dict[str, dict[str, int]]] = {}
    for field in fields:
        table: dict[str, Counter[int]] = defaultdict(Counter)
        for program in programs:
            value = program.get(field, program["nuisance"].get(field))
            table[canonical_json(value)][program["label"]] += 1
        rendered = {
            value: {"0": counts[0], "1": counts[1]}
            for value, counts in sorted(table.items())
        }
        if any(counts["0"] != counts["1"] for counts in rendered.values()):
            raise StructuralKataError(f"label-correlated generator field {field!r}")
        contingencies[field] = rendered

    split_counts = {
        split: {
            family: sum(
                program["split"] == split and program["family"] == family
                for program in programs
            )
            for family in KATA_FAMILIES
        }
        for split in PAIR_SPLITS
    }
    expected = {split: count * 2 for split, count in PAIR_SPLITS.items()}
    if any(
        split_counts[split][family] != expected[split]
        for split in PAIR_SPLITS
        for family in KATA_FAMILIES
    ):
        raise StructuralKataError("split family counts are not exactly balanced")

    pair_labels = {
        pair["pair_id"]: sorted(
            program["label"]
            for program in programs
            if program["pair_id"] == pair["pair_id"]
        )
        for pair in pairs
    }
    if any(labels != [0, 1] for labels in pair_labels.values()):
        raise StructuralKataError("every pair must contain opposite labels")

    return {
        "split_counts": split_counts,
        "overlap": overlap,
        "label_contingencies": contingencies,
        "pair_label_balance": True,
        "identity_fields_tensorized": [],
        "definition_references": 0,
    }


def build_suite(
    source: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if sha256_bytes(COMPILER_PATH.read_bytes()) != EXPECTED_COMPILER_SHA256:
        raise StructuralKataError("compiler bytes differ from the pinned contract")
    if sha256_bytes(SCHEMA_PATH.read_bytes()) != EXPECTED_SCHEMA_SHA256:
        raise StructuralKataError(
            "learning-schema bytes differ from the pinned contract"
        )
    oracle = validate_oracle_fixtures()
    ir, coverage = compile_source(source)
    pack = _pack_from_ir(ir)
    spec = build_spec(pack)
    binder = SymbolicProgramBinder(pack, spec)
    schema = LearningSchema.load()

    provisional: list[dict[str, Any]] = []
    pair_members: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row_index, program in enumerate(pack.ir.programs):
        definition = pack.ir.definitions[int(program["definition_index"])]
        definition_key = str(definition["semantic_key"])
        meta = metadata[definition_key]
        start = int(pack.catalog.program_offsets[row_index])
        end = int(pack.catalog.program_offsets[row_index + 1])
        token_ids = binder.encode_program(row_index)
        token_symbols = tuple(spec.token_vocabulary[token] for token in token_ids)
        if any(symbol.startswith("definition_ref:") for symbol in token_symbols):
            raise StructuralKataError("definition reference entered a kata input")
        projection = project_static_relations(token_symbols)
        normalized = _normalized_program(program)
        label = oracle_label(str(meta["family"]), program)
        program_hash = sha256_json(normalized)
        multiset_hash = sha256_json(sorted(Counter(token_symbols).items()))
        record = {
            "family": meta["family"],
            "label": label,
            "nuisance": deepcopy(meta["nuisance"]),
            "normalized_program": normalized,
            "program_hash": program_hash,
            "source_definition_key": definition_key,
            "source_program_key": program["semantic_key"],
            "skeleton_id": meta["family"],
            "token_length": len(token_ids),
            "token_multiset_hash": multiset_hash,
            "opcode_histogram": _histogram(
                row["op_name"] for row, _ in _walk_instructions(program["instructions"])
            ),
            "value_histogram": _histogram(
                symbol
                for symbol in token_symbols
                if symbol.startswith(("integer:", "boolean:", "enum:", "role:"))
            ),
            "selector_histogram": _selector_histogram(program),
            "model_input": {
                "token_ids": list(token_ids),
                "token_kinds": [
                    int(value) for value in pack.catalog.token_kind[start:end]
                ],
                "token_symbols": list(token_symbols),
                **projection.to_dict(),
            },
        }
        if record["token_length"] > 72:
            raise StructuralKataError(
                f"{definition_key}: {record['token_length']} tokens exceed budget"
            )
        provisional.append(record)
        pair_members[(str(meta["family"]), int(meta["pair_index"]))].append(record)

    pairs: list[dict[str, Any]] = []
    for (family, pair_index), members in sorted(pair_members.items()):
        if len(members) != 2:
            raise StructuralKataError(f"{family}.{pair_index}: expected two members")
        if sorted(member["label"] for member in members) != [0, 1]:
            raise StructuralKataError(f"{family}.{pair_index}: labels are not opposite")
        if len({member["token_multiset_hash"] for member in members}) != 1:
            raise StructuralKataError(f"{family}.{pair_index}: token multisets differ")
        if len({tuple(member["model_input"]["token_ids"]) for member in members}) != 2:
            raise StructuralKataError(f"{family}.{pair_index}: token sequences match")
        nuisance = members[0]["nuisance"]
        if any(member["nuisance"] != nuisance for member in members[1:]):
            raise StructuralKataError(f"{family}.{pair_index}: nuisance mismatch")
        nuisance_signature = sha256_json({"family": family, "nuisance": nuisance})
        template_signature = sha256_json(
            {"family": family, "skeleton": family, "nuisance": nuisance}
        )
        pair_id = sha256_json(
            {
                "family": family,
                "oracle_query": QUERY_NAMES[family],
                "nuisance": nuisance,
                "program_hashes": sorted(member["program_hash"] for member in members),
            }
        )
        pair = {
            "family": family,
            "nuisance_signature": nuisance_signature,
            "pair_id": pair_id,
            "pair_index": pair_index,
            "pair_template_signature": template_signature,
            "program_hashes": sorted(member["program_hash"] for member in members),
            "query": QUERY_NAMES[family],
        }
        pairs.append(pair)
        for member in members:
            member["pair_id"] = pair_id
            member["nuisance_signature"] = nuisance_signature
            member["pair_template_signature"] = template_signature

    split_by_pair = _split_for_pairs(pairs)
    for pair in pairs:
        pair["split"] = split_by_pair[pair["pair_id"]]
    for program in provisional:
        program["split"] = split_by_pair[program["pair_id"]]

    if len({program["program_hash"] for program in provisional}) != len(provisional):
        raise StructuralKataError("normalized program hashes are not unique")
    if len({pair["nuisance_signature"] for pair in pairs}) != len(pairs):
        raise StructuralKataError("nuisance signatures are not unique")
    if len({pair["pair_template_signature"] for pair in pairs}) != len(pairs):
        raise StructuralKataError("pair template signatures are not unique")

    programs = sorted(provisional, key=lambda row: row["program_hash"])
    pairs = sorted(pairs, key=lambda row: row["pair_id"])
    audit = _audit(programs, pairs)
    source_bytes = pretty_json(source).encode("utf-8")
    suite: dict[str, Any] = {
        "schema_version": 1,
        "id": SUITE_ID,
        "authority": {
            "compiler_path": str(COMPILER_PATH.relative_to(ROOT)),
            "compiler_sha256": EXPECTED_COMPILER_SHA256,
            "learning_schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
            "learning_schema_sha256": EXPECTED_SCHEMA_SHA256,
            "compiler_ir_hash": ir["ir_hash"],
            "compiler_source_hash": coverage["source_hash"],
            "oracle": oracle,
            "source_path": str(SOURCE_PATH.relative_to(ROOT)),
            "source_sha256": sha256_bytes(source_bytes),
        },
        "claim_boundary": (
            "Static discriminability of five known structural relations only; "
            "not recombination, dynamic binding, definition-reference semantics, "
            "or gameplay evidence."
        ),
        "families": [
            {"family": family, "query": QUERY_NAMES[family]} for family in KATA_FAMILIES
        ],
        "splits": PAIR_SPLITS,
        "token_vocabulary": list(spec.token_vocabulary),
        "token_kind_count": max(schema.token_kinds.values()) + 1,
        "pairs": pairs,
        "programs": programs,
        "audit": audit,
    }
    suite["suite_hash"] = sha256_json(suite)
    validate_suite(suite)
    return suite


def validate_suite(suite: Mapping[str, Any]) -> None:
    if suite.get("schema_version") != 1 or suite.get("id") != SUITE_ID:
        raise StructuralKataError("unknown structural kata suite")
    unhashed = dict(suite)
    declared = unhashed.pop("suite_hash", None)
    if declared != sha256_json(unhashed):
        raise StructuralKataError("suite_hash is invalid")
    programs = suite.get("programs")
    pairs = suite.get("pairs")
    if not isinstance(programs, list) or len(programs) != 800:
        raise StructuralKataError("suite must contain exactly 800 programs")
    if not isinstance(pairs, list) or len(pairs) != 400:
        raise StructuralKataError("suite must contain exactly 400 pairs")
    if any(
        symbol.startswith("definition_ref:")
        for program in programs
        for symbol in program["model_input"]["token_symbols"]
    ):
        raise StructuralKataError("definition-reference token found")
    if any(program["token_length"] > 72 for program in programs):
        raise StructuralKataError("program exceeds token budget")
    _audit(programs, pairs)


def build_contract(
    source_bytes: bytes, suite_bytes: bytes, oracle_bytes: bytes
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": SUITE_ID,
        "status": "pre_registered_results_pending",
        "claim_boundary": (
            "This experiment may nominate one static encoder candidate for W2-213. "
            "It does not establish semantic recombination, dynamic binding, card "
            "transfer, executable rules parity, gameplay strength, or integration readiness."
        ),
        "authority": {
            "compiler_path": str(COMPILER_PATH.relative_to(ROOT)),
            "compiler_sha256": EXPECTED_COMPILER_SHA256,
            "learning_schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
            "learning_schema_sha256": EXPECTED_SCHEMA_SHA256,
            "oracle_path": str(ORACLE_PATH.relative_to(ROOT)),
            "oracle_sha256": sha256_bytes(oracle_bytes),
            "source_path": str(SOURCE_PATH.relative_to(ROOT)),
            "source_sha256": sha256_bytes(source_bytes),
            "suite_path": str(SUITE_PATH.relative_to(ROOT)),
            "suite_sha256": sha256_bytes(suite_bytes),
        },
        "dataset": {
            "dataset_seed": DATASET_SEED,
            "families": list(KATA_FAMILIES),
            "pairs_per_family": PAIRS_PER_FAMILY,
            "splits": PAIR_SPLITS,
            "maximum_program_tokens": 72,
            "definition_references": "forbidden",
            "opaque_identity_features": "forbidden",
        },
        "model_seeds": [21_401, 21_402, 21_403, 21_404, 21_405],
        "models": {
            "bag_v1": {
                "hidden_dim": 32,
                "pooling": "masked_mean_max",
                "probe_dim": 24,
            },
            "relational_semantic_encoder_v1": {
                "d_model": 24,
                "heads": 2,
                "d_ff": 28,
                "blocks": 1,
                "dropout": 0.0,
                "position": "sinusoidal_preorder_and_depth",
                "relations": list(RELATION_NAMES),
                "context_projection": "24x24_tanh",
                "probe_dim": 24,
            },
            "probe": {
                "family_heads": 5,
                "candidates_per_head": 2,
                "head_type": "dot_product",
                "family_is_route_only": True,
            },
        },
        "training": {
            "python_minor": "3.12",
            "device": "cpu",
            "torch_threads": 1,
            "deterministic_algorithms": True,
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "batch_size": 64,
            "maximum_steps_per_run": 800,
            "validation_interval": 20,
            "checkpoint_selection": "lowest_validation_nll_earliest_step_tiebreak",
            "family_loss_weights": {family: 0.2 for family in KATA_FAMILIES},
        },
        "budget": {
            "arms": 2,
            "seeds": 5,
            "maximum_optimizer_steps": 8_000,
            "maximum_presented_examples": 512_000,
            "maximum_wall_clock_seconds": 1_800,
            "cpu_cores": 1,
        },
        "performance": {
            "batch1_warmups": 200,
            "batch1_samples": 2_000,
            "throughput_batch_size": 128,
            "throughput_warmups": 20,
            "throughput_samples": 100,
        },
        "predictions": {
            "bag_accuracy_every_family_seed": 0.5,
            "bag_brier": 0.25,
            "bag_nll": math.log(2),
            "structural_aggregate_accuracy_minimum": 0.95,
            "structural_per_family_accuracy_minimum": 0.90,
            "per_family_uplift_minimum": 0.40,
            "structural_brier_maximum": 0.10,
            "structural_nll_maximum": 0.35,
        },
        "gates": {
            "parameter_difference_fraction_maximum": 0.05,
            "structural_batch1_p95_ratio_maximum": 2.5,
            "structural_batch128_throughput_ratio_minimum": 0.40,
            "structural_train_accuracy_each_seed_minimum": 0.99,
            "structural_validation_accuracy_each_seed_minimum": 0.95,
            "structural_aggregate_test_accuracy_minimum": 0.95,
            "structural_per_family_test_accuracy_minimum": 0.90,
            "per_family_uplift_minimum": 0.40,
            "aggregate_uplift_t95_lower_minimum_exclusive": 0.35,
            "structural_brier_maximum": 0.10,
            "structural_nll_maximum": 0.35,
        },
        "branches": [
            "nominate_for_w2_213",
            "instrument_invalid",
            "teacher_or_label_error",
            "optimization_or_capacity_unresolved",
            "missing_structural_relation",
            "encoder_redesign",
            "cost_redesign",
        ],
    }


def artifact_bytes() -> tuple[bytes, bytes, bytes]:
    source, metadata = build_source()
    source_bytes = pretty_json(source).encode("utf-8")
    suite = build_suite(source, metadata)
    # The suite carries dense static edge metadata for 800 programs.  Keep its
    # checked representation canonical and compact so the diagnostic remains a
    # small repository artifact rather than pretty-printing millions of array
    # delimiters.
    suite_bytes = (canonical_json(suite) + "\n").encode("utf-8")
    oracle_bytes = ORACLE_PATH.read_bytes()
    contract = build_contract(source_bytes, suite_bytes, oracle_bytes)
    return source_bytes, suite_bytes, pretty_json(contract).encode("utf-8")


def load_contract(path: Path = CONTRACT_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if contract.get("schema_version") != 1 or contract.get("id") != SUITE_ID:
        raise StructuralKataError("unknown structural kata contract")
    if len(contract.get("model_seeds", [])) < 3:
        raise StructuralKataError("contract requires at least three model seeds")
    return contract, sha256_bytes(raw)


def load_suite(path: Path = SUITE_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    suite = json.loads(raw)
    validate_suite(suite)
    return suite, sha256_bytes(raw)


def records_to_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    include_relations: bool = True,
) -> KataBatch:
    if not records:
        raise StructuralKataError("cannot tensorize an empty kata split")
    width = max(int(record["token_length"]) for record in records)
    batch_size = len(records)
    token_ids = torch.zeros((batch_size, width), dtype=torch.long)
    token_kinds = torch.zeros((batch_size, width), dtype=torch.long)
    token_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    depth = torch.zeros((batch_size, width), dtype=torch.long)
    relations = torch.zeros(
        (batch_size, len(RELATION_NAMES), width, width), dtype=torch.bool
    )
    families = torch.empty(batch_size, dtype=torch.long)
    candidate_orders = torch.empty((batch_size, 2), dtype=torch.long)
    labels = torch.empty(batch_size, dtype=torch.long)
    for index, record in enumerate(records):
        model_input = record["model_input"]
        size = int(record["token_length"])
        token_ids[index, :size] = torch.tensor(model_input["token_ids"])
        token_kinds[index, :size] = torch.tensor(model_input["token_kinds"])
        token_mask[index, :size] = True
        depth[index, :size] = torch.tensor(model_input["depth"])
        if include_relations:
            for name, edges in model_input["relations"].items():
                relation = RELATION_INDEX[name]
                for source, target in edges:
                    relations[index, relation, source, target] = True
        families[index] = FAMILY_INDEX[record["family"]]
        flip = (
            int(
                hashlib.sha256(
                    f"{record['pair_id']}:{seed}".encode("utf-8")
                ).hexdigest(),
                16,
            )
            % 2
        )
        candidate_orders[index] = torch.tensor((0, 1) if flip == 0 else (1, 0))
        labels[index] = int(record["label"])
    return KataBatch(
        token_ids=token_ids,
        token_kinds=token_kinds,
        token_mask=token_mask,
        depth=depth,
        relations=relations,
        families=families,
        candidate_orders=candidate_orders,
        labels=labels,
    )


class KataTensorCatalog:
    """Content-addressed static tensors with seed-specific ordering at selection."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        suite_sha256: str,
    ) -> None:
        if not records:
            raise StructuralKataError("cannot build an empty tensor catalog")
        if len({str(record["program_hash"]) for record in records}) != len(records):
            raise StructuralKataError("tensor catalog program hashes must be unique")
        self.suite_sha256 = suite_sha256
        self._records = tuple(records)
        self._row_by_hash = {
            str(record["program_hash"]): index
            for index, record in enumerate(self._records)
        }
        self._lengths = tuple(int(record["token_length"]) for record in self._records)
        self._pair_ids = tuple(str(record["pair_id"]) for record in self._records)
        self._full = records_to_batch(self._records, seed=0, include_relations=True)
        self._empty_relations = torch.zeros_like(self._full.relations)
        self._candidate_orders_by_seed: dict[int, torch.Tensor] = {}

    def _candidate_orders(self, seed: int) -> torch.Tensor:
        orders = self._candidate_orders_by_seed.get(seed)
        if orders is None:
            rows = []
            for pair_id in self._pair_ids:
                flip = (
                    int(
                        hashlib.sha256(f"{pair_id}:{seed}".encode("utf-8")).hexdigest(),
                        16,
                    )
                    % 2
                )
                rows.append((0, 1) if flip == 0 else (1, 0))
            orders = torch.tensor(rows, dtype=torch.long)
            self._candidate_orders_by_seed[seed] = orders
        return orders

    def batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        include_relations: bool = True,
    ) -> KataBatch:
        if not records:
            raise StructuralKataError("cannot select an empty tensor-catalog batch")
        try:
            row_values = [
                self._row_by_hash[str(record["program_hash"])] for record in records
            ]
        except KeyError as error:
            raise StructuralKataError("record is absent from tensor catalog") from error
        indexes = torch.tensor(row_values, dtype=torch.long)
        width = max(self._lengths[index] for index in row_values)
        relation_source = (
            self._full.relations if include_relations else self._empty_relations
        )
        return KataBatch(
            token_ids=self._full.token_ids[indexes, :width],
            token_kinds=self._full.token_kinds[indexes, :width],
            token_mask=self._full.token_mask[indexes, :width],
            depth=self._full.depth[indexes, :width],
            relations=relation_source[indexes, :, :width, :width],
            families=self._full.families[indexes],
            candidate_orders=self._candidate_orders(seed)[indexes],
            labels=self._full.labels[indexes],
        )


def canonical_probabilities(model: nn.Module, batch: KataBatch) -> torch.Tensor:
    logits = model(batch)
    ordered = torch.softmax(logits, dim=-1)
    canonical = torch.zeros_like(ordered)
    canonical.scatter_(1, batch.candidate_orders, ordered)
    return canonical


def metric_summary(model: nn.Module, batch: KataBatch) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        probabilities = canonical_probabilities(model, batch)
    predicted = probabilities.argmax(dim=1)
    correct = predicted == batch.labels
    label_probability = probabilities.gather(1, batch.labels.unsqueeze(1)).squeeze(1)
    p1 = probabilities[:, 1]
    nll = -torch.log(label_probability.clamp_min(1e-12))
    brier = (p1 - batch.labels.to(torch.float32)).square()
    confidence = probabilities.max(dim=1).values
    ece = 0.0
    bins = torch.linspace(0.0, 1.0, 6)
    for bin_index in range(5):
        lower = bins[bin_index]
        upper = bins[bin_index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if bin_index == 4 else confidence < upper
        )
        if bool(mask.any()):
            ece += float(mask.to(torch.float32).mean()) * abs(
                float(correct[mask].to(torch.float32).mean())
                - float(confidence[mask].mean())
            )
    return {
        "accuracy": float(correct.to(torch.float32).mean()),
        "brier": float(brier.mean()),
        "ece_5_bin": ece,
        "nll": float(nll.mean()),
        "total": len(batch.labels),
    }


def evaluate_model(model: nn.Module, batch: KataBatch) -> dict[str, Any]:
    result = metric_summary(model, batch)
    result["by_family"] = {
        family: metric_summary(
            model,
            batch.select(torch.where(batch.families == family_index)[0]),
        )
        for family, family_index in FAMILY_INDEX.items()
    }
    return result


def _family_balanced_indexes(batch: KataBatch, step: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed * 100_000 + step)
    extra_families = {(step * 4 + offset) % 5 for offset in range(4)}
    indexes: list[int] = []
    for family_index in range(5):
        candidates = torch.where(batch.families == family_index)[0].tolist()
        count = 13 if family_index in extra_families else 12
        indexes.extend(
            int(value) for value in rng.choice(candidates, size=count, replace=False)
        )
    rng.shuffle(indexes)
    return torch.tensor(indexes, dtype=torch.long)


def _loss(model: nn.Module, batch: KataBatch) -> torch.Tensor:
    logits = model(batch)
    targets = (
        (batch.candidate_orders == batch.labels.unsqueeze(1)).to(torch.long).argmax(1)
    )
    losses = []
    for family_index in range(5):
        mask = batch.families == family_index
        losses.append(nn.functional.cross_entropy(logits[mask], targets[mask]))
    return torch.stack(losses).mean()


def train_model(
    model: KataProbeModel,
    train: KataBatch,
    validation: KataBatch,
    *,
    seed: int,
    steps: int,
    validation_interval: int,
    learning_rate: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )
    best_nll = math.inf
    best_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    maximum_training_accuracy = 0.0
    first_99_train_step: int | None = None
    trajectory: list[dict[str, float | int]] = []
    started = perf_counter_ns()
    for step in range(1, steps + 1):
        model.train()
        indexes = _family_balanced_indexes(train, step, seed)
        minibatch = train.select(indexes)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, minibatch)
        loss.backward()
        optimizer.step()
        if step % validation_interval == 0:
            training_accuracy = metric_summary(model, train)["accuracy"]
            validation_nll = metric_summary(model, validation)["nll"]
            maximum_training_accuracy = max(
                maximum_training_accuracy, training_accuracy
            )
            if first_99_train_step is None and training_accuracy >= 0.99:
                first_99_train_step = step
            trajectory.append(
                {
                    "step": step,
                    "training_accuracy": training_accuracy,
                    "validation_nll": validation_nll,
                }
            )
            if validation_nll < best_nll:
                best_nll = validation_nll
                best_step = step
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
    if best_state is None:
        raise StructuralKataError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_validation_nll": best_nll,
        "optimizer_steps": steps,
        "presented_examples": steps * 64,
        "selected_checkpoint_step": best_step,
        "maximum_training_accuracy": maximum_training_accuracy,
        "first_99_train_step": first_99_train_step,
        "trajectory": trajectory,
        "wall_clock_seconds": (perf_counter_ns() - started) / 1_000_000_000,
    }


def paired_symmetry_audit(
    model: nn.Module,
    batch: KataBatch,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        probabilities = canonical_probabilities(model, batch)
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_pair[record["pair_id"]].append(index)
    maximum = 0.0
    disagreements = 0
    for indexes in by_pair.values():
        if len(indexes) != 2:
            raise StructuralKataError("symmetry audit requires complete pairs")
        left, right = indexes
        maximum = max(
            maximum, float((probabilities[left] - probabilities[right]).abs().max())
        )
        disagreements += int(
            probabilities[left].argmax() != probabilities[right].argmax()
        )
    return {
        "maximum_paired_probability_difference": maximum,
        "paired_prediction_disagreements": disagreements,
        "pairs": len(by_pair),
    }


def _percentiles(values: Sequence[int]) -> dict[str, int]:
    return {
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "samples": len(values),
    }


def performance_metrics(
    model: nn.Module,
    single: KataBatch,
    throughput: KataBatch,
    *,
    warmups: int,
    samples: int,
    throughput_warmups: int,
    throughput_samples: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmups):
            model(single)
        timings = []
        for _ in range(samples):
            started = perf_counter_ns()
            model(single)
            timings.append(perf_counter_ns() - started)
        for _ in range(throughput_warmups):
            model(throughput)
        started = perf_counter_ns()
        for _ in range(throughput_samples):
            model(throughput)
        elapsed = (perf_counter_ns() - started) / 1_000_000_000
    examples = len(throughput.labels) * throughput_samples
    tokens = int(throughput.token_mask.sum()) * throughput_samples
    return {
        "model_batch1_latency_ns": _percentiles(timings),
        "batch128_examples_per_second": examples / elapsed,
        "batch128_tokens_per_second": tokens / elapsed,
        "parameter_count": trainable_parameter_count(model),
        "parameter_bytes": trainable_parameter_bytes(model),
    }


def end_to_end_performance(
    model: nn.Module,
    single_record: Mapping[str, Any],
    throughput_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    include_relations: bool,
    warmups: int,
    samples: int,
    throughput_warmups: int,
    throughput_samples: int,
) -> dict[str, Any]:
    """Measure canonical-record materialization plus the timed model call."""

    model.eval()

    def single_call() -> None:
        batch = records_to_batch(
            [single_record], seed=seed, include_relations=include_relations
        )
        with torch.no_grad():
            model(batch)

    def throughput_call() -> None:
        batch = records_to_batch(
            throughput_records, seed=seed, include_relations=include_relations
        )
        with torch.no_grad():
            model(batch)

    for _ in range(warmups):
        single_call()
    timings = []
    for _ in range(samples):
        started = perf_counter_ns()
        single_call()
        timings.append(perf_counter_ns() - started)
    for _ in range(throughput_warmups):
        throughput_call()
    started = perf_counter_ns()
    for _ in range(throughput_samples):
        throughput_call()
    elapsed = (perf_counter_ns() - started) / 1_000_000_000
    examples = len(throughput_records) * throughput_samples
    tokens = (
        sum(int(record["token_length"]) for record in throughput_records)
        * throughput_samples
    )
    return {
        "catalog_projector_plus_model_batch1_latency_ns": _percentiles(timings),
        "catalog_projector_plus_model_batch128_examples_per_second": (
            examples / elapsed
        ),
        "catalog_projector_plus_model_batch128_tokens_per_second": tokens / elapsed,
    }


def t_interval(values: Sequence[float]) -> dict[str, float]:
    if len(values) != 5:
        raise StructuralKataError("the primary t interval requires five seeds")
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    margin = 2.7764451051977987 * stdev / math.sqrt(len(values))
    return {
        "mean": mean,
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
        "t95_low": mean - margin,
        "t95_high": mean + margin,
    }
