from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manabot.semantic.compiler import canonical_json
from manabot.semantic.learning import (
    DEFAULT_IR_PATH,
    DEFAULT_SCHEMA_PATH,
    ArtifactCompatibilityError,
    BoundSemanticPack,
    ContentPackBindingError,
    LearningSchema,
    SemanticArtifactHeader,
    SemanticProjectionError,
    UnknownOpcodeError,
    UnknownSchemaError,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(*, reverse_ids: bool = False) -> dict:
    ir = _json(DEFAULT_IR_PATH)
    rows = list(range(len(ir["definitions"])))
    card_ids = list(reversed(rows)) if reverse_ids else rows
    return {
        "schema_version": 1,
        "content_digest": "content-pack-test-digest",
        "definitions": [
            {
                "card_def_id": card_ids[row],
                "registry_name": definition["content_pack_binding"]["value"],
            }
            for row, definition in enumerate(ir["definitions"])
        ],
    }


def _write_rehashed_ir(path: Path, ir: dict) -> None:
    unhashed = deepcopy(ir)
    unhashed.pop("ir_hash", None)
    ir["ir_hash"] = (
        __import__("hashlib")
        .sha256(canonical_json(unhashed).encode("utf-8"))
        .hexdigest()
    )
    path.write_text(json.dumps(ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_observation(card_ids: list[int], *, hidden_marker: str = "a"):
    return SimpleNamespace(
        agent_cards=[SimpleNamespace(registry_key=card_ids[0])],
        opponent_cards=[SimpleNamespace(registry_key=card_ids[1])],
        stack_objects=[
            SimpleNamespace(kind=1, source_card_registry_key=card_ids[0]),
            SimpleNamespace(kind=0, source_card_registry_key=card_ids[1]),
        ],
        # The projector must never inspect this private-state stand-in.
        opponent_hidden_hand=hidden_marker,
    )


def _span(catalog, offsets: np.ndarray, row: int) -> tuple[bytes, bytes]:
    start, end = (int(offsets[row]), int(offsets[row + 1]))
    return (
        catalog.token_kind[start:end].tobytes(),
        catalog.token_value[start:end].tobytes(),
    )


def _first_instruction_span(
    pack: BoundSemanticPack, program_row: int
) -> tuple[bytes, bytes]:
    catalog = pack.catalog
    start = int(catalog.program_offsets[program_row])
    end = int(catalog.program_offsets[program_row + 1])
    kinds = catalog.token_kind[start:end]
    values = catalog.token_value[start:end]
    structure_kind = pack.schema.token_kinds["structure"]
    begin_value = pack.schema.structures["instruction_begin"]
    end_value = pack.schema.structures["instruction_end"]
    begin = next(
        index
        for index, (kind, value) in enumerate(zip(kinds, values))
        if int(kind) == structure_kind and int(value) == begin_value
    )
    finish = next(
        index
        for index, (kind, value) in enumerate(zip(kinds[begin:], values[begin:]), begin)
        if int(kind) == structure_kind and int(value) == end_value
    )
    return kinds[begin : finish + 1].tobytes(), values[begin : finish + 1].tobytes()


def test_schema_and_catalog_are_deterministic_and_cover_compiler_opcodes():
    first = BoundSemanticPack.bind(_manifest())
    second = BoundSemanticPack.bind(deepcopy(_manifest()))

    assert first.schema.schema_hash == second.schema.schema_hash
    assert first.semantic_pack_hash == second.semantic_pack_hash
    used_opcodes = {
        instruction["op_name"]: instruction["opcode"]
        for program in first.ir.programs
        for instruction in _walk(program["instructions"])
    }
    assert used_opcodes == first.schema.opcodes
    for field in first.catalog.__dict__:
        left = getattr(first.catalog, field)
        right = getattr(second.catalog, field)
        assert left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()
        assert not left.flags.writeable

    assert len(first.catalog.definition_offsets) == len(first.ir.definitions) + 1
    assert len(first.catalog.program_offsets) == len(first.ir.programs) + 1
    assert first.catalog.definition_ref_source_tokens.size > 0
    assert first.catalog.definition_ref_source_tokens.size == (
        first.catalog.definition_ref_target_rows.size
    )


def _walk(instructions):
    for instruction in instructions:
        yield instruction
        for field in ("then", "otherwise", "body"):
            yield from _walk(instruction.get(field, []))


def test_same_instruction_recombines_identically_across_card_programs():
    pack = BoundSemanticPack.bind(_manifest())
    rows = {
        program["semantic_key"]: index for index, program in enumerate(pack.ir.programs)
    }

    invasion = _first_instruction_span(
        pack, rows["tla.invasion_reinforcements.create_ally"]
    )
    warriors = _first_instruction_span(pack, rows["tla.kyoshi_warriors.create_ally"])
    assert invasion == warriors

    kinds, values = _span(
        pack.catalog, pack.catalog.program_offsets, rows["tla.accumulate_wisdom.spell"]
    )
    structure_kind = np.frombuffer(kinds, dtype=np.uint16)
    structure_value = np.frombuffer(values, dtype=np.int32)
    controls = [
        int(value)
        for kind, value in zip(structure_kind, structure_value)
        if int(kind) == pack.schema.token_kinds["structure"]
        and int(value)
        in {
            pack.schema.structures["then_begin"],
            pack.schema.structures["then_end"],
            pack.schema.structures["otherwise_begin"],
            pack.schema.structures["otherwise_end"],
        }
    ]
    assert controls == [
        pack.schema.structures["otherwise_begin"],
        pack.schema.structures["otherwise_end"],
        pack.schema.structures["then_begin"],
        pack.schema.structures["then_end"],
    ]


def test_catalog_exposes_every_required_semantic_token_family():
    pack = BoundSemanticPack.bind(_manifest())
    kinds = pack.catalog.token_kind
    values = pack.catalog.token_value

    present_kinds = set(int(value) for value in kinds)
    assert present_kinds == set(pack.schema.token_kinds.values()) - {
        pack.schema.token_kinds["padding"]
    }

    structure_values = {
        int(value)
        for kind, value in zip(kinds, values)
        if int(kind) == pack.schema.token_kinds["structure"]
    }
    for boundary in (
        "definition_begin",
        "program_begin",
        "cost_begin",
        "target_begin",
        "selector_begin",
        "predicate_begin",
        "condition_begin",
        "then_begin",
        "otherwise_begin",
        "body_begin",
        "instruction_begin",
    ):
        assert pack.schema.structures[boundary] in structure_values

    field_values = {
        int(value)
        for kind, value in zip(kinds, values)
        if int(kind) == pack.schema.token_kinds["field"]
    }
    for field in (
        "cost",
        "definition_ref",
        "max",
        "min",
        "role",
        "selector",
        "target",
        "then",
        "otherwise",
        "body",
    ):
        assert pack.schema.fields[field] in field_values

    program_kinds = {
        int(value)
        for kind, value in zip(kinds, values)
        if int(kind) == pack.schema.token_kinds["program_kind"]
    }
    opcodes = {
        int(value)
        for kind, value in zip(kinds, values)
        if int(kind) == pack.schema.token_kinds["opcode"]
    }
    assert program_kinds == set(pack.schema.program_kinds.values())
    assert opcodes == set(pack.schema.opcodes.values())


def test_semantic_only_ablation_is_independent_of_card_def_id_allocation():
    regular = BoundSemanticPack.bind(_manifest())
    permuted = BoundSemanticPack.bind(_manifest(reverse_ids=True))
    assert regular.semantic_pack_hash == permuted.semantic_pack_hash
    assert regular.binding_hash != permuted.binding_hash
    assert regular.catalog.token_kind.tobytes() == permuted.catalog.token_kind.tobytes()
    assert (
        regular.catalog.token_value.tobytes() == permuted.catalog.token_value.tobytes()
    )

    regular_ids = [
        next(
            card_id
            for card_id, row in regular.definition_row_by_card_def_id.items()
            if row == wanted
        )
        for wanted in (0, 1)
    ]
    permuted_ids = [
        next(
            card_id
            for card_id, row in permuted.definition_row_by_card_def_id.items()
            if row == wanted
        )
        for wanted in (0, 1)
    ]
    regular_projection = regular.project_observation(
        _fake_observation(regular_ids), identity_mode="semantic_only"
    )
    permuted_projection = permuted.project_observation(
        _fake_observation(permuted_ids), identity_mode="semantic_only"
    )
    assert regular_projection.object_definition_rows.tobytes() == (
        permuted_projection.object_definition_rows.tobytes()
    )
    assert (
        regular_projection.object_roles.tobytes()
        == permuted_projection.object_roles.tobytes()
    )
    assert regular_projection.opaque_identity_ids.tolist() == [-1, -1, -1]
    assert not regular_projection.opaque_identity_valid.any()
    assert permuted_projection.opaque_identity_ids.tolist() == [-1, -1, -1]
    assert not permuted_projection.opaque_identity_valid.any()

    identity = regular.project_observation(
        _fake_observation(regular_ids), identity_mode="identity"
    )
    assert identity.opaque_identity_ids.tolist() == [
        regular_ids[0],
        regular_ids[1],
        regular_ids[0],
    ]
    assert identity.opaque_identity_valid.all()


def test_projection_uses_only_viewer_objects_and_batches_with_stable_masks():
    pack = BoundSemanticPack.bind(_manifest())
    card_ids = sorted(pack.definition_row_by_card_def_id)[:2]
    first = pack.project_observation(
        _fake_observation(card_ids, hidden_marker="first"),
        identity_mode="semantic_only",
    )
    second = pack.project_observation(
        _fake_observation(card_ids, hidden_marker="different-private-state"),
        identity_mode="semantic_only",
    )
    assert (
        first.object_definition_rows.tobytes()
        == second.object_definition_rows.tobytes()
    )
    assert first.object_roles.tobytes() == second.object_roles.tobytes()
    assert first.object_slots.tobytes() == second.object_slots.tobytes()

    empty = pack.project_observation(
        SimpleNamespace(agent_cards=[], opponent_cards=[], stack_objects=[]),
        identity_mode="semantic_only",
    )
    batch = pack.batch([first, empty, second])
    assert batch.sample_offsets.tolist() == [0, 3, 3, 6]
    padded = batch.pad()
    assert padded["object_mask"].tolist() == [
        [True, True, True],
        [False, False, False],
        [True, True, True],
    ]
    assert padded["object_definition_rows"][1].tolist() == [-1, -1, -1]
    assert not padded["opaque_identity_valid"].any()
    assert (padded["opaque_identity_ids"] == -1).all()

    padded_catalog = pack.pad_catalog()
    assert padded_catalog["definition_token_mask"].dtype == np.bool_
    assert padded_catalog["program_token_mask"].dtype == np.bool_
    assert padded_catalog["definition_ref_mask"].any()
    assert (padded_catalog["definition_ref_target_rows"] >= -1).all()


def test_artifact_headers_round_trip_in_dataset_and_checkpoint(tmp_path):
    pack = BoundSemanticPack.bind(_manifest())
    header = pack.artifact_header("semantic_only")
    assert header.binding_hash is None
    assert "registry_name" not in header.to_json()
    assert "card_def_id" not in header.to_json()

    dataset_path = tmp_path / "semantic.npz"
    np.savez_compressed(dataset_path, semantic_projection=np.array(header.to_json()))
    with np.load(dataset_path) as dataset:
        loaded_dataset_header = str(dataset["semantic_projection"])
    assert (
        pack.validate_artifact_header(
            loaded_dataset_header, identity_mode="semantic_only"
        )
        == header
    )

    checkpoint_path = tmp_path / "semantic.pt"
    torch.save({"semantic_projection": header.to_dict()}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert (
        pack.validate_artifact_header(
            checkpoint["semantic_projection"], identity_mode="semantic_only"
        )
        == header
    )

    mutated = header.to_dict()
    mutated["content_pack_hash"] = "different"
    with pytest.raises(ArtifactCompatibilityError, match="content_pack_hash"):
        pack.validate_artifact_header(mutated, identity_mode="semantic_only")
    with pytest.raises(ArtifactCompatibilityError, match="invalid semantic artifact"):
        SemanticArtifactHeader.from_json("not-json")


def test_unknown_schema_opcode_and_content_binding_fail_closed(tmp_path):
    schema = _json(DEFAULT_SCHEMA_PATH)
    schema["schema_version"] = 999
    bad_schema = tmp_path / "schema.json"
    bad_schema.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(UnknownSchemaError, match="expected 1, got 999"):
        LearningSchema.load(bad_schema)

    ir = _json(DEFAULT_IR_PATH)
    ir["programs"][0]["instructions"][0].update(op_name="future_op", opcode=999)
    bad_ir = tmp_path / "ir.json"
    _write_rehashed_ir(bad_ir, ir)
    with pytest.raises(UnknownOpcodeError, match="unknown opcode name"):
        BoundSemanticPack.bind(_manifest(), ir_path=bad_ir)

    ir = _json(DEFAULT_IR_PATH)
    ir["programs"][0]["instructions"][0]["opcode"] = 999
    mismatched_ir = tmp_path / "mismatched-ir.json"
    _write_rehashed_ir(mismatched_ir, ir)
    with pytest.raises(UnknownOpcodeError, match="disagrees"):
        BoundSemanticPack.bind(_manifest(), ir_path=mismatched_ir)

    missing = _manifest()
    missing["definitions"].pop()
    with pytest.raises(ContentPackBindingError, match="does not bind"):
        BoundSemanticPack.bind(missing)

    duplicate = _manifest()
    duplicate["definitions"][1]["card_def_id"] = duplicate["definitions"][0][
        "card_def_id"
    ]
    with pytest.raises(ContentPackBindingError, match="duplicate CardDefId"):
        BoundSemanticPack.bind(duplicate)

    pack = BoundSemanticPack.bind(_manifest())
    with pytest.raises(SemanticProjectionError, match="different identity modes"):
        pack.batch(
            [
                pack.project_observation(
                    _fake_observation([0, 1]), identity_mode="semantic_only"
                ),
                pack.project_observation(
                    _fake_observation([0, 1]), identity_mode="identity"
                ),
            ]
        )
