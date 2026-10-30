"""The learning projection binds the engine's own card tokens to live packs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manabot.semantic.learning import (
    DEFAULT_SCHEMA_PATH,
    PROGRAM_KINDS,
    TOKEN_KINDS,
    ArtifactCompatibilityError,
    BoundSemanticPack,
    ContentPackBindingError,
    LearningSchema,
    SemanticArtifactHeader,
    SemanticProjectionError,
    UnknownSchemaError,
    token_value,
)
import managym

PACK_KEY = "ur-lessons-vs-gw-allies"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(*, reverse_ids: bool = False, pack_key: str = PACK_KEY) -> dict:
    pool = managym.content_pool(pack_key)
    rows = sorted(pool["definitions"], key=lambda row: row["card_def_id"])
    ids = [row["card_def_id"] for row in rows]
    if reverse_ids:
        ids.reverse()
    return {
        "schema_version": 1,
        "content_digest": pool["content_digest"],
        "pack_key": pack_key,
        "definitions": [
            {"card_def_id": card_def_id, "registry_name": row["registry_name"]}
            for card_def_id, row in zip(ids, rows, strict=True)
        ],
    }


def _row(pack: BoundSemanticPack, name: str, pack_key: str = PACK_KEY) -> int:
    pool = managym.content_pool(pack_key)
    card_def_id = next(
        row["card_def_id"]
        for row in pool["definitions"]
        if row["registry_name"] == name
    )
    return pack.definition_row(card_def_id)


def _tokens(
    pack: BoundSemanticPack, offsets: np.ndarray, row: int
) -> list[tuple[int, int]]:
    start, end = int(offsets[row]), int(offsets[row + 1])
    return list(
        zip(
            pack.catalog.token_kind[start:end].tolist(),
            pack.catalog.token_value[start:end].tolist(),
            strict=True,
        )
    )


def _card_tokens(pack: BoundSemanticPack, row: int) -> list[tuple[int, int]]:
    tokens = _tokens(pack, pack.catalog.definition_offsets, row)
    for program in pack.program_rows(row):
        tokens += _tokens(pack, pack.catalog.program_offsets, program)
    return tokens


def _variant(name: str) -> tuple[int, int]:
    return (TOKEN_KINDS["variant"], token_value(name))


def _fake_observation(card_ids: list[int], *, hidden_marker: str = "a"):
    return SimpleNamespace(
        agent_cards=[SimpleNamespace(registry_key=card_ids[0])],
        agent_sideboard=[],
        opponent_cards=[SimpleNamespace(registry_key=card_ids[1])],
        stack_objects=[
            SimpleNamespace(kind=1, source_card_registry_key=card_ids[0]),
            SimpleNamespace(kind=0, source_card_registry_key=card_ids[1]),
        ],
        continuation=[],
        # The projector must never inspect this private-state stand-in.
        opponent_hidden_hand=hidden_marker,
    )


def test_catalog_is_deterministic_and_spans_every_definition() -> None:
    first = BoundSemanticPack.bind(_manifest())
    second = BoundSemanticPack.bind(_manifest())
    assert first.schema.raw == _json(DEFAULT_SCHEMA_PATH)
    for field in first.catalog.__dict__:
        assert (
            getattr(first.catalog, field).tobytes()
            == getattr(second.catalog, field).tobytes()
        )
    assert first.catalog.definitions == len(_manifest()["definitions"])
    assert first.catalog.definition_ref_source_tokens.size > 0
    assert first.catalog.definition_ref_source_tokens.size == (
        first.catalog.definition_ref_target_rows.size
    )
    present = set(first.catalog.token_kind.tolist())
    assert present == set(TOKEN_KINDS.values()) - {TOKEN_KINDS["padding"]}
    kinds = {
        int(first.catalog.token_value[int(offset)])
        for offset in first.catalog.program_offsets[:-1]
    }
    assert kinds <= set(PROGRAM_KINDS.values())
    for offset in first.catalog.program_offsets[:-1]:
        assert int(first.catalog.token_kind[int(offset)]) == TOKEN_KINDS["program_kind"]


def test_tokens_are_the_engine_definition_without_identity_or_printed_text() -> None:
    pack = BoundSemanticPack.bind(_manifest())
    values = set(pack.catalog.token_value.tolist())
    for row in managym.content_pool(PACK_KEY)["definitions"]:
        # Single-word names such as "Ally" are also printed subtypes.
        if " " in row["registry_name"]:
            assert token_value(row["registry_name"]) not in values
    for field in ("key", "scryfall_id", "name", "text_box"):
        assert (TOKEN_KINDS["field"], token_value(field)) not in set(
            zip(pack.catalog.token_kind.tolist(), values, strict=False)
        )
    bolt = _card_tokens(pack, _row(pack, "Igneous Inspiration"))
    assert _variant("Effect::DealDamage") in bolt
    assert _variant("TargetSpec::CreatureOrPlayer") in bolt
    assert (TOKEN_KINDS["integer"], 3) in bolt


def test_same_effect_encodes_identically_across_cards() -> None:
    pack = BoundSemanticPack.bind(_manifest())
    draw = [
        _variant("Effect::DrawCards"),
        (TOKEN_KINDS["field"], token_value("count")),
    ]

    def has_draw(tokens: list[tuple[int, int]]) -> bool:
        return any(tokens[i : i + 2] == draw for i in range(len(tokens) - 1))

    drawing = [
        row for row in range(pack.definition_count) if has_draw(_card_tokens(pack, row))
    ]
    assert len(drawing) >= 2


def test_definition_tokens_do_not_depend_on_the_pack() -> None:
    small = BoundSemanticPack.bind(_manifest())
    world = BoundSemanticPack.bind(_manifest(pack_key="world"))
    for name in ("Forest", "Igneous Inspiration", "Invasion Reinforcements"):
        assert _card_tokens(small, _row(small, name)) == _card_tokens(
            world, _row(world, name, "world")
        )


def test_created_tokens_are_definition_references() -> None:
    pack = BoundSemanticPack.bind(_manifest())
    (program,) = pack.program_rows(_row(pack, "Invasion Reinforcements"))
    padded = pack.pad_catalog()
    targets = padded["definition_ref_target_rows"][program]
    mask = padded["definition_ref_mask"][program]
    assert targets[mask].tolist() == [_row(pack, "Ally")]


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
        SimpleNamespace(
            agent_cards=[],
            agent_sideboard=[],
            opponent_cards=[],
            stack_objects=[],
            continuation=[],
        ),
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


def test_unknown_schema_and_content_binding_fail_closed(tmp_path: Path) -> None:
    schema = _json(DEFAULT_SCHEMA_PATH)
    schema["schema_version"] = 999
    bad_schema = tmp_path / "schema.json"
    bad_schema.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(UnknownSchemaError, match="got 999"):
        LearningSchema.load(bad_schema)

    schema = _json(DEFAULT_SCHEMA_PATH)
    schema["token_kinds"]["opcode"] = 10
    bad_kinds = tmp_path / "kinds.json"
    bad_kinds.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(SemanticProjectionError, match="TokenKind"):
        LearningSchema.load(bad_kinds)

    stale = {**_manifest(), "content_digest": "0" * 64}
    with pytest.raises(ContentPackBindingError, match="differs from the live pack"):
        BoundSemanticPack.bind(stale)

    missing = _manifest()
    missing["definitions"].pop()
    with pytest.raises(ContentPackBindingError, match="do not all bind"):
        BoundSemanticPack.bind(missing)

    unknown = _manifest()
    unknown["definitions"][0]["registry_name"] = "Not A Card"
    with pytest.raises(ContentPackBindingError, match="does not bind"):
        BoundSemanticPack.bind(unknown)

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


def test_sideboard_program_projection_is_independent_of_definition_ids():
    manifest = _manifest()
    permuted = deepcopy(manifest)
    for row in permuted["definitions"]:
        row["card_def_id"] += 1000
    pack = BoundSemanticPack.bind(manifest)
    reordered = BoundSemanticPack.bind(permuted)
    ids = sorted(pack.definition_row_by_card_def_id)
    observation = _fake_observation(ids[:2])
    observation.agent_sideboard = [SimpleNamespace(registry_key=ids[2])]
    other = _fake_observation([value + 1000 for value in ids[:2]])
    other.agent_sideboard = [SimpleNamespace(registry_key=ids[2] + 1000)]
    first = pack.project_observation(observation)
    second = reordered.project_observation(other)
    assert (
        first.object_definition_rows.tolist() == second.object_definition_rows.tolist()
    )
    assert first.object_roles.tolist() == second.object_roles.tolist()
    assert first.object_slots.tolist() == second.object_slots.tolist()
    assert not first.opaque_identity_valid.any()
    assert not second.opaque_identity_valid.any()


def test_learning_schema_rejects_the_pre_sideboard_contract(tmp_path: Path) -> None:
    schema = _json(DEFAULT_SCHEMA_PATH)
    schema["schema_version"] = 1
    schema["object_roles"].pop("agent_sideboard")
    path = tmp_path / "old-schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(UnknownSchemaError, match="got 1"):
        LearningSchema.load(path)


def test_waiting_delayed_trigger_projects_its_source_programs() -> None:
    pack = BoundSemanticPack.bind(_manifest())
    card_ids = sorted(pack.definition_row_by_card_def_id)[:2]
    observation = _fake_observation(card_ids)
    observation.continuation = [
        SimpleNamespace(kind=1, source=SimpleNamespace(definition=card_ids[1])),
        SimpleNamespace(kind=10, source=SimpleNamespace(definition=card_ids[0])),
    ]
    projection = pack.project_observation(observation, identity_mode="semantic_only")
    role = pack.schema.object_roles["delayed_trigger"]
    delayed = projection.object_roles == role
    assert delayed.sum() == 1
    assert projection.object_slots[delayed].tolist() == [1]
    assert projection.object_definition_rows[delayed].tolist() == [
        pack.definition_row(card_ids[0])
    ]
