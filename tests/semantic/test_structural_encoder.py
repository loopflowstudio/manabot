from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from manabot.semantic.structural import (
    KATA_FAMILIES,
    StructuralSemanticError,
    build_matched_models,
    project_static_relations,
    trainable_parameter_count,
)
from manabot.semantic.structural_katas import (
    evaluate_model,
    load_suite,
    paired_symmetry_audit,
    records_to_batch,
)


def _test_records() -> tuple[dict, list[dict]]:
    suite, _ = load_suite()
    records = [row for row in suite["programs"] if row["split"] == "test"]
    return suite, records


def test_projector_rejects_definition_references_and_malformed_links() -> None:
    with pytest.raises(StructuralSemanticError, match="definition-reference"):
        project_static_relations(
            [
                "structure:program_begin",
                "definition_ref:kata.identity",
                "structure:program_end",
            ]
        )
    with pytest.raises(StructuralSemanticError, match="unclosed"):
        project_static_relations(["structure:program_begin", "opcode:draw_cards"])
    with pytest.raises(StructuralSemanticError, match="dangling local role"):
        project_static_relations(
            [
                "structure:program_begin",
                "field:target",
                "role:local:0",
                "structure:program_end",
            ]
        )


def test_structural_suite_exposes_required_relation_families() -> None:
    _, records = _test_records()
    relation_presence = defaultdict(set)
    for record in records:
        for name, edges in record["model_input"]["relations"].items():
            if edges:
                relation_presence[record["family"]].add(name)

    assert all(
        {"parent_to_child", "ancestor_to_descendant", "field_to_value"}
        <= relation_presence[family]
        for family in KATA_FAMILIES
    )
    for family in ("argument_binding", "target_choice_role"):
        assert {
            "role_declaration_to_reference",
            "role_reference_to_declaration",
        } <= relation_presence[family]


def test_models_are_capacity_matched_and_probe_heads_start_identically() -> None:
    suite, records = _test_records()
    bag, structural = build_matched_models(
        token_count=len(suite["token_vocabulary"]),
        token_kind_count=suite["token_kind_count"],
        seed=21401,
    )
    bag_count = trainable_parameter_count(bag)
    structural_count = trainable_parameter_count(structural)

    assert abs(structural_count - bag_count) / bag_count <= 0.05
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            bag.probe.parameters(), structural.probe.parameters(), strict=True
        )
    )

    batch = records_to_batch(records[:8], seed=21401)
    assert bag(batch).shape == structural(batch).shape == (8, 2)


def test_bag_negative_control_has_exact_pair_symmetry_and_ceiling() -> None:
    suite, records = _test_records()
    bag, _ = build_matched_models(
        token_count=len(suite["token_vocabulary"]),
        token_kind_count=suite["token_kind_count"],
        seed=21401,
    )
    batch = records_to_batch(records, seed=21401)
    result = evaluate_model(bag, batch)
    symmetry = paired_symmetry_audit(bag, batch, records)

    assert symmetry["maximum_paired_probability_difference"] <= 1e-6
    assert symmetry["paired_prediction_disagreements"] == 0
    assert result["accuracy"] == 0.5
    assert all(result["by_family"][family]["accuracy"] == 0.5 for family in KATA_FAMILIES)
