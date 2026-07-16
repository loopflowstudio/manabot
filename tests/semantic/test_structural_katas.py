from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from manabot.semantic.structural import KATA_FAMILIES
from manabot.semantic.structural_katas import (
    artifact_bytes,
    load_contract,
    load_suite,
    validate_oracle_fixtures,
)


def test_checked_artifacts_rebuild_byte_identically() -> None:
    source, suite, contract = artifact_bytes()

    from manabot.semantic.structural_katas import (  # local to keep paths explicit
        CONTRACT_PATH,
        SOURCE_PATH,
        SUITE_PATH,
    )

    assert SOURCE_PATH.read_bytes() == source
    assert SUITE_PATH.read_bytes() == suite
    assert CONTRACT_PATH.read_bytes() == contract


def test_suite_contains_five_pair_safe_content_addressed_katas() -> None:
    suite, suite_digest = load_suite()
    contract, contract_digest = load_contract()

    assert len(suite_digest) == len(contract_digest) == 64
    assert len(suite["pairs"]) == 400
    assert len(suite["programs"]) == 800
    assert set(row["family"] for row in suite["families"]) == set(KATA_FAMILIES)
    assert contract["model_seeds"] == [21401, 21402, 21403, 21404, 21405]

    programs_by_pair = defaultdict(list)
    for program in suite["programs"]:
        programs_by_pair[program["pair_id"]].append(program)
        model_input = program["model_input"]
        assert program["token_length"] <= 72
        assert not any(
            symbol.startswith("definition_ref:")
            for symbol in model_input["token_symbols"]
        )
        assert len(model_input["token_ids"]) == program["token_length"]
        assert len(model_input["token_kinds"]) == program["token_length"]
        assert len(model_input["depth"]) == program["token_length"]

    for members in programs_by_pair.values():
        assert len(members) == 2
        assert {row["label"] for row in members} == {0, 1}
        assert len({row["split"] for row in members}) == 1
        assert len({row["token_multiset_hash"] for row in members}) == 1
        assert Counter(members[0]["model_input"]["token_ids"]) == Counter(
            members[1]["model_input"]["token_ids"]
        )
        assert members[0]["model_input"]["token_ids"] != members[1]["model_input"][
            "token_ids"
        ]


def test_anti_template_audit_is_fail_closed_and_explicit() -> None:
    suite, _ = load_suite()
    audit = suite["audit"]

    for field in (
        "normalized_program_hash",
        "nuisance_signature",
        "pair_template_signature",
    ):
        assert audit["overlap"][field] == {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        }
    assert audit["overlap"]["skeleton_id"] == {
        "train_validation": 5,
        "train_test": 5,
        "validation_test": 5,
    }
    assert audit["identity_fields_tensorized"] == []
    assert audit["definition_references"] == 0
    assert audit["pair_label_balance"]
    assert all(
        counts["0"] == counts["1"]
        for table in audit["label_contingencies"].values()
        for counts in table.values()
    )


def test_hand_oracles_cover_each_family_independently() -> None:
    result = validate_oracle_fixtures()

    assert len(result["sha256"]) == 64
    assert set(result["checked"]) == {
        f"{family}.{case}"
        for family in KATA_FAMILIES
        for case in ("negative", "positive")
    }


def test_contract_pins_budget_predictions_and_terminal_branches() -> None:
    contract, _ = load_contract()

    assert contract["budget"] == {
        "arms": 2,
        "seeds": 5,
        "maximum_optimizer_steps": 8000,
        "maximum_presented_examples": 512000,
        "maximum_wall_clock_seconds": 1800,
        "cpu_cores": 1,
    }
    assert contract["training"]["python_minor"] == "3.12"
    assert contract["training"]["batch_size"] == 64
    assert contract["gates"]["structural_aggregate_test_accuracy_minimum"] == 0.95
    assert contract["gates"]["parameter_difference_fraction_maximum"] == 0.05
    assert contract["branches"] == [
        "nominate_for_w2_213",
        "instrument_invalid",
        "teacher_or_label_error",
        "optimization_or_capacity_unresolved",
        "missing_structural_relation",
        "encoder_redesign",
        "cost_redesign",
    ]


def test_unknown_suite_shape_is_rejected() -> None:
    from manabot.semantic.structural_katas import StructuralKataError, validate_suite

    with pytest.raises(StructuralKataError, match="unknown structural kata suite"):
        validate_suite({"schema_version": 2, "id": "invented"})
