from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from manabot.semantic.learning import DEFAULT_IR_PATH, BoundSemanticPack
from manabot.semantic.transfer import (
    ARMS,
    SEMANTIC_CONTROLS,
    RuntimeTables,
    SemanticTransferSpec,
    SymbolicProgramBinder,
    TransferExperimentError,
    aggregate_arm,
    build_examples,
    build_records,
    build_spec,
    holdout_audit,
    load_workload,
    paired_cluster_contrast,
    permuted_runtime_projection,
    robustness_controls,
    run_arm,
    validate_workload,
)

WORKLOAD = Path("experiments/workloads/opcode-alignment-v1.json")


def _manifest() -> dict:
    ir = json.loads(DEFAULT_IR_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "content_digest": "opcode-alignment-test-pack",
        "definitions": [
            {
                "card_def_id": row,
                "registry_name": definition["content_pack_binding"]["value"],
            }
            for row, definition in enumerate(ir["definitions"])
        ],
    }


def _fast_workload() -> dict:
    workload, _ = load_workload(WORKLOAD)
    workload["training"]["epochs"] = 2
    workload["training"]["limited_retraining_steps"] = 1
    workload["performance"] = {"warmup": 1, "samples": 2}
    return workload


def test_workload_preregisters_balanced_causal_controls() -> None:
    workload, digest = load_workload(WORKLOAD)

    assert len(digest) == 64
    assert workload["evaluation"]["seats"] == [0, 1]
    assert len(workload["model_seeds"]) >= 3
    assert {key for rows in workload["held_out_definitions"].values() for key in rows}
    assert all(len(rows) == 2 for rows in workload["held_out_definitions"].values())
    assert workload["gates"]["minimum_legal_choices"] > 32


def test_workload_rejects_unbalanced_or_underpowered_controls() -> None:
    workload, _ = load_workload(WORKLOAD)
    unbalanced = deepcopy(workload)
    unbalanced["held_out_definitions"]["ur_lessons"] = ["stx.pop_quiz"]
    with pytest.raises(TransferExperimentError, match="two definitions"):
        validate_workload(unbalanced)

    too_few_seeds = deepcopy(workload)
    too_few_seeds["model_seeds"] = [1, 2]
    with pytest.raises(TransferExperimentError, match=">=3"):
        validate_workload(too_few_seeds)


def test_spec_is_symbolic_and_holdout_uses_only_seen_operations() -> None:
    pack = BoundSemanticPack.bind(_manifest())
    workload, _ = load_workload(WORKLOAD)
    spec = build_spec(pack)
    records = build_records(pack, spec, workload)

    assert spec.definition_symbols != tuple(
        definition["semantic_key"] for definition in pack.ir.definitions
    )
    assert spec.max_program_tokens == max(len(row.token_ids) for row in records)
    assert len([row for row in records if row.held_out]) == 4
    training_ops = {row.target_opcode for row in records if not row.held_out}
    assert all(row.target_opcode in training_ops for row in records if row.held_out)

    first = build_examples(records, workload, held_out=True)
    second = build_examples(records, workload, held_out=True)
    assert first == second
    assert {row.seat for row in first} == {0, 1}
    assert len({row.candidate_order for row in first}) > 1

    audit = holdout_audit(pack, spec, workload)
    assert audit["normalized_ast_novel_programs"] == 2
    assert audit["full_symbolic_primitive_closure_programs"] == 2
    assert not audit["all_normalized_asts_novel"]
    assert not audit["full_symbolic_primitive_closure"]


def test_rebinding_and_numeric_permutation_are_exact_but_unknowns_fail_closed() -> None:
    pack = BoundSemanticPack.bind(_manifest())
    spec = build_spec(pack)
    control = robustness_controls(pack, spec)

    assert control["content_pack_reorder_checkpoint_rebind_rate"] == 1.0
    assert control["token_kind_and_opcode_id_permutation_rate"] == 1.0
    assert control["unknown_opcode_fail_closed"]
    assert control["enum_domain_schema_change_fail_closed"]
    assert not control["arbitrary_enum_id_rebinding_supported"]

    tables, kinds, values = permuted_runtime_projection(pack)
    bad_tables = RuntimeTables(
        token_kinds={**tables.token_kinds, "invented": 999},
        opcodes=tables.opcodes,
    )
    with pytest.raises(TransferExperimentError, match="token-kind symbols"):
        SymbolicProgramBinder(pack, spec, runtime_tables=bad_tables)

    bad_spec = SemanticTransferSpec.from_dict(
        {**spec.to_dict(), "enum_schema_hash": "0" * 64}
    )
    with pytest.raises(TransferExperimentError, match="enum-domain schema"):
        SymbolicProgramBinder(pack, bad_spec)

    assert kinds.shape == pack.catalog.token_kind.shape
    assert values.shape == pack.catalog.token_value.shape


@pytest.mark.parametrize("arm", (*ARMS, *SEMANTIC_CONTROLS))
def test_every_arm_runs_without_changing_the_production_abi(arm: str) -> None:
    pack = BoundSemanticPack.bind(_manifest())
    result = run_arm(pack, _fast_workload(), arm=arm, model_seed=7)

    assert result["arm"] == arm
    assert result["training_examples"] > result["heldout_examples"]
    assert result["zero_shot"]["total"] == result["heldout_examples"]
    assert result["limited_retraining"]["total"] == result["heldout_examples"]
    assert result["checkpoint_rebind"]["match_rate"] == 1.0
    assert result["performance"]["latency_ns"]["p50"] > 0
    assert result["semantic_input_spec_digest"]


def test_identity_only_arms_receive_no_semantic_program_input() -> None:
    pack = BoundSemanticPack.bind(_manifest())
    workload = _fast_workload()
    legacy = run_arm(pack, workload, arm="card_id_legacy", model_seed=8)
    structured = run_arm(pack, workload, arm="card_id_structured", model_seed=8)
    semantic = run_arm(pack, workload, arm="semantic_only_structured", model_seed=8)

    assert legacy["input"] == {
        "semantic_program": False,
        "opaque_card_def_identity": True,
    }
    assert structured["input"] == legacy["input"]
    assert semantic["input"] == {
        "semantic_program": True,
        "opaque_card_def_identity": False,
    }
    assert legacy["head"] == "legacy_fixed"
    assert structured["head"] == semantic["head"] == "structured_ragged"


def _cluster_run(seed: int, successes: dict[str, bool]) -> dict:
    rows = []
    for program, correct in successes.items():
        for seat in (0, 1):
            for replicate in range(4):
                rows.append(
                    {
                        "program": program,
                        "seat": seat,
                        "replicate": replicate,
                        "correct": correct,
                    }
                )
    return {"model_seed": seed, "zero_shot_rows": rows}


def test_uncertainty_uses_program_by_seed_clusters_not_repeated_rows() -> None:
    programs = ["a", "b", "c", "d"]
    runs = [
        _cluster_run(seed, {program: True for program in programs}) for seed in (1, 2)
    ]
    runs.append(_cluster_run(3, {"a": True, "b": True, "c": True, "d": False}))

    summary = aggregate_arm(runs, "zero_shot")
    assert summary["successes"] == 11
    assert summary["total"] == summary["cluster_count"] == 12
    assert summary["accuracy"] == pytest.approx(11 / 12)
    assert summary["ci95_low"] == pytest.approx(0.646120088858883)
    assert summary["ci95_high"] == pytest.approx(0.9851349055950829)
    assert summary["repeated_rows_excluded_from_uncertainty"]

    control = [
        _cluster_run(seed, {program: False for program in programs})
        for seed in (1, 2, 3)
    ]
    contrast = paired_cluster_contrast(runs, control)
    assert contrast["wins"] == 11
    assert contrast["losses"] == 0
    assert contrast["ties"] == 1
    assert contrast["two_sided_exact_sign_p"] == pytest.approx(1 / 1024)
