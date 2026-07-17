"""Frozen contract checks for the INT-4 visit iteration orchestrator."""

from pathlib import Path

from experiments.runners.run_visit_teacher_iteration import (
    _iteration_runtime,
    _load_contract,
)
from manabot.sim.teacher1_evidence import validate_runtime_fingerprints


def test_int4_contract_binds_runtime_and_both_execution_profiles() -> None:
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/contracts/int-4-visit-teacher-iteration-v1.json"
    )
    contract, _, iteration, _ = _load_contract(contract_path, "iteration")
    _, _, smoke, _ = _load_contract(contract_path, "smoke")
    runtime = _iteration_runtime(seed=int(contract["runtime_seed"]))

    validate_runtime_fingerprints(contract["expected_fingerprints"], runtime)
    assert contract["supersedes"] == "w2-234-teacher1-pilot-v1"
    assert iteration["teacher_budgets"] == [8, 32, 128]
    assert iteration["dataset_games"] == 256
    assert iteration["training_seeds"] == [197, 419, 887]
    assert sum(block["games"] for block in iteration["arena_seed_blocks"]) == 48
    assert len(iteration["sampled_search_roots"]) == 8
    assert smoke["evidence_class"] == "engineering_smoke_non_admission"
