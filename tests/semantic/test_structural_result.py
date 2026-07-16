from __future__ import annotations

import json
from pathlib import Path

RESULT = Path("experiments/data/structural-semantic-katas-v1.json")


def test_checked_result_is_complete_and_selects_registered_branch() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["decision"]["terminal"] == (
        "REDESIGN optimization_or_capacity_unresolved"
    )
    assert result["budget"]["optimizer_steps"] == 8000
    assert result["budget"]["presented_examples"] == 512000
    assert result["budget"]["wall_clock_seconds"] < 1800
    assert len(result["runs"]) == 10
    assert {run["model_seed"] for run in result["runs"]} == {
        21401,
        21402,
        21403,
        21404,
        21405,
    }


def test_bag_symmetry_holds_but_structural_trainability_does_not() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    evidence = result["decision"]["evidence"]
    bag_runs = [run for run in result["runs"] if run["arm"] == "bag_v1"]
    structural_runs = [
        run
        for run in result["runs"]
        if run["arm"] == "relational_semantic_encoder_v1"
    ]

    assert evidence["bag_symmetry"]
    assert not evidence["structural_trainability"]
    assert all(run["test"]["accuracy"] == 0.5 for run in bag_runs)
    assert all(run["symmetry"]["paired_prediction_disagreements"] == 0 for run in bag_runs)
    assert all(
        run["test"]["by_family"]["order"]["accuracy"] == 1.0
        and run["test"]["by_family"]["hierarchy"]["accuracy"] == 1.0
        for run in structural_runs
    )
    assert any(run["train"]["accuracy"] < 0.99 for run in structural_runs)


def test_result_preserves_scope_and_instrument_audits() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    audit = result["suite_audit"]

    assert audit["definition_references"] == 0
    assert audit["identity_fields_tensorized"] == []
    assert all(
        overlap == 0
        for family in (
            "normalized_program_hash",
            "nuisance_signature",
            "pair_template_signature",
        )
        for overlap in audit["overlap"][family].values()
    )
    assert "gameplay" in result["claim_boundary"]
