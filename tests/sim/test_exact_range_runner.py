"""Pinned evidence-boundary checks for the INT-9 substrate contract."""

from pathlib import Path

import pytest

from experiments.runners.run_exact_range_player import (
    ArtifactUnavailable,
    load_contract,
    resolve_artifact,
)

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "contracts"
    / "int-9-exact-range-v1.json"
)


def test_contract_pins_world_arena_evidence_and_exclusions() -> None:
    contract, smoke = load_contract(CONTRACT, "smoke")
    _, arena = load_contract(CONTRACT, "arena")

    assert contract["world"] == "w2"
    assert contract["arena"]["id"] == "w2-interactive-belief-v1"
    assert contract["arena"]["required_primary_cell"] == "belief_vs_compatible_prior"
    assert contract["algorithm"]["belief_state"].startswith(
        "normalized_probability_over_managym_possible_world_space"
    )
    assert contract["expected_fingerprints"]["semantic_decision_version"] == 3
    assert contract["expected_fingerprints"]["possible_world_space_version"] == 1
    assert {
        "engine_source_sha256",
        "engine_extension_sha256",
        "engine_extension_name",
        "int9_source_sha256",
    }.issubset(contract["expected_fingerprints"])
    assert "public_action_alphabet" not in contract["algorithm"]
    assert "determinization" not in contract["algorithm"]
    assert smoke["evidence_class"] == "engineering_smoke_non_admission"
    assert arena["evidence_class"] == "preregistered_arena"
    assert set(contract["required_evidence"]) == {
        "gameplay",
        "calibration",
        "integrity",
        "competencies",
        "systems",
    }
    assert "public_belief_solving" in contract["exclusions"]
    assert "broad_deck_generalization" in contract["exclusions"]
    compute_gate = contract["gates"]["matched_compute"]
    assert compute_gate["configured_worlds_per_action_ratio_min"] == 1.0
    assert compute_gate["configured_worlds_per_action_ratio_max"] == 1.0
    assert compute_gate["configured_rollouts_per_world_ratio_min"] == 1.0
    assert compute_gate["configured_rollouts_per_world_ratio_max"] == 1.0
    assert "raw_playout_count_ratio_min" not in compute_gate


def test_unresolved_likelihood_artifact_fails_closed() -> None:
    contract, _ = load_contract(CONTRACT, "smoke")

    with pytest.raises(ArtifactUnavailable, match="not byte-locked"):
        resolve_artifact(contract, "likelihood_checkpoint")
