"""Evidence-boundary checks for the frozen INT-17 calibration runner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.runners.run_belief_calibration import (
    _load_inputs,
    _resolve_locked_file,
    load_contract,
    parse_args,
    preflight,
)
from manabot.sim.teacher1_evidence import (
    ContractError,
    canonical_sha256,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "experiments/contracts/int-17-belief-calibration-v1.json"
FAILURE_DIR = (
    ROOT
    / "experiments/data/int-17-belief-calibration-v1/sha256"
    / "78bde491e16957b743a59cebe6f87fd519dc982793d5d6f7dbb649a98d57e027"
)


def test_contract_locks_real_likelihood_selected_trace_and_negative_policy() -> None:
    contract = load_contract(CONTRACT)

    checkpoint = _resolve_locked_file(
        contract["likelihood_checkpoint"], "likelihood checkpoint"
    )

    assert checkpoint.name == "visit_policy_only-seed-197.pt"
    assert contract["likelihood_checkpoint"]["sha256"].startswith("1673a237")
    assert contract["algorithm"]["model_fallback"] == "forbidden"
    assert contract["algorithm"]["truth_access"] == ("post_update_authority_audit_only")
    assert contract["cohort"] == {
        "games": 1,
        "game_seed": 0,
        "viewers": [0, 1],
        "commands": 132,
        "commitments": 62,
        "unadmitted_commands": 70,
        "points_per_viewer": 133,
        "total_points": 266,
    }
    assert contract["prediction"]["failure_policy"] == (
        "retain_refutation_without_checkpoint_substitution"
    )
    assert "arena_strength" in contract["exclusions"]
    assert "belief_head_training" in contract["exclusions"]


def test_preflight_reconstructs_the_rul11_identity_and_exact_row_cap() -> None:
    contract = load_contract(CONTRACT)
    trace, _ = _load_inputs(contract)

    result = preflight(contract, trace)

    assert result.to_dict() == contract["preflight"]
    assert result.world_rows == 903_063
    assert result.commitments == 62


def test_checkpoint_hash_mismatch_fails_before_inference(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    tampered = deepcopy(contract["likelihood_checkpoint"])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not the retained checkpoint")
    tampered.update({"path": str(checkpoint), "sha256": "0" * 64})

    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        _resolve_locked_file(tampered, "likelihood checkpoint")


def test_test_only_likelihood_cannot_be_registered(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text())
    payload["algorithm"]["model_fallback"] = "test-only-likelihood/v1"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ContractError, match="forbid likelihood fallback"):
        load_contract(path)


def test_runner_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--contract",
                str(CONTRACT),
                "--out-dir",
                "result",
                "--preflight-only",
                "--verify-only",
            ]
        )


def test_rul13_requires_a_new_execution_contract_with_identical_science() -> None:
    contract = load_contract(CONTRACT)
    failure = json.loads((FAILURE_DIR / "failure.json").read_text())
    prerequisite = json.loads((FAILURE_DIR / "provider-prerequisite.json").read_text())
    execution = prerequisite["rerun"]["execution_contract"]

    assert execution["action"] == "preregister_new_version_after_provider_lands"
    assert execution["new_contract_path"].endswith("int-17-belief-calibration-v2.json")
    assert (
        execution["source_contract_file_sha256"]
        == failure["frozen_inputs"]["contract_sha256"]
    )
    assert file_sha256(CONTRACT) == execution["source_contract_file_sha256"]
    assert canonical_sha256(contract) == execution["source_contract_canonical_sha256"]
    assert set(execution["scientific_inputs_to_copy_exactly"]) == {
        "trace",
        "likelihood_checkpoint",
        "algorithm",
        "cohort",
        "preflight",
        "metrics",
        "prediction",
        "caps",
        "arena_interpretation",
        "exclusions",
    }
    assert all(
        key in contract for key in execution["scientific_inputs_to_copy_exactly"]
    )
    assert execution["provider_bound_fields_to_refresh"] == [
        "provider_receipt",
        "expected_runtime",
    ]
    assert execution["required_identity_continuity"] == {
        "identity_stream_sha256": contract["preflight"]["identity_stream_sha256"],
        "rules_provider_gaps": 0,
    }
