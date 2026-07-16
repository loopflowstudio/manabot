"""Fail-closed contract tests for the Teacher-1 admission runner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.runners.run_teacher1_pilot import (
    _HOST_IDENTITY_FIELDS,
    _load_contract,
    _load_control_lock,
    _play_cell,
)
from manabot.sim.teacher1_evidence import (
    ContractError,
    canonical_sha256,
    file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "experiments/contracts/w2-234-teacher1-pilot-v1.json"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _host_identity(contract: dict) -> dict:
    return {field: contract["host"][field] for field in _HOST_IDENTITY_FIELDS}


def _write_valid_control_lock(
    tmp_path: Path,
    contract: dict,
    contract_hash: str,
    *,
    recovery_status: str = "completed_pass",
) -> tuple[Path, dict, Path]:
    host = _host_identity(contract)
    checkpoint = tmp_path / "policy-value.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    recovery = tmp_path / "recovery-manifest.json"
    _write_json(recovery, {"status": recovery_status})
    calibration = tmp_path / "calibration.json"
    matches = {
        str(budget): {
            "flat_sims_per_action": index + 1,
            "teacher_p50_decision_ms": float((index + 1) * 10),
            "flat_p50_decision_ms": float((index + 1) * 10.5),
            "relative_p50_gap": 0.05,
        }
        for index, budget in enumerate(contract["teacher"]["budgets"])
    }
    _write_json(calibration, {"host": host, "matches": matches})
    lock = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "host": host,
        "checkpoint_control": {
            "arm": "policy_value",
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "recovery_manifest": {
            "path": str(recovery),
            "sha256": file_sha256(recovery),
        },
        "latency_calibration": {
            "path": str(calibration),
            "sha256": file_sha256(calibration),
        },
        "flat_sims_per_action_by_teacher_budget": {
            str(budget): index + 1
            for index, budget in enumerate(contract["teacher"]["budgets"])
        },
    }
    lock_path = tmp_path / "control-lock.json"
    _write_json(lock_path, lock)
    return lock_path, lock, calibration


def test_missing_control_lock_blocks_before_search(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)

    with pytest.raises(ContractError, match="required file does not exist"):
        _load_control_lock(
            tmp_path / "missing.json",
            contract=contract,
            contract_hash=contract_hash,
            current_host=_host_identity(contract),
        )


def test_matchup_cell_reports_three_seed_blocks_and_fixed_aggregate() -> None:
    result = _play_cell(
        {"kind": "random"},
        {"kind": "random"},
        seed_blocks=[
            {"id": "one", "seed": 701, "games": 2},
            {"id": "two", "seed": 709, "games": 2},
            {"id": "three", "seed": 719, "games": 2},
        ],
    )

    assert result["num_games"] == 6
    assert list(result["seed_blocks"]) == ["one", "two", "three"]
    assert [block["num_games"] for block in result["seed_blocks"].values()] == [
        2,
        2,
        2,
    ]
    assert [block["seed"] for block in result["seed_blocks"].values()] == [
        701,
        709,
        719,
    ]


def test_control_lock_binds_terminal_recovery_and_latency_calibration(
    tmp_path: Path,
) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, lock, calibration = _write_valid_control_lock(
        tmp_path, contract, contract_hash
    )

    loaded, lock_hash = _load_control_lock(
        lock_path,
        contract=contract,
        contract_hash=contract_hash,
        current_host=_host_identity(contract),
    )
    assert loaded == lock
    assert len(lock_hash) == 64

    calibration_payload = json.loads(calibration.read_text())
    calibration_payload["matches"]["128"]["flat_p50_decision_ms"] = 33.3
    calibration_payload["matches"]["128"]["relative_p50_gap"] = 0.11
    _write_json(calibration, calibration_payload)
    lock["latency_calibration"]["sha256"] = file_sha256(calibration)
    _write_json(lock_path, lock)
    with pytest.raises(ContractError, match="exceeds contract"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=_host_identity(contract),
        )


def test_nonterminal_recovery_cannot_be_frozen_as_control(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, _, _ = _write_valid_control_lock(
        tmp_path,
        contract,
        contract_hash,
        recovery_status="running",
    )

    with pytest.raises(ContractError, match="not terminal"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=_host_identity(contract),
        )


def test_foreign_host_calibration_is_rejected(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, lock, calibration = _write_valid_control_lock(
        tmp_path, contract, contract_hash
    )
    foreign_host = {**_host_identity(contract), "machine": "foreign-machine"}
    calibration_payload = json.loads(calibration.read_text())
    calibration_payload["host"] = foreign_host
    _write_json(calibration, calibration_payload)
    lock["host"] = foreign_host
    lock["latency_calibration"]["sha256"] = file_sha256(calibration)
    _write_json(lock_path, lock)

    with pytest.raises(ContractError, match="does not bind to the contract"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=_host_identity(contract),
        )


def test_current_runtime_host_must_match_contract(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, _, _ = _write_valid_control_lock(tmp_path, contract, contract_hash)
    foreign_runtime = {**_host_identity(contract), "machine": "foreign-machine"}

    with pytest.raises(ContractError, match="current host identity"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=foreign_runtime,
        )


def test_same_architecture_foreign_chip_must_match_contract(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, _, _ = _write_valid_control_lock(tmp_path, contract, contract_hash)
    foreign_runtime = {**_host_identity(contract), "chip": "Apple M4 Pro"}

    with pytest.raises(ContractError, match="current host identity"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=foreign_runtime,
        )


@pytest.mark.parametrize(
    "field",
    [
        "teacher_p50_decision_ms",
        "flat_p50_decision_ms",
        "relative_p50_gap",
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), -0.01])
def test_invalid_latency_measurements_are_rejected(
    tmp_path: Path, field: str, invalid: float
) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, lock, calibration = _write_valid_control_lock(
        tmp_path, contract, contract_hash
    )
    calibration_payload = json.loads(calibration.read_text())
    calibration_payload["matches"]["128"][field] = invalid
    _write_json(calibration, calibration_payload)
    lock["latency_calibration"]["sha256"] = file_sha256(calibration)
    _write_json(lock_path, lock)

    with pytest.raises(ContractError, match="finite and nonnegative"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=_host_identity(contract),
        )


def test_gap_threshold_comes_from_contract(tmp_path: Path) -> None:
    contract, _ = _load_contract(CONTRACT)
    contract = deepcopy(contract)
    contract["gates"]["max_realized_p50_gap"] = 0.20
    contract_hash = canonical_sha256(contract)
    lock_path, lock, calibration = _write_valid_control_lock(
        tmp_path, contract, contract_hash
    )
    calibration_payload = json.loads(calibration.read_text())
    calibration_payload["matches"]["128"]["flat_p50_decision_ms"] = 34.5
    calibration_payload["matches"]["128"]["relative_p50_gap"] = 0.15
    _write_json(calibration, calibration_payload)
    lock["latency_calibration"]["sha256"] = file_sha256(calibration)
    _write_json(lock_path, lock)

    loaded, _ = _load_control_lock(
        lock_path,
        contract=contract,
        contract_hash=contract_hash,
        current_host=_host_identity(contract),
    )
    assert loaded == lock


def test_nonpositive_teacher_p50_is_rejected(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, lock, calibration = _write_valid_control_lock(
        tmp_path, contract, contract_hash
    )
    calibration_payload = json.loads(calibration.read_text())
    calibration_payload["matches"]["128"]["teacher_p50_decision_ms"] = 0.0
    _write_json(calibration, calibration_payload)
    lock["latency_calibration"]["sha256"] = file_sha256(calibration)
    _write_json(lock_path, lock)

    with pytest.raises(ContractError, match="must be positive"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=_host_identity(contract),
        )


def test_reported_gap_must_match_measured_p50_values(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    lock_path, lock, calibration = _write_valid_control_lock(
        tmp_path, contract, contract_hash
    )
    calibration_payload = json.loads(calibration.read_text())
    calibration_payload["matches"]["128"]["flat_p50_decision_ms"] = 1000.0
    calibration_payload["matches"]["128"]["relative_p50_gap"] = 0.05
    _write_json(calibration, calibration_payload)
    lock["latency_calibration"]["sha256"] = file_sha256(calibration)
    _write_json(lock_path, lock)

    with pytest.raises(ContractError, match="does not match derived gap"):
        _load_control_lock(
            lock_path,
            contract=contract,
            contract_hash=contract_hash,
            current_host=_host_identity(contract),
        )
