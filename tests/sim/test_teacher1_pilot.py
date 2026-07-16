"""Fail-closed contract tests for the Teacher-1 admission runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.run_teacher1_pilot import (
    _load_contract,
    _load_control_lock,
)
from manabot.sim.teacher1_evidence import ContractError, file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "experiments/contracts/w2-234-teacher1-pilot-v1.json"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_missing_control_lock_blocks_before_search(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)

    with pytest.raises(ContractError, match="required file does not exist"):
        _load_control_lock(
            tmp_path / "missing.json",
            contract_hash=contract_hash,
            budgets=contract["teacher"]["budgets"],
        )


def test_control_lock_binds_terminal_recovery_and_latency_calibration(
    tmp_path: Path,
) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    checkpoint = tmp_path / "policy-value.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    recovery = tmp_path / "recovery-manifest.json"
    _write_json(recovery, {"status": "completed_pass"})
    calibration = tmp_path / "calibration.json"
    matches = {
        str(budget): {
            "flat_sims_per_action": index + 1,
            "relative_p50_gap": 0.05,
        }
        for index, budget in enumerate(contract["teacher"]["budgets"])
    }
    _write_json(calibration, {"host": "test-host", "matches": matches})
    lock = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "host": "test-host",
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

    loaded, lock_hash = _load_control_lock(
        lock_path,
        contract_hash=contract_hash,
        budgets=contract["teacher"]["budgets"],
    )
    assert loaded == lock
    assert len(lock_hash) == 64

    calibration_payload = json.loads(calibration.read_text())
    calibration_payload["matches"]["128"]["relative_p50_gap"] = 0.11
    _write_json(calibration, calibration_payload)
    lock["latency_calibration"]["sha256"] = file_sha256(calibration)
    _write_json(lock_path, lock)
    with pytest.raises(ContractError, match="not within 10%"):
        _load_control_lock(
            lock_path,
            contract_hash=contract_hash,
            budgets=contract["teacher"]["budgets"],
        )


def test_nonterminal_recovery_cannot_be_frozen_as_control(tmp_path: Path) -> None:
    contract, contract_hash = _load_contract(CONTRACT)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    recovery = tmp_path / "recovery.json"
    _write_json(recovery, {"status": "running"})
    calibration = tmp_path / "calibration.json"
    _write_json(calibration, {"host": "test", "matches": {}})
    lock_path = tmp_path / "lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "contract_sha256": contract_hash,
            "host": "test",
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
                str(budget): 1 for budget in contract["teacher"]["budgets"]
            },
        },
    )

    with pytest.raises(ContractError, match="not terminal"):
        _load_control_lock(
            lock_path,
            contract_hash=contract_hash,
            budgets=contract["teacher"]["budgets"],
        )
