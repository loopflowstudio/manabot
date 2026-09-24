"""Locked evidence checks for the RUL-13 prepared materializer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.run_rul13_prepared_materializer import (
    EXPECTED_SUPPORT_SIZE,
    IDENTITY_STREAM_SHA256,
    V1_CONTRACT,
    V1_FAILURE,
    Rul13Error,
    verify_receipt,
)
from manabot.sim.teacher1_evidence import file_sha256

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "experiments/data/rul-13-prepared-possible-world-materializer-v1.json"


def test_locked_provider_receipt_proves_linear_bounded_materialization() -> None:
    receipt = json.loads(RECEIPT.read_text())
    verify_receipt(receipt)
    prepared = receipt["prepared_materializer"]
    diagnostics = prepared["diagnostics"]

    assert prepared["support_size"] == EXPECTED_SUPPORT_SIZE
    assert prepared["full_support_materialized"] is True
    assert diagnostics["space_constructions"] == 1
    assert diagnostics["requested_rows"] == diagnostics["materialized_rows"]
    assert diagnostics["maximum_returned_batch"] == 256
    assert diagnostics["current_live_prepared_branches"] == 0
    assert diagnostics["maximum_live_prepared_branches"] == 256
    assert prepared["peak_rss_bytes"] < 2 * 1024**3
    assert prepared["full_pass_rows_per_second"] > 0
    assert [cell["threshold_rows"] for cell in prepared["scaling"]] == [
        256,
        4096,
        32768,
        EXPECTED_SUPPORT_SIZE,
    ]
    assert len(prepared["parity"]) == 4
    assert set(prepared["rejections"]) == {
        "empty",
        "length_mismatch",
        "oversize",
        "index",
        "stale_root",
        "identity",
    }
    assert receipt["public_commitment_provider"] == {
        "path": "conformance/public-commitment-parity-v1/release-live-replay-hypothesis-seed-0.json",
        "sha256": "4522794188d4a93eb49a9fd6862ddd1d1345dd127a4a1fbb62dae9a5ad2f57da",
        "rules_provider_gaps": 0,
        "identity_stream_sha256": IDENTITY_STREAM_SHA256,
    }


def test_int17_v1_contract_and_failure_remain_immutable() -> None:
    assert file_sha256(V1_CONTRACT) == (
        "221ef4979ae0e4421efb5953473276b125cb5fdaf91983f4a66660b065d44473"
    )
    assert file_sha256(V1_FAILURE) == (
        "78bde491e16957b743a59cebe6f87fd519dc982793d5d6f7dbb649a98d57e027"
    )


@pytest.mark.parametrize(
    "field",
    [
        "engine_source_sha256",
        "int17_source_sha256",
        "provider_source_sha256",
        "engine_extension_sha256",
    ],
)
def test_receipt_rejects_runtime_drift(field: str) -> None:
    receipt = json.loads(RECEIPT.read_text())
    receipt["runtime"][field] = "0" * 64
    with pytest.raises(Rul13Error, match="runtime drifted"):
        verify_receipt(receipt, check_extension=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("materialized_rows", 1),
        ("requested_rows", 1),
        ("current_live_prepared_branches", 1),
        ("maximum_live_prepared_branches", 257),
        ("space_constructions", 2),
    ],
)
def test_receipt_rejects_incomplete_or_unbounded_work(field: str, value: int) -> None:
    receipt = json.loads(RECEIPT.read_text())
    receipt["prepared_materializer"]["diagnostics"][field] = value
    with pytest.raises(Rul13Error):
        verify_receipt(receipt)
