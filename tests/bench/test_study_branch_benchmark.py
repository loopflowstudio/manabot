from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "bench_study_branch", ROOT / "scripts/bench_study_branch.py"
)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def _rehash(payload: dict) -> dict:
    payload["artifact_sha256"] = bench.artifact_hash(payload)
    return payload


def test_percentile_uses_nearest_rank() -> None:
    assert bench.percentile([4, 1, 3, 2], 0.50) == 2
    assert bench.percentile([4, 1, 3, 2], 0.95) == 4


def test_artifact_hash_omits_only_its_own_field() -> None:
    payload = {"schema": "x", "value": 1, "artifact_sha256": "old"}
    first = bench.artifact_hash(payload)
    payload["artifact_sha256"] = "new"
    assert bench.artifact_hash(payload) == first
    payload["value"] = 2
    assert bench.artifact_hash(payload) != first


@pytest.fixture(scope="module")
def smoke_payload() -> dict:
    payload = bench.run_measurement(
        iterations=3,
        retained_count=4,
        warmup=1,
        argv=["bench_study_branch.py", "measure", "--smoke"],
    )
    bench.verify(
        payload,
        check_source=True,
        require_canonical=False,
        enforce_gates=False,
    )
    return payload


def test_smoke_measurement_exercises_all_contract_surfaces(smoke_payload: dict) -> None:
    assert smoke_payload["run"]["canonical"] is False
    assert smoke_payload["exactness"]["source_digest_mismatches"] == 0
    assert smoke_payload["privacy"]["opponent_hand_exposures"] == 0
    assert smoke_payload["incarnation"]["projection_tamper_checks"] == 1
    assert {case["id"] for case in smoke_payload["failures"]["cases"]} == (
        bench.EXPECTED_FAILURE_CASES
    )
    assert smoke_payload["execution"]["study_totals"]["fallback_commands"] == 0


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("privacy", "opponent_hand_exposures"), "opponent_hand_exposures"),
        (("incarnation", "object_ref_mismatches"), "object_ref_mismatches"),
        (("execution", "study_totals", "fallback_commands"), "fallback"),
    ],
)
def test_verifier_rejects_nonzero_safety_counters(
    smoke_payload: dict,
    path: tuple[str, ...],
    message: str,
) -> None:
    payload = copy.deepcopy(smoke_payload)
    cursor = payload
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = 1
    _rehash(payload)
    with pytest.raises(RuntimeError, match=message):
        bench.verify(
            payload,
            check_source=False,
            require_canonical=False,
            enforce_gates=False,
        )


def test_verifier_recomputes_latency_summaries(smoke_payload: dict) -> None:
    payload = copy.deepcopy(smoke_payload)
    payload["performance"]["sequential"]["summary"]["fork_ns"]["p95_ns"] += 1
    _rehash(payload)
    with pytest.raises(RuntimeError, match="latency summary mismatch"):
        bench.verify(
            payload,
            check_source=False,
            require_canonical=False,
            enforce_gates=False,
        )


def test_verifier_requires_consuming_failure_evidence(smoke_payload: dict) -> None:
    payload = copy.deepcopy(smoke_payload)
    unknown = next(
        case
        for case in payload["failures"]["cases"]
        if case["id"] == "unknown_offer_consumes_offer_set"
    )
    del unknown["offer_set_consumed"]
    _rehash(payload)
    with pytest.raises(RuntimeError, match="offer set remained reusable"):
        bench.verify(
            payload,
            check_source=False,
            require_canonical=False,
            enforce_gates=False,
        )


def test_checked_canonical_artifact_verifies() -> None:
    payload = json.loads(bench.DEFAULT_RAW.read_text(encoding="utf-8"))
    bench.verify(
        payload,
        check_source=True,
        require_canonical=True,
        enforce_gates=True,
    )
