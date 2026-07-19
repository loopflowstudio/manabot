from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.runners import run_rul12_release_stack_budget as rul12

ROOT = Path(__file__).resolve().parents[2]


def test_contract_retains_rul9_workloads_budgets_and_measurement_boundary() -> None:
    contract = rul12.load_contract()

    assert contract["release"]["repetitions"] == 10
    assert contract["release"]["commands_per_game"] == 132
    assert contract["release"]["latency_boundary"] == (
        "TestClient WebSocket client send through accepted acknowledgment"
    )
    assert contract["release"]["summary_cache"] is False
    assert contract["release"]["terminal_persistence"] == (
        "synchronous and inside measured game"
    )
    assert contract["budgets"]["release"]["live_command_p95_ms_max"] == 100.0
    assert contract["budgets"]["release"]["live_games_per_second_min"] == 1.0
    assert contract["training"]["driver"] == "full_clone/current_game_v1"


def test_contract_rejects_budget_boundary_or_training_drift() -> None:
    for mutate, message in (
        (
            lambda contract: contract["budgets"]["release"].__setitem__(
                "live_command_p95_ms_max", 101.0
            ),
            "budgets",
        ),
        (
            lambda contract: contract["release"].__setitem__("summary_cache", True),
            "summary cache",
        ),
        (
            lambda contract: contract["training"].__setitem__("workers", 3),
            "training",
        ),
    ):
        contract = deepcopy(rul12.load_contract())
        mutate(contract)
        with pytest.raises(rul12.Rul12Error, match=message):
            rul12.validate_contract(contract)


def test_frozen_rul9_inputs_are_byte_and_artifact_bound_without_rerun() -> None:
    contract = rul12.load_contract()
    frozen = rul12.verify_frozen_inputs(contract)

    assert frozen["rul9_measurement_origin"]["artifact_sha256"] == (
        "498df1eda031f1d6ea68f72f792dfd68195bb10eab884b006aa7772b586564da"
    )
    assert all(entry["rerun"] is False for entry in frozen.values())


def test_checked_correctness_projection_keeps_all_semantic_and_privacy_gates() -> None:
    evidence = rul12.correctness_evidence(rul12.load_contract(), rerun=False)

    authored = evidence["authored_match_parity"]
    assert authored["summary"]["viewer_projection_checks"] == 798
    assert authored["summary"]["spectator_admitted"] is False
    assert authored["stale_object_proof"]["current_rejection"]["code"] == (
        "stale_object"
    )
    assert (
        authored["stale_object_proof"]["retained_command_rejection"]["code"]
        == "stale_revision"
    )

    public = evidence["public_commitment_parity"]
    assert public["summary"]["commitments"] == 62
    assert public["summary"]["rules_provider_gaps"] == 0
    assert len(set(public["surface_identity_stream_sha256"].values())) == 1
    assert all(
        proof["rejection"] == "RulesProviderGap"
        for proof in public["atomic_negative_proof"].values()
    )


@pytest.mark.parametrize(
    ("filename", "artifact_sha256"),
    (
        (
            "rul-12-release-stack-budget-v1.contended-host-load.json",
            "71641a8c10691dbf1fc7f07e819d3e6514b80630a1ea0bb92a5801f0e11d0ed9",
        ),
        (
            "rul-12-release-stack-budget-v1.control-recovered-then-contended.json",
            "c14f3a43b17840217113f9edeaf8e03f9c430e9a725a62222b57ed2fa47b1959",
        ),
    ),
)
def test_contended_run_is_preserved_as_an_honest_structural_miss(
    filename: str, artifact_sha256: str
) -> None:
    receipt = json.loads((ROOT / "experiments" / "data" / filename).read_text())
    contract = rul12.load_contract()

    verdict = rul12.verify_receipt(
        contract,
        receipt,
        check_current=False,
        require_pass=False,
    )

    assert receipt["artifact_sha256"] == artifact_sha256
    assert verdict["overall"] == "miss"
    assert verdict["capacity"]["status"] == "pass"
    assert verdict["fallbacks"]["status"] == "pass"
    with pytest.raises(rul12.Rul12Error, match="product budgets missed"):
        rul12.verify_receipt(
            contract,
            receipt,
            check_current=False,
            require_pass=True,
        )
