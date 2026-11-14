"""Checked selected-match public-commitment provider closure."""

import json
from pathlib import Path
from typing import Any

import pytest

from conformance import public_commitment_parity
from conformance.public_commitment_parity import (
    AUTHORITY_RECEIPT_SHA256,
    INT12_FIXTURE_SHA256,
    RUL9_MEASUREMENT_ARTIFACT_SHA256,
    verify_receipt,
)
from manabot.belief.range import BeliefState
import managym
from managym.possible_worlds import PossibleWorldSpace


@pytest.mark.experiment_data(
    "experiments/data/rul-9-played-workloads-v1.measurement.json"
)
def test_checked_public_commitment_receipt_closes_selected_trace(
    checked_receipt: dict[str, Any],
) -> None:
    receipt = checked_receipt

    assert receipt["summary"] == {
        "commands": 135,
        "commitments": 66,
        "unadmitted_commands": 69,
        "tracker_records_per_viewer": {"0": 68, "1": 135},
        "consumed_commitments": 53,
        "rules_provider_gaps": 0,
        "identity_stream_mismatches": 0,
        "negative_proof_mutations": 0,
    }
    assert receipt["identity"]["authority_receipt_sha256"] == (AUTHORITY_RECEIPT_SHA256)
    assert receipt["identity_stream"][30]["public_commitment"] == {
        "kind": "discard",
        "card": "Island",
    }
    assert receipt["identity_stream"][102]["public_commitment"] == {
        "kind": "decline_learn"
    }
    stream_hashes = {
        surface["identity_stream_sha256"] for surface in receipt["surfaces"].values()
    }
    assert len(stream_hashes) == 1
    assert {
        viewer: tracker["records"] for viewer, tracker in receipt["trackers"].items()
    } == {"0": 68, "1": 135}


@pytest.mark.experiment_data(
    "experiments/data/rul-9-played-workloads-v1.measurement.json"
)
def test_public_commitment_receipt_binds_hypothesis_atomicity_and_frozen_evidence(
    checked_receipt: dict[str, Any],
) -> None:
    receipt = checked_receipt
    assert public_commitment_parity._frozen_evidence() == receipt["frozen_evidence"]

    materialized = receipt["materialized_hypothesis"]
    assert materialized["support_size"] == 484
    assert materialized["public_commitment"] == {
        "kind": "discard",
        "card": "Island",
    }
    assert materialized["source_match_witness_unchanged"]
    assert materialized["source_semantic_event_cursor_unchanged"]
    assert materialized["source_viewer_observation_unchanged"]
    for proof in receipt["atomic_negative_proof"].values():
        assert proof["rejection"] == "RulesProviderGap"
        assert all(value is True for key, value in proof.items() if key != "rejection")

    frozen = receipt["frozen_evidence"]
    assert frozen["rul9_measurement"]["artifact_sha256"] == (
        RUL9_MEASUREMENT_ARTIFACT_SHA256
    )
    assert frozen["rul9_measurement"]["rerun"] is False
    assert frozen["int12_fixture"]["file_sha256"] == INT12_FIXTURE_SHA256
    assert frozen["int12_fixture"]["rewritten"] is False


def test_historical_source_normalization_is_comparison_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checked = {
        "identity": {"relevant_source": {"sha256": "historical"}},
        "summary": {"commitments": 62, "rules_provider_gaps": 0},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(checked))
    monkeypatch.setattr(public_commitment_parity, "RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(
        public_commitment_parity,
        "build_receipt",
        lambda: {
            "identity": {"relevant_source": {"sha256": "current"}},
            "summary": {"commitments": 62, "rules_provider_gaps": 0},
        },
    )
    verify_receipt.cache_clear()
    try:
        assert verify_receipt() == checked
    finally:
        verify_receipt.cache_clear()


def test_historical_source_normalization_keeps_non_source_fields_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checked = {
        "identity": {"relevant_source": {"sha256": "historical"}},
        "summary": {"commitments": 62, "rules_provider_gaps": 0},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(checked))
    monkeypatch.setattr(public_commitment_parity, "RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(
        public_commitment_parity,
        "build_receipt",
        lambda: {
            "identity": {"relevant_source": {"sha256": "current"}},
            "summary": {"commitments": 61, "rules_provider_gaps": 1},
        },
    )
    verify_receipt.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="receipt is stale"):
            verify_receipt()
    finally:
        verify_receipt.cache_clear()


@pytest.fixture
def checked_receipt() -> dict[str, Any]:
    return json.loads(public_commitment_parity.RECEIPT_PATH.read_text())


def test_live_and_persisted_public_commitments_match(
    checked_receipt: dict[str, Any],
) -> None:
    decisions = public_commitment_parity._load_authority()["decisions"]
    live, replay = public_commitment_parity._run_live(decisions)
    assert [row.command.model_dump(mode="json") for row in replay.decisions] == [
        row["command"] for row in decisions
    ]
    assert live.pop("identity_stream") == checked_receipt["identity_stream"]
    assert live == checked_receipt["surfaces"]["live"]
    replay_rows = [
        {**row, "command": persisted.command.model_dump(mode="json")}
        for row, persisted in zip(decisions, replay.decisions, strict=True)
    ]
    persisted, _ = public_commitment_parity._run_engine(
        replay_rows, track_beliefs=False
    )
    assert persisted.pop("identity_stream") == checked_receipt["identity_stream"]
    assert persisted == checked_receipt["surfaces"]["persisted_replay"]


@pytest.mark.parametrize("viewer", [0, 1])
def test_public_commitment_tracker_matches_retained_viewer(
    checked_receipt: dict[str, Any],
    viewer: int,
) -> None:
    # Independent audits keep the complete retained prefix for each viewer.
    decisions = public_commitment_parity._load_authority()["decisions"]
    headless, discard_root = public_commitment_parity._run_engine(
        decisions, track_beliefs=True, viewers=(viewer,)
    )
    assert headless["identity_stream"] == checked_receipt["identity_stream"]
    for key in (
        "commands",
        "commitments",
        "identity_stream_sha256",
        "rules_provider_gaps",
    ):
        assert headless[key] == checked_receipt["surfaces"]["headless"][key]
    retained = checked_receipt["trackers"][str(viewer)]
    assert headless["tracker_receipts"] == {str(viewer): retained}
    assert headless["likelihood_identity_audits"] == retained["consumed_commitments"]
    if viewer == 1:
        assert discard_root is not None
        assert (
            headless["atomic_negative_proof"]
            == checked_receipt["atomic_negative_proof"]
        )
        assert (
            public_commitment_parity._materialized_discard_proof(discard_root)
            == checked_receipt["materialized_hypothesis"]
        )


def test_identity_audit_rejects_another_native_root() -> None:
    roots: list[managym.Env] = []
    for forests in (8, 9):
        env = managym.Env(seed=0)
        env.reset(
            [
                managym.PlayerConfig("Hero", {"Mountain": 8, "Gray Ogre": 4}),
                managym.PlayerConfig(
                    "Villain", {"Forest": forests, "Llanowar Elves": 4}
                ),
            ]
        )
        roots.append(env)
    belief = BeliefState.compatible_prior(PossibleWorldSpace.from_engine(roots[0], 0))
    likelihood = public_commitment_parity.IdentityAuditLikelihood()
    with pytest.raises(managym.AgentError, match="space identity mismatch"):
        likelihood.evaluate(
            roots[1],
            viewer=0,
            commitment={"kind": "decline_learn"},
            belief=belief,
        )
    assert likelihood.calls == 0
