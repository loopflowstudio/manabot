"""Checked selected-match public-commitment provider closure."""

from etude.public_commitment_parity import (
    AUTHORITY_RECEIPT_SHA256,
    INT12_FIXTURE_SHA256,
    RUL9_MEASUREMENT_ARTIFACT_SHA256,
    verify_receipt,
)


def test_checked_public_commitment_receipt_closes_selected_trace() -> None:
    receipt = verify_receipt()

    assert receipt["summary"] == {
        "commands": 132,
        "commitments": 62,
        "unadmitted_commands": 70,
        "tracker_records_per_viewer": 132,
        "consumed_commitments": 62,
        "rules_provider_gaps": 0,
        "identity_stream_mismatches": 0,
        "negative_proof_mutations": 0,
    }
    assert receipt["identity"]["authority_receipt_sha256"] == (AUTHORITY_RECEIPT_SHA256)
    assert receipt["identity_stream"][29]["public_commitment"] == {
        "kind": "discard",
        "card": "Island",
    }
    assert receipt["identity_stream"][40]["public_commitment"] == {
        "kind": "decline_discard"
    }
    stream_hashes = {
        surface["identity_stream_sha256"] for surface in receipt["surfaces"].values()
    }
    assert len(stream_hashes) == 1
    assert {tracker["records"] for tracker in receipt["trackers"].values()} == {132}


def test_public_commitment_receipt_binds_hypothesis_atomicity_and_frozen_evidence() -> (
    None
):
    receipt = verify_receipt()

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
