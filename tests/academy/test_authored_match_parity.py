from __future__ import annotations

from academy.authored_match_parity import ParityDivergence, verify_receipt


def test_checked_authored_match_parity_receipt() -> None:
    receipt = verify_receipt()
    assert receipt["summary"] == {
        "commands_per_surface": 135,
        "checkpoints_per_surface": 136,
        "ordered_transition_groups_per_surface": 135,
        "viewer_projection_checks": 816,
        "canonical_player_projections": 2,
        "spectator_admitted": False,
        "first_divergence": None,
    }
    stale = receipt["stale_object_proof"]
    assert stale["captured_render_ref"] == {"entity": 96, "incarnation": 3}
    assert stale["current_rejection"] == {
        "code": "stale_object",
        "state_witness_unchanged": True,
        "semantic_event_cursor_unchanged": True,
    }
    assert stale["retained_command_rejection"] == {
        "code": "stale_revision",
        "state_witness_unchanged": True,
        "semantic_event_cursor_unchanged": True,
    }
    assert receipt["surfaces"]["live"]["presentation"] == {
        "0": {
            "events": 91,
            "sha256": "deaf7a86fa332dbd74e6da24109f5f4fa1b9c0548b624e0aef35dd35351c44fd",
        },
        "1": {
            "events": 91,
            "sha256": "e784cbdc7099108c94f0e7d2b121925206649e2273e11deed81115a86f69b084",
        },
    }


def test_first_divergence_names_both_source_identities() -> None:
    message = str(ParityDivergence("headless", 37, "state_witness", "a", "b"))
    assert "surface=headless from_revision=37 field=state_witness" in message
    assert "authority_receipt_sha256=" in message
    assert "relevant_source_sha256=" in message
    assert "unavailable" not in message


def test_divergence_survives_context_manager_unwinding() -> None:
    from contextlib import contextmanager

    import pytest

    @contextmanager
    def scope():
        yield

    with pytest.raises(ParityDivergence, match="state_witness"):
        with scope():
            raise ParityDivergence("headless", 37, "state_witness", "a", "b")
