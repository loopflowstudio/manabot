from __future__ import annotations

from etude.authored_match_parity import ParityDivergence, verify_receipt


def test_checked_authored_match_parity_receipt() -> None:
    receipt = verify_receipt()
    assert receipt["summary"] == {
        "commands_per_surface": 132,
        "checkpoints_per_surface": 133,
        "ordered_transition_groups_per_surface": 132,
        "viewer_projection_checks": 798,
        "canonical_player_projections": 2,
        "spectator_admitted": False,
        "first_divergence": None,
    }
    stale = receipt["stale_object_proof"]
    assert stale["captured_render_ref"] == {"entity": 102, "incarnation": 2}
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
            "events": 61,
            "sha256": "1fe2a948748f34f6972c345c20a3d13d055dcda8865996d56d02d6643ec76bb9",
        },
        "1": {
            "events": 61,
            "sha256": "8594022c1bf96287d79b378f2295b7c48f1dcdda1913b09db8002496f4874aa1",
        },
    }


def test_first_divergence_names_both_source_identities() -> None:
    message = str(ParityDivergence("headless", 37, "state_witness", "a", "b"))
    assert "surface=headless from_revision=37 field=state_witness" in message
    assert "authority_receipt_sha256=" in message
    assert "relevant_source_sha256=" in message
    assert "unavailable" not in message
