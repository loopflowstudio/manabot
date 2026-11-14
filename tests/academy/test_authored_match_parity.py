from __future__ import annotations

import json
from pathlib import Path

import pytest

from conformance import authored_match_parity
from conformance.authored_match_parity import ParityDivergence, verify_receipt


def test_checked_authored_match_parity_receipt() -> None:
    receipt = verify_receipt()
    assert receipt["summary"] == {
        "commands_per_surface": 182,
        "checkpoints_per_surface": 183,
        "ordered_transition_groups_per_surface": 182,
        "viewer_projection_checks": 1098,
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
            "events": 130,
            "sha256": "791b93d678f4b774020ad12fef9fc9524f6dfe2a3d48d35fd0c580d45109afb6",
        },
        "1": {
            "events": 130,
            "sha256": "5b8f7ef8273485c19f76be99f2668c3d06c1006ae99ece55e7d31f140e0292d4",
        },
    }


def test_first_divergence_names_both_source_identities() -> None:
    message = str(ParityDivergence("headless", 37, "state_witness", "a", "b"))
    assert "surface=headless from_revision=37 field=state_witness" in message
    assert "authority_receipt_sha256=" in message
    assert "relevant_source_sha256=" in message
    assert "unavailable" not in message


@pytest.mark.parametrize("commands", [135, 134])
def test_relocated_source_preserves_receipt_and_rejects_behavior_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, commands: int
) -> None:
    checked = {
        "identity": {"relevant_source": {"sha256": "historical"}},
        "summary": {"commands_per_surface": 135},
    }
    receipt_path = tmp_path / "receipt.json"
    original = json.dumps(checked).encode()
    receipt_path.write_bytes(original)
    monkeypatch.setattr(authored_match_parity, "RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(
        authored_match_parity,
        "build_receipt",
        lambda: {
            "identity": {"relevant_source": {"sha256": "relocated"}},
            "summary": {"commands_per_surface": commands},
        },
    )
    if commands == 135:
        assert verify_receipt() == checked
    else:
        with pytest.raises(RuntimeError, match="receipt is stale"):
            verify_receipt()
    assert receipt_path.read_bytes() == original


def test_divergence_survives_context_manager_unwinding() -> None:
    from contextlib import contextmanager

    import pytest

    @contextmanager
    def scope():
        yield

    with pytest.raises(ParityDivergence, match="state_witness"):
        with scope():
            raise ParityDivergence("headless", 37, "state_witness", "a", "b")
