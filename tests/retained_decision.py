"""Replay retained choices through naming and explicit source-payment decisions."""

import json

from manabot.managym.decision import SEMANTIC_DECISION_VERSION


def assert_retained_frame(
    current_frame: str, retained_frame: str, *, added_payments: int = 0
) -> None:
    """Compare schema-7 fields with the exact count of inserted payment revisions."""
    current = json.loads(current_frame)
    retained = json.loads(retained_frame)
    assert retained.pop("schema_version") == 7
    assert current.pop("schema_version") == SEMANTIC_DECISION_VERSION
    assert current.pop("fingerprint") != retained.pop("fingerprint")
    assert current["revision"] == retained["revision"] + added_payments
    retained["revision"] = current["revision"]
    assert current == retained
