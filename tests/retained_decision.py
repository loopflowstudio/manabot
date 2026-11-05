"""Exact replay comparison across the naming decision schema transition."""

import json

from manabot.managym.decision import SEMANTIC_DECISION_VERSION


def assert_retained_frame(current_frame: str, retained_frame: str) -> None:
    """Preserve every schema-7 field except the declared version/fingerprint change."""
    current = json.loads(current_frame)
    retained = json.loads(retained_frame)
    assert retained.pop("schema_version") == 7
    assert current.pop("schema_version") == SEMANTIC_DECISION_VERSION
    assert current.pop("fingerprint") != retained.pop("fingerprint")
    assert current == retained
