"""Typed native capacity errors preserve retained ordinary error identities."""

from manabot.env.replay import matches_retained_error
import managym


def test_retained_agent_error_matches_typed_capacity_subclass() -> None:
    capacity = managym.CapacityError(
        "combat_solver capacity exhausted: limit 1, visited 2"
    )
    for error in (managym.AgentError("ordinary rejection"), capacity):
        for retained_class in ("AgentError", "PyAgentError"):
            assert matches_retained_error(error, retained_class, str(error)) is True
        assert matches_retained_error(error, "NativePanic", str(error)) is False
        assert matches_retained_error(error, "AgentError", "different error") is False
    assert type(capacity).__name__ == "CapacityError"
