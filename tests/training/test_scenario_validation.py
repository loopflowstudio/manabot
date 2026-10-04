"""Historical mechanisms do not silently become selected-world policy scores."""

from manabot.infra.hypers import MatchHypers
from manabot.verify.scenario_validation import validate_scenarios


def test_historical_lines_resolve_but_selected_match_is_unsupported() -> None:
    records = validate_scenarios(selected_match=MatchHypers())
    assert len(records) == 5
    for record in records:
        assert record.status == "unsupported_selected_setup"
        assert not record.selected_setup_supported
        assert record.content_sha256 is not None
        assert {attempt.line for attempt in record.attempts} == {
            "reference",
            "contrast",
        }
        assert all(
            attempt.expected_outcome and attempt.error is None
            for attempt in record.attempts
        )
        assert record.attempts[0].outcome != record.attempts[1].outcome
