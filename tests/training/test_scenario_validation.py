"""Historical mechanisms do not silently become selected-world policy scores."""

<<<<<<< HEAD
from manabot.infra.hypers import MatchHypers
=======
from manabot.infra.config import MatchConfig
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from manabot.verify.scenario_validation import validate_scenarios


def test_historical_lines_resolve_but_selected_match_is_unsupported() -> None:
<<<<<<< HEAD
    records = validate_scenarios(selected_match=MatchHypers())
=======
    records = validate_scenarios(selected_match=MatchConfig())
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
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
