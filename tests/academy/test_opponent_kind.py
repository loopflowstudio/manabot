"""An opponent is named with how it decides, wherever Academy shows it."""

from managym.academy.opponent import opponent_kind


def test_scripted_policies_have_no_artifact() -> None:
    assert (
        opponent_kind({"artifacts": {}, "config": {"kind": "heuristic"}}) == "scripted"
    )


def test_a_checkpoint_names_its_world_when_recorded() -> None:
    details = [{"label": "Run", "value": "r"}, {"label": "World", "value": "w17"}]
    assert opponent_kind({"artifacts": {"actor": "ab"}, "details": details}) == (
        "checkpoint w17"
    )
    unrecorded = [{"label": "World", "value": "not recorded"}]
    assert opponent_kind({"artifacts": {"actor": "ab"}, "details": unrecorded}) == (
        "checkpoint"
    )


def test_search_over_a_policy_is_search() -> None:
    assert opponent_kind({"artifacts": {}, "base_policy": {"name": "x"}}) == "search"
