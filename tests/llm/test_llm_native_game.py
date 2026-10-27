"""The full-game harness must retain native replay and explicit incomplete runs."""

from pathlib import Path

from manabot.verify.llm_game import GameConfig, play


def test_native_random_control_completes_and_replays(tmp_path: Path) -> None:
    summary = play(GameConfig(), tmp_path / "game", None)
    assert summary.completed and summary.failure is None
    assert summary.replay_matches
    assert summary.decisions > 100
    assert summary.model_decisions == 0
    assert summary.origins == {"random-control": summary.decisions}
    assert (
        len(list((tmp_path / "game").glob("decision-*/transition.json")))
        == summary.decisions
    )


def test_decision_limit_is_not_a_completed_game(tmp_path: Path) -> None:
    summary = play(GameConfig(max_decisions=2), tmp_path / "game", None)
    assert not summary.completed
    assert summary.failure is not None and "decision limit" in summary.failure
    assert summary.decisions == 2
    assert (tmp_path / "game" / "summary.json").is_file()
