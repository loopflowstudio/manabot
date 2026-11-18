"""The full-game harness must retain native replay and explicit incomplete runs."""

import json
from pathlib import Path

import pytest

from manabot.backends import resolve
from manabot.llm.schema import InferenceConfig
from manabot.llm.session import InferenceSession
from manabot.verify.llm_game import GameConfig, play
from tests.fixtures.llm_process import FirstReferenceRuntime


@pytest.mark.parametrize("decision_memory", [False, True])
def test_native_random_control_completes_and_replays(
    tmp_path: Path, decision_memory: bool
) -> None:
    """The decision log is opt-in, and a game with it on still replays exactly."""
    assert GameConfig().decision_memory is False
    summary = play(
        GameConfig(decision_memory=decision_memory),
        tmp_path / "game",
        None,
        backend=resolve("torch"),
    )
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
    summary = play(
        GameConfig(max_decisions=2), tmp_path / "game", None, backend=resolve("torch")
    )
    assert not summary.completed
    assert summary.failure is not None and "decision limit" in summary.failure
    assert summary.decisions == 2
    assert (tmp_path / "game" / "summary.json").is_file()


def test_game_receipts_bind_the_model_behavior_version(tmp_path: Path) -> None:
    config = InferenceConfig(
        model_id="test",
        model_revision="a" * 40,
        tokenizer_id="test",
        tokenizer_revision="a" * 40,
        device="cpu",
        precision="float32",
        context_tokens=64,
        max_new_tokens=4,
        deadline_seconds=5,
        fallback_seed=1,
    )
    with InferenceSession(
        config, factory=FirstReferenceRuntime, load_seconds=10
    ) as session:
        summary = play(
            GameConfig(max_decisions=8),
            tmp_path / "game",
            session,
            backend=resolve("torch"),
        )
    assert summary.model_decisions > 0
    assert summary.model_behavior_version == session.behavior_version
    assert summary.opponent_sha256 is None
    versions = {
        json.loads(path.read_text())["behavior_version"]
        for path in (tmp_path / "game").glob("decision-*/selection.json")
    }
    assert versions == {session.behavior_version}
