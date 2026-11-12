"""The self-play loop: same-seed parity with the regime executor, resume and refusal."""

from pathlib import Path

import pytest
import torch

from manabot.infra.config import ModelConfig
from manabot.recipes.builders import gray_ogre_match
from manabot.training.execution import execute_regime
from manabot.training.loops.self_play import (
    SelfPlay,
    Stopped,
    every,
    load_state,
    self_play,
)
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.verify.store import VerifyStore

SEED = 197


def _settings(learning: Learning | DampedPolicyLearning) -> SelfPlay:
    return SelfPlay(
        match=gray_ogre_match(),
        model=ModelConfig(hidden_dim=16, num_attention_heads=2),
        learning=learning,
        streams=2,
        transitions=16,
    )


def _weights(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]


def _assert_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


@pytest.mark.parametrize(
    "learning",
    [
        Learning(epochs=1, minibatches=1, ema=0.9),
        DampedPolicyLearning(gradient="damped_policy"),
    ],
    ids=["ppo", "damped"],
)
def test_loop_matches_the_executor_with_the_same_seed(
    tmp_path: Path, learning: Learning | DampedPolicyLearning
) -> None:
    settings = _settings(learning)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        record = execute_regime(
            settings.regime(id="parity", updates=2), SEED, tmp_path / "executor", store
        )
    assert record.status == "completed", record.error

    final = self_play(settings, tmp_path / "loop", seed=SEED, updates=2)

    raw = torch.load(tmp_path / "loop" / "state.pt", weights_only=False)["raw"]
    _assert_equal(raw, _weights(tmp_path / "executor" / "self-play-raw.pt"))
    _assert_equal(_weights(final), _weights(tmp_path / "executor" / "self-play-ema.pt"))


def test_a_rerun_resumes_and_a_finished_loop_returns_at_once(tmp_path: Path) -> None:
    settings = _settings(Learning(epochs=1, minibatches=1, ema=0.9)).model_copy(
        update={"milestones": every(1)}
    )
    out = tmp_path / "run"
    self_play(settings, out, seed=SEED, updates=1)
    (out / "final.pt").unlink()

    final = self_play(settings, out, seed=SEED, updates=2)

    state = load_state(out / "state.pt", settings, SEED)
    assert state is not None and state.update == 2
    assert sorted(p.name for p in (out / "checkpoints").glob("*-raw.pt")) == [
        "00000000-raw.pt",
        "00000001-raw.pt",
        "00000002-raw.pt",
    ]
    written = final.stat().st_mtime_ns
    assert self_play(settings, out, seed=SEED, updates=2) == final
    assert final.stat().st_mtime_ns == written


def test_a_state_for_other_settings_or_seed_is_refused(tmp_path: Path) -> None:
    settings = _settings(Learning(epochs=1, minibatches=1))
    self_play(settings, tmp_path, seed=SEED, updates=1)

    with pytest.raises(ValueError, match="another seed"):
        load_state(tmp_path / "state.pt", settings, SEED + 1)
    changed = settings.model_copy(update={"transitions": 32})
    with pytest.raises(ValueError, match="another settings"):
        load_state(tmp_path / "state.pt", changed, SEED)
    moved = settings.model_copy(update={"threads": 2})
    assert load_state(tmp_path / "state.pt", moved, SEED) is not None


class _StopAfterOne:
    """A tracker that asks the loop to stop once one update is logged."""

    def __init__(self) -> None:
        self.updates: list[float] = []
        self.checkpoints: list[Path] = []

    def log(self, values: dict[str, object], /, **clocks: float) -> None:
        self.updates.append(clocks["updates"])

    def log_table(
        self, name: str, rows: list[dict[str, object]], /, **clocks: float
    ) -> None:
        pass

    def checkpoint(self, path: Path, /, **clocks: float) -> None:
        self.checkpoints.append(path)

    def stopping(self) -> bool:
        return len(self.updates) >= 1


def test_a_stop_saves_at_an_update_boundary_and_the_next_call_finishes(
    tmp_path: Path,
) -> None:
    settings = _settings(Learning(epochs=1, minibatches=1, ema=0.9))
    tracker = _StopAfterOne()

    with pytest.raises(Stopped) as stopped:
        self_play(settings, tmp_path, seed=SEED, updates=3, run=tracker)

    assert stopped.value.update == 1 and tracker.updates == [1]
    final = self_play(settings, tmp_path, seed=SEED, updates=3)
    state = load_state(tmp_path / "state.pt", settings, SEED)
    assert final.exists() and state is not None and state.update == 3
