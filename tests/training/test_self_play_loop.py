"""Self-play resume, immutable exports, refusal and failure evidence."""

from collections.abc import Sequence
import json
from pathlib import Path

from etude.stopping import Stopped
import numpy as np
import pytest
import torch

from manabot.backends.torch.rl import NumericalError
from manabot.infra.config import ModelConfig
from manabot.recipes.builders import gray_ogre_match
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.loops import self_play as loop
from manabot.training.loops.self_play import LoopState, SelfPlay, load_state, self_play
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates.selection import UpdateDiagnostics

SEED = 197


def _settings(learning: Learning | DampedPolicyLearning) -> SelfPlay:
    return SelfPlay(
        match=gray_ogre_match(),
        model=ModelConfig(hidden_dim=16, num_attention_heads=2),
        learning=learning,
        streams=2,
        transitions=16,
    )


def test_a_rerun_resumes_and_a_finished_loop_returns_at_once(tmp_path: Path) -> None:
    settings = _settings(
        Learning(horizon=2, epochs=1, minibatches=1, ema=0.9)
    ).model_copy(update={"checkpoint_every": 1})
    out = tmp_path / "run"
    earlier = self_play(settings, out, seed=SEED, updates=1)
    original = earlier.read_bytes()

    final = self_play(settings, out, seed=SEED, updates=2)

    state = load_state(out / "state.pt", settings, SEED)
    assert state is not None and state.update == 2
    assert sorted(p.name for p in (out / "checkpoints").glob("*-raw.pt")) == [
        "00000000-raw.pt",
        "00000001-raw.pt",
        "00000002-raw.pt",
    ]
    assert earlier.read_bytes() == original
    # A different launch target must not change the same first update's rates.
    self_play(settings, tmp_path / "uninterrupted", seed=SEED, updates=2)
    for kind in ("raw", "ema"):
        name = f"00000001-{kind}.pt"
        first = torch.load(out / "checkpoints" / name, weights_only=False)
        other = torch.load(
            tmp_path / "uninterrupted/checkpoints" / name, weights_only=False
        )
        for key, value in first["model_state_dict"].items():
            torch.testing.assert_close(
                value, other["model_state_dict"][key], rtol=0, atol=0
            )
    assert final.name == "00000002-ema.pt"
    written = final.stat().st_mtime_ns
    assert self_play(settings, out, seed=SEED, updates=2) == final
    assert final.stat().st_mtime_ns == written


def test_a_state_for_other_settings_or_seed_is_refused(tmp_path: Path) -> None:
    settings = _settings(Learning(horizon=2, epochs=1, minibatches=1))
    self_play(settings, tmp_path, seed=SEED, updates=1)

    with pytest.raises(ValueError, match="another seed"):
        load_state(tmp_path / "state.pt", settings, SEED + 1)
    changed = settings.model_copy(update={"transitions": 32})
    with pytest.raises(ValueError, match="transitions: saved 16, requested 32"):
        load_state(tmp_path / "state.pt", changed, SEED)
    changed = settings.model_copy(
        update={"learning": settings.learning.model_copy(update={"policy_lambda": 0.8})}
    )
    with pytest.raises(
        ValueError, match=r"learning.policy_lambda: saved 0.95, requested 0.8"
    ):
        load_state(tmp_path / "state.pt", changed, SEED)
    changed = settings.model_copy(
        update={"learning": settings.learning.model_copy(update={"horizon": 3})}
    )
    with pytest.raises(ValueError, match=r"learning.horizon: saved 2, requested 3"):
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

    def checkpoint(
        self, path: Path, /, *, tags: Sequence[str] = (), **clocks: float
    ) -> None:
        self.checkpoints.append(path)

    def stopping(self) -> bool:
        return len(self.updates) >= 1


def test_a_stop_saves_at_an_update_boundary_and_the_next_call_finishes(
    tmp_path: Path,
) -> None:
    settings = _settings(Learning(horizon=2, epochs=1, minibatches=1, ema=0.9))
    tracker = _StopAfterOne()

    with pytest.raises(Stopped) as stopped:
        self_play(settings, tmp_path, seed=SEED, updates=3, run=tracker)

    assert "update 1" in str(stopped.value) and tracker.updates == [1]
    final = self_play(settings, tmp_path, seed=SEED, updates=3)
    state = load_state(tmp_path / "state.pt", settings, SEED)
    assert final.exists() and state is not None and state.update == 3


@pytest.mark.parametrize("capture_failure", [False, True])
def test_numerical_failure_retains_diagnostics_without_replacing_valid_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_failure: bool
) -> None:
    settings = _settings(Learning(horizon=2, epochs=1, minibatches=1, ema=0.9))
    policy = self_play(settings, tmp_path, seed=SEED, updates=1)
    original = {path: path.read_bytes() for path in (policy, tmp_path / "state.pt")}

    def fail(
        state: LoopState, batch: RolloutBatch, rng: np.random.Generator
    ) -> UpdateDiagnostics:
        raise NumericalError(
            "fixture_nonfinite", {"values": torch.tensor([float("nan")])}
        )

    monkeypatch.setattr(loop, "learn", fail)
    if capture_failure:
        # A regular file prevents creating the diagnostic directory.
        (tmp_path / "numerical-incident").touch()
    with pytest.raises(NumericalError, match="fixture_nonfinite") as caught:
        self_play(settings, tmp_path, seed=SEED, updates=2)
    for path, content in original.items():
        assert path.read_bytes() == content
    assert not (tmp_path / "checkpoints/00000002-raw.pt").exists()
    if capture_failure:
        assert any("incident capture failed" in note for note in caught.value.__notes__)
        return
    receipt = next((tmp_path / "numerical-incident").glob("*/incident.json"))
    metadata = json.loads(receipt.read_text())
    assert metadata["iteration"] == 2 and metadata["completed_updates"] == 1
    assert metadata["invariant"] == "fixture_nonfinite"
    assert "packed_actions/offsets" in metadata["tensors"]
    evidence = torch.load(receipt.with_name("tensors.pt"), weights_only=True)
    assert torch.isnan(evidence["values"]).any()
    diagnostic = torch.load(receipt.with_name("state.pt"), weights_only=False)
    assert diagnostic["optimizer"]["state"] and diagnostic["ema"]


def test_live_ppo_requires_a_horizon_but_retained_records_omit_it() -> None:
    learning = Learning()
    assert "horizon" not in learning.model_dump()
    assert "horizon" not in learning.model_dump_json()
    with pytest.raises(ValueError, match="PPO self-play requires a learning horizon"):
        _settings(learning)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            Learning.model_validate({"horizon": invalid})
    fixed = Learning(horizon=2)
    assert Learning.model_validate_json(fixed.model_dump_json()) == fixed
    assert fixed.learning_rate.at(1) == fixed.learning_rate.at(2)
    assert fixed.tau.at(1) == fixed.tau.at(2)
