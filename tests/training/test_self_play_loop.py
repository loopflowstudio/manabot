"""Self-play uses real Etude saves and preserves world-bound policy evidence."""

import json
from pathlib import Path

from etude.train.read import read
from etude.train.run import Stopped, track
import numpy as np
import pytest
import torch

from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rl import NumericalError
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.loops import self_play as loop
from manabot.training.loops.self_play import CLOCKS, Counts, SelfPlay, self_play
from manabot.training.rollout.collection import Collector, RolloutBatch
from manabot.training.updates.selection import UpdateDiagnostics

SEED = 197


def _settings(learning: Learning | DampedPolicyLearning) -> SelfPlay:
    values = gray_ogre_self_play(horizon=2).model_dump()
    values["learning"] = learning
    return SelfPlay.model_validate(values)


@pytest.mark.parametrize("averaging", [None, 0.9])
def test_resume_keeps_adam_ema_counts_and_immutable_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, averaging: float | None
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={
            "checkpoint_every": 1,
            "learning": Learning(horizon=2, epochs=1, minibatches=1, ema=averaging),
        }
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with track(
        experiment="test", run="policy", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        earlier = self_play(settings, run, updates=1)
    before = earlier.read_bytes()
    with track(
        experiment="test",
        run="policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        final = self_play(settings, run, updates=2)
    assert earlier.read_bytes() == before
    assert final.name == ("00000002-raw.pt" if averaging is None else "00000002-ema.pt")
    with track(
        experiment="test",
        run="policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        learner, ema = loop.create(settings, seed=run.seed)
        objects: dict[str, object] = {
            "learner": learner.agent,
            "optimizer": learner.optimizer,
        }
        if ema is not None:
            objects["ema"] = ema
        counts = run.resume(**objects, extra=Counts())
        assert counts.updates == 2 and counts.optimizer_updates > 0
        assert counts.training_seconds > 0 and learner.optimizer.state
        saved = torch.load(final.with_name("00000002-raw.pt"), weights_only=False)[
            "model_state_dict"
        ]
        for name, value in learner.agent.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    if ema is not None:
        averaged = torch.load(final, weights_only=False)["model_state_dict"]
        for name, value in ema.state_dict().items():
            torch.testing.assert_close(value, averaged[name], rtol=0, atol=0)
    recorded = read(tmp_path / "track", "test", "policy")
    assert list(recorded.points("coordinates/updates", "updates").x) == [1, 2]
    assert {cp.at["updates"] for cp in recorded.checkpoints} == {0, 1, 2}


def test_stop_keeps_registered_counts_at_the_completed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = gray_ogre_self_play(horizon=2)
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    stop = tmp_path / "stop"
    monkeypatch.setenv("ETUDE_STOP_FILE", str(stop))
    original_step = loop.step

    def stop_after_step(
        settings: SelfPlay,
        learner: Learner,
        ema: ActorCritic | None,
        collector: Collector,
        counts: Counts,
        rng: np.random.Generator,
        *,
        out: Path,
    ) -> dict[str, object]:
        row = original_step(settings, learner, ema, collector, counts, rng, out=out)
        stop.touch()
        return row

    with monkeypatch.context() as patch:
        patch.setattr(loop, "step", stop_after_step)
        with pytest.raises(Stopped):
            with track(
                experiment="test",
                run="stop",
                clocks=CLOCKS,
                directory=tmp_path / "track",
            ) as run:
                self_play(settings, run, updates=2)
    pointer = json.loads((tmp_path / "out/state/current.json").read_text())
    saved = json.loads(
        (tmp_path / "out/state" / pointer["save"] / "state.json").read_text()
    )
    assert saved["extra"]["updates"] == saved["at"]["updates"] == 1
    stop.unlink()
    with track(
        experiment="test",
        run="stop",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        assert self_play(settings, run, updates=2).name == "00000002-raw.pt"


@pytest.mark.parametrize("capture_failure", [False, True])
def test_numerical_failure_keeps_previous_policy_and_private_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_failure: bool
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={"learning": Learning(horizon=2, epochs=1, minibatches=1, ema=0.9)}
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    with track(
        experiment="test", run="failure", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        policy = self_play(settings, run, updates=1)
    original = policy.read_bytes()
    pointer = (tmp_path / "out/state/current.json").read_bytes()

    def fail(
        settings: SelfPlay,
        learner: Learner,
        ema: ActorCritic | None,
        counts: Counts,
        batch: RolloutBatch,
        rng: np.random.Generator,
    ) -> UpdateDiagnostics:
        raise NumericalError(
            "fixture_nonfinite", {"values": torch.tensor([float("nan")])}
        )

    monkeypatch.setattr(loop, "learn", fail)
    if capture_failure:
        (tmp_path / "out/numerical-incident").touch()
    with pytest.raises(NumericalError, match="fixture_nonfinite") as caught:
        with track(
            experiment="test",
            run="failure",
            clocks=CLOCKS,
            directory=tmp_path / "track",
            resume=True,
        ) as run:
            self_play(settings, run, updates=2)
    assert policy.read_bytes() == original
    assert (tmp_path / "out/state/current.json").read_bytes() == pointer
    assert not (tmp_path / "out/policies/00000002-raw.pt").exists()
    if capture_failure:
        assert any("incident capture failed" in note for note in caught.value.__notes__)
    else:
        receipt = next((tmp_path / "out/numerical-incident").glob("*/incident.json"))
        metadata = json.loads(receipt.read_text())
        assert metadata["iteration"] == 2 and metadata["completed_updates"] == 1
        assert "packed_actions/offsets" in metadata["tensors"]
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


def test_device_memory_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 12)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 45)
    cuda = loop.resource_metrics("cuda")
    assert cuda["memory/peak_allocated_cuda_bytes"] == 123
    assert cuda["memory/peak_reserved_cuda_bytes"] == 456
    mps = loop.resource_metrics("mps")
    assert mps["memory/allocated_mps_bytes"] == 12
    assert mps["memory/driver_allocated_mps_bytes"] == 45
