"""Real captures replay the production update's flattened targets and loss."""

import json
from pathlib import Path

from etude.train.tracking import track
import numpy as np
import pytest
import torch

from manabot.backends.parity import backend, backend_module, minibatch_loss, minibatches
from manabot.backends.torch.capture import capture
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.policy.checkpoint import from_torch_state
from manabot.training.learning import DampedPolicyLearning
from manabot.training.loops.self_play import CLOCKS, SelfPlay
import managym


def test_capture_replays_both_seat_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = SelfPlay(
        match=MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        model=ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            value_kind="scalar",
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
        streams=2,
        transitions=4,
        bucketed_shapes=True,
        learning=DampedPolicyLearning(
            gradient="damped_policy",
            min_advantage=0,
            advantage_quantile=0,
            max_grad_norm=0.01,
        ),
    )
    out = tmp_path / "capture"
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "run"))
    monkeypatch.setenv("ETUDE_SEED", "197")
    with track(
        experiment="capture-test", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        capture(out, settings, run)
    meta = json.loads((out / "capture.json").read_text())
    assert meta["calls"] > 0 and meta["minibatches"] > 1
    assert meta["engine"] >= 0
    agent, space = load_checkpoint(str(out / "initial.pt"), bucketed_shapes=True)
    weights = from_torch_state(
        agent.config,
        space.encoder.config,
        {k: v.numpy() for k, v in agent.state_dict().items()},
    )
    policy, learner = backend("torch").load(weights)
    rows = 0
    for obs, targets, settings, recorded in minibatches(out, None):
        assert targets.actions.shape == targets.advantages.shape
        assert targets.value_targets.shape == targets.actions.shape
        assert len(targets.actions) <= 2
        loss = minibatch_loss(
            backend_module("torch"),
            policy,
            learner,
            policy.prepare(obs),
            targets,
            settings,
        )
        np.testing.assert_allclose(
            float(loss.detach()), recorded["loss"], atol=1e-5, rtol=1e-4
        )
        learner.set_learning_rate(settings.learning_rate)
        norm = learner.step(loss, settings.max_grad_norm)
        assert torch.isfinite(norm)
        rows += len(targets.actions)
    assert rows == meta["learner_decisions"]
