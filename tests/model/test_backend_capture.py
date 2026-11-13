"""Real captures replay the production update's flattened targets and loss."""

import json
from pathlib import Path

import numpy as np
import torch

from manabot.backends.capture import capture
from manabot.backends.parity import backend, backend_module, minibatch_loss, minibatches
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.policy.checkpoint import from_torch_state
from manabot.training.regime import (
    DampedPolicyLearning,
    Execution,
    TrainingRegime,
    TrainSelfPlay,
)
import managym


def test_capture_replays_both_seat_update(tmp_path: Path) -> None:
    regime = TrainingRegime(
        id="capture-test",
        world=managym.WORLD_VERSION,
        match=MatchConfig(
            hero_deck={"Mountain": 8, "Gray Ogre": 8},
            villain_deck={"Forest": 8, "Llanowar Elves": 8},
        ),
        agent=ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            value_kind="scalar",
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
        stages=[
            TrainSelfPlay(
                id="fit",
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=4,
                learning_seats="both",
                execution=Execution(
                    device="cpu",
                    threads=1,
                    bucketed_shapes=True,
                    memory_bytes=2 * 2**30,
                    wall_seconds=120,
                ),
                learning=DampedPolicyLearning(
                    gradient="damped_policy",
                    min_advantage=0,
                    advantage_quantile=0,
                    max_grad_norm=0.01,
                ),
            )
        ],
        wall_seconds=120,
    )
    out = tmp_path / "capture"
    capture(out, regime, 197, experiment="capture-test")
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
