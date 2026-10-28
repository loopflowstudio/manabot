"""Admitted checkpoints become trainable on the shared optimizer path."""

from pathlib import Path

import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.training.learning.learner import Learner


def test_checkpoint_admission_restores_training_mode(tmp_path: Path) -> None:
    space = ObservationSpace()
    path = tmp_path / "policy.pt"
    save_checkpoint(
        ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2)),
        space,
        path,
        player_configs=Match(MatchConfig()).to_rust(),
    )
    admitted, _ = load_checkpoint(str(path))
    assert not admitted.training
    learner = Learner(admitted, "cpu", minibatch_rows=2)
    assert all(module.training for module in learner.agent.modules())
    assert all(group["eps"] == 1e-5 for group in learner.optimizer.param_groups)
    assert isinstance(learner.optimizer, torch.optim.Adam)
