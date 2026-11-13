"""A published policy supplies weights, while a fresh loop owns Adam and counters."""

from pathlib import Path

import pytest
import torch

from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import new_state, self_play


def test_pretrained_weights_start_with_fresh_optimizer_and_counts(
    tmp_path: Path,
) -> None:
    settings = gray_ogre_self_play()
    path = self_play(settings, tmp_path / "parent", seed=197, updates=1)
    state = new_state(settings.model_copy(update={"pretrained": path}), 198)
    saved = torch.load(path, weights_only=False)["model_state_dict"]
    for name, value in state.learner.agent.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    assert state.update == 0 and state.counts.optimizer_updates == 0
    assert not state.learner.optimizer.state
    changed = settings.model_copy(
        update={
            "model": settings.model.model_copy(update={"hidden_dim": 32}),
            "pretrained": path,
        }
    )
    with pytest.raises(ValueError, match="other model or observation"):
        new_state(changed, 198)
