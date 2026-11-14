"""A published policy supplies weights, while a fresh loop owns Adam and counters."""

from pathlib import Path

from etude.train.tracking import track
import pytest
import torch

from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import CLOCKS, create, self_play


def test_pretrained_weights_start_with_fresh_optimizer_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = gray_ogre_self_play(horizon=1)
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "parent"))
    with track(
        experiment="test", run="pretrained", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        path = self_play(settings, run, updates=1)
    learner, _ = create(settings.model_copy(update={"pretrained": path}), seed=198)
    saved = torch.load(path, weights_only=False)["model_state_dict"]
    for name, value in learner.agent.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    assert not learner.optimizer.state
    changed = settings.model_copy(
        update={
            "model": settings.model.model_copy(update={"hidden_dim": 32}),
            "pretrained": path,
        }
    )
    with pytest.raises(ValueError, match="other model or observation"):
        create(changed, seed=198)
