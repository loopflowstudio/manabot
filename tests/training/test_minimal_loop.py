"""The example loop trains from the public pieces alone, end to end."""

from pathlib import Path
from urllib.parse import urlparse

from etude.train.read import read
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.training.loops.minimal import MinimalConfig, train


def test_minimal_loop_publishes_moved_finite_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    tiny = ModelConfig(attention_on=False, hidden_dim=8)
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("ETUDE_EXPERIMENT", "minimal")
    monkeypatch.setenv("ETUDE_RUN", "minimal-seed-3")
    monkeypatch.setenv("ETUDE_SEED", "3")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    train(MinimalConfig(updates=2, streams=2, decisions=8, model=tiny))
    recorded = read(tmp_path / "track", "minimal", "minimal-seed-3")
    assert len(recorded.checkpoints) == 1
    checkpoint = recorded.checkpoints[0]
    # The minimal example publishes ordinary framework weights in one folder.
    folder = Path(urlparse(checkpoint.artifact.uri).path)
    weights = torch.load(folder / "policy.pt", weights_only=True)
    torch.manual_seed(3)
    initial = ActorCritic(ObservationSpace(), tiny)
    assert all(torch.isfinite(value).all() for value in weights.values())
    assert any(
        not torch.equal(value, initial.state_dict()[name])
        for name, value in weights.items()
    )


def test_guide_shows_the_loop_as_written() -> None:
    root = Path(__file__).resolve().parents[2]
    loop = (root / "manabot/training/loops/minimal.py").read_text()
    assert loop in (root / "docs/training-loop.md").read_text()
