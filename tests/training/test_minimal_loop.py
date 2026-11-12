"""The example loop trains from the public pieces alone, end to end."""

from pathlib import Path

import torch

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.infra.config import ModelConfig
from manabot.training.loops.minimal import train


def test_minimal_loop_returns_a_moved_finite_average(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    tiny = ModelConfig(attention_on=False, hidden_dim=8)
    trained = train(2, streams=2, decisions=8, seed=3, model=tiny)
    torch.manual_seed(3)
    initial = ActorCritic(trained.observation_space, tiny)
    assert all(torch.isfinite(p).all() for p in trained.parameters())
    assert any(
        not torch.equal(after, before)
        for after, before in zip(
            trained.parameters(), initial.parameters(), strict=True
        )
    )

    # The guide's way to keep a run: bound to the decks it played.
    path = tmp_path / "policy.pt"
    save_checkpoint(
        trained, trained.observation_space, path, player_configs=Match().to_rust()
    )
    loaded, _ = load_checkpoint(str(path))
    for (name, kept), saved in zip(
        loaded.state_dict().items(), trained.state_dict().values(), strict=True
    ):
        assert torch.equal(kept, saved), name


def test_guide_shows_the_loop_as_written() -> None:
    root = Path(__file__).resolve().parents[2]
    loop = (root / "manabot/training/loops/minimal.py").read_text()
    assert loop in (root / "docs/training-loop.md").read_text()
