"""The closure and lambda studies play one held-out deal through the real arena."""

from pathlib import Path
from types import SimpleNamespace

from etude.eval.records import Code
import pytest
import torch

from experiments import learning_signal
from manabot.env.match import Match as GameMatch
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.sim.matchup import selected_match


def test_a_held_out_deal_scores_all_eight_deck_seat_legs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = ObservationSpace()
    torch.manual_seed(83)
    agent = ActorCritic(space, ModelConfig(world=learning_signal.PACK, hidden_dim=8))
    path = tmp_path / "fixture.pt"
    save_checkpoint(
        agent, space, path, player_configs=GameMatch(selected_match()).to_rust()
    )
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(learning_signal, "local_checkpoint_file", lambda _: path)
    match = SimpleNamespace(
        key="deal",
        games=8,
        conditions={"deal": learning_signal.DEALS[0]},
        players=(
            SimpleNamespace(name="candidate"),
            Code(name="heuristic", source=[Path(learning_signal.__file__)]),
        ),
    )
    tables = learning_signal.play_deal(match)  # type: ignore[arg-type]
    plays = tables["plays"]
    assert len(plays) == 16
    assert {row["matchup"] for row in plays} == {"LvA", "AvL", "LvL", "AvA"}
    assert all(row["valid"] for row in plays)


def test_scored_deals_are_outside_the_monitoring_namespace() -> None:
    assert not set(learning_signal.DEALS) & set(range(1_910_101_000, 1_910_101_025))
