"""Frozen opponents retain their bytes and a pre-training attacker baseline."""

from pathlib import Path

import pytest
import torch

from manabot.arena.models import file_sha256
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentHypers, MatchHypers
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.models import (
    FrozenOpponent,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
import managym


def _regime(path: Path) -> TrainingRegime:
    return TrainingRegime(
        id="frozen-proof",
        world=managym.WORLD_VERSION,
        match=MatchHypers(),
        agent=AgentHypers(hidden_dim=8, num_attention_heads=2),
        stages=[
            TrainSelfPlay(
                id="attack",
                operation="train_self_play",
                behavior="frozen",
                opponent=FrozenOpponent(path=str(path), sha256=file_sha256(path)),
                streams=2,
                updates=1,
                transitions=8,
                learning=Learning(epochs=1, minibatches=1, retained_fraction=1),
            )
        ],
    )


def test_frozen_contract_rejects_missing_opponent_and_changed_continuation(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires exactly"):
        TrainSelfPlay(id="attack", operation="train_self_play", behavior="frozen")
    path = tmp_path / "opponent.pt"
    path.write_bytes(b"fixture")
    regime = _regime(path)
    first = regime.stages[0]
    assert isinstance(first, TrainSelfPlay)
    continuation = first.model_copy(update={"id": "next", "initial": "attack"})
    assert continuation.opponent is not None
    continuation.opponent = FrozenOpponent(path=str(path), sha256="f" * 64)
    with pytest.raises(ValueError, match="preserve"):
        TrainingRegime.model_validate(
            regime.model_copy(update={"stages": [first, continuation]}).model_dump()
        )


def test_frozen_run_preserves_opponent_and_initial_attacker(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    path = tmp_path / "opponent.pt"
    space = ObservationSpace()
    opponent = Agent(space, AgentHypers(hidden_dim=8, num_attention_heads=2))
    save_bc_checkpoint(
        opponent, space, path, player_configs=Match(MatchHypers()).to_rust()
    )
    before = file_sha256(path)
    regime = _regime(path)
    first = regime.stages[0]
    assert isinstance(first, TrainSelfPlay)
    regime.stages.append(first.model_copy(update={"id": "next", "initial": "attack"}))
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 41, tmp_path / "run", store)
    assert run.status == "completed"
    assert file_sha256(path) == before
    assert run.stages[0].inputs["frozen_opponent"]["sha256"] == before
    assert "initial_raw" not in run.stages[1].artifacts
    initial, _ = load_checkpoint_agent(run.stages[0].artifacts["initial_raw"]["path"])
    trained, _ = load_checkpoint_agent(run.stages[1].artifacts["raw"]["path"])
    assert any(
        not torch.equal(initial.state_dict()[key], value)
        for key, value in trained.state_dict().items()
    )
    assert all(stage.learner_transitions == 16 for stage in run.stages)
    assert run.selected_artifact == run.stages[-1].artifacts["raw"]


def test_compound_opponent_is_rejected_before_training(tmp_path: Path) -> None:
    space = ObservationSpace()
    opponent = Agent(
        space, AgentHypers(hidden_dim=8, num_attention_heads=2, compound_decisions=True)
    )
    path = tmp_path / "compound.pt"
    save_bc_checkpoint(
        opponent, space, path, player_configs=Match(MatchHypers()).to_rust()
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(ValueError, match="compound opponent submissions"):
            execute_regime(_regime(path), 41, tmp_path / "run", store)
