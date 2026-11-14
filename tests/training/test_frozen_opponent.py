"""Frozen opponents retain their bytes and a pre-training attacker baseline."""

from pathlib import Path

import pytest

from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import file_sha256
from manabot.training.learning import Learning
from manabot.training.regime import FrozenOpponent, TrainingRegime, TrainSelfPlay
import managym


def _regime(path: Path) -> TrainingRegime:
    return TrainingRegime(
        id="frozen-proof",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=ModelConfig(hidden_dim=8, num_attention_heads=2),
        stages=[
            TrainSelfPlay(
                id="attack",
                operation="train_self_play",
                behavior="frozen",
                opponent=FrozenOpponent(path=str(path), sha256=file_sha256(path)),
                batch_closure="exact-quota",
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
