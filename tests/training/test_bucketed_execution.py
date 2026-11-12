"""`Execution.bucketed_shapes`: serialization, admission and the trained policy."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.recipes.baseline import baseline_test
from manabot.training import execution
from manabot.training.regime import (
    Execution,
    FrozenOpponent,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
import managym


def _regime(**stage_execution: object) -> dict[str, object]:
    data = baseline_test(2).regime.model_dump(mode="json")
    stages = data["stages"]
    assert isinstance(stages, list)
    stages[0]["execution"].update(stage_execution)
    return data


def test_unset_field_is_omitted_and_keeps_regime_bytes() -> None:
    unset = baseline_test(2).regime
    assert "bucketed_shapes" not in Execution().model_dump(mode="json")
    explicit = TrainingRegime.model_validate(_regime(bucketed_shapes=False))
    assert canonical_sha256(explicit.model_dump(mode="json")) == canonical_sha256(
        unset.model_dump(mode="json")
    )
    enabled = TrainingRegime.model_validate(_regime(bucketed_shapes=True))
    dumped = enabled.model_dump(mode="json")
    assert dumped["stages"][0]["execution"]["bucketed_shapes"] is True
    assert TrainingRegime.model_validate(dumped) == enabled
    assert canonical_sha256(dumped) != canonical_sha256(unset.model_dump(mode="json"))


def test_pooled_token_stream_rejects_bucketed_shapes() -> None:
    data = _regime(bucketed_shapes=True)
    agent = data["agent"]
    assert isinstance(agent, dict)
    agent["token_stream"] = "pooled"
    with pytest.raises(ValidationError, match="does not support bucketed shapes"):
        TrainingRegime.model_validate(data)


def test_live_continuation_preserves_bucketed_shapes() -> None:
    data = _regime()
    stages = data["stages"]
    assert isinstance(stages, list)
    follow = dict(stages[0], id="continued", initial=stages[0]["id"])
    follow["execution"] = dict(stages[0]["execution"], bucketed_shapes=True)
    stages.append(follow)
    with pytest.raises(ValidationError, match="bucketed shapes"):
        TrainingRegime.model_validate(data)


class _Built(Exception):
    pass


def test_stage_execution_builds_the_trained_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The self-play learner is built with the stage's setting."""
    regime = TrainingRegime.model_validate(_regime(bucketed_shapes=True))
    assert isinstance(regime.stages[0], TrainSelfPlay)
    built: list[bool] = []

    def build(
        observation_space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
    ) -> ActorCritic:
        built.append(bucketed_shapes)
        raise _Built

    monkeypatch.setattr(execution, "ActorCritic", build)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(_Built):
            execution.execute_regime(regime, 197, tmp_path / "run", store)
    assert built == [True]


def test_frozen_opponent_runs_the_stage_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen opponent's collection policy follows the learner's setting."""
    model = ModelConfig(hidden_dim=8, num_attention_heads=2)
    space = ObservationSpace()
    path = tmp_path / "opponent.pt"
    save_checkpoint(
        ActorCritic(space, model),
        space,
        path,
        player_configs=Match(MatchConfig()).to_rust(),
    )
    regime = TrainingRegime(
        id="frozen-bucketed",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=model,
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
                execution=Execution(bucketed_shapes=True),
            )
        ],
    )
    opponents: list[bool] = []
    learners: list[bool] = []

    def load(
        checkpoint: str, *, bucketed_shapes: bool = False
    ) -> tuple[ActorCritic, ObservationSpace]:
        agent, loaded = load_checkpoint(checkpoint, bucketed_shapes=bucketed_shapes)
        opponents.append(agent.bucketed_shapes)
        return agent, loaded

    def build(
        observation_space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
    ) -> ActorCritic:
        learners.append(bucketed_shapes)
        raise _Built

    monkeypatch.setattr(execution, "load_checkpoint", load)
    monkeypatch.setattr(execution, "ActorCritic", build)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(_Built):
            execution.execute_regime(regime, 41, tmp_path / "run", store)
    assert opponents == [True]
    assert learners == [True]
