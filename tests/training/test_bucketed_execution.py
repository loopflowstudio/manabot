"""`Execution.bucketed_shapes` and `Execution.live_tokens`: serialization,
admission and the trained policy."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.recipes.baseline import baseline_test
from manabot.training import execution
from manabot.training.regime import (
    Execution,
    FrozenOpponent,
    Learning,
    Stage,
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


@pytest.mark.parametrize("field", ["bucketed_shapes", "live_tokens"])
def test_unset_field_is_omitted_and_keeps_regime_bytes(field: str) -> None:
    unset = baseline_test(2).regime
    assert field not in Execution().model_dump(mode="json")
    explicit = TrainingRegime.model_validate(_regime(**{field: False}))
    assert canonical_sha256(explicit.model_dump(mode="json")) == canonical_sha256(
        unset.model_dump(mode="json")
    )
    enabled = TrainingRegime.model_validate(_regime(**{field: True}))
    dumped = enabled.model_dump(mode="json")
    assert dumped["stages"][0]["execution"][field] is True
    assert TrainingRegime.model_validate(dumped) == enabled
    assert canonical_sha256(dumped) != canonical_sha256(unset.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("field", "name"),
    [("bucketed_shapes", "bucketed shapes"), ("live_tokens", "live tokens")],
)
def test_live_continuation_preserves_layout(field: str, name: str) -> None:
    data = _regime()
    stages = data["stages"]
    assert isinstance(stages, list)
    follow = dict(stages[0], id="continued", initial=stages[0]["id"])
    follow["execution"] = dict(stages[0]["execution"], **{field: True})
    stages.append(follow)
    with pytest.raises(ValidationError, match=name):
        TrainingRegime.model_validate(data)


def test_live_tokens_require_self_play() -> None:
    with pytest.raises(ValidationError, match="live tokens require train_self_play"):
        Stage.model_validate({"id": "plain", "execution": {"live_tokens": True}})


class _Built(Exception):
    pass


def test_frozen_opponent_runs_the_stage_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The learner and the frozen opponent are both built with the stage's setting."""
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
                execution=Execution(bucketed_shapes=True, live_tokens=True),
            )
        ],
    )
    opponents: list[tuple[bool, bool]] = []
    learners: list[tuple[bool, bool]] = []

    def load(
        checkpoint: str, *, bucketed_shapes: bool = False, live_tokens: bool = False
    ) -> tuple[ActorCritic, ObservationSpace]:
        agent, loaded = load_checkpoint(
            checkpoint, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
        )
        opponents.append((agent.bucketed_shapes, agent.live_tokens))
        return agent, loaded

    def build(
        observation_space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> ActorCritic:
        learners.append((bucketed_shapes, live_tokens))
        raise _Built

    monkeypatch.setattr(execution, "load_checkpoint", load)
    monkeypatch.setattr(execution, "ActorCritic", build)
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(_Built):
            execution.execute_regime(regime, 41, tmp_path / "run", store)
    assert opponents == [(True, True)]
    assert learners == [(True, True)]
