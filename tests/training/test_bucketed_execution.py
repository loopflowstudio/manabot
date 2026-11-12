"""`Execution.bucketed_shapes`: serialization, admission and the trained policy."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.baseline import baseline_test
from manabot.training import execution
from manabot.training.regime import Execution, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore


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
