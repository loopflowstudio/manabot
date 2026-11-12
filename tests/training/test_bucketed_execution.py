"""`Execution.bucketed_shapes` and `Execution.live_tokens`: `auto` resolution,
serialization, admission and the trained policy."""

from pathlib import Path
from typing import Literal

from pydantic import ValidationError
import pytest

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.recipes.baseline import baseline_test
from manabot.recipes.builders import with_agent
from manabot.training import execution
from manabot.training.regime import (
    Execution,
    FrozenOpponent,
    Layout,
    Learning,
    Stage,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
import managym

LAYOUT = ("bucketed_shapes", "live_tokens")


def _regime(**stage_execution: object) -> dict[str, object]:
    data = baseline_test(2).regime.model_dump(mode="json")
    stages = data["stages"]
    assert isinstance(stages, list)
    stages[0]["execution"].update(stage_execution)
    return data


def _layout(regime: TrainingRegime, stage: int = 0) -> tuple[Layout, Layout]:
    execution = regime.stages[stage].execution
    return execution.bucketed_shapes, execution.live_tokens


def _fresh(device: Literal["cpu", "cuda"], **agent: object) -> TrainingRegime:
    """A regime authored in code, as recipes build them, so `auto` applies."""
    regime = baseline_test(2).regime
    stage = regime.stages[0]
    return TrainingRegime(
        **{
            **regime.model_dump(exclude={"stages", "agent"}),
            "agent": regime.agent.model_copy(update=agent),
            "stages": [
                stage.model_copy(
                    update={
                        "execution": Execution(
                            device=device, wall_seconds=stage.execution.wall_seconds
                        )
                    }
                )
            ],
        }
    )


def test_auto_resolves_on_for_cuda_self_play_and_records_it() -> None:
    cuda = _fresh("cuda")
    assert _layout(cuda) == (True, True)
    dumped = cuda.model_dump(mode="json")
    execution = dumped["stages"][0]["execution"]
    assert {name: execution[name] for name in LAYOUT} == dict.fromkeys(LAYOUT, True)
    assert TrainingRegime.model_validate(dumped) == cuda


def test_auto_resolves_off_on_cpu_and_keeps_regime_bytes() -> None:
    cpu = _fresh("cpu")
    assert _layout(cpu) == (False, False)
    dumped = cpu.model_dump(mode="json")
    assert not set(LAYOUT) & set(dumped["stages"][0]["execution"])
    written = TrainingRegime.model_validate(_regime(**dict.fromkeys(LAYOUT, False)))
    assert canonical_sha256(dumped) == canonical_sha256(written.model_dump(mode="json"))


def test_auto_resolves_off_under_the_pooled_stream() -> None:
    assert _layout(_fresh("cuda", token_stream="pooled")) == (False, False)


def test_auto_resolves_off_for_other_operations() -> None:
    stage = Stage(id="plain")
    assert stage.execution.bucketed_shapes == "auto"
    for name in LAYOUT:
        with pytest.raises(ValueError, match="auto is resolved by the regime"):
            getattr(stage.execution, f"resolved_{name}")


def test_saved_regime_omitting_the_layout_reads_off() -> None:
    """A run.json saved before the default trained padded, even on CUDA."""
    saved = _fresh("cuda").model_dump(mode="json")
    execution = saved["stages"][0]["execution"]
    for name in LAYOUT:
        del execution[name]
    assert _layout(TrainingRegime.model_validate(saved)) == (False, False)


def test_builders_resolve_auto_again_and_keep_written_values() -> None:
    cuda = _fresh("cuda")
    pooled = with_agent(
        cuda,
        id="pooled",
        agent=cuda.agent.model_copy(update={"token_stream": "pooled"}),
    )
    assert _layout(pooled) == (False, False)
    assert _layout(with_agent(pooled, id="tables", agent=cuda.agent)) == (True, True)
    data = _fresh("cuda").model_dump()
    data["stages"][0]["execution"].update(dict.fromkeys(LAYOUT, False))
    padded = TrainingRegime.model_validate(data)
    assert _layout(with_agent(padded, id="copy", agent=padded.agent)) == (False, False)


def test_live_tokens_require_cuda() -> None:
    with pytest.raises(ValidationError, match="live tokens require CUDA"):
        Execution(device="cpu", live_tokens=True)
    with pytest.raises(ValidationError, match="live tokens require CUDA"):
        Execution(device="mps", live_tokens=True)
    assert Execution(device="cpu", bucketed_shapes=True).bucketed_shapes is True


@pytest.mark.parametrize(
    ("field", "name"),
    [("bucketed_shapes", "bucketed shapes"), ("live_tokens", "live tokens")],
)
def test_pooled_token_stream_rejects_layout(field: str, name: str) -> None:
    data = _regime(device="cuda", **{field: True})
    agent = data["agent"]
    assert isinstance(agent, dict)
    agent["token_stream"] = "pooled"
    with pytest.raises(ValidationError, match=f"does not support {name}"):
        TrainingRegime.model_validate(data)


def test_continuation_resolves_auto_to_its_parent_layout() -> None:
    regime = _fresh("cuda")
    parent = regime.stages[0]
    follow = parent.model_copy(
        update={
            "id": "continued",
            "initial": parent.id,
            "execution": Execution(
                device="cuda", wall_seconds=parent.execution.wall_seconds
            ),
        }
    )
    continued = TrainingRegime(
        **{
            **regime.model_dump(exclude={"stages", "agent"}),
            "agent": regime.agent,
            "stages": [parent, follow],
        }
    )
    assert _layout(continued, 1) == (True, True)


@pytest.mark.parametrize(
    ("field", "name"),
    [("bucketed_shapes", "bucketed shapes"), ("live_tokens", "live tokens")],
)
def test_live_continuation_preserves_layout(field: str, name: str) -> None:
    data = _fresh("cuda").model_dump(mode="json")
    stages = data["stages"]
    assert isinstance(stages, list)
    follow = dict(stages[0], id="continued", initial=stages[0]["id"])
    follow["execution"] = dict(stages[0]["execution"], **{field: False})
    stages.append(follow)
    with pytest.raises(ValidationError, match=name):
        TrainingRegime.model_validate(data)


def test_live_tokens_require_self_play() -> None:
    with pytest.raises(ValidationError, match="live tokens require train_self_play"):
        Stage.model_validate(
            {"id": "plain", "execution": {"device": "cuda", "live_tokens": True}}
        )


class _Built(Exception):
    pass


def test_frozen_opponent_runs_the_stage_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The learner and the frozen opponent are both built with the stage's layout.

    A frozen opponent trains on CPU only, where live tokens are refused.
    """
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
    assert opponents == [(True, False)]
    assert learners == [(True, False)]
