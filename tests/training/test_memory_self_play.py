"""Self-play with decision memory: the decision clock and a control-to-memory start."""

from pathlib import Path

import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.recipes.builders import gray_ogre_match
from manabot.training.learning import (
    DampedPolicyLearning,
    IterationSchedule,
    LearnerDecisionSchedule,
)
from manabot.training.loops.self_play import CLOCKS, SelfPlay, create, self_play
import managym
from tests.fixtures.training_records import recording

MEMORY = ObservationConfig(
    max_decision_records=256,
    max_decision_reference_details=4096,
    max_decision_references=128,
    max_decision_members=2048,
)


def model(encoder: str, hidden_dim: int = 16) -> ModelConfig:
    return ModelConfig(
        world=managym.WORLD_SETUP,
        hidden_dim=hidden_dim,
        num_attention_heads=2,
        attention_on=True,
        policy_film=True,
        decision_memory=encoder,
    )


def settings(encoder: str = "none", **changes: object) -> SelfPlay:
    """A two-stream damped recipe whose control keeps the log without reading it."""
    declared: dict[str, object] = {
        "match": gray_ogre_match(),
        "observation": MEMORY,
        "model": model(encoder),
        "learning": DampedPolicyLearning(
            gradient="damped_policy",
            advantage_quantile=0,
            min_advantage=0,
            decision_schedule=LearnerDecisionSchedule(transitions=40),
        ),
        "streams": 2,
        "transitions": 8,
        "collection_rows": 2,
        "bootstrap_rows": 2,
        "learner_rows": 2,
    }
    return SelfPlay(**(declared | changes))


def train(
    recipe: SelfPlay, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> tuple[Path, int]:
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / name))
    with recording(
        experiment="test", run=name, clocks=CLOCKS, directory=tmp_path / f"{name}-track"
    ) as run:
        path = self_play(recipe, run, updates=50)
    return path, load_checkpoint(path)[1].encoder.config.max_decision_records


def test_rates_follow_decisions_independently_of_update_count() -> None:
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        decision_schedule=LearnerDecisionSchedule(transitions=1000),
    )
    assert learning.rates(1, learner_transitions=0) == (1e-4, 0.05)
    assert learning.rates(2, learner_transitions=500) == learning.rates(
        99, learner_transitions=500
    )
    assert learning.rates(20, learner_transitions=1000) == learning.rates(
        21, learner_transitions=1200
    )
    with pytest.raises(ValueError, match="completed transitions"):
        learning.rates(1)
    with pytest.raises(ValueError, match="nonnegative"):
        learning.rates(1, learner_transitions=-1)
    with pytest.raises(ValueError, match="choose an iteration"):
        DampedPolicyLearning(
            gradient="damped_policy",
            iteration_schedule=IterationSchedule(steps=100),
            decision_schedule=LearnerDecisionSchedule(transitions=1000),
        )
    assert (
        "decision_schedule"
        not in DampedPolicyLearning(gradient="damped_policy").model_dump()
    )


def test_memory_settings_refuse_incomplete_declarations() -> None:
    with pytest.raises(ValueError, match="explicit collection_rows"):
        settings(learner_rows=None)
    with pytest.raises(ValueError, match="all four decision capacities"):
        SelfPlay.model_validate(
            settings("gru").model_dump() | {"observation": ObservationConfig()}
        )
    with pytest.raises(ValueError, match="damped decision schedule"):
        SelfPlay.model_validate(
            settings(learner_transitions=40).model_dump()
            | {"learning": {"gradient": "damped_policy"}}
        )
    assert (
        "collection_rows"
        not in SelfPlay(
            match=gray_ogre_match(), learning={"gradient": "damped_policy"}
        ).model_dump()
    )


def test_decision_target_stops_the_loop_and_memory_branch_starts_from_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = settings(learner_transitions=40)
    trunk, records = train(control, tmp_path, monkeypatch, "trunk")
    assert records == MEMORY.max_decision_records
    published = load_checkpoint(trunk)[0]

    branch = settings("gru", pretrained=trunk)
    learner, _ = create(branch, seed=198)
    fresh = learner.agent.state_dict()
    saved = published.state_dict()
    memory = set(fresh) - set(saved)
    assert memory and all(
        name.startswith("decision_memory.")
        or name.endswith((".memory_gate", ".memory_projection.weight"))
        for name in memory
    )
    for name, value in saved.items():
        torch.testing.assert_close(fresh[name], value, rtol=0, atol=0)
    assert set(learner.agent.memory_gates().values()) == {0.0}
    assert not learner.optimizer.state

    # The branch trains through collection, bootstrap and the update with memory.
    train(
        branch.model_copy(update={"learner_transitions": 40}),
        tmp_path,
        monkeypatch,
        "branch",
    )

    with pytest.raises(ValueError, match="differ from requested arm"):
        create(
            settings("gru", pretrained=trunk).model_copy(
                update={"model": model("gru", 32)}
            ),
            seed=198,
        )


def test_update_ceiling_does_not_claim_the_decision_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = settings(learner_transitions=10_000)
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "short"))
    with (
        recording(
            experiment="test", run="short", clocks=CLOCKS, directory=tmp_path / "track"
        ) as run,
        pytest.raises(RuntimeError, match="before learner-decision target"),
    ):
        self_play(recipe, run, updates=1)
