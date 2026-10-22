"""Builders compose cumulative stages, preserve controls and reject unsupported targets."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from experiments.ataraxos_move import ataraxos_move_scalar
from manabot.infra.hypers import AgentSpec
from manabot.training.builders import (
    ataraxos_baseline,
    move_checkpoints,
    ppo_checkpoints,
    value_outputs,
    with_agent,
    with_capacity,
    with_value_aggregation,
)
from manabot.training.models import Learning, TrainingRegime, TrainSelfPlay

ROOT = Path(__file__).resolve().parents[2]


def _baseline() -> TrainingRegime:
    return ataraxos_move_scalar()


def test_python_baseline_preserves_existing_resolved_recipe() -> None:
    original = _baseline()
    resolved = ataraxos_baseline(
        id=original.id,
        world=original.world,
        match=original.match,
        observation=original.observation,
        agent=original.agent,
        checkpoints=2,
        updates=1,
        transitions=64,
        streams=4,
        stage_seconds=80,
        wall_seconds=180,
    )
    assert resolved.model_dump() == original.model_dump()
    resolved.match.hero_sideboard.clear()
    resolved.agent.hidden_dim = 32
    assert original.match.hero_sideboard
    assert original.agent.hidden_dim == 16


def test_value_cross_has_independent_models_and_unchanged_controls() -> None:
    base = _baseline()
    base.id = "base"
    snapshot = base.model_dump()
    arms = value_outputs((base,), ("scalar", "categorical_wdl"))
    assert len(arms) == 2
    for name, arm in arms.items():
        assert arm.id == name
        assert arm.match == base.match
        assert arm.observation == base.observation
        assert arm.stages == base.stages
        assert arm.wall_seconds == base.wall_seconds
        assert TrainingRegime.model_validate_json(arm.model_dump_json()) == arm
    first = arms["base-scalar"]
    assert isinstance(first.stages[0], TrainSelfPlay)
    first.stages[0].execution.wall_seconds = 1
    first.match.hero_deck.clear()
    assert base.model_dump() == snapshot
    assert arms["base-categorical-wdl"].match.hero_deck
    assert arms["base-categorical-wdl"].stages == base.stages


def test_invalid_value_cross_fails_without_mutating_baseline() -> None:
    base = _baseline()
    with pytest.raises(ValueError, match="unique regime IDs"):
        value_outputs((base, base.model_copy(deep=True)), ("scalar",))
    for stage in base.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.learning = Learning()
    before = base.model_dump()
    with pytest.raises(ValidationError, match="requires damped_policy"):
        value_outputs((base,), ("scalar", "categorical_wdl"))
    assert base.model_dump() == before
    with pytest.raises(ValueError, match="nonempty"):
        value_outputs((base,), ("scalar", "scalar"))


def test_variations_revalidate_mutated_input_objects() -> None:
    base = _baseline()
    agent = AgentSpec()
    agent.compound_decisions = True
    with pytest.raises(ValidationError, match="compound"):
        with_agent(base, id="invalid", agent=agent)
    assert not base.agent.compound_decisions
    assert base.id == "ataraxos-move-scalar"


def test_architecture_variants_reject_unsupported_models() -> None:
    base = _baseline()
    with pytest.raises(ValidationError, match="divisible"):
        with_capacity(base, id="invalid", width=65, depth=2, heads=4)
    base.agent.attention_on = False
    with pytest.raises(ValidationError, match="require attention"):
        with_value_aggregation(base, id="invalid", aggregation="value_token")
    with pytest.raises(ValidationError, match="require attention"):
        with_capacity(base, id="invalid", width=64, depth=2, heads=4)


def test_checkpoint_builders_chain_stages_with_independent_settings() -> None:
    moves = move_checkpoints(3, updates=5, seconds=10)
    assert [(s.id, s.initial) for s in moves] == [
        ("policy-0", None),
        ("policy-1", "policy-0"),
        ("policy-2", "policy-1"),
    ]
    learning = Learning()
    stages = ppo_checkpoints(2, learning)
    first, second = stages
    assert isinstance(first, TrainSelfPlay) and isinstance(second, TrainSelfPlay)
    first.learning.max_grad_norm = 0.123
    assert second.learning.max_grad_norm == learning.max_grad_norm
    with pytest.raises(ValueError, match="checkpoint"):
        move_checkpoints(0, updates=1, seconds=1)
    with pytest.raises(ValueError, match="checkpoint"):
        ppo_checkpoints(0, learning)
