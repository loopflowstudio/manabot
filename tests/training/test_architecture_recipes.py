"""Recipe composition preserves controls and rejects unsupported value targets."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from experiments.runners.model_capacity import regimes as capacity_regimes
from experiments.runners.omitted_controls import resolve_contrast
from manabot.infra.hypers import AgentSpec
from manabot.training.models import Learning, TrainingRegime, TrainSelfPlay
from manabot.training.recipes import (
    ataraxos_baseline,
    value_outputs,
    with_agent,
    with_capacity,
    with_value_aggregation,
    with_value_output,
)

ROOT = Path(__file__).resolve().parents[2]


def _baseline() -> TrainingRegime:
    return TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/ataraxos-move-scalar.json").read_text()
    )


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
    with pytest.raises(ValidationError, match="requires ataraxos_move"):
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


def test_existing_value_contrast_uses_same_targets() -> None:
    contrast = resolve_contrast("paper-value-head")
    retained = _baseline()
    retained.id = "paper-value-head-control"
    retained.wall_seconds = 150
    for stage in retained.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = 2
        stage.execution.wall_seconds = 60
    assert contrast.baseline == retained
    expected = with_value_output(
        contrast.baseline, id="paper-value-head-treatment", output="categorical_wdl"
    )
    assert contrast.treatment == expected
    assert contrast.baseline.agent.value_kind == "scalar"


def test_capacity_examples_preserve_controls_and_snapshot_the_baseline() -> None:
    base = with_value_aggregation(_baseline(), id="token", aggregation="value_token")
    before = base.model_dump()
    arms = capacity_regimes(base)
    assert [(a.agent.hidden_dim, a.agent.attention_layers) for a in arms.values()] == [
        (64, 1),
        (64, 2),
        (128, 2),
    ]
    for arm in arms.values():
        assert arm.agent.value_aggregation == "value_token"
        assert arm.agent.semantic_pack == base.agent.semantic_pack
        assert arm.stages == base.stages
        assert arm.match == base.match
        assert arm.observation == base.observation
        assert arm.wall_seconds == base.wall_seconds
        assert TrainingRegime.model_validate_json(arm.model_dump_json()) == arm
    arms["w64-d1"].match.hero_deck.clear()
    assert arms["w128-d2"].match.hero_deck
    assert base.model_dump() == before


def test_architecture_variants_reject_unsupported_models() -> None:
    base = _baseline()
    with pytest.raises(ValidationError, match="divisible"):
        with_capacity(base, id="invalid", width=65, depth=2, heads=4)
    base.agent.attention_on = False
    with pytest.raises(ValidationError, match="require attention"):
        with_value_aggregation(base, id="invalid", aggregation="value_token")
    with pytest.raises(ValidationError, match="require attention"):
        with_capacity(base, id="invalid", width=64, depth=2, heads=4)
