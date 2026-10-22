"""Keep grid contrasts on the baseline and bind budgets to actual experience."""

from pathlib import Path

import pytest

from experiments.runners.batching_grid import (
    CAPACITIES,
    STREAMS,
    recipe,
    transition_checkpoints,
)
from manabot.training.models import AtaraxosMoveLearning, TrainingRegime, TrainSelfPlay


def test_grid_preserves_baseline_model_and_learning_at_four_streams() -> None:
    original = TrainingRegime.model_validate_json(
        Path("experiments/regimes/early-learning-rate-mini-source.json").read_text()
    )
    actual = recipe(CAPACITIES[0], 4, 131_072)
    assert actual.agent == original.agent
    assert actual.match == original.match
    assert actual.observation == original.observation
    assert actual.stages[0].learning == original.stages[0].learning
    assert actual.stages[0].matchup_curriculum == "cross-balanced"
    assert actual.world == "w6"


def test_all_grid_cells_have_equal_experience_and_schedule_endpoints() -> None:
    for capacity in CAPACITIES:
        for streams in STREAMS:
            spec = recipe(capacity, streams, 131_072)
            stage = spec.stages[0]
            assert isinstance(stage, TrainSelfPlay)
            assert stage.updates * stage.streams * stage.transitions == 131_072
            assert isinstance(stage.learning, AtaraxosMoveLearning)
            reference = recipe(CAPACITIES[0], 4, 131_072).stages[0].learning
            assert isinstance(reference, AtaraxosMoveLearning)
            assert stage.learning.rates(1) == reference.rates(1)
            assert stage.learning.rates(stage.updates) == reference.rates(512)
            assert spec.agent.value_aggregation == "masked_mean"


def test_grid_rejects_fractional_or_empty_iterations() -> None:
    for budget in (0, 1, 131_071):
        with pytest.raises(ValueError):
            recipe(CAPACITIES[0], 512, budget)


def test_width_depth_contrasts_preserve_all_other_training_settings() -> None:
    capacities = {capacity.name: capacity for capacity in CAPACITIES}
    for original_name, added_name, depth in (
        ("small", "small-deep", 4),
        ("large", "large-shallow", 2),
    ):
        original = recipe(capacities[original_name], 128, 131_072)
        added = recipe(capacities[added_name], 128, 131_072)
        original.id = added.id
        original.agent.attention_layers = depth
        assert added == original


def test_checkpoints_compare_exact_experience_across_stream_counts() -> None:
    milestones = (32_768, 65_536, 131_072)
    for streams in STREAMS:
        updates = transition_checkpoints(streams, 131_072, milestones)
        assert tuple(n * streams * 64 for n in updates) == milestones[:-1]


@pytest.mark.parametrize(
    "milestones", [(0,), (32_769,), (65_536, 32_768), (32_768, 32_768), (262_144,)]
)
def test_checkpoint_milestones_cannot_silently_change_experience(
    milestones: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        transition_checkpoints(512, 131_072, milestones)


def test_batched_baseline_is_the_tested_small_64_stream_recipe() -> None:
    checked_in = TrainingRegime.model_validate_json(
        Path("experiments/regimes/batched-baseline.json").read_text()
    ).model_dump()
    rebuilt = recipe(CAPACITIES[0], 64, 524_288, seconds=1740).model_dump()
    assert checked_in == rebuilt
    stage = checked_in["stages"][0]
    assert (
        checked_in["agent"]["hidden_dim"],
        checked_in["agent"]["attention_layers"],
    ) == (64, 2)
    assert (stage["streams"], stage["transitions"], stage["updates"]) == (64, 64, 128)
    assert stage["learning"]["iteration_schedule"] == {
        "steps": 128,
        "reference_steps": 2048,
    }
