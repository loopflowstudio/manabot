"""Bounded evidence for held-out fitting, immutable bindings and exact scoring."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from manabot.belief.range import BeliefState
from manabot.belief.sampling import (
    AutoregressiveBeliefSampler,
    SamplerInput,
    SamplerSchema,
)
from manabot.belief.sampling_data import (
    SamplerDataset,
    SamplerExample,
    SamplerGame,
    Split,
)
from manabot.belief.sampling_fit import (
    compare_exact_reference,
    evaluate_sampler,
    evaluate_sampler_cohort,
    fit_belief_sampler,
    load_belief_sampler,
    save_belief_sampler,
)
from managym.possible_worlds import PossibleWorldSpace


def _dataset() -> SamplerDataset:
    schema = SamplerSchema("fixture-v1", ("A", "B"), 1, max_count=2)
    games: list[SamplerGame] = []
    splits: tuple[Split, ...] = ("train", "validation", "test")
    for index, split in enumerate(splits):
        rows = tuple(
            SamplerExample(
                SamplerInput(
                    schema.vocabulary_identity, (2, 2), (0, 0), 1, (float(sign),)
                ),
                (1, 0) if sign > 0 else (0, 1),
                0,
                position,
                f"obs-{index}-{position}",
            )
            for position, sign in enumerate((-1, 1))
        )
        games.append(SamplerGame(f"game-{index}", index, 0, split, rows))
    return SamplerDataset(schema, "frozen-policy", "frozen-world", tuple(games))


def test_fit_generalizes_public_history_and_reloads_bound_bytes(tmp_path: Path) -> None:
    dataset = _dataset()
    result = fit_belief_sampler(
        dataset,
        steps=50,
        batch_size=8,
        hidden_size=8,
        learning_rate=0.03,
        evaluation_samples=16,
        seed=7,
    )
    assert result.optimizer_exposures == 400
    assert (
        result.metrics.test.learned.joint_nll
        < result.metrics.test.physical_baseline.joint_nll - 0.3
    )
    assert result.metrics.test.learned.support_violations == 0
    assert result.metrics.test.learned.sampled_hands == 32
    assert result.metrics.parameter_bytes > 0
    assert result.metrics.test.learned.peak_python_bytes > 0
    checkpoint = tmp_path / "belief.pt"
    identity = save_belief_sampler(checkpoint, result, dataset)
    bindings = {
        "expected_dataset_identity": dataset.identity,
        "expected_policy_identity": dataset.policy_identity,
        "expected_world_identity": dataset.world_identity,
        "expected_schema_identity": dataset.schema.identity,
        "expected_checkpoint_identity": identity,
    }
    loaded = load_belief_sampler(checkpoint, **bindings)
    rows = dataset.games[2].examples
    inputs = [row.inputs for row in rows]
    targets = torch.tensor([row.target_hand for row in rows])
    torch.testing.assert_close(
        loaded.log_prob(inputs, targets),
        result.model.log_prob(inputs, targets),
        rtol=0,
        atol=0,
    )
    for key in bindings:
        with pytest.raises(ValueError, match="identity mismatch"):
            load_belief_sampler(checkpoint, **{**bindings, key: "changed"})
    with pytest.raises(FileExistsError):
        save_belief_sampler(checkpoint, result, dataset)
    with pytest.raises(ValueError, match="different dataset"):
        save_belief_sampler(
            tmp_path / "wrong.pt", result, replace(dataset, policy_identity="foreign")
        )


def test_held_out_labels_do_not_change_fit_and_dropout_is_training_only() -> None:
    dataset = _dataset()
    changed = replace(
        dataset,
        games=tuple(
            game
            if game.split == "train"
            else replace(
                game,
                examples=tuple(
                    replace(row, target_hand=tuple(reversed(row.target_hand)))
                    for row in game.examples
                ),
            )
            for game in dataset.games
        ),
    )
    first = fit_belief_sampler(
        dataset,
        steps=3,
        batch_size=4,
        hidden_size=4,
        history_dropout=0.5,
        evaluation_samples=2,
        seed=4,
    )
    second = fit_belief_sampler(
        changed,
        steps=3,
        batch_size=4,
        hidden_size=4,
        history_dropout=0.5,
        evaluation_samples=2,
        seed=4,
    )
    for name, tensor in first.model.state_dict().items():
        torch.testing.assert_close(
            tensor, second.model.state_dict()[name], rtol=0, atol=0
        )
    first.model.train()
    a = evaluate_sampler(first.model, dataset.games[2].examples, samples=4, seed=10)
    b = evaluate_sampler(first.model, dataset.games[2].examples, samples=4, seed=10)
    assert a.learned.joint_nll == b.learned.joint_nll
    assert a.learned.inclusion_brier == b.learned.inclusion_brier
    assert first.model.training


def test_budget_callback_stops_before_fit() -> None:
    def expired() -> None:
        raise TimeoutError("bounded attempt exhausted")

    with pytest.raises(TimeoutError, match="bounded attempt"):
        fit_belief_sampler(_dataset(), steps=100, check=expired)


def test_exact_reference_scores_joint_posterior_without_refitting() -> None:
    dataset = _dataset()
    model = AutoregressiveBeliefSampler(dataset.schema, hidden_size=4)
    reference = BeliefState.from_probabilities(
        PossibleWorldSpace.from_fixture(
            viewer=0,
            source_revision=0,
            source_viewer_state_hash="root",
            pool={"A": 2, "B": 2},
            hands=[({"A": 1}, 2), ({"B": 1}, 2)],
        ),
        "exact-tracker",
        (0.8, 0.2),
    )
    result = compare_exact_reference(
        model, dataset.games[0].examples[0].inputs, reference
    )
    assert result.learned_cross_entropy == pytest.approx(result.physical_cross_entropy)
    assert result.learned_kl == pytest.approx(result.physical_kl)
    assert result.learned_reference_support_mass == pytest.approx(1.0)
    assert result.reference_identity == reference.digest
    with pytest.raises(ValueError, match="physical domain"):
        compare_exact_reference(
            model,
            replace(dataset.games[0].examples[0].inputs, pool_counts=(3, 2)),
            reference,
        )


def test_foreign_baiting_is_identified_and_safe_baseline_retained() -> None:
    dataset = _dataset()
    result = fit_belief_sampler(
        dataset,
        steps=35,
        batch_size=8,
        hidden_size=8,
        learning_rate=0.03,
        evaluation_samples=4,
        seed=7,
    )
    # Synthetic adversarial policy reverses the training history/hand relation.
    bait = tuple(
        replace(row, target_hand=tuple(reversed(row.target_hand)))
        for row in dataset.games[2].examples
    )
    evidence = evaluate_sampler_cohort(
        result,
        bait,
        evaluation_policy_identity="synthetic-baiting-policy",
        cohort_identity="synthetic-opposite-history-v1",
        world_identity=dataset.world_identity,
        adversarial_selection=True,
        samples=8,
    )
    assert evidence.distribution_shift and evidence.adversarial_selection
    assert evidence.training_policy_identity == dataset.policy_identity
    assert (
        evidence.metrics.learned.joint_nll
        > evidence.metrics.physical_baseline.joint_nll
    )
    assert evidence.metrics.physical_baseline.support_violations == 0
    with pytest.raises(ValueError, match="world differs"):
        evaluate_sampler_cohort(
            result,
            bait,
            evaluation_policy_identity="foreign",
            cohort_identity="cohort",
            world_identity="different",
        )
