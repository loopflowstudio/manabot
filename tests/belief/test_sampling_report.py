"""Offline report acceptance uses fixed weights and labels, never optimizer steps."""

from collections.abc import Sequence
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from manabot.belief.sampling import (
    AutoregressiveBeliefSampler,
    SamplerInput,
    SamplerSchema,
)
from manabot.belief.sampling_cohort import CohortSpec, FitAttempt, report_sampler_cohort
from manabot.belief.sampling_data import (
    SamplerDataset,
    SamplerExample,
    SamplerGame,
    Split,
    save_dataset,
)
from manabot.belief.sampling_fit import CalibrationBin, SamplerArmMetrics
from manabot.belief.sampling_report import _pool_games, main, report_saved_sampler


def _dataset(width: int = 2, capacity: int = 2) -> SamplerDataset:
    schema = SamplerSchema(
        "synthetic-counts", tuple(f"card-{i}" for i in range(width)), 1, capacity
    )
    row = SamplerExample(
        SamplerInput(
            schema.vocabulary_identity,
            (capacity,) * width,
            (1,) + (0,) * (width - 1),
            capacity,
            (0.5,),
        ),
        (capacity,) + (0,) * (width - 1),
        0,
        0,
        "synthetic-observation",
    )
    splits: tuple[Split, ...] = ("train", "validation", "test")
    return SamplerDataset(
        schema,
        "fixture-policy",
        "fixture-world",
        tuple(
            SamplerGame(f"game-{i}", i, 0, split, (row,))
            for i, split in enumerate(splits)
        ),
    )


def _checkpoint(path: Path, dataset: SamplerDataset, seed: int = 19) -> str:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AutoregressiveBeliefSampler(dataset.schema, hidden_size=4)
        # Fixed nonzero corrections exercise learned inference without training.
        with torch.no_grad():
            model.correction.bias.copy_(
                torch.linspace(-0.4, 0.4, dataset.schema.max_count + 1)
            )
    torch.save(
        {
            "format": "manabot.autoregressive-belief-sampler/v1",
            "schema": asdict(dataset.schema),
            "schema_identity": dataset.schema.identity,
            "dataset_identity": dataset.identity,
            "policy_identity": dataset.policy_identity,
            "world_identity": dataset.world_identity,
            "hidden_size": 4,
            "history_dropout": 0.0,
            "state_dict": model.state_dict(),
            "metrics": {"seed": seed},
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statistical(report: object) -> object:
    if isinstance(report, dict):
        return {
            key: _statistical(value)
            for key, value in report.items()
            if key not in {"sampling_seconds", "peak_python_bytes"}
        }
    if isinstance(report, list):
        return [_statistical(value) for value in report]
    return report


def test_saved_cli_reproduces_statistics_and_refuses_overwrite(tmp_path: Path) -> None:
    dataset = _dataset()
    data, checkpoint = tmp_path / "data.json", tmp_path / "sampler.pt"
    save_dataset(dataset, data)
    digest = _checkpoint(checkpoint, dataset)
    args = [
        "--dataset",
        str(data),
        "--training-dataset",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        digest,
        "--samples",
        "4",
        "12",
    ]
    outputs: list[dict[str, object]] = []
    for index in range(2):
        target = tmp_path / f"report-{index}.json"
        main([*args, "--out", str(target)])
        outputs.append(json.loads(target.read_text()))
    assert _statistical(outputs[0]) == _statistical(outputs[1])
    report = report_saved_sampler(
        data,
        checkpoint=checkpoint,
        checkpoint_identity=digest,
        training_dataset_path=data,
        sample_counts=(4, 12),
    )
    assert report.policy_relationship == "same_policy"
    assert len(report.measurements) == 6
    for measurement in report.measurements:
        assert measurement.game_ids
        assert measurement.learned is not None
        for arm in (measurement.learned, measurement.physical_prior):
            assert arm.support_violations == 0
            assert arm.sampled_hands == measurement.samples_per_example
            assert arm.sampling_seconds > 0 and arm.peak_python_bytes > 0
            assert (
                arm.largest_sample_tensor_bytes
                == measurement.samples_per_example * 2 * 8
            )
    with pytest.raises(FileExistsError):
        main([*args, "--out", str(tmp_path / "report-0.json")])
    with pytest.raises(ValueError, match="byte identity"):
        report_saved_sampler(
            data,
            checkpoint=checkpoint,
            checkpoint_identity="bad",
            training_dataset_path=data,
        )


def test_foreign_policy_label_changes_do_not_enter_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = _dataset()
    train_path, checkpoint = tmp_path / "train.json", tmp_path / "model.pt"
    save_dataset(training, train_path)
    digest = _checkpoint(checkpoint, training)
    reports = []
    samples: list[torch.Tensor] = []
    original_sample = AutoregressiveBeliefSampler.sample

    def capture_sample(
        self: AutoregressiveBeliefSampler,
        inputs: Sequence[SamplerInput],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        draws = original_sample(self, inputs, generator=generator)
        samples.append(draws.clone())
        return draws

    monkeypatch.setattr(AutoregressiveBeliefSampler, "sample", capture_sample)
    for index, label in enumerate(((2, 0), (1, 1))):
        foreign = replace(
            training,
            policy_identity="foreign-policy",
            games=tuple(
                replace(
                    game,
                    game_id=f"foreign-{game.game_id}",
                    seed=game.seed + 100,
                    examples=tuple(
                        replace(row, target_hand=label) for row in game.examples
                    ),
                )
                for game in training.games
            ),
        )
        path = tmp_path / f"foreign-{index}.json"
        save_dataset(foreign, path)
        reports.append(
            report_saved_sampler(
                path,
                checkpoint=checkpoint,
                checkpoint_identity=digest,
                training_dataset_path=train_path,
                sample_counts=(8,),
            )
        )
    assert all(report.policy_relationship == "foreign_policy" for report in reports)
    assert len(samples) == 6
    for first, second in zip(samples[:3], samples[3:], strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    # Likelihood changes because the private label changes; sampling constraints
    # and output size do not. Direct inference is identical for identical inputs.
    a, b = reports[0].measurements[0].learned, reports[1].measurements[0].learned
    assert a is not None and b is not None
    assert a.joint_nll != b.joint_nll
    assert a.support_violations == b.support_violations == 0
    assert a.largest_sample_tensor_bytes == b.largest_sample_tensor_bytes
    overlap = tmp_path / "overlap.json"
    save_dataset(replace(training, policy_identity="foreign"), overlap)
    with pytest.raises(ValueError, match="overlaps"):
        report_saved_sampler(
            overlap,
            checkpoint=checkpoint,
            checkpoint_identity=digest,
            training_dataset_path=train_path,
        )
    wrong = tmp_path / "wrong.json"
    save_dataset(replace(training, world_identity="other-world"), wrong)
    with pytest.raises(ValueError, match="schema/world"):
        report_saved_sampler(
            wrong,
            checkpoint=checkpoint,
            checkpoint_identity=digest,
            training_dataset_path=train_path,
        )


def test_wide_count_space_uses_bounded_sampling(tmp_path: Path) -> None:
    # 40 names x 80 copies, 80-card hand: enumerating support is infeasible.
    dataset = _dataset(40, 80)
    data, checkpoint = tmp_path / "wide.json", tmp_path / "wide.pt"
    save_dataset(dataset, data)
    digest = _checkpoint(checkpoint, dataset)
    report = report_saved_sampler(
        data,
        checkpoint=checkpoint,
        checkpoint_identity=digest,
        training_dataset_path=data,
        sample_counts=(2, 5),
    )
    for row in report.measurements:
        assert row.learned is not None and row.learned.support_violations == 0
        assert row.physical_prior.support_violations == 0
        assert (
            row.learned.largest_sample_tensor_bytes == row.samples_per_example * 40 * 8
        )
    prior = report_saved_sampler(data, sample_counts=(2,))
    assert prior.policy_relationship == "prior_only"
    assert all(row.learned is None for row in prior.measurements)
    assert {item.metric for item in prior.unavailable} >= {
        "arbitrary_queries",
        "exact_posterior_error",
    }


@pytest.mark.parametrize("counts", [(), (0,), (2, 2)])
def test_invalid_sensitivity_rejected_before_read(
    tmp_path: Path, counts: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="sample counts"):
        report_saved_sampler(tmp_path / "missing", sample_counts=counts)


def test_report_counts_invalid_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset()
    data, checkpoint = tmp_path / "data.json", tmp_path / "model.pt"
    save_dataset(dataset, data)
    digest = _checkpoint(checkpoint, dataset)

    def invalid_sample(
        self: AutoregressiveBeliefSampler,
        inputs: Sequence[SamplerInput],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # One invalid hand violates minima/size; another violates capacity/size.
        return torch.tensor([(0, 0), (3, 0)], dtype=torch.int64)

    monkeypatch.setattr(AutoregressiveBeliefSampler, "sample", invalid_sample)
    report = report_saved_sampler(
        data,
        checkpoint=checkpoint,
        checkpoint_identity=digest,
        training_dataset_path=data,
        sample_counts=(2,),
    )
    for row in report.measurements:
        assert row.learned is not None and row.learned.support_violations == 2
        assert row.physical_prior.support_violations == 0


def test_saved_fixed_weight_fits_feed_offline_cohort(tmp_path: Path) -> None:
    dataset = _dataset()
    game = dataset.games[-1]
    extra = replace(
        game,
        game_id="extra-test",
        seed=77,
        examples=(
            replace(game.examples[0], revision=1, viewer=0),
            replace(game.examples[0], revision=2, viewer=1, target_hand=(1, 1)),
        ),
    )
    dataset = replace(dataset, games=(*dataset.games, extra))
    data = tmp_path / "data.json"
    save_dataset(dataset, data)
    attempts: list[FitAttempt] = []
    reports = []
    for seed in (10, 11):
        checkpoint = tmp_path / f"model-{seed}.pt"
        digest = _checkpoint(checkpoint, dataset, seed)
        report = report_saved_sampler(
            data,
            checkpoint=checkpoint,
            checkpoint_identity=digest,
            training_dataset_path=data,
            sample_counts=(4,),
            seed=7,
        )
        reports.append(report)
        assert report.training_seed == seed
        panel = next(m for m in report.measurements if m.split == "test")
        assert [g.examples for g in panel.games] == [1, 2]
        assert panel.examples == 3 and panel.learned is not None
        assert panel.learned.sampled_hands == 12
        assert len(panel.learned.calibration_bins) == 10
        path = tmp_path / f"report-{seed}.json"
        encoded = json.dumps(asdict(report), allow_nan=False).encode()
        path.write_bytes(encoded)
        attempts.append(
            FitAttempt(
                attempt_id=f"fixed-{seed}",
                treatment="fixed",
                training_seed=seed,
                fit_receipt_identity=f"fixture-{seed}",
                configuration_identity="fixed-untrained-weights",
                training_dataset_identity=dataset.identity,
                status="complete",
                report=path,
                report_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        )
    first = reports[0]
    spec = CohortSpec(
        description="Fixed-weight saved fixtures, not trained fits",
        producer_policy_identity=dataset.policy_identity,
        evaluation_policy_identity=dataset.policy_identity,
        evaluation_dataset_identity=dataset.identity,
        world_identity=dataset.world_identity,
        schema_identity=dataset.schema.identity,
        evaluator_identity=first.evaluator_identity,
        samples_per_example=4,
        sampling_seed=7,
        bootstrap_replicates=100,
        attempts=tuple(attempts),
    )
    result = report_sampler_cohort(spec)
    assert result.complete
    assert all(e.fit_seeds == (10, 11) for e in result.estimates)
    assert result == report_sampler_cohort(spec)


def test_duplicate_saved_views_rejected(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset = replace(
        dataset, games=tuple(replace(g, examples=g.examples * 2) for g in dataset.games)
    )
    path = tmp_path / "duplicates.json"
    save_dataset(dataset, path)
    with pytest.raises(ValueError, match="duplicate saved observation"):
        report_saved_sampler(path, sample_counts=(2,))


def test_pooled_ece_uses_bins_not_mean_game_ece() -> None:
    # Same forecast bin, opposite errors: pooled calibration cancels although
    # the mean game ECE is .5. Retain enough statistics to distinguish them.
    bins_a = (
        (CalibrationBin(0, 0, 0),) * 5
        + (CalibrationBin(2, 1, 0),)
        + (CalibrationBin(0, 0, 0),) * 4
    )
    bins_b = (
        (CalibrationBin(0, 0, 0),) * 5
        + (CalibrationBin(2, 1, 2),)
        + (CalibrationBin(0, 0, 0),) * 4
    )
    arm = SamplerArmMetrics(1, 0.25, 0.5, None, 0, 4, 0, 0, 0, bins_a)
    pooled = _pool_games([arm, replace(arm, calibration_bins=bins_b)], [1, 1])
    assert pooled.inclusion_ece == 0
    assert pooled.conjunction_brier is None
