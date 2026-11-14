"""Offline report acceptance uses fixed weights and labels, never optimizer steps."""

from collections.abc import Sequence
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.tla_belief_sampler import demo, report_panels
from manabot.belief.checkpoint import load_belief_sampler
from manabot.belief.sampling import AutoregressiveBeliefSampler
from manabot.belief.sampling_cohort import (
    CohortSpec,
    EvaluationGroup,
    FitAttempt,
    _estimate,
    report_sampler_cohort,
)
from manabot.belief.sampling_collection import (
    CaptureSlot,
    close_capture_collection,
    freeze_capture_dataset,
)
from manabot.belief.sampling_data import (
    SamplerDataset,
    SamplerExample,
    SamplerGame,
    Split,
    save_dataset,
)
from manabot.belief.sampling_fit import CalibrationBin, SamplerArmMetrics
from manabot.belief.sampling_types import SamplerInput, SamplerSchema, _digest
from manabot.reports.belief import _equal_games, _pool_games, main, report_saved_sampler
from tests.belief.test_sampling_data import _collected_directory, _snapshot_fixture


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
    payload = {
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
    }
    if dataset.binding is not None:
        payload["format"] = "manabot.autoregressive-belief-sampler/v2"
        del payload["policy_identity"]
        payload["snapshot_binding"] = asdict(dataset.binding)
    torch.save(payload, path)
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
    grouped = spec.model_copy(
        update={
            "groups": (
                EvaluationGroup(
                    group_id="deal-a", game_ids=("game-2", "extra-test", "missing-leg")
                ),
                EvaluationGroup(
                    group_id="deal-b", game_ids=("missing-both-0", "missing-both-1")
                ),
            )
        }
    )
    grouped_result = report_sampler_cohort(grouped)
    assert grouped_result.group_coverage[0].missing_games == ("missing-leg",)
    assert grouped_result.group_coverage[1].retained_games == ()
    assert all(e.game_interval is None for e in grouped_result.estimates)
    # Within-pair differences cancel. Resampling legs independently would
    # fabricate evaluation uncertainty although every retained group averages 5.
    estimate = _estimate(
        "paired",
        "joint_nll",
        (10, 11),
        [[0, 10, 0, 10]] * 2,
        complete=True,
        spec=spec,
        groups=((0, 1), (2, 3)),
    )
    assert estimate.mean == 5
    assert estimate.game_interval is not None
    assert estimate.game_interval.low == estimate.game_interval.high == 5
    with pytest.raises(ValueError, match="membership must be unique"):
        spec.model_validate(
            {
                **spec.model_dump(),
                "groups": [
                    {"group_id": "a", "game_ids": ["game-2"]},
                    {"group_id": "b", "game_ids": ["game-2"]},
                ],
            }
        )


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


def test_tla_saved_demo_is_public_and_never_optimizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _collected_directory(tmp_path)
    close_capture_collection(root / "main", reason="controlled fixture complete")
    dataset = freeze_capture_dataset(root)
    fit = tmp_path / "fits/357"
    fit.mkdir(parents=True)
    digest = _checkpoint(fit / "sampler.pt", dataset, 357)
    binding = {"dataset_identity": dataset.identity, "config": {}, "seed": 357}
    (fit / "started.json").write_text(json.dumps(binding))
    (fit / "complete.json").write_text(
        json.dumps({"binding": binding, "checkpoint_identity": digest})
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline demo attempted optimization")

    monkeypatch.setattr("manabot.belief.sampling_fit.fit_belief_sampler", forbidden)
    path = demo(tmp_path)
    public = json.loads(path.read_text())
    assert public["public_commitment_features"]
    assert all(
        feature["kind"] == "pass_priority"
        for feature in public["public_commitment_features"]
    )
    assert set(public["inclusion_probabilities"]) == {
        "prior",
        "learned",
        "zero_history",
    }
    assert "target_hand" not in path.read_text() and "seed" not in path.read_text()
    assert len(public["inclusion_probabilities"]["learned"]) == len(
        dataset.schema.card_names
    )
    before = (tmp_path / "demo/quality.json").read_bytes()
    assert demo(tmp_path) == path
    assert (tmp_path / "demo/quality.json").read_bytes() == before
    (tmp_path / "demo/quality.json").write_text("{}")
    with pytest.raises(ValueError, match="report identity"):
        demo(tmp_path)


def test_schedule_panels_retain_zero_row_and_unattempted_games(tmp_path: Path) -> None:
    slots = tuple(
        CaptureSlot(str(i), i, i % 2, split, ("a", "b"), panel)
        for i, (split, panel) in enumerate(
            (
                ("train", None),
                ("validation", None),
                ("test", "same-field"),
                ("test", "reserved"),
                ("test", "reserved"),
            )
        )
    )
    root = _collected_directory(tmp_path, slots)
    shard_path = tmp_path / "main/game-000003.game.json"
    shard = json.loads(shard_path.read_text())
    shard["game"] = None
    shard_path.write_text(json.dumps(shard))
    receipt_path = tmp_path / "main/game-000003.result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["shard_identity"] = _digest(shard)
    receipt_path.write_text(json.dumps(receipt))
    for path in (tmp_path / "main").glob("game-000004.*.json"):
        path.unlink()
    close_capture_collection(root / "main", reason="wall bound")
    dataset = freeze_capture_dataset(root)
    checkpoint = tmp_path / "fixed.pt"
    digest = _checkpoint(checkpoint, dataset, 357)
    report = report_saved_sampler(
        tmp_path / "dataset.json",
        checkpoint=checkpoint,
        checkpoint_identity=digest,
        training_dataset_path=tmp_path / "dataset.json",
        sample_counts=(4,),
        seed=7,
    )
    data = json.dumps(asdict(report)).encode()
    (tmp_path / "quality.json").write_bytes(data)
    spec = CohortSpec(
        description="Fixed saved fixture",
        producer_policy_identity=dataset.policy_identity,
        evaluation_policy_identity=dataset.policy_identity,
        evaluation_dataset_identity=dataset.identity,
        world_identity=dataset.world_identity,
        schema_identity=dataset.schema.identity,
        evaluator_identity=report.evaluator_identity,
        samples_per_example=4,
        sampling_seed=7,
        bootstrap_replicates=100,
        attempts=(
            FitAttempt(
                attempt_id="fixed",
                treatment="fixed",
                training_seed=357,
                fit_receipt_identity="fixture",
                configuration_identity="fixed",
                training_dataset_identity=dataset.identity,
                status="complete",
                report=Path("quality.json"),
                report_sha256=hashlib.sha256(data).hexdigest(),
            ),
        ),
    )
    panels = report_panels(root, spec)
    assert panels["same-field"].group_coverage[0].retained_games == ("2",)
    assert [group.missing_games for group in panels["reserved"].group_coverage] == [
        ("3",),
        ("4",),
    ]
    assert all(estimate.mean is None for estimate in panels["reserved"].estimates)
    closure = json.loads((tmp_path / "main/closure.json").read_text())
    assert closure["statuses"][-2:] == ["completed", "unattempted"]
    fit = tmp_path / "fits/357"
    fit.mkdir(parents=True)
    (fit / "sampler.pt").write_bytes(checkpoint.read_bytes())
    binding = {"dataset_identity": dataset.identity, "config": {}, "seed": 357}
    (fit / "started.json").write_text(json.dumps(binding))
    (fit / "complete.json").write_text(
        json.dumps({"binding": binding, "checkpoint_identity": digest})
    )
    demo(tmp_path, cohort=spec)
    saved_panels = json.loads((tmp_path / "demo/panels.json").read_text())
    assert saved_panels["reserved"]["group_coverage"][0]["missing_games"] == ["3"]


def test_snapshot_checkpoint_reload_and_report_name_producers(tmp_path: Path) -> None:
    dataset, path = _snapshot_fixture(tmp_path)
    checkpoint = tmp_path / "sampler.pt"
    digest = _checkpoint(checkpoint, dataset, 357)
    model = load_belief_sampler(
        checkpoint,
        expected_dataset_identity=dataset.identity,
        expected_snapshot_binding=dataset.binding,
        expected_schema_identity=dataset.schema.identity,
        expected_world_identity=dataset.world_identity,
        expected_checkpoint_identity=digest,
    )
    assert not model.training
    with pytest.raises(ValueError, match="policy_identity"):
        load_belief_sampler(
            checkpoint,
            expected_dataset_identity=dataset.identity,
            expected_policy_identity="frozen-policy",
            expected_schema_identity=dataset.schema.identity,
            expected_world_identity=dataset.world_identity,
            expected_checkpoint_identity=digest,
        )
    assert dataset.binding is not None
    with pytest.raises(ValueError, match="snapshot identity"):
        load_belief_sampler(
            checkpoint,
            expected_dataset_identity=dataset.identity,
            expected_snapshot_binding=replace(dataset.binding, watermark="another-cut"),
            expected_schema_identity=dataset.schema.identity,
            expected_world_identity=dataset.world_identity,
            expected_checkpoint_identity=digest,
        )
    report = report_saved_sampler(
        path,
        checkpoint=checkpoint,
        checkpoint_identity=digest,
        training_dataset_path=path,
        sample_counts=(2,),
    )
    assert report.format == "manabot.sampler-quality/v3"
    assert report.policy_relationship == "same_producers"
    assert (
        report.training_policy_identity is None
        and report.evaluation_policy_identity is None
    )
    assert report.training_snapshot == dataset.binding == report.evaluation_snapshot
    assert report.measurements[0].zero_row_game_ids == (dataset.games[0].game_id,)
    assert report.measurements[0].equal_game_prior is not None


def test_equal_game_losses_keep_pooled_rows_separate() -> None:
    # Equal-game ECE pools normalized per-game forecasts, so these cancel.
    bins_a = (CalibrationBin(1, 0.5, 0),) + (CalibrationBin(0, 0, 0),) * 9
    bins_b = (CalibrationBin(8, 4, 8),) + (CalibrationBin(0, 0, 0),) * 9
    first = SamplerArmMetrics(1, 0.1, 0.5, 0.2, 0, 1, 0, 0, 0, bins_a)
    second = SamplerArmMetrics(3, 0.3, 0.5, 0.4, 0, 8, 0, 0, 0, bins_b)
    equal = _equal_games((first, second))
    assert equal.joint_nll == 2
    assert equal.inclusion_brier == pytest.approx(0.2)
    assert equal.inclusion_ece == 0
    assert _pool_games((first, second), (1, 8)).joint_nll == pytest.approx(25 / 9)


def test_snapshot_cohort_binds_manifest_groups_and_fit_distribution(
    tmp_path: Path,
) -> None:
    dataset, path = _snapshot_fixture(tmp_path)
    checkpoint = tmp_path / "fixed.pt"
    digest = _checkpoint(checkpoint, dataset, 357)
    quality = report_saved_sampler(
        path,
        checkpoint=checkpoint,
        checkpoint_identity=digest,
        training_dataset_path=path,
        sample_counts=(2,),
    )
    data = json.dumps(asdict(quality)).encode()
    (tmp_path / "quality.json").write_bytes(data)
    assert dataset.binding is not None
    member = dataset.binding.games[-1]
    spec = CohortSpec(
        format="manabot.sampler-cohort/v2",
        description="Fixed snapshot with independent groups",
        producer_snapshot_identity=dataset.binding.identity,
        evaluation_snapshot_identity=dataset.binding.identity,
        evaluation_dataset_identity=dataset.identity,
        world_identity=dataset.world_identity,
        schema_identity=dataset.schema.identity,
        evaluator_identity=quality.evaluator_identity,
        samples_per_example=2,
        sampling_seed=0,
        bootstrap_replicates=100,
        groups=(
            EvaluationGroup(group_id=member.group, game_ids=(member.key.identity,)),
        ),
        attempts=(
            FitAttempt(
                attempt_id="fit",
                treatment="fixed",
                training_seed=357,
                fit_receipt_identity="receipt",
                configuration_identity="config",
                training_dataset_identity=dataset.identity,
                status="complete",
                report=Path("quality.json"),
                report_sha256=hashlib.sha256(data).hexdigest(),
            ),
        ),
    )
    report = report_sampler_cohort(spec, root=tmp_path)
    assert report.complete
    assert report.group_coverage[0].retained_games == (member.key.identity,)
    with pytest.raises(ValueError, match="snapshot identity"):
        report_sampler_cohort(
            spec.model_copy(update={"producer_snapshot_identity": "other"}),
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="groups differ"):
        report_sampler_cohort(
            spec.model_copy(
                update={
                    "groups": (
                        EvaluationGroup(
                            group_id="unrelated", game_ids=(member.key.identity,)
                        ),
                    )
                }
            ),
            root=tmp_path,
        )
