"""Offline quality reports over immutable whole-game sampler evidence.

``report_saved_sampler`` admits saved artifacts and preserves split membership.
Inference receives only SamplerInput; retrospective labels enter log scoring and
calibration in sampling_fit. No game generation, optimizer or enumeration runs.
The JSON contains deterministic statistical measurements and separate, naturally
variable timing/allocation fields inside each arm's metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Literal, Sequence

import torch

from manabot.belief.learning import _deterministic_torch_cpu
from manabot.belief.sampling_data import Split, read_dataset
from manabot.belief.sampling_fit import (
    SamplerArmMetrics,
    evaluate_physical_sampler,
    evaluate_sampler,
    load_belief_sampler,
)


@dataclass(frozen=True, slots=True)
class UnavailableMetric:
    metric: str
    reason: str


@dataclass(frozen=True, slots=True)
class SplitMeasurement:
    split: Split
    game_ids: tuple[str, ...]
    examples: int
    samples_per_example: int
    physical_prior: SamplerArmMetrics
    learned: SamplerArmMetrics | None


@dataclass(frozen=True, slots=True)
class SamplerQualityReport:
    format: str
    evaluation_dataset_identity: str
    training_dataset_identity: str | None
    evaluation_policy_identity: str
    training_policy_identity: str | None
    policy_relationship: Literal["same_policy", "foreign_policy", "prior_only"]
    world_identity: str
    schema_identity: str
    checkpoint_identity: str | None
    evaluator_identity: str
    seed: int
    torch_version: str
    python_version: str
    device: str
    threads: int
    parameter_bytes: int
    measurements: tuple[SplitMeasurement, ...]
    unavailable: tuple[UnavailableMetric, ...]


def report_saved_sampler(
    dataset_path: Path,
    *,
    checkpoint: Path | None = None,
    checkpoint_identity: str | None = None,
    training_dataset_path: Path | None = None,
    sample_counts: Sequence[int] = (16, 64),
    seed: int = 0,
) -> SamplerQualityReport:
    """Evaluate every saved split; reject incomplete or incompatible bindings.

    Train-split scores are descriptive, never holdout evidence. Foreign datasets
    retain their own policy identity; overlap with fitting games is rejected.
    Sample counts restart the same RNG seed, without promising nested samples.
    Run serially: Torch thread settings and Python allocation tracing are global.
    """
    if not sample_counts or any(type(n) is not int or n < 1 for n in sample_counts):
        raise ValueError("sample counts must be positive integers")
    if len(set(sample_counts)) != len(sample_counts):
        raise ValueError("sample counts must be unique")
    bindings = (checkpoint, checkpoint_identity, training_dataset_path)
    if any(value is not None for value in bindings) and not all(
        value is not None for value in bindings
    ):
        raise ValueError(
            "learned reporting requires checkpoint, digest and training dataset"
        )
    dataset = read_dataset(dataset_path)
    training = (
        None if training_dataset_path is None else read_dataset(training_dataset_path)
    )
    model = None
    if training is not None:
        if (
            training.world_identity != dataset.world_identity
            or training.schema.identity != dataset.schema.identity
        ):
            raise ValueError("evaluation schema/world differs from training dataset")
        if training.identity != dataset.identity:
            fitting_ids = {
                game.game_id for game in training.games if game.split == "train"
            }
            if any(game.game_id in fitting_ids for game in dataset.games):
                raise ValueError("evaluation dataset overlaps sampler fitting games")
        assert checkpoint is not None and checkpoint_identity is not None
        # Initialization during loading must not perturb the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            model = load_belief_sampler(
                checkpoint,
                expected_dataset_identity=training.identity,
                expected_policy_identity=training.policy_identity,
                expected_schema_identity=training.schema.identity,
                expected_world_identity=training.world_identity,
                expected_checkpoint_identity=checkpoint_identity,
            )
    measurements: list[SplitMeasurement] = []
    splits: tuple[Split, ...] = ("train", "validation", "test")
    with _deterministic_torch_cpu():
        for split in splits:
            games = tuple(game for game in dataset.games if game.split == split)
            rows = tuple(row for game in games for row in game.examples)
            for count in sample_counts:
                if model is None:
                    prior = evaluate_physical_sampler(rows, samples=count, seed=seed)
                    learned = None
                else:
                    result = evaluate_sampler(model, rows, samples=count, seed=seed)
                    prior, learned = result.physical_baseline, result.learned
                measurements.append(
                    SplitMeasurement(
                        split,
                        tuple(game.game_id for game in games),
                        len(rows),
                        count,
                        prior,
                        learned,
                    )
                )
    return SamplerQualityReport(
        format="manabot.sampler-quality/v1",
        evaluation_dataset_identity=dataset.identity,
        training_dataset_identity=None if training is None else training.identity,
        evaluation_policy_identity=dataset.policy_identity,
        training_policy_identity=None if training is None else training.policy_identity,
        policy_relationship="prior_only"
        if training is None
        else (
            "same_policy"
            if training.policy_identity == dataset.policy_identity
            else "foreign_policy"
        ),
        world_identity=dataset.world_identity,
        schema_identity=dataset.schema.identity,
        checkpoint_identity=checkpoint_identity,
        evaluator_identity=hashlib.sha256(
            b"".join(
                Path(__file__).with_name(name).read_bytes()
                for name in (
                    "sampling_report.py",
                    "sampling_fit.py",
                    "sampling.py",
                    "sampling_data.py",
                    "learning.py",
                )
            )
        ).hexdigest(),
        seed=seed,
        torch_version=str(torch.__version__),
        python_version=platform.python_version(),
        device="cpu",
        threads=1,
        parameter_bytes=0
        if model is None
        else sum(p.numel() * p.element_size() for p in model.parameters()),
        measurements=tuple(measurements),
        unavailable=(
            UnavailableMetric(
                "arbitrary_queries",
                "Only card presence and adjacent vocabulary-pair conjunctions are measured; no typed-query interpreter.",
            ),
            UnavailableMetric(
                "exact_posterior_error",
                "No external exact posterior supplied; joint NLL scores observed labels, not posterior KL.",
            ),
            UnavailableMetric(
                "native_peak_memory",
                "Python allocation peaks and returned tensor bytes exclude native Torch workspace and process RSS.",
            ),
            UnavailableMetric(
                "strength_and_transfer_acceptance",
                "Descriptive saved-cohort metrics are not independent-seed calibration or playing-strength evidence; ETU-99 owns scientific acceptance.",
            ),
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Write a new JSON report, refusing to overwrite retained evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    report = report_saved_sampler(
        args.dataset,
        checkpoint=args.checkpoint,
        checkpoint_identity=args.checkpoint_sha256,
        training_dataset_path=args.training_dataset,
        sample_counts=args.samples,
        seed=args.seed,
    )
    # Serialize before creating output so invalid numerical evidence cannot leave
    # a misleading partially written report. Artifact digests bind the inputs.
    encoded = (
        json.dumps(asdict(report), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(encoded)
    print(f"Report {args.out} (sha256 {hashlib.sha256(encoded.encode()).hexdigest()})")


if __name__ == "__main__":
    main()
