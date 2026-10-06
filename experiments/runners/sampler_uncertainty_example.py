"""Write explicitly synthetic saved reports and their offline cohort analysis.

No fitted models, gameplay or sampled hands: fabricated numbers demonstrate
pairing and weighting only. This is a review fixture, not calibration evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

from manabot.belief.sampling_cohort import (
    CohortSpec,
    Contrast,
    FitAttempt,
    main as cohort_main,
)
from manabot.belief.sampling_fit import CalibrationBin, SamplerArmMetrics
from manabot.belief.sampling_report import (
    GameMeasurement,
    SamplerQualityReport,
    SplitMeasurement,
    _pool_games,
)


def synthetic_report(seed: int, treatment: str) -> SamplerQualityReport:
    """Correlated game effects cancel under paired treatment differences."""
    games: list[GameMeasurement] = []
    for i, count in enumerate((2, 8, 20)):
        # Each game has many dependent views; longer games must not dominate NLL.
        nll = (1.0, 5.0, 9.0)[i]
        effect = (-0.2, -0.1, -0.3)[seed - 11]
        if treatment == "dropout":
            effect -= 0.1
        bins = (
            (CalibrationBin(0, 0, 0),) * 4
            + (CalibrationBin(count * 2, count * 0.8, count),)
            + (CalibrationBin(0, 0, 0),) * 5
        )

        def arm(score: float) -> SamplerArmMetrics:
            return SamplerArmMetrics(
                score, 0.26, 0.1, 0.25, 0, count * 16, 0.0, 0, 0, bins
            )

        games.append(
            GameMeasurement(
                f"synthetic-game-{i}", i, 0, count, arm(nll), arm(nll + effect)
            )
        )
    sizes = [g.examples for g in games]
    learned = [g.learned for g in games if g.learned is not None]
    panel = SplitMeasurement(
        "test",
        tuple(g.game_id for g in games),
        sum(sizes),
        16,
        _pool_games([g.physical_prior for g in games], sizes),
        _pool_games(learned, sizes),
        tuple(games),
    )
    return SamplerQualityReport(
        format="manabot.sampler-quality/v2",
        evaluation_dataset_identity="SYNTHETIC-eval",
        training_dataset_identity="SYNTHETIC-train",
        evaluation_policy_identity="SYNTHETIC-policy",
        training_policy_identity="SYNTHETIC-policy",
        policy_relationship="same_policy",
        world_identity="SYNTHETIC-world",
        schema_identity="SYNTHETIC-schema",
        checkpoint_identity=f"SYNTHETIC-{treatment}-{seed}",
        evaluator_identity="SYNTHETIC-evaluator",
        seed=7,
        torch_version="SYNTHETIC",
        python_version="SYNTHETIC",
        device="cpu",
        threads=1,
        parameter_bytes=0,
        measurements=(panel,),
        unavailable=(),
        training_seed=seed,
    )


def write_example(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    attempts: list[FitAttempt] = []
    for treatment in ("baseline", "dropout"):
        for seed in (11, 12, 13):
            name = f"{treatment}-{seed}"
            report = synthetic_report(seed, treatment)
            data = (
                json.dumps(asdict(report), sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode()
            with (root / f"{name}.json").open("xb") as output:
                output.write(data)
            attempts.append(
                FitAttempt(
                    attempt_id=name,
                    treatment=treatment,
                    training_seed=seed,
                    fit_receipt_identity=f"SYNTHETIC-receipt-{name}",
                    configuration_identity=f"SYNTHETIC-config-{treatment}",
                    training_dataset_identity="SYNTHETIC-train",
                    status="complete",
                    report=Path(f"{name}.json"),
                    report_sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    spec = CohortSpec(
        description="SYNTHETIC REVIEW FIXTURE. Fabricated scores and fit identities; no training, games or calibration evidence.",
        producer_policy_identity="SYNTHETIC-policy",
        evaluation_policy_identity="SYNTHETIC-policy",
        evaluation_dataset_identity="SYNTHETIC-eval",
        world_identity="SYNTHETIC-world",
        schema_identity="SYNTHETIC-schema",
        evaluator_identity="SYNTHETIC-evaluator",
        samples_per_example=16,
        sampling_seed=7,
        bootstrap_seed=42,
        attempts=tuple(attempts),
        contrasts=(Contrast(left="dropout", right="baseline"),),
    )
    manifest = root / "manifest.json"
    with manifest.open("x") as output:
        output.write(spec.model_dump_json(indent=2) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = write_example(args.out)
    cohort_main(["--manifest", str(manifest), "--out", str(args.out / "analysis.json")])


if __name__ == "__main__":
    main()
