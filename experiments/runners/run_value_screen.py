"""Resolve or launch the separate eight-hour scalar value-token screen.

Plan generation initializes the environment for identities but never trains.
Execution requires an explicit saved plan; no smoke or scientific plan is adapted.
"""

import argparse
from pathlib import Path

import torch

from experiments.runners.run_training_regimes import ROOT, run_study
from experiments.runners.run_value_models import smoke_baseline
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import canonical_sha256
from manabot.env import ObservationSpace
from manabot.sim.teacher1_evidence import runtime_fingerprints, source_bundle_sha256
from manabot.training.analysis import report
from manabot.training.models import TrainingRegime, TrainSelfPlay
from manabot.training.recipes import with_value_aggregation


def screen_plan() -> ResolvedStudy:
    """Pin a prospective fixed-count cohort with phase watchdogs, without scoring."""
    base = smoke_baseline()
    base.wall_seconds = 2400
    for stage in base.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = 400
        stage.execution.wall_seconds = 1190
    base = TrainingRegime.model_validate(base.model_dump())
    recipes = tuple(
        with_value_aggregation(
            base, id=f"screen-{name.replace('_', '-')}", aggregation=name
        )
        for name in ("historical_mean", "masked_mean", "value_token")
    )
    resolved = tuple(r.model_dump(mode="json") for r in recipes)
    identities = runtime_fingerprints(
        10611,
        match_hypers=base.match,
        observation_space=ObservationSpace(base.observation),
    )
    keys = (
        "engine_extension_sha256",
        "engine_source_sha256",
        "content_manifest_sha256",
        "observation_abi_sha256",
        "action_abi_sha256",
        "matchup_sha256",
    )
    runtime: dict[str, str] = {}
    for key in keys:
        value = identities[key]
        if not isinstance(value, str):
            raise TypeError(f"runtime identity {key} must be text")
        runtime[key] = value
    runtime["training_source_sha256"] = source_bundle_sha256(
        sorted((ROOT / "manabot").rglob("*.py"))
    )
    runtime["study_source_sha256"] = source_bundle_sha256(
        sorted((ROOT / "experiments/runners").glob("*.py"))
        + [
            ROOT / "experiments/study/training-regimes.ipynb",
            ROOT / "uv.lock",
            ROOT / "pyproject.toml",
        ]
    )
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study="value-token-screen",
            purpose="screening",
            regime_digests=tuple(canonical_sha256(r) for r in resolved),
            training_seeds=(10611, 10612, 10613),
            paired_deals=(),
            anchor_deals=tuple(range(961060, 961085)),
            anchors=("scripted-greedy",),
            process_seconds=28800,
            uncertainty="paired-seed-descriptive",
        ),
        recipes=resolved,
        allocation_seconds=28800,
        prior_campaign_seconds=0,
        runtime_identities=runtime,
        projected_disk_bytes=2 * 1024**3,
        calibration_evidence="Prospective screen: 400 updates per half from bounded ETU-106 smoke; long-run rate unverified. Two-hour conservative stop; no strength evidence.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-plan", type=Path)
    action.add_argument("--out", type=Path)
    action.add_argument("--report-only", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.out is not None:
        if args.plan is None:
            parser.error("execution requires --plan")
        plan = ResolvedStudy.model_validate_json(args.plan.read_text())
        if plan.protocol.study != "value-token-screen":
            parser.error("requires a value-token-screen plan")
        torch.set_num_threads(1)
        run_study("value-token-screen", args.out.resolve(), plan)
    elif args.plan is not None:
        parser.error("--plan requires --out")
    elif args.report_only:
        report(args.report_only)
    else:
        with args.write_plan.open("x") as stream:
            stream.write(screen_plan().model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
