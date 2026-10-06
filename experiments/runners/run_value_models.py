"""Resolve the bounded value-model factorial through the existing study executor.

Generation is deterministic and does not train. Only the smoke profile is
allocated; scientific scoring needs a separately reviewed protocol.
"""

import argparse
import json
from pathlib import Path
import shutil

from experiments.runners.run_training_regimes import run_study
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import file_sha256
from manabot.infra.hypers import AgentSpec
from manabot.training.analysis import report, verify_saved_inputs
from manabot.training.execution import atomic_json
from manabot.training.experiments import (
    Axis,
    Baseline,
    Case,
    Experiment,
    Model,
    ResolvedExperiment,
)
from manabot.training.models import TrainingRegime, TrainingRun

ROOT = Path(__file__).resolve().parents[2]


def recover_evaluation(source: Path, out: Path) -> None:
    """Finish a pre-arena failure using immutable runs and the remaining budget.

    The original attempt is never modified. Runtime, protocol and run admission
    remain owned by run_study's resume path, which cannot execute training.
    """
    source = source.resolve()
    out = out.resolve()
    study = json.loads((source / "study.json").read_text())
    verify_saved_inputs(source, study)
    plan = ResolvedStudy.model_validate_json(
        (source / "resolved-plan.json").read_text()
    )
    if (
        study["study"] != "value-models"
        or study["status"] != "failed"
        or study["comparisons"]
        or study["measurements"]
        or plan.protocol.purpose != "workflow-smoke"
        or plan.protocol.study != "value-models"
    ):
        raise ValueError(
            "recovery requires a failed value-model smoke before arena play"
        )
    for entry in study["runs"]:
        run = TrainingRun.model_validate_json(Path(entry["path"]).read_text())
        if run.status != "completed":
            raise ValueError("evaluation recovery requires completed training")
        for stage in run.stages:
            for artifact in stage.artifacts.values():
                if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError("retained checkpoint digest mismatch")
    if study["seconds"] >= plan.protocol.process_seconds:
        raise ValueError("original smoke process allocation is exhausted")
    out.mkdir(parents=True, exist_ok=False)
    files = ("study.json", "recipes.json", "protocol.json", "resolved-plan.json")
    for name in files:
        shutil.copyfile(source / name, out / name)
    atomic_json(
        out / "recovery.json",
        {
            "source": str(source),
            "source_files": {name: file_sha256(source / name) for name in files},
            "operation": "evaluation-only",
            "arena_id_mapping": "underscore-to-hyphen; training identities unchanged",
            "runner_sha256": file_sha256(Path(__file__)),
            "study_runner_sha256": file_sha256(
                ROOT / "experiments/runners/run_training_regimes.py"
            ),
            "remaining_seconds": plan.protocol.process_seconds - study["seconds"],
        },
    )
    run_study("value-models", out, plan, resume=True)


def smoke_baseline() -> TrainingRegime:
    """Frozen ETU-106 settings; future defaults cannot alter the registered cross."""
    return Baseline(
        "value-model-baseline-v1",
        (ROOT / "experiments/regimes/value-model-baseline-v1.json").read_text(),
    ).regime()


def experiment() -> Experiment:
    """Declare pooling/depth cases crossed independently with value output."""
    return Experiment(
        name="value",
        baseline=Baseline.capture("value-model-baseline-v1", smoke_baseline()),
        cases=(
            Case("historical-mean-1", label="Historical mean"),
            Case(
                "masked-mean-1",
                (Model(AgentSpec(value_aggregation="masked_mean")),),
                "Masked mean",
            ),
            Case(
                "value-token-1",
                (Model(AgentSpec(value_aggregation="value_token")),),
                "Token ×1",
            ),
            Case(
                "value-token-2",
                (
                    Model(
                        AgentSpec(value_aggregation="value_token", attention_layers=2)
                    ),
                ),
                "Token ×2",
            ),
        ),
        matrix=(
            Axis(
                "output",
                (
                    Case("scalar", (Model(AgentSpec(value_kind="scalar")),), "Scalar"),
                    Case(
                        "categorical-wdl",
                        (Model(AgentSpec(value_kind="categorical_wdl")),),
                        "WDL",
                    ),
                ),
            ),
        ),
    )


def smoke_plan(resolution: ResolvedExperiment | None = None) -> ResolvedStudy:
    cells = resolution if resolution is not None else experiment().resolve()
    resolved = tuple(
        recipe.model_dump(mode="json") for recipe in cells.regimes.values()
    )
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study="value-models",
            regime_digests=cells.digests,
            training_seeds=(1061,),
            paired_deals=(910106,),
            anchor_deals=(920106,),
            process_seconds=900,
        ),
        recipes=resolved,
        allocation_seconds=900,
        prior_campaign_seconds=0,
        calibration_evidence="ETU-106 bounded workflow only; strength unresolved",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke",), default="smoke")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-plan", type=Path)
    action.add_argument("--out", type=Path)
    action.add_argument("--report-only", type=Path)
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument(
        "--write-provenance",
        type=Path,
        help="Export authoring receipt alongside --write-plan",
    )
    args = parser.parse_args()
    if args.write_provenance and args.write_plan is None:
        parser.error("--write-provenance requires --write-plan")
    if args.recover_from:
        if args.out is None:
            parser.error("--recover-from requires --out")
        recover_evaluation(args.recover_from, args.out)
    elif args.report_only:
        report(args.report_only)
    elif args.write_plan:
        cells = experiment().resolve()
        with args.write_plan.open("x") as stream:
            stream.write(smoke_plan(cells).model_dump_json(indent=2) + "\n")
        if args.write_provenance:
            with args.write_provenance.open("x") as stream:
                stream.write(json.dumps(cells.receipt(), indent=2) + "\n")
    else:
        run_study("value-models", args.out.resolve(), smoke_plan())


if __name__ == "__main__":
    main()
