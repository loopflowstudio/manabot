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
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.training.analysis import report, verify_saved_inputs
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime, TrainingRun
from manabot.training.recipes import (
    ataraxos_baseline,
    value_outputs,
    with_capacity,
    with_value_aggregation,
)

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
    """Explicit workload shared by the value cross and non-executing capacity example."""
    return ataraxos_baseline(
        id="value-historical-mean-1",
        world="w4",
        match=MatchHypers.authored(
            "ur-lessons-vs-gw-allies",
            "ur_lessons",
            "gw_allies",
            hero="arena-seat-0",
            villain="arena-seat-1",
        ),
        observation=ObservationSpaceHypers(),
        agent=AgentSpec(
            hidden_dim=64,
            num_attention_heads=4,
            semantic_pack="ur-lessons-vs-gw-allies",
        ),
        checkpoints=2,
        updates=1,
        transitions=64,
        streams=4,
        stage_seconds=30,
        wall_seconds=60,
    )


def smoke_plan() -> ResolvedStudy:
    base = smoke_baseline()
    token = with_value_aggregation(
        base, id="value-value-token-1", aggregation="value_token"
    )
    models = (
        base,
        with_value_aggregation(
            base, id="value-masked-mean-1", aggregation="masked_mean"
        ),
        token,
        with_capacity(token, id="value-value-token-2", width=64, depth=2, heads=4),
    )
    recipes = value_outputs(models, ("scalar", "categorical_wdl"))
    resolved = tuple(recipe.model_dump(mode="json") for recipe in recipes.values())
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study="value-models",
            regime_digests=tuple(canonical_sha256(recipe) for recipe in resolved),
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
    args = parser.parse_args()
    if args.recover_from:
        if args.out is None:
            parser.error("--recover-from requires --out")
        recover_evaluation(args.recover_from, args.out)
    elif args.report_only:
        report(args.report_only)
    elif args.write_plan:
        with args.write_plan.open("x") as stream:
            stream.write(smoke_plan().model_dump_json(indent=2) + "\n")
    else:
        run_study("value-models", args.out.resolve(), smoke_plan())


if __name__ == "__main__":
    main()
