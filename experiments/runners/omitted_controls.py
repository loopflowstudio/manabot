"""Resolve ETU-93 independent contrasts; generation never launches training.

Only workflow smoke is allocated here. Scientific counts, seeds, deal families
and total cost require a separately frozen ResolvedStudy after calibration.
The ordinary study runner owns execution, arena evidence and offline reporting.
"""

import argparse
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import canonical_sha256
from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.training.models import (
    Learning,
    Schedule,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.recipes import ataraxos_baseline, with_value_output

ROOT = Path(__file__).resolve().parents[2]
ContrastName = Literal[
    "discount",
    "policy-trace",
    "paper-estimators",
    "reference",
    "collection-kl",
    "paper-filter",
    "filter-ties",
    "filter-scope",
    "evaluation-ema",
    "behavior-ema",
    "lr-decay",
    "regularization-decay",
    "combined-no-filter",
    "combined-no-decay",
    "combined-no-value-trace",
    "paper-value-head",
]


class Contrast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: ContrastName
    baseline: TrainingRegime
    treatment: TrainingRegime
    evaluation_variants: tuple[Literal["raw", "ema"], ...] = ("raw",)
    disposition: Literal["unresolved"] = "unresolved"


def resolve_contrast(name: ContrastName) -> Contrast:
    """Return fully resolved arms; every delta is explicit in their digests."""
    if name == "paper-value-head":
        base = ataraxos_baseline(
            id=f"{name}-control",
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
                hidden_dim=16,
                num_attention_heads=2,
                semantic_pack="ur-lessons-vs-gw-allies",
            ),
            checkpoints=2,
            updates=2,
            transitions=64,
            streams=4,
            stage_seconds=60,
            wall_seconds=150,
        )
        return Contrast(
            name=name,
            baseline=base,
            treatment=with_value_output(
                base, id=f"{name}-treatment", output="categorical_wdl"
            ),
        )

    base_name = "rl-control"
    if name in {"paper-filter", "filter-ties", "filter-scope"}:
        base_name = "advantage-filtering"
    elif name.startswith("combined-no-"):
        base_name = "combined"
    base = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes" / f"{base_name}.json").read_text()
    )
    base.id = f"{name}-control"
    for stage in base.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = 2
        stage.transitions = 64
        stage.execution.wall_seconds = 60
        if isinstance(stage.learning, Learning):
            stage.learning.epochs = 1
        if name in {"evaluation-ema", "behavior-ema"}:
            stage.learning.ema = 0.999
    base.wall_seconds = 150
    treatment = base.model_copy(deep=True)
    treatment.id = f"{name}-treatment"
    for stage in treatment.stages:
        assert isinstance(stage, TrainSelfPlay)
        learning = stage.learning
        assert isinstance(learning, Learning)
        match name:
            case "discount":
                learning.gamma = 0.99
            case "policy-trace":
                learning.policy_lambda = 0.99
            case "paper-estimators":
                learning.policy_lambda, learning.value_lambda = 0.5, 0.8
            case "reference":
                learning.reference = "action_type_uniform"
            case "collection-kl":
                learning.collection_kl = 0
            case "paper-filter":
                learning.filter_kind = "quantile"
                learning.retained_fraction, learning.min_advantage = 0.25, 0.01
            case "filter-ties":
                learning.filter_kind = "quantile"
            case "filter-scope":
                learning.filter_scope = "actor"
            case "behavior-ema":
                stage.behavior = "ema-self"
            case "lr-decay":
                learning.learning_rate = Schedule(initial=2.5e-4, decay=9)
            case "regularization-decay":
                learning.tau = Schedule(initial=0.01, decay=9)
            case "combined-no-filter":
                learning.retained_fraction = 1
            case "combined-no-decay":
                learning.learning_rate.decay = learning.tau.decay = 0
            case "combined-no-value-trace":
                learning.value_lambda = 0.95
            case "evaluation-ema":
                pass
    return Contrast(
        name=name,
        baseline=TrainingRegime.model_validate(base.model_dump()),
        treatment=TrainingRegime.model_validate(treatment.model_dump()),
        evaluation_variants=("raw", "ema")
        if name in {"evaluation-ema", "behavior-ema"}
        else ("raw",),
    )


def smoke_plan(name: ContrastName) -> ResolvedStudy:
    contrast = resolve_contrast(name)
    recipes = (
        (contrast.baseline,)
        if name == "evaluation-ema"
        else (contrast.baseline, contrast.treatment)
    )
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study="omitted-controls",
            purpose="workflow-smoke",
            regime_digests=tuple(
                canonical_sha256(r.model_dump(mode="json")) for r in recipes
            ),
            training_seeds=(693,),
            paired_deals=(963001,),
            anchor_deals=(964001,),
            evaluation_variants=contrast.evaluation_variants,
            selection=(
                "all-completed-cutoffs-raw-and-ema"
                if "ema" in contrast.evaluation_variants
                else "all-completed-cutoffs-raw"
            ),
            process_seconds=600,
        ),
        recipes=tuple(r.model_dump(mode="json") for r in recipes),
        allocation_seconds=600,
        prior_campaign_seconds=0,
        calibration_evidence="Bounded ETU-93 workflow proof; no scientific allocation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", choices=get_args(ContrastName), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = smoke_plan(args.contrast)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(plan.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
