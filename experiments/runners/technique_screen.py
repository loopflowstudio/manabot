"""Export independent ETU-105 software plans; never execute a study.

The landed Experiment authoring API owns composition and provenance. Existing
ResolvedStudy/omitted-controls owns pairwise execution and reporting. These tiny
workloads validate wiring only; scientific counts require a separate protocol.
"""

import argparse
import json
from pathlib import Path
from typing import Literal, get_args

from experiments.runners.omitted_controls import resolve_contrast
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.infra.hypers import AgentSpec
from manabot.training.experiments import (
    Baseline,
    Case,
    Experiment,
    LearningRule,
    Model,
    Pipeline,
    ResolvedExperiment,
)
from manabot.training.models import AtaraxosMoveLearning, Learning, TrainSelfPlay
from manabot.training.presets import ataraxos_mtg_v1

ContrastName = Literal[
    "policy-trace",
    "value-trace",
    "reference",
    "collection-kl",
    "ratio-clip",
    "gradient-clip",
    "filter-quantile",
    "filter-minimum",
    "filter-off",
    "filter-scope",
    "filter-ties",
    "lr-constant",
    "tau-constant",
    "schedules-constant",
    "evaluation-ema",
    "behavior-ema",
    "discount",
    "ppo-package",
]


def experiment(name: ContrastName) -> Experiment:
    """One factor per pair except explicitly named package/interaction controls.

    PPO is scalar in both arms; its package contrast changes several optimizer
    conventions. Discount stays in the delivered scalar PPO instrument: adding
    gamma to categorical outcome mixtures would invent a new estimator contract.
    """
    if name not in get_args(ContrastName):
        raise ValueError(f"unknown technique contrast: {name}")
    if name == "discount":
        existing = resolve_contrast("discount")
        stage = existing.treatment.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        return Experiment(
            name="technique-discount",
            baseline=Baseline.capture("etu93-discount-control", existing.baseline),
            cases=(Case("control"), Case("treatment", (LearningRule(stage.learning),))),
        )

    baseline = ataraxos_mtg_v1()
    stage = baseline.regime().stages[0]
    assert isinstance(stage, TrainSelfPlay)
    rule = stage.learning
    assert isinstance(rule, AtaraxosMoveLearning)
    match name:
        case "policy-trace":
            rule.policy_lambda = 0.95
        case "value-trace":
            rule.value_lambda = 1.0
        case "reference":
            rule.reference = "uniform"
        case "collection-kl":
            rule.collection_kl = 0.0
        case "ratio-clip":
            rule.clip = 0.1
        case "gradient-clip":
            rule.max_grad_norm = 0.5
        case "filter-quantile":
            rule.advantage_quantile = 0.5
        case "filter-minimum":
            rule.min_advantage = 0.0
        case "filter-off":
            rule.advantage_quantile, rule.min_advantage = 0.0, 0.0
        case "filter-scope":
            rule.filter_scope = "actor"
        case "filter-ties":
            rule.filter_kind = "top_count"
        case "lr-constant" | "tau-constant" | "schedules-constant":
            if name != "tau-constant":
                # Same initial effective rate. Holding scale=.5 with power=0
                # would also clamp forever, but obscures the constant's units.
                rule.learning_rate_scale = rule.learning_rate_max
                rule.learning_rate_power = 0.0
            if name != "lr-constant":
                rule.tau_power = 0.0
        case "behavior-ema":
            stages = baseline.regime().stages
            for item in stages:
                assert isinstance(item, TrainSelfPlay)
                item.behavior = "ema-self"
            return Experiment(
                name=f"technique-{name}",
                baseline=baseline,
                cases=(Case("control"), Case("treatment", (Pipeline(tuple(stages)),))),
            )
        case "evaluation-ema":
            # One training cohort, two correlated artifacts; never train an
            # identical second arm to manufacture another apparent replicate.
            return Experiment(name=f"technique-{name}", baseline=baseline)
        case "ppo-package":
            ppo = resolve_contrast("discount").baseline.stages[0]
            assert isinstance(ppo, TrainSelfPlay)
            assert isinstance(ppo.learning, Learning)
            return Experiment(
                name=f"technique-{name}",
                baseline=baseline,
                overrides=(Model(AgentSpec(value_kind="scalar")),),
                cases=(Case("move"), Case("ppo", (LearningRule(ppo.learning),))),
            )
    return Experiment(
        name=f"technique-{name}",
        baseline=baseline,
        cases=(Case("control"), Case("treatment", (LearningRule(rule),))),
    )


def smoke_plan(name: ContrastName, cells: ResolvedExperiment) -> ResolvedStudy:
    """Bind a resolved contrast to the existing bounded workflow protocol.

    No campaign allocation is inherited. The 900-second cap is a proposed
    future software execution ceiling, not permission to train during export.
    """
    expected = experiment(name).resolve()
    if cells.identity != expected.identity:
        raise ValueError("resolution does not match the named technique contrast")
    variants: tuple[Literal["raw", "ema"], ...] = (
        ("raw", "ema") if name in {"evaluation-ema", "behavior-ema"} else ("raw",)
    )
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study="omitted-controls",
            purpose="workflow-smoke",
            regime_digests=cells.digests,
            training_seeds=(1051,),
            paired_deals=(951051,),
            anchor_deals=(952051,),
            evaluation_variants=variants,
            selection="all-completed-cutoffs-raw-and-ema"
            if "ema" in variants
            else "all-completed-cutoffs-raw",
            process_seconds=900,
        ),
        recipes=tuple(r.model_dump(mode="json") for r in cells.regimes.values()),
        allocation_seconds=900,
        prior_campaign_seconds=0,
        calibration_evidence="ETU-105 unexecuted software plan; scientific allocation zero",
    )


def export_plan(name: ContrastName, out: Path) -> None:
    """Retain the exact recipe/provenance pair without overwriting any attempt."""
    cells = experiment(name).resolve()
    plan = smoke_plan(name, cells)
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps(cells.receipt(), indent=2) + "\n")
    lines = [
        f"# {name}: plan admission",
        "",
        "Status: resolved, unexecuted. No training, checkpoints or arena results.",
        "Purpose: workflow smoke; no technique benefit or empirical disposition.",
        "",
        f"Authoring identity: `{cells.identity}`.",
        "",
        "| Cell | Regime digest |",
        "| --- | --- |",
        *(f"| {c.name} | `{c.digest}` |" for c in cells.cases),
        "",
        "The plan binds configuration and proposed workflow seeds/budget only.",
        "Code/runtime/world/data/checkpoint identities are admitted by TrainingRun",
        "and the arena at execution; none are fabricated by this planning report.",
        "Scientific budgets and interpretations: experiments/ataraxos-technique-screen.md.",
    ]
    (out / "admission.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", choices=get_args(ContrastName), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export_plan(args.contrast, args.out)


if __name__ == "__main__":
    main()
