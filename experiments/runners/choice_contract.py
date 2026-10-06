"""Bounded ETU-107 software workflow on the current native choice contract.

Reuses the four compound credit/target recipes and the existing exact-replayed
study arena. Every attempt has one thread and a 900-second process ceiling;
this is a workflow fixture, not a strength study or a scientific allocation.
"""

import argparse
from pathlib import Path

import torch

from experiments.runners.run_training_regimes import STUDIES, run_study, smoke_recipe
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import canonical_sha256
from manabot.training.models import TrainingRegime
import managym


def workflow_plan() -> ResolvedStudy:
    recipes: list[TrainingRegime] = []
    for name in STUDIES["compound-decisions"]:
        recipe = smoke_recipe(name)
        recipe.world = str(managym.WORLD_VERSION)
        recipe.agent.compound_features = "objects"
        recipe.wall_seconds = 120
        recipes.append(TrainingRegime.model_validate(recipe.model_dump()))
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study="compound-decisions",
            regime_digests=tuple(
                canonical_sha256(r.model_dump(mode="json")) for r in recipes
            ),
            process_seconds=900,
        ),
        recipes=tuple(r.model_dump(mode="json") for r in recipes),
        allocation_seconds=900,
        prior_campaign_seconds=0,
        calibration_evidence="ETU-107 approved bounded software fixture; no scientific allocation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_study("compound-decisions", args.out.resolve(), workflow_plan())


if __name__ == "__main__":
    main()
