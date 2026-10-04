"""Generate a bounded frozen-policy diagnostic recipe or regenerate its report.

Execution remains `uv run manabot train --regime ... --out ...`; VerifyStore
retains attempts. Existing checkpoints can be selected by Run/stage identity
without any training. The default includes a tiny workflow-only policy stage.
"""

import argparse
from pathlib import Path

from manabot.training.analysis import report_selection_run
from manabot.training.models import (
    CollectSelection,
    Execution,
    Learning,
    SelectionGameSpec,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore

from .omitted_controls import resolve_contrast


def diagnostic_recipe(
    source: TrainingRegime | None = None,
    *,
    source_run: str | None = None,
    policy: str = "policy-0",
) -> TrainingRegime:
    """Four unique deals, each partition contains both deck assignments.

    One checkpoint, two games per split: workflow evidence only. No scientific
    allocation, technique decision or method-level interval follows this recipe.
    """
    recipe = (source or resolve_contrast("filter-ties").baseline).model_copy(deep=True)
    recipe.id = "held-out-selection-workflow"
    recipe.wall_seconds = 180
    recipe.recovery_max_microsteps = None
    collection = CollectSelection(
        id="selection",
        operation="collect_selection",
        policy=policy,
        source_run=source_run,
        execution=Execution(wall_seconds=120),
        population=tuple(
            SelectionGameSpec(
                seed=930001 + i,
                action_seed=940001 + i,
                assignment=i % 2,
                split="development" if i < 2 else "held_out",
            )
            for i in range(4)
        ),
        learning=Learning(
            filter_kind="quantile", retained_fraction=0.5, min_advantage=0
        ),
    )
    if source_run:
        recipe.stages = [collection]
    else:
        recipe.agent.hidden_dim = 8
        recipe.stages = [
            TrainSelfPlay(
                id=policy,
                operation="train_self_play",
                updates=1,
                streams=2,
                transitions=32,
                execution=Execution(wall_seconds=60),
            ),
            collection,
        ]
    return TrainingRegime.model_validate(recipe.model_dump())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--policy-stage", default="policy-0")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--report-only", type=Path, help="TrainingRun JSON export")
    args = parser.parse_args()
    if args.report_only:
        report_selection_run(args.report_only, "selection", args.out)
        return
    source = None
    if args.source_run:
        if args.store is None:
            parser.error(
                "--source-run requires --store; execution must use the same store"
            )
        with VerifyStore(args.store) as store:
            source = store.training_run(args.source_run).regime
    recipe = diagnostic_recipe(
        source, source_run=args.source_run, policy=args.policy_stage
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(recipe.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
