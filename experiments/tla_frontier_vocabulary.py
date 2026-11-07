"""Task c5fba4 fact-embedding check: catalog against hashed record ids at w64.

Before either 54-hour frontier run, two short runs of the frozen w64 regime
differ only in `ModelConfig.record_vocabulary`: `catalog` embeds fact and
shared-effect ids through the card encoder's exact token vocabulary (Task
30df0c, the fresh default), `hashed` learns 4096 hashed rows per id column.
Same recipe, seed, decks, producers and updates. The protocol's Before freeze
item 5 owns the decision; this file fixes its criterion before the check runs.

`runs` launches both arms for `UPDATES` updates, exporting at `EXPORTS`.
Once both arms have ended, `eval` plays three comparisons on the frontier's
16 frozen pairs through the frontier's own pass job: each arm's update
`UPDATES` against its own update 0, and catalog against hashed at update
`UPDATES`, catalog first. `read_check` (the CLI) reads them with each arm's
value-loss curve and applies `verdict`.

The criterion fails catalog when any of these holds:

1. **Head-to-head.** Catalog loses to hashed under every failed-leg outcome:
   the head-to-head's `best` upper bound is below 0.5.
2. **Learning.** Hashed's learning against its initialization is established
   (`worst` lower bound above 0.5) and catalog's is rejected (`best` lower bound
   at or below 0.5).
3. **Learning curve.** Catalog's median `value_loss` over updates
   `LATE_UPDATES` is more than `VALUE_LOSS_RATIO` times hashed's.

Otherwise catalog passes. A reading missing any input (an arm short of
`UPDATES`, an unplayed comparison or an absent loss) is unread, `passed` None,
which is no pass: a recorded decision reruns or reviews it before a long run.
Each failure names its rule, and a failure returns the embedding choice to
review. These games sit under this experiment and an environment of their own,
so they touch no frontier slot, match key or score.

PR #437 (Task 30df0c) adds `record_vocabulary`; until it merges, `runs`
refuses rather than launching two identical arms.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Literal

from etude.eval.query import statsdb
from etude.eval.records import Benchmark, Environment, Matchup
from etude.experiment import Checkpoint, ExperimentState, TrainingRun, observe
from etude.launch.records import Job, Run
from etude.run.listing import as_storage
from etude.train.read import RunData, read_all

from experiments.tla_frontier import (
    SMALL,
    Protocol,
    ScheduledComparison,
    capped_runs,
    checkpoint_inference,
    evaluation_pass,
    frozen,
    last_update,
    paired_matchup,
    pass_name,
    with_launcher_ends,
)
from experiments.tla_frontier_analysis import (
    Bounded,
    bootstrap_indices,
    learned,
    policy_actions_by_match,
    read_comparison,
)
from experiments.tla_pilot_analysis import game_columns
from manabot.infra.config import ModelConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.infra.runtime import regime_tag
from manabot.training.regime import TrainingRegime, TrainSelfPlay

EXPERIMENT = Path(__file__).stem
type Vocabulary = Literal["catalog", "hashed"]
VOCABULARIES: tuple[Vocabulary, Vocabulary] = ("catalog", "hashed")
UPDATES = 100
EXPORTS = (25, 50, 75, 100)
# 100 updates fit 8,400 usable seconds under the 600 s stop reserve at up to
# 84 s/update. Calibration reads w64's cost first; a slower reading changes the
# length here, by recorded decision, before the check runs.
WALL_SECONDS = 9000.0
# Etude's 15-minute setup allowance and the wall fit each arm's ceiling at the
# USD0.67/hour the training budget admits; USD4 together, from the reserve.
MAX_HOURS = 3.0
MAX_USD = Decimal("2")
# Rule 3: the second half of the run, past the first export's warm-up.
LATE_UPDATES = range(51, UPDATES + 1)
VALUE_LOSS_RATIO = 1.10
VALUE_LOSS = "value_loss"


def _regime_id(vocabulary: Vocabulary) -> str:
    return f"tla-frontier-c5fba4-vocabulary-{vocabulary}"


def vocabulary_regime(regime: TrainingRegime, vocabulary: Vocabulary) -> TrainingRegime:
    """The frozen w64 regime with its identity, length and record vocabulary changed.

    Raises before PR #437, whose `ModelConfig.record_vocabulary` the arms need.
    """
    if "record_vocabulary" not in ModelConfig.model_fields:
        raise ValueError("record_vocabulary needs Task 30df0c (PR #437) merged")
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    execution = stage.execution.model_copy(update={"wall_seconds": WALL_SECONDS})
    return TrainingRegime.model_validate(
        regime.model_copy(
            update={
                "id": _regime_id(vocabulary),
                "wall_seconds": WALL_SECONDS,
                "agent": regime.agent.model_copy(
                    update={"record_vocabulary": vocabulary}
                ),
                "stages": [
                    stage.model_copy(
                        update={"updates": UPDATES, "execution": execution}
                    )
                ],
            }
        ).model_dump()
    )


def runs(revision: str) -> list[Run]:
    """One capped allocation per arm, at the frontier's training seed."""
    regimes, protocol = frozen()
    return [
        run
        for vocabulary in VOCABULARIES
        for run in capped_runs(
            vocabulary_regime(regimes[SMALL], vocabulary),
            experiment=EXPERIMENT,
            revision=revision,
            seed=protocol.training_seed,
            max_hours=MAX_HOURS,
            max_usd=MAX_USD,
            milestones=EXPORTS,
        )
    ]


# -- Evaluation ---------------------------------------------------------------


def arm_run(state: ExperimentState, vocabulary: Vocabulary) -> TrainingRun | None:
    """The arm's run, known by its regime tag since stored names are launcher ids."""
    tag = regime_tag(_regime_id(vocabulary))
    found = [run for run in state.runs if run.checkpoints.tagged(tag)]
    if len(found) > 1:
        raise ValueError(f"the check admits one {vocabulary} run, found {len(found)}")
    return found[0] if found else None


def _at(run: TrainingRun | None, update: int) -> Checkpoint | None:
    found = run.checkpoints.at("updates", update) if run else ()
    return found[0] if found else None


def check_schedule(
    state: ExperimentState, protocol: Protocol
) -> list[ScheduledComparison]:
    """The three comparisons, each unplayable until both arms ended at `UPDATES`."""
    arms = {vocabulary: arm_run(state, vocabulary) for vocabulary in VOCABULARIES}
    ready = all(
        run is not None and run.ended and last_update(run) == UPDATES
        for run in arms.values()
    )
    inference = checkpoint_inference()

    def slot(
        tag: str, name: str, first: Checkpoint | None, second: Checkpoint | None
    ) -> ScheduledComparison:
        matchup = (
            paired_matchup(
                name, (first.model(inference), second.model(inference)), tag, protocol
            )
            if ready and first is not None and second is not None
            else None
        )
        return ScheduledComparison(tag=tag, update=UPDATES, matchup=matchup)

    end = {vocabulary: _at(arms[vocabulary], UPDATES) for vocabulary in VOCABULARIES}
    return [
        *(
            slot(
                f"initial-{vocabulary}",
                f"{vocabulary} u{UPDATES} vs {vocabulary} u0",
                end[vocabulary],
                _at(arms[vocabulary], 0),
            )
            for vocabulary in VOCABULARIES
        ),
        slot(
            "head-to-head",
            f"catalog u{UPDATES} vs hashed u{UPDATES}",
            end["catalog"],
            end["hashed"],
        ),
    ]


def _environment(protocol: Protocol) -> Environment:
    return Environment(
        name="tla-frontier-c5fba4-vocabulary-v1",
        files=canonical_sha256(protocol.model_dump(mode="json")),
        config={"purpose": "record-vocabulary-check-not-frontier-scored"},
    )


def eval(state: ExperimentState) -> list[Job]:
    """One pass once both arms ended at `UPDATES`; nothing before."""
    if state.experiment != EXPERIMENT:
        raise ValueError("the vocabulary check requires its own runs")
    _, protocol = frozen()
    matchups: tuple[Matchup, ...] = tuple(
        planned.matchup
        for planned in check_schedule(with_launcher_ends(state), protocol)
        if planned.matchup is not None
    )
    if not matchups:
        return []
    return [
        evaluation_pass(
            Benchmark(
                name=pass_name("tla-frontier-c5fba4-vocabulary-v1"),
                environment=_environment(protocol),
                matchups=matchups,
            )
        )
    ]


# -- Reading ------------------------------------------------------------------


class ArmReading(StrictModel):
    """One arm; None means not yet read, never a negative result."""

    vocabulary: Vocabulary
    run: str | None
    state: str | None
    last_update: int | None
    # The learning curve: (update, value loss) for every logged update.
    value_loss: list[tuple[int, float]]
    late_value_loss: float | None
    against_initial: Bounded | None
    learned: bool | None


class VocabularyCheck(StrictModel):
    arms: list[ArmReading]
    # Catalog's score against hashed at `UPDATES`.
    head_to_head: Bounded | None
    loss_ratio: float | None
    failures: list[str]
    # None: unread, which is no pass.
    passed: bool | None


def verdict(
    catalog: ArmReading, hashed: ArmReading, head_to_head: Bounded | None
) -> tuple[list[str], bool | None]:
    """The fixed criterion: failing rules by name, and whether catalog passed."""
    if (
        head_to_head is None
        or catalog.against_initial is None
        or hashed.against_initial is None
        or catalog.late_value_loss is None
        or hashed.late_value_loss is None
    ):
        return [], None
    failures: list[str] = []
    if head_to_head.best.upper < 0.5:
        failures.append("head-to-head: catalog loses to hashed under every outcome")
    if hashed.learned is True and catalog.learned is False:
        failures.append("learning: hashed learned and catalog did not")
    if catalog.late_value_loss > VALUE_LOSS_RATIO * hashed.late_value_loss:
        failures.append(
            f"learning curve: catalog's late value loss exceeds {VALUE_LOSS_RATIO}x hashed's"
        )
    return failures, not failures


def _curve(data: RunData) -> list[tuple[int, float]]:
    rows = data.points(VALUE_LOSS, "updates")
    return [
        (int(update), float(loss))
        for update, loss in zip(rows.x.tolist(), rows.y.tolist(), strict=True)
    ]


def _late(curve: list[tuple[int, float]]) -> float | None:
    late = [loss for update, loss in curve if update in LATE_UPDATES]
    return median(late) if len(late) == len(LATE_UPDATES) else None


def read_check(state: ExperimentState, source: Path | str) -> VocabularyCheck:
    """Both arms, the three comparisons and the verdict, from `state` and its store."""
    _, protocol = frozen()
    storage = as_storage(source)
    curves = {
        data.name: _curve(data)
        for data in read_all(storage)
        if data.experiment == EXPERIMENT
    }
    environment_id = _environment(protocol).identity
    indices = bootstrap_indices(protocol)
    with statsdb(EXPERIMENT, source) as db:
        columns = game_columns(db)
        actions = policy_actions_by_match(db, environment_id)
        comparisons = {
            planned.tag: read_comparison(
                planned, db, protocol, environment_id, columns, actions, indices
            )
            for planned in check_schedule(state, protocol)
        }
    arms: dict[Vocabulary, ArmReading] = {}
    for vocabulary in VOCABULARIES:
        run = arm_run(state, vocabulary)
        curve = curves.get(run.name, []) if run else []
        initial = comparisons[f"initial-{vocabulary}"].result
        arms[vocabulary] = ArmReading(
            vocabulary=vocabulary,
            run=run.name if run else None,
            state=run.state if run else None,
            last_update=last_update(run) if run else None,
            value_loss=curve,
            late_value_loss=_late(curve),
            against_initial=initial,
            learned=learned(initial),
        )
    catalog, hashed = arms["catalog"], arms["hashed"]
    head_to_head = comparisons["head-to-head"].result
    failures, passed = verdict(catalog, hashed, head_to_head)
    return VocabularyCheck(
        arms=[catalog, hashed],
        head_to_head=head_to_head,
        loss_ratio=catalog.late_value_loss / hashed.late_value_loss
        if catalog.late_value_loss is not None and hashed.late_value_loss
        else None,
        failures=failures,
        passed=passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Etude training and evaluation store"
    )
    args = parser.parse_args()
    state = with_launcher_ends(observe(args.source, EXPERIMENT))
    print(read_check(state, args.source).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
