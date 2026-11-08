"""Task c5fba4 fact-embedding check: catalog against hashed record ids at w64.

Before the freeze, two short runs of the protocol's w64 regime differ only in
`ModelConfig.record_vocabulary`: `catalog` embeds fact and shared-effect ids
through the card encoder's exact token vocabulary (Task 30df0c, the fresh
default), `hashed` learns 4096 hashed rows per id column.
Same recipe, seed, decks, producers and updates. The protocol's Before freeze
item 5 owns the decision; this file fixes its criterion before the check runs.

`runs` launches both arms for `UPDATES` updates, exporting at the interior
`MILESTONES` and at the final update, where training stops and exports.
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
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Literal

from etude.eval.query import statsdb
from etude.eval.records import Environment
from etude.experiment import Checkpoint, ExperimentState, TrainingRun, observe
from etude.launch.records import Job, Run
from etude.run.listing import as_storage
from etude.run.storage import Storage, storage_from_uri
from etude.train.read import RunData, read_all

from experiments.tla_frontier import (
    SMALL,
    Protocol,
    ScheduledComparison,
    capped_runs,
    checkpoint_inference,
    environment,
    evaluation_jobs,
    frozen,
    last_update,
    one_run,
    paired_matchup,
    shortened,
    with_launcher_ends,
)
from experiments.tla_frontier_analysis import (
    Bounded,
    Comparison,
    learned,
    read_comparisons,
)
from manabot.infra.identity import StrictModel
from manabot.training.regime import TrainingRegime

EXPERIMENT = Path(__file__).stem
type Vocabulary = Literal["catalog", "hashed"]
VOCABULARIES: tuple[Vocabulary, Vocabulary] = ("catalog", "hashed")
UPDATES = 75
# Interior exports; the job exports update UPDATES itself when training stops.
MILESTONES = (25, 50)
# Calibration read w64 at 167 s/update. 75 updates fit 15,000 usable seconds under
# the 600 s stop reserve at up to 200 s/update, a fifth above that reading.
WALL_SECONDS = 15600.0
# The wall, finalization, job setup and Etude's 15-minute allowance fit each arm's
# ceiling at the on-demand's USD0.60/hour limit; USD6 together, from the reserve.
MAX_HOURS = 4.6
MAX_USD = Decimal("3")
# Rule 3: the second half of the run, past the first export's warm-up.
LATE_UPDATES = range(38, UPDATES + 1)
VALUE_LOSS_RATIO = 1.10


def _regime_id(vocabulary: Vocabulary) -> str:
    return f"tla-frontier-c5fba4-vocabulary-{vocabulary}"


def vocabulary_regime(regime: TrainingRegime, vocabulary: Vocabulary) -> TrainingRegime:
    """The w64 regime with its identity, length and record vocabulary changed."""
    return shortened(
        regime,
        id=_regime_id(vocabulary),
        updates=UPDATES,
        wall_seconds=WALL_SECONDS,
        agent=regime.agent.model_copy(update={"record_vocabulary": vocabulary}),
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
            milestones=MILESTONES,
        )
    ]


# -- Evaluation ---------------------------------------------------------------


def arm_run(state: ExperimentState, vocabulary: Vocabulary) -> TrainingRun | None:
    return one_run(state, _regime_id(vocabulary))


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


def early_schedule(
    state: ExperimentState, protocol: Protocol
) -> list[ScheduledComparison]:
    """The same three comparisons at each interior export both arms have stored.

    Descriptive only: they show progress before the endpoint and enter no rule of
    the criterion, which reads the endpoint slots alone.
    """
    arms = {vocabulary: arm_run(state, vocabulary) for vocabulary in VOCABULARIES}
    inference = checkpoint_inference()
    slots: list[ScheduledComparison] = []
    for update in MILESTONES:
        at = {vocabulary: _at(arms[vocabulary], update) for vocabulary in VOCABULARIES}
        initial = {vocabulary: _at(arms[vocabulary], 0) for vocabulary in VOCABULARIES}
        pairs: list[tuple[str, str, Checkpoint | None, Checkpoint | None]] = [
            *(
                (
                    f"early-initial-{vocabulary}-u{update}",
                    f"{vocabulary} u{update} vs {vocabulary} u0",
                    at[vocabulary],
                    initial[vocabulary],
                )
                for vocabulary in VOCABULARIES
            ),
            (
                f"early-head-to-head-u{update}",
                f"catalog u{update} vs hashed u{update}",
                at["catalog"],
                at["hashed"],
            ),
        ]
        ready = all(checkpoint is not None for checkpoint in at.values())
        for tag, name, first, second in pairs:
            matchup = (
                paired_matchup(
                    name,
                    (first.model(inference), second.model(inference)),
                    tag,
                    protocol,
                )
                if ready and first is not None and second is not None
                else None
            )
            slots.append(ScheduledComparison(tag=tag, update=update, matchup=matchup))
    return slots


def _environment(protocol: Protocol) -> Environment:
    return environment(
        protocol,
        "tla-frontier-c5fba4-vocabulary-v1",
        "record-vocabulary-check-not-frontier-scored",
    )


def eval(state: ExperimentState) -> list[Job]:
    """Early comparisons as interior exports arrive; the endpoint once both ended."""
    if state.experiment != EXPERIMENT:
        raise ValueError("the vocabulary check requires its own runs")
    _, protocol = frozen()
    return evaluation_jobs(
        "tla-frontier-c5fba4-vocabulary-v1",
        _environment(protocol),
        [
            planned.matchup
            for planned in (
                *early_schedule(state, protocol),
                *check_schedule(with_launcher_ends(state), protocol),
            )
            if planned.matchup is not None
        ],
    )


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
    # Descriptive comparisons at interior exports; no rule reads them.
    early: list[Comparison]
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
    rows = data.points("value_loss", "updates")
    return [
        (int(update), float(loss))
        for update, loss in zip(rows.x.tolist(), rows.y.tolist(), strict=True)
    ]


def _late(curve: list[tuple[int, float]]) -> float | None:
    late = [loss for update, loss in curve if update in LATE_UPDATES]
    return median(late) if len(late) == len(LATE_UPDATES) else None


def read_check(state: ExperimentState, source: Storage | Path | str) -> VocabularyCheck:
    """Both arms, the three comparisons and the verdict, from `state` and its store."""
    _, protocol = frozen()
    storage = as_storage(source)
    curves = {
        data.name: _curve(data)
        for data in read_all(storage)
        if data.experiment == EXPERIMENT
    }
    with statsdb(EXPERIMENT, source) as db:
        comparisons = {
            comparison.tag: comparison
            for comparison in read_comparisons(
                check_schedule(state, protocol),
                db,
                protocol,
                _environment(protocol).identity,
            )
        }
        early = read_comparisons(
            early_schedule(state, protocol),
            db,
            protocol,
            _environment(protocol).identity,
        )
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
        early=early,
        failures=failures,
        passed=passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Etude training and evaluation store"
    )
    args = parser.parse_args()
    source = storage_from_uri(args.source)
    state = with_launcher_ends(observe(source, EXPERIMENT))
    print(read_check(state, source).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
