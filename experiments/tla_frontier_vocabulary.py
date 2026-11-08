"""Task c5fba4 fact-embedding check: catalog against hashed record ids at w64.

Before the freeze, two short runs of the protocol's w64 regime differ only in
`ModelConfig.record_vocabulary`: `catalog` embeds fact and shared-effect ids
through the card encoder's exact token vocabulary (Task 30df0c, the fresh
default), `hashed` learns 4096 hashed rows per id column.
Same recipe, seed, decks, producers and updates. The protocol's Before freeze
item 5 owns the decision; this file fixes its criterion before the check runs.

`runs` launched both arms for `FIRST_UPDATES` updates, exporting at the interior
`MILESTONES` and at the final update. Catalog's pod read slow and stopped at 64,
leaving that endpoint unread, so Jack Heart's extension continues each arm from
its own final raw/EMA/Adam exports to `UPDATES` (`runs` with `extend_from`): a
second segment per arm under its own regime id, exporting at
`EXTENSION_MILESTONES`. The arm is both segments' checkpoints together.
Once both extensions have ended, `eval` plays three comparisons on the frontier's
16 frozen pairs through the frontier's own pass job: each arm's update
`UPDATES` against its own update 0, and catalog against hashed at update
`UPDATES`, catalog first. As exports arrive it also plays the same comparisons at
`EARLY_UPDATES`, which no rule reads. `read_check` (the CLI) reads them with each arm's
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
import hashlib
from pathlib import Path
from statistics import median
from typing import Literal

from etude.eval.query import statsdb
from etude.eval.records import Environment
from etude.experiment import Checkpoint, ExperimentState, TrainingRun, observe
from etude.launch.records import Job, Run
from etude.run.listing import as_storage
from etude.run.storage import S3Storage, Storage, run_prefix, storage_from_uri
from etude.train.read import RunData, read_all

from experiments.tla_frontier import (
    ROOT,
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
from manabot.experiment.job import LearnerContinuation, StagedExport, TrainingJob
from manabot.infra.identity import StrictModel
from manabot.training.records import TrainingRun as LearnerRun
from manabot.training.regime import TrainingRegime

EXPERIMENT = Path(__file__).stem
type Vocabulary = Literal["catalog", "hashed"]
VOCABULARIES: tuple[Vocabulary, Vocabulary] = ("catalog", "hashed")
# The first segment's target and exports.
FIRST_UPDATES = 75
MILESTONES = (25, 50)
# Calibration read w64 at 167 s/update. 75 updates fit 15,000 usable seconds under
# the 600 s stop reserve at up to 200 s/update, a fifth above that reading.
WALL_SECONDS = 15600.0
# The wall, finalization, job setup and Etude's 15-minute allowance fit each arm's
# ceiling at the on-demand's USD0.60/hour limit; USD6 together, from the reserve.
MAX_HOURS = 4.6
MAX_USD = Decimal("3")
# The extension's endpoint, the criterion's, and its interior exports.
UPDATES = 150
EXTENSION_MILESTONES = (100, 125)
# Catalog continues from 64, so the extension needs 86 updates; 20,400 usable
# seconds cover them at the 234 s/update its slow first pod read. USD4 per arm
# and the USD2 pass fill Jack Heart's USD10.
EXTENSION_WALL_SECONDS = 21000.0
EXTENSION_MAX_HOURS = 6.1
EXTENSION_MAX_USD = Decimal("4")
# Descriptive comparisons at interior exports; no rule reads them.
EARLY_UPDATES = (25, 50, 100)
# Rule 3: the second half of the run, past the first export's warm-up.
LATE_UPDATES = range(76, UPDATES + 1)
VALUE_LOSS_RATIO = 1.10


def _regime_id(vocabulary: Vocabulary, *, extension: bool = False) -> str:
    base = f"tla-frontier-c5fba4-vocabulary-{vocabulary}"
    return f"{base}-to{UPDATES}" if extension else base


def vocabulary_regime(
    regime: TrainingRegime, vocabulary: Vocabulary, *, extension: bool = False
) -> TrainingRegime:
    """The w64 regime with its identity, length and record vocabulary changed.

    The extension differs from the first segment only in identity, target and
    wall; it keeps the first segment's damping horizon.
    """
    return shortened(
        regime,
        id=_regime_id(vocabulary, extension=extension),
        updates=UPDATES if extension else FIRST_UPDATES,
        wall_seconds=EXTENSION_WALL_SECONDS if extension else WALL_SECONDS,
        agent=regime.agent.model_copy(update={"record_vocabulary": vocabulary}),
    )


def runs(revision: str, extend_from: str = "") -> list[Run]:
    """One capped allocation per arm, at the frontier's training seed.

    With `extend_from`, the Etude store holding the first segments, each arm
    instead continues from its first segment's final exports to `UPDATES`.
    """
    regimes, protocol = frozen()
    if not extend_from:
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
    storage = storage_from_uri(extend_from)
    state = observe(storage, EXPERIMENT)
    jobs: list[Run] = []
    for vocabulary in VOCABULARIES:
        first = one_run(state, _regime_id(vocabulary))
        if first is None:
            raise ValueError(f"{vocabulary} has no first segment to extend")
        continuation = stored_continuation(storage, first.name)
        jobs.extend(
            TrainingJob(
                regime=vocabulary_regime(regimes[SMALL], vocabulary, extension=True),
                experiment=EXPERIMENT,
                milestones=tuple(
                    n for n in EXTENSION_MILESTONES if n > continuation.from_update
                ),
                finalization_seconds=600,
                stop_reserve_seconds=600,
                validate_numerics=True,
                continuation=continuation,
            )
            .job(
                repo=ROOT,
                revision=revision,
                max_hours=EXTENSION_MAX_HOURS,
                max_usd=EXTENSION_MAX_USD,
            )
            .runs(protocol.training_seed)
        )
    return jobs


# A worker's final learner exports, beside its run record, under each attempt.
_EXPORTS = {
    "raw": "self-play-raw.pt",
    "ema": "self-play-ema.pt",
    "optimizer": "self-play-optimizer.pt",
}


def stored_continuation(storage: Storage, run: str) -> LearnerContinuation:
    """Continue `run` from the raw/EMA/Adam exports its run record published.

    The exports stay where its worker stored them; each is checked against the
    record's digest. A run with more than one attempt holding a record is
    refused, since which one is final would be a guess.
    """
    if not isinstance(storage, S3Storage):
        raise ValueError("an extension reads its parents from the Etude bucket")
    folder = run_prefix(run)
    records = [
        key
        for key in storage.keys(folder)
        if key.startswith(f"{folder}attempts/") and key.endswith("/training/run.json")
    ]
    if len(records) != 1:
        raise ValueError(f"{run} has {len(records)} attempt records, not one")
    record = records[0]
    payload = storage.get(record)
    learner = LearnerRun.model_validate_json(payload)
    if learner.status not in {"completed", "paused"}:
        raise ValueError(f"{run} did not end its segment: {learner.status}")
    training = record.removesuffix("run.json")
    published = learner.stages[0].artifacts
    staged: dict[str, StagedExport] = {}
    for role, name in _EXPORTS.items():
        artifact = published.get(role)
        if artifact is None or Path(artifact["path"]).name != name:
            raise ValueError(f"{run} published no {role} export named {name}")
        data = storage.get(training + name)
        if (hashlib.sha256(data).hexdigest(), len(data)) != (
            artifact["sha256"],
            artifact["bytes"],
        ):
            raise ValueError(f"{run}'s stored {name} differs from its record")
        staged[role] = StagedExport(
            key=_bucket_key(storage, training + name),
            sha256=artifact["sha256"],
            bytes=artifact["bytes"],
        )
    return LearnerContinuation(
        from_update=learner.updates_through(),
        source_stage=learner.stages[0].id,
        source_run=StagedExport(
            key=_bucket_key(storage, record),
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        ),
        **staged,
    )


def _bucket_key(storage: S3Storage, key: str) -> str:
    """A storage key as an Etude input names it: relative to the bucket."""
    prefix = storage.prefix.strip("/")
    return f"{prefix}/{key}" if prefix else key


# -- Evaluation ---------------------------------------------------------------


def arm_runs(state: ExperimentState, vocabulary: Vocabulary) -> list[TrainingRun]:
    """The arm's stored segments, first segment first."""
    return [
        run
        for run in (
            one_run(state, _regime_id(vocabulary)),
            one_run(state, _regime_id(vocabulary, extension=True)),
        )
        if run is not None
    ]


def _at(segments: list[TrainingRun], update: int) -> Checkpoint | None:
    for run in segments:
        found = run.checkpoints.at("updates", update)
        if found:
            return found[0]
    return None


def _complete(state: ExperimentState, vocabulary: Vocabulary) -> bool:
    """The arm's extension ended at `UPDATES`."""
    extension = one_run(state, _regime_id(vocabulary, extension=True))
    return (
        extension is not None and extension.ended and last_update(extension) == UPDATES
    )


def check_schedule(
    state: ExperimentState, protocol: Protocol
) -> list[ScheduledComparison]:
    """The three comparisons, each unplayable until both extensions ended at `UPDATES`."""
    arms = {vocabulary: arm_runs(state, vocabulary) for vocabulary in VOCABULARIES}
    ready = all(_complete(state, vocabulary) for vocabulary in VOCABULARIES)
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
    """The same three comparisons at `EARLY_UPDATES` once both arms stored them.

    Descriptive only: they show progress before the endpoint and enter no rule of
    the criterion, which reads the endpoint slots alone.
    """
    arms = {vocabulary: arm_runs(state, vocabulary) for vocabulary in VOCABULARIES}
    inference = checkpoint_inference()
    slots: list[ScheduledComparison] = []
    for update in EARLY_UPDATES:
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
        segments = arm_runs(state, vocabulary)
        run = segments[-1] if segments else None
        # Both segments' losses by update; the extension's own log wins a repeat.
        curve = sorted(
            {
                update: loss
                for segment in segments
                for update, loss in curves.get(segment.name, [])
            }.items()
        )
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
