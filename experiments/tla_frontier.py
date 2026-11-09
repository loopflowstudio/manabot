"""Task c5fba4: frontier self-play on the 17Lands TLA trophy corpus at two model sizes.

One seed per size trains under the same damped self-play rule, population,
input v9, four producers and wall budget; the large size is the frozen w64
regime with only its width, depth, heads and damping horizon replaced (see the
protocol `experiments/protocols/tla-frontier-c5fba4-v1.md`). Primary progress is
each learner against its latest trained checkpoint at or before half its elapsed
training hours, on the frozen random 17Lands deck pairs. `eval` also plays, at
every shared update milestone, each size against the matched first-64 heuristic and
the larger size against the smaller at equal updates; at the endpoints it adds
each size against its own initialization and the two endpoints against each
other, which is equal wall budget only when both runs finished. At every
w160l8 milestone and its endpoint it also pairs w160l8 with w64's last
checkpoint at or before the same training hours (`checkpoint_hours` reads
Etude's `Checkpoint.wall_ms`). A failed run's endpoint is its last stored
checkpoint, so its reach is still played and read.
Only a worker writes a run's end, so `with_launcher_ends` joins the launcher's
record of a lost pod to what the store kept before anything reads the schedule.
Training exports more often than `eval` plays: the export cadence bounds what a
lost rental can cost, and only milestones and endpoints are evaluated.

A launched run is stored under its launcher id, so `size_run` knows a size by
the regime tag on its checkpoints. `pod_reading` reads each run's step phase,
host and game-error counts, which are reported and void nothing. A lost run stops at its last export, which is its endpoint even
below the first milestone; then its endpoint comparisons are unreached.

A pass's launcher id is a digest of its pending matches, and the launcher refuses
a known id, so `pass_name` gives a relaunched pass a new tournament name. Match
keys exclude that name, so a relaunch plays only what no pass attempted.

The MTG-213 v10 runner (`experiments.tla_pilot_v10`) is executed evidence and
stays unchanged, so its evaluation callback, whose identities name that
pilot's source and artifacts, is mirrored here rather than shared.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
from statistics import median
from tempfile import TemporaryDirectory
from typing import Any, Literal

from etude.eval.match import Row, Tables
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Inference,
    Match,
    Matchup,
    Model,
)
from etude.experiment import Checkpoint, ExperimentState, TrainingRun
from etude.jobs import evaluation
from etude.launch import client
from etude.launch.records import Job, Run
from etude.report import Curve, Matches, Report, Score
from etude.run.storage import Storage
from etude.train.read import RunData
import torch

from experiments import tla_frontier
from experiments.tla_pilot_v10 import HeuristicSpec, Pair, RuntimeWorld
from manabot.env.observation import ObservationSpace
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TOOLCHAIN_ENV, TrainingJob, machine_setup
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.infra.runtime import regime_tag
from manabot.model.checkpoint import load_checkpoint, local_checkpoint_file
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.training.regime import (
    DampedPolicyLearning,
    IterationSchedule,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.state.game_errors import GameErrorState
import managym

ROOT = Path(__file__).resolve().parents[1]
# `etude eval` and `etude launch` both name the experiment for this file.
EXPERIMENT = Path(__file__).stem
PROTOCOL = ROOT / "experiments/protocols/tla-frontier-c5fba4-v1.json"
REGIME = ROOT / "experiments/regimes/tla-frontier-c5fba4-w64.json"
GAMES_PER_PAIR = 4
# One single-threaded game per worker on a 32-vCPU rental, leaving room for its
# host. At cpu32's USD0.96/hour, these hours and Etude's 15-minute setup
# allowance fit the pass ceiling.
EVALUATION_WORKERS = 28
EVALUATION_HOURS = 1.75
EVALUATION_PASS_USD = Decimal("2")
EVALUATION_ATTEMPTS = 3
# Set to the next attempt's number to relaunch a pass; see `pass_name`.
PASS_ATTEMPT = "C5FBA4_PASS_ATTEMPT"
# Native stepping and encoding, which hold no network work, so pods training
# different sizes compare directly. Update 1 carries warm-up.
STEP_PHASE = "time/collection/environment_step_and_encoding"
HOST_UPDATES = range(2, 7)
HEURISTIC: HeuristicSpec = {
    "kind": "heuristic",
    "max_actions": 64,
    "action_overflow": "first-legal",
}

type SizeLabel = Literal["w64", "w160l8"]
SMALL: SizeLabel = "w64"
LARGE: SizeLabel = "w160l8"
SIZES: tuple[SizeLabel, SizeLabel] = (SMALL, LARGE)


# -- Frozen contract ----------------------------------------------------------


class LargeSize(StrictModel):
    """The large arm's compute network and damping horizon; all else is w64's."""

    hidden_dim: int
    attention_layers: int
    num_attention_heads: int
    # `IterationSchedule.steps`: the updates calibration expects it to reach.
    expected_updates: int


class Protocol(StrictModel):
    runtime_world: RuntimeWorld
    corpus_sha256: str
    observation_shapes: dict[str, tuple[int, ...]]
    regime_sha256: str
    large: LargeSize
    training_seed: int
    deck_seed: int
    pairs: tuple[Pair, ...]
    milestones: tuple[int, ...]
    # Updates between exports; milestones set what is played, this what a lost
    # rental can lose.
    export_every: int
    game_seconds: float
    max_commands: int
    bootstrap_seed: int
    # Whole-Task paid ceiling; training allocations are per size, evaluation
    # is one cumulative allocation across every pass.
    max_paid_usd: Decimal
    training_usd: Decimal
    training_hours: float
    evaluation_usd: Decimal


def _regime_id(label: SizeLabel) -> str:
    return f"tla-frontier-c5fba4-{label}"


def frozen() -> tuple[dict[SizeLabel, TrainingRegime], Protocol]:
    """Both regimes and the protocol, rejecting drift from the frozen bytes.

    The hash fixes the regime's contents; `tests/eval/test_tla_frontier.py`
    checks what those bytes must declare. Only the world and the tensor shapes
    the encoder derives can still move underneath them.
    """
    protocol = Protocol.model_validate_json(PROTOCOL.read_bytes())
    if protocol.runtime_world.model_dump() != managym.world_identity():
        raise ValueError("frontier runtime world differs from the frozen protocol")
    if file_sha256(REGIME) != protocol.regime_sha256:
        raise ValueError("frontier regime differs from frozen protocol")
    small = TrainingRegime.model_validate_json(REGIME.read_bytes())
    if ObservationSpace(small.observation).shapes != protocol.observation_shapes:
        raise ValueError("frontier tensor capacities differ from frozen protocol")
    stage = _stage(small)
    size = protocol.large.model_dump(exclude={"expected_updates"})
    schedule = _schedule(small).model_copy(
        update={"steps": protocol.large.expected_updates}
    )
    learning = stage.learning.model_copy(update={"iteration_schedule": schedule})
    large = TrainingRegime.model_validate(
        small.model_copy(
            update={
                "id": _regime_id(LARGE),
                "agent": small.agent.model_copy(update=size),
                "stages": [stage.model_copy(update={"learning": learning})],
            }
        ).model_dump()
    )
    return {SMALL: small, LARGE: large}, protocol


def _stage(regime: TrainingRegime) -> TrainSelfPlay:
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    return stage


def _schedule(regime: TrainingRegime) -> IterationSchedule:
    learning = _stage(regime).learning
    assert isinstance(learning, DampedPolicyLearning)
    assert learning.iteration_schedule is not None
    return learning.iteration_schedule


def shortened(
    regime: TrainingRegime,
    *,
    id: str,
    updates: int,
    wall_seconds: float,
    workers: Literal[1, 2, 4] | None = None,
    agent: ModelConfig | None = None,
) -> TrainingRegime:
    """`regime` as a short run of its own: new identity, update target and wall.

    Calibration and the vocabulary check each change one thing more, `workers`
    or `agent`.
    """
    stage = _stage(regime)
    execution = stage.execution.model_copy(
        update={
            "wall_seconds": wall_seconds,
            "workers": workers or stage.execution.workers,
        }
    )
    return TrainingRegime.model_validate(
        regime.model_copy(
            update={
                "id": id,
                "wall_seconds": wall_seconds,
                "agent": agent or regime.agent,
                "stages": [
                    stage.model_copy(
                        update={"updates": updates, "execution": execution}
                    )
                ],
            }
        ).model_dump()
    )


def horizon(regime: TrainingRegime) -> int:
    """The size's damping horizon, `IterationSchedule.steps`."""
    return _schedule(regime).steps


def update_cap(regime: TrainingRegime) -> int:
    return _stage(regime).updates


def export_updates(regime: TrainingRegime, protocol: Protocol) -> tuple[int, ...]:
    """Every update the run exports: the cadence joined with the milestones."""
    cadence = range(protocol.export_every, update_cap(regime), protocol.export_every)
    return tuple(sorted({*cadence, *protocol.milestones}))


def capped_runs(
    regime: TrainingRegime,
    *,
    experiment: str,
    revision: str,
    seed: int,
    max_hours: float,
    max_usd: Decimal,
    milestones: tuple[int, ...] = (),
) -> list[Run]:
    """One capped allocation; calibration measures the job the frontier launches."""
    return (
        TrainingJob(
            regime=regime,
            experiment=experiment,
            milestones=milestones,
            finalization_seconds=600,
            stop_reserve_seconds=600,
            validate_numerics=True,
        )
        .job(repo=ROOT, revision=revision, max_hours=max_hours, max_usd=max_usd)
        .runs(seed)
    )


def runs(revision: str) -> list[Run]:
    """One capped GPU allocation per size; no automatic retry or budget expansion."""
    regimes, protocol = frozen()
    return [
        run
        for regime in regimes.values()
        for run in capped_runs(
            regime,
            experiment=EXPERIMENT,
            revision=revision,
            seed=protocol.training_seed,
            max_hours=protocol.training_hours,
            max_usd=protocol.training_usd,
            milestones=export_updates(regime, protocol),
        )
    ]


def _source() -> list[Path]:
    return [
        *sorted((ROOT / "manabot").rglob("*.py")),
        Path(__file__),
        ROOT / "experiments/tla_pilot_v10.py",
        REGIME,
        PROTOCOL,
    ]


def checkpoint_inference() -> Inference:
    """How a stored checkpoint plays, for frontier passes and calibration's rehearsal."""
    return Inference(source=_source())


def environment(
    protocol: Protocol,
    name: str = "tla-frontier-c5fba4-v1",
    purpose: str = "fixed-corpus-frontier-no-heldout-deck-claim",
) -> Environment:
    """The frontier's environment, or a named one of the checks beside it."""
    return Environment(
        name=name,
        # The protocol names the world and corpus, and `frozen` rejects drift.
        files=canonical_sha256(protocol.model_dump(mode="json")),
        config={"purpose": purpose},
    )


# -- Host speed ---------------------------------------------------------------


class HostSpeed(StrictModel):
    """How fast a run's pod stepped; None means the run has no such reading.

    A median over `HOST_UPDATES` as far as the run reached them, per row
    because rows per update follow game length. `step_spread` is the window's
    slowest update over its fastest, so one pod's own variation is not read as
    a host difference. `cpu_model` is on a run's receipt only once its attempt
    finishes, on revisions that record it.
    """

    step_us_per_row: float | None
    step_spread: float | None
    cpu_model: str | None


class GameErrorCounts(StrictModel):
    """A run's settled game attempts by outcome, from `TrainingRun.game_errors`.

    `engine_error` and `resource_limit` are the aborted games `recover_capacity`
    keeps apart; their samples are discarded and training continued.
    """

    attempted: int
    completed: int
    engine_error: int
    resource_limit: int
    discarded_samples: int
    stop_reason: str | None


def _game_error_counts(found: object) -> GameErrorCounts | None:
    """Per-label totals over every deck/seat cell; None without a stored state."""
    if found is None:
        return None
    state = GameErrorState.model_validate(found)
    totals = Counter[str]()
    for cell in state.by_cell.values():
        totals.update(cell)
    return GameErrorCounts(
        attempted=state.attempted,
        completed=state.completed,
        engine_error=totals["engine_error"],
        resource_limit=totals["resource_limit"],
        discarded_samples=state.discarded_samples,
        stop_reason=state.stop_reason,
    )


def _host_speed(data: RunData, cpu_model: str | None) -> HostSpeed:
    steps = data.points(STEP_PHASE, "updates")
    rows = data.points("rows", "updates")
    counts = dict(zip(rows.x.tolist(), rows.y.tolist(), strict=True))
    window = [
        seconds * 1e6 / counts[update]
        for update, seconds in zip(steps.x.tolist(), steps.y.tolist(), strict=True)
        if update in HOST_UPDATES and counts.get(update)
    ]
    return HostSpeed(
        step_us_per_row=median(window) if window else None,
        step_spread=max(window) / min(window) if window else None,
        cpu_model=cpu_model,
    )


class PodReading(StrictModel):
    """What a run's pod did beside its scores: reported, voiding nothing."""

    host: HostSpeed
    game_errors: GameErrorCounts | None


def pod_reading(data: RunData, storage: Storage) -> PodReading:
    """The run's rows and its `TrainingRun` export, written once its attempt finishes."""
    exports = [
        key for key in storage.keys(data.name) if key.endswith("training/run.json")
    ]
    receipt: dict[str, Any] = json.loads(storage.get(max(exports))) if exports else {}
    return PodReading(
        host=_host_speed(data, (receipt.get("host") or {}).get("cpu_model")),
        game_errors=_game_error_counts(receipt.get("game_errors")),
    )


# -- Evaluation schedule ------------------------------------------------------


# Hours since each run's first checkpoint (update 0), by stored run name, then
# by the checkpoint's update. Etude stores each checkpoint's `wall_ms`; this is
# that clock rebased per run, so pre-training setup counts for no size.
type CheckpointHours = Mapping[str, Mapping[int, float]]


@dataclass(frozen=True)
class ScheduledComparison:
    """One expected comparison; a missing checkpoint leaves an unplayable slot.

    `update` is the first player's, None for an endpoint its run has not
    reached. Head-to-head rows put the large size first, so their score is the
    large size's. `unreached` marks a milestone past an ended run's endpoint,
    or an endpoint comparison of a run that ended below the first milestone,
    which no later evidence can fill and no missing evidence explains. An
    equal-hours slot's `hours` are the large size's and its w64 partner's.
    """

    tag: str
    update: int | None
    matchup: Matchup | None
    unreached: bool = False
    hours: tuple[float, float] | None = None


def paired_matchup(
    name: str, players: tuple[Model, Model | Code], tag: str, protocol: Protocol
) -> Matchup:
    """One comparison: every frozen pair, four legs each, as `play` admits."""
    return Matchup(
        name=name,
        players=players,
        conditions=tuple(pair.model_dump(mode="json") for pair in protocol.pairs),
        games=GAMES_PER_PAIR,
        tags=(tag,),
    )


def with_launcher_ends(state: ExperimentState) -> ExperimentState:
    """`state` with each run the launcher closed unsuccessfully ended as failed.

    Only a worker writes a run's end, so a pod lost outright or deleted by its
    guardian reads `running` in the store for good, and its size could never
    reach an endpoint. The launcher owns whether an attempt is over and the store
    what it kept, so such a run ends at its last export. A `succeeded` run with
    no end state is delivery lag and stays running. A running run the launcher
    does not know raises, so a read away from the launcher fails loudly rather
    than taking a dead pod for a live one.
    """

    def lost(run: TrainingRun) -> bool:
        if run.ended:
            return False
        found = client.status(run.name)
        return found.done and found.outcome != "succeeded"

    return replace(
        state,
        runs=tuple(
            replace(run, state="failed") if lost(run) else run for run in state.runs
        ),
    )


def size_label(run: TrainingRun) -> SizeLabel | None:
    """The size a run's checkpoints name; a run that stored none names no size."""
    for label in SIZES:
        if run.checkpoints.tagged(regime_tag(_regime_id(label))):
            return label
    return None


def checkpoint_update(checkpoint: Checkpoint) -> int:
    return int(checkpoint.clocks["updates"])


def last_update(run: TrainingRun) -> int | None:
    """The run's last stored update; None if it stored no checkpoint."""
    last = run.checkpoints.last
    return checkpoint_update(last) if last else None


def tagged_runs(state: ExperimentState, regime: str) -> list[TrainingRun]:
    """The regime's runs: the checkpoints' tag names them, not their stored names."""
    return [run for run in state.runs if run.checkpoints.tagged(regime_tag(regime))]


def one_run(state: ExperimentState, regime: str) -> TrainingRun | None:
    """The regime's one run; a second is a new protocol version, so it raises."""
    found = tagged_runs(state, regime)
    if len(found) > 1:
        raise ValueError(f"{regime} admits one run, found {len(found)}")
    return found[0] if found else None


def size_run(state: ExperimentState, label: SizeLabel) -> TrainingRun | None:
    """The size's one run; a run lost at any update is still its size's run."""
    return one_run(state, _regime_id(label))


def endpoint(run: TrainingRun | None) -> int | None:
    """An ended run's last stored update, failed or not; None while it may save more."""
    return last_update(run) if run and run.ended else None


def checkpoint_hours(state: ExperimentState) -> CheckpointHours | None:
    """Stored wall time since each learner's update zero, excluding setup.

    A missing historical timestamp leaves that checkpoint unavailable to timed
    comparisons. Missing update zero leaves the whole run unavailable; never
    substitute another run's origin or infer elapsed time from update counts.
    """
    result: dict[str, dict[int, float]] = {}
    for run in state.runs:
        if size_label(run) is None:
            continue
        initial = run.checkpoints.at("updates", 0).last
        if initial is None or initial.wall_ms is None:
            continue
        elapsed: dict[int, float] = {}
        previous = initial.wall_ms
        for saved in sorted(run.checkpoints, key=checkpoint_update):
            if saved.wall_ms is None:
                continue
            if saved.wall_ms < previous:
                raise ValueError(f"checkpoint wall time went backwards in {run.name}")
            elapsed[checkpoint_update(saved)] = (
                saved.wall_ms - initial.wall_ms
            ) / 3_600_000
            previous = saved.wall_ms
        result[run.name] = elapsed
    return result


def _partner(
    hours: float, small: Mapping[int, float], small_ended: bool
) -> tuple[int, float] | Literal["unreached"] | None:
    """W64's last checkpoint at or before `hours`; None while it may still change.

    The pair is fixed once w64 has stored a later checkpoint or has ended. A w64
    that ended before `hours` never reaches the slot.
    """
    if not small_ended and all(saved <= hours for saved in small.values()):
        return None
    before = [(saved, update) for update, saved in small.items() if saved <= hours]
    if not before or (small_ended and max(small.values()) < hours):
        return "unreached"
    saved, update = max(before)
    return update, saved


def comparison_schedule(
    state: ExperimentState,
    protocol: Protocol,
    hours: CheckpointHours | None = None,
) -> list[ScheduledComparison]:
    """Every expected comparison, playable or not, in one order for play and analysis.

    Milestones past a wall-bounded endpoint remain expected but unplayable, so
    analysis can report how far each size reached. A run that ended below the
    first milestone reached none, and its endpoint comparisons are unreached
    too: an update-0 endpoint against initialization would play a policy
    against itself and read as learning rejected. A milestone that is also an
    endpoint is scheduled once, as a milestone. Equal-hours slots need `hours`
    for both sizes' checkpoints; without them they stay unplayable.
    """
    inference = checkpoint_inference()
    heuristic = Code(name="heuristic", source=_source(), config=HEURISTIC)
    trained = {label: size_run(state, label) for label in SIZES}
    endpoints = {label: endpoint(run) for label, run in trained.items()}
    saved: dict[SizeLabel, dict[int, Checkpoint]] = {
        label: {checkpoint_update(c): c for c in run.checkpoints} if run else {}
        for label, run in trained.items()
    }

    def model(label: SizeLabel, update: int | None) -> Model | None:
        found = saved[label].get(update) if update is not None else None
        return found.model(inference) if found else None

    schedule: list[ScheduledComparison] = []

    def expect(
        tag: str,
        label: SizeLabel,
        update: int | None,
        opponent: Model | Code | None,
        name: str,
        *,
        unreached: bool = False,
    ) -> None:
        candidate = model(label, update)
        schedule.append(
            ScheduledComparison(
                tag=tag,
                update=update,
                matchup=paired_matchup(name, (candidate, opponent), tag, protocol)
                if candidate is not None and opponent is not None and not unreached
                else None,
                unreached=unreached,
            )
        )

    def past_endpoint(label: SizeLabel, update: int) -> bool:
        end = endpoints[label]
        return end is not None and update > end

    def short(label: SizeLabel) -> bool:
        return past_endpoint(label, min(protocol.milestones))

    updates = (0, *protocol.milestones)
    for update in updates:
        for label in SIZES:
            expect(
                f"heuristic-{label}",
                label,
                update,
                heuristic,
                f"{label} u{update} vs heuristic",
                unreached=past_endpoint(label, update),
            )
        if update:
            expect(
                "head-to-head",
                LARGE,
                update,
                model(SMALL, update),
                f"{LARGE} u{update} vs {SMALL} u{update}",
                unreached=any(past_endpoint(label, update) for label in SIZES),
            )
    for label in SIZES:
        end = endpoints[label]
        if end not in updates:
            expect(
                f"heuristic-{label}",
                label,
                end,
                heuristic,
                f"{label} final u{end} vs heuristic",
                unreached=short(label),
            )
        expect(
            f"initial-{label}",
            label,
            end,
            model(label, 0),
            f"{label} final u{end} vs {label} u0",
            unreached=short(label),
        )

    def run_hours(label: SizeLabel) -> Mapping[int, float]:
        run = trained[label]
        return hours.get(run.name, {}) if hours is not None and run else {}

    def equal_hours(update: int | None) -> ScheduledComparison:
        """W160l8 at `update` against w64's last checkpoint at or before its hours."""
        unplayable = ScheduledComparison(
            "equal-hours",
            update,
            None,
            unreached=short(LARGE)
            or (update is not None and past_endpoint(LARGE, update)),
        )
        if unplayable.unreached:
            return unplayable
        small_run, small_hours = trained[SMALL], run_hours(SMALL)
        at = run_hours(LARGE).get(update) if update is not None else None
        if at is None or small_run is None or not small_hours:
            return unplayable
        found = _partner(at, small_hours, small_run.ended)
        if found == "unreached":
            return replace(unplayable, unreached=True)
        candidate = model(LARGE, update)
        partner = model(SMALL, found[0]) if found is not None else None
        if found is None or candidate is None or partner is None:
            return unplayable
        point = f"u{update}" if update in protocol.milestones else f"final u{update}"
        return ScheduledComparison(
            tag="equal-hours",
            update=update,
            matchup=paired_matchup(
                f"{LARGE} {point} vs {SMALL} u{found[0]} at equal hours",
                (candidate, partner),
                "equal-hours",
                protocol,
            ),
            hours=(at, found[1]),
        )

    def half_training(label: SizeLabel, update: int | None) -> ScheduledComparison:
        """Progress against this learner's latest trained checkpoint by half time."""
        tag = f"half-training-{label}"
        slot = ScheduledComparison(
            tag,
            update,
            None,
            unreached=short(label)
            or (update is not None and past_endpoint(label, update)),
        )
        if slot.unreached or update is None:
            return slot
        clock = run_hours(label)
        at = clock.get(update)
        if at is None:
            return slot
        eligible = [(t, u) for u, t in clock.items() if 0 < u < update and t <= at / 2]
        if not eligible:
            # The first export may have no trained half-time predecessor. It is
            # unavailable, never replaced with initialization or a later policy.
            missing = any(0 < u < update and u not in clock for u in saved[label])
            return replace(slot, unreached=not missing)
        partner_hours, partner_update = max(eligible)
        if any(partner_update < u < update and u not in clock for u in saved[label]):
            return slot
        candidate, partner = model(label, update), model(label, partner_update)
        if candidate is None or partner is None:
            return slot
        return ScheduledComparison(
            tag,
            update,
            paired_matchup(
                f"{label} u{update} vs {label} u{partner_update} at half training",
                (candidate, partner),
                tag,
                protocol,
            ),
            hours=(at, partner_hours),
        )

    # Primary progress comparisons come first so limited evaluation spend serves
    # the newest learner against its own earlier policy before secondary baselines.
    progress = [
        half_training(label, update)
        for update in protocol.milestones
        for label in SIZES
    ]
    progress.extend(
        half_training(label, endpoints[label])
        for label in SIZES
        if endpoints[label] not in protocol.milestones
    )

    # Equal wall budget when both finished; analysis reads no scale otherwise.
    expect(
        "endpoint-head-to-head",
        LARGE,
        endpoints[LARGE],
        model(SMALL, endpoints[SMALL]),
        f"{LARGE} final vs {SMALL} final",
        unreached=any(short(label) for label in SIZES),
    )
    # Last, so a pair that is also an equal-update or endpoint pair keeps that
    # comparison's name and tag when `eval` plays it once.
    schedule.extend(equal_hours(update) for update in protocol.milestones)
    if endpoints[LARGE] not in protocol.milestones:
        schedule.append(equal_hours(endpoints[LARGE]))
    return progress + schedule


def eval(state: ExperimentState) -> list[Job]:
    """Play the available comparisons; Etude owns already-delivered matches."""
    if state.experiment != EXPERIMENT:
        raise ValueError("frontier evaluation requires this experiment's runs")
    _, protocol = frozen()
    closed = with_launcher_ends(state)
    return evaluation_jobs(
        "tla-frontier-c5fba4-paired-v1",
        environment(protocol),
        _distinct(
            comparison.matchup
            for comparison in comparison_schedule(
                closed, protocol, checkpoint_hours(closed)
            )
            if comparison.matchup is not None
        ),
    )


def _distinct(matchups: Iterable[Matchup]) -> tuple[Matchup, ...]:
    """Each set of matches once, under its first comparison's name.

    An equal-hours pair can be an equal-update or endpoint pair as well, most
    often at the endpoints when both runs stop at the wall. Analysis reads games
    by match key, so both slots read the one set of games.
    """
    seen: set[frozenset[str]] = set()
    kept: list[Matchup] = []
    for matchup in matchups:
        keys = frozenset(match.key for match in matchup.matches())
        if keys not in seen:
            seen.add(keys)
            kept.append(matchup)
    return tuple(kept)


def pass_name(name: str) -> str:
    """A benchmark's name under this attempt; with no attempt set, `name` itself.

    `etude eval` names a pass for its tournament and pending matches, so without
    a new name a pass that delivered nothing could never launch again. The name
    sits outside match keys and environments, so an attempt skips stored matches,
    leaves failed ones failed and changes nothing scored.
    """
    attempt = os.environ.get(PASS_ATTEMPT)
    return f"{name}-attempt-{attempt}" if attempt else name


def evaluation_jobs(
    name: str, env: Environment, matchups: Sequence[Matchup]
) -> list[Job]:
    """One rented pass of `play` over `matchups`, none without any.

    Every check beside the frontier launches this same job.
    """
    if not matchups:
        return []
    job = evaluation(
        Benchmark(name=pass_name(name), environment=env, matchups=tuple(matchups)),
        # Etude discovers this file with runpy; callbacks need an importable
        # module identity even when eval itself came from <run_path>.
        play=tla_frontier.play,
        workers=EVALUATION_WORKERS,
        max_hours=EVALUATION_HOURS,
        # One pass's ceiling across its attempts; the operator keeps the
        # cumulative evaluation spend under `protocol.evaluation_usd`.
        max_usd=EVALUATION_PASS_USD,
        # Etude retries a machine lost or refused after admission, resuming the
        # same run and skipping stored matches, so no game result can be replaced.
        # A refused first create raises instead, which ends the watch loop.
        max_attempts=EVALUATION_ATTEMPTS,
        setup=machine_setup(" --extra evaluation"),
    )
    # Etude's default command omits the extra holding its result writer (duckdb).
    command = job.command.replace(
        "uv run python",
        f"{TOOLCHAIN_ENV}\nuv run --locked --extra evaluation python",
        1,
    )
    return [job.model_copy(update={"command": command})]


# -- Evaluation callback ------------------------------------------------------


def deck_matchup(regime: TrainingRegime, protocol: Protocol, pair: Pair) -> DeckMatchup:
    """The frozen pair's two lists, as the regime's population names them."""
    source = regime.deck_source
    assert source is not None and source.entry_ids is not None
    return DeckMatchup(
        population_sha256=protocol.corpus_sha256,
        entries=(source.entry_ids[pair.first], source.entry_ids[pair.second]),
        lists=(source.decks[pair.first], source.decks[pair.second]),
    )


def _registration(
    player: Model | Code,
    name: Literal["candidate", "opponent"],
    matchup: DeckMatchup,
    observation: ObservationConfig,
) -> tuple[PlayerRegistration, Path | None]:
    setup = matchup.match()
    fingerprints = runtime_fingerprints(match_config=setup)
    common = dict(
        player_id=name,
        display_name=player.name,
        role="challenger" if name == "candidate" else "anchor",
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=fingerprints["observation_abi_sha256"],
        action_abi_sha256=fingerprints["action_abi_sha256"],
        matchup_sha256=canonical_sha256(setup.model_dump(mode="json")),
        player_seed_derivation_id="arena-pair-deal-player-v1",
        compute_class_id="policy-cpu-one-thread-one-pass",
    )
    if isinstance(player, Code):
        if player.config != HEURISTIC:
            raise ValueError("heuristic support differs from frontier protocol")
        return PlayerRegistration(
            **common,
            runner_kind="code",
            player_spec=dict(HEURISTIC),
            source_sha256=canonical_sha256(
                {str(p.relative_to(ROOT)): file_sha256(p) for p in _source()}
            ),
        ), None
    path = local_checkpoint_file(player)
    agent, space = load_checkpoint(str(path))
    if space.encoder.config != observation:
        raise ValueError("checkpoint observation differs from frontier protocol")
    digest = file_sha256(path)
    return PlayerRegistration(
        **common,
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "device": "cpu",
            "batch_size": 1,
            "deterministic": False,
            "allow_deck_repetition": True,
        },
        checkpoint_sha256=digest,
        checkpoint_bytes=path.stat().st_size,
        parameter_count=sum(p.numel() for p in agent.parameters()),
        artifact_id=f"c5fba4/{digest}",
    ), path


@contextmanager
def _memory_journal(retained: Path) -> Iterator[Path]:
    """A pair's arena directory in memory, copied to `retained` even on failure.

    The arena syncs every record, which a rented container disk serves at about
    .1 s, and that time counts against the game limit, so a game a slow size
    could finish would fail instead. `experiments.learning_signal` measured this
    and is mirrored here because its source is hashed into retained environments.
    """
    memory = Path("/dev/shm")
    with TemporaryDirectory(dir=memory if memory.is_dir() else None) as journal:
        out_dir = Path(journal) / "deal"
        try:
            yield out_dir
        finally:
            if out_dir.is_dir():
                shutil.copytree(out_dir, retained, dirs_exist_ok=True)


def play(match: Match) -> Tables:
    """One sampled pair/deal, four policy/deck/seat legs, with failure denominators."""
    torch.set_num_threads(1)
    regimes, protocol = frozen()
    # The sizes share observation and population; only their models differ.
    regime = regimes[SMALL]
    pair = Pair.model_validate(match.conditions)
    if pair not in protocol.pairs or match.games != GAMES_PER_PAIR:
        raise ValueError("evaluation conditions differ from frozen schedule")
    matchup = deck_matchup(regime, protocol, pair)
    candidate, candidate_path = _registration(
        match.players[0], "candidate", matchup, regime.observation
    )
    opponent, opponent_path = _registration(
        match.players[1], "opponent", matchup, regime.observation
    )
    paths = {
        name: str(path)
        for name, path in (("candidate", candidate_path), ("opponent", opponent_path))
        if path is not None
    }
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="tla-frontier-c5fba4-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    retained = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    with _memory_journal(retained) as out_dir:
        result = evaluate(
            EvaluationRequest(
                key=key,
                candidate=candidate,
                opponent=opponent,
                checkpoint_paths=paths,
                deal_seeds=(pair.deal,),
                authored_matchup=matchup,
                game_seconds=protocol.game_seconds,
                max_commands=protocol.max_commands,
                bootstrap_seed=protocol.bootstrap_seed,
                purpose="tla-frontier-fixed-corpus",
            ),
            out_dir=out_dir,
        )
    by_leg = {row.leg: row for row in result.rows}
    rows: list[Row] = []
    for leg in range(GAMES_PER_PAIR):
        row = by_leg.get(leg)
        valid = row is not None and row.valid
        score = row.score_a if valid else None
        actions = row.policy_action_counts if row else None
        decks = matchup.decks(leg)
        for seat in (0, 1):
            game_seat = (leg + seat) % 2
            rows.append(
                {
                    "seat": seat,
                    "game": leg,
                    "game_seat": game_seat,
                    "deck": decks[game_seat],
                    "opponent_deck": decks[1 - game_seat],
                    "deal_seed": pair.deal,
                    "corpus_sha256": protocol.corpus_sha256,
                    "score": score if seat == 0 or score is None else 1 - score,
                    "valid": valid,
                    "attempted": row is not None,
                    "terminated": row.terminated if row else False,
                    "truncated": row.truncated if row else False,
                    "failure": row.failure if row else result.error or "missing game",
                    "seconds": row.game_seconds if row else None,
                    # The game's, repeated on each seat like `seconds`. A game
                    # killed at its limit returns no count.
                    "policy_decisions": actions.decisions if actions else None,
                    "policy_dropped_actions": actions.dropped_actions
                    if actions
                    else None,
                }
            )
    return {"games": rows}


def report() -> Report:
    """Progress curves by update; `experiments.tla_frontier_analysis` owns claims.

    Etude's report axes are checkpoint clocks, and hours are not one yet, so the
    equal-hours curve is drawn at w160l8's update; a pair that is also an
    equal-update or endpoint pair is played once, under that tag.
    """
    return Report(
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(
                title="Width 64 vs itself at half the training hours (primary)",
                matches=Matches(tag=f"half-training-{SMALL}"),
            ),
            Curve(
                title="Width 160, 8 layers vs itself at half the training hours (primary)",
                matches=Matches(tag=f"half-training-{LARGE}"),
            ),
            Curve(
                title="Width 64 vs fixed heuristic",
                matches=Matches(tag=f"heuristic-{SMALL}"),
            ),
            Curve(
                title="Width 160, 8 layers vs fixed heuristic",
                matches=Matches(tag=f"heuristic-{LARGE}"),
            ),
            Curve(
                title="Width 160, 8 layers vs width 64 at equal updates",
                matches=Matches(tag="head-to-head"),
            ),
            # Drawn at w160l8's updates; each pair's two hours are in analysis.
            Curve(
                title="Width 160, 8 layers vs width 64 at equal training hours",
                matches=Matches(tag="equal-hours"),
            ),
        ),
    )
