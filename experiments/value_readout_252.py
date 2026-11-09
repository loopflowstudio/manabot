"""MTG-252 critic readout arms, derived from the MTG-183 Elemental recipe.

The base is `elemental-long-183-v1.json` (ten Elemental decks, uniform
per-game pairings, two-seat self-play) with only its world re-pinned to the
native runtime's. The comparison then changes one serialized field,
`agent.value_aggregation`, and nothing else: depth, width, heads, capacities,
learning rule, filter, lambdas and execution stay those of that base. The
study design lives in `experiments/value-readout-252.md`; this module owns the
arm derivation, its single-axis check and the static per-arm model costs that
the protocol reports before any training, and `runs()`, the Etude launch of
every arm under one shared seed list (3 arms x 3 seeds x 150 updates inside a
USD20 study cap that also covers evaluation):

    doppler run -p etude -c prd -- doppler run -p manabot -c prd \\
        --preserve-env=RUNPOD_API_KEY -- \\
        uv run --with "etude-sdk[s3,eval]==0.2.14" etude launch \\
        experiments/value_readout_252.py --rent 07c490-l4 --cohort mtg252 \\
        --parallel 9 --max-usd 12.15 --revision COMMIT

The first segments paused at their watchdog after 92-112 updates, so each run
continues to update 150 with its raw/EMA/Adam state. `etude continue` cannot do
this: it carries a policy checkpoint only, and the worker refuses it. With each
parent's `run.json` and exports fetched to `DIR/<parent run>/`:

    uv run python -m experiments.value_readout_252 stage --parents DIR \\
        --out continuations.json

then launch as above with `--continuations continuations.json --hours 1.5
--usd-per-run 1.05`, at the parents' revision. The third segment (150 -> 300)
stages the update-150 runs with `--after 2` and launches with `--after 2
--updates 300 --recover-capacity yes --hours 3.75 --usd-per-run 2.4`.

Evaluation (`eval`) scores updates 20, 40 and 75 of each first segment and 150 of
its continuation against the heuristic, and the three final policies of each seed
against each other, all over 07c490's fifty fixed random Elemental pairings
(`learning_signal.ELEMENTAL_PAIRS`, four cross legs each), on one rented CPU host:

    etude eval experiments/value_readout_252.py --rent cpu32 --source r2://manabot/jobs
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations, product
import json
from pathlib import Path

from etude.eval.records import Code, Inference, Matchup
from etude.experiment import Checkpoint, ExperimentState
from etude.launch.records import Job, Run as EtudeRun
from etude.run.storage import storage_from_uri
import torch

from experiments.elemental_long import stage_parent
from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    FINALIZATION_SECONDS,
    SETUP_SECONDS,
    Conditions,
    benchmark_jobs,
    training_runs,
)
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import LearnerContinuation, TrainingJob
from manabot.infra.config import ValueAggregation
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.builders import with_capacity, with_value_aggregation
from manabot.training.execution import validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym

ROOT = Path(__file__).resolve().parents[1]
BASE_REGIME = ROOT / "experiments/regimes/elemental-long-183-v1.json"
# Canonical order; the baseline is the 183 recipe's own masked mean.
ARMS: tuple[ValueAggregation, ...] = ("masked_mean", "value_token", "attention_readout")
EXPERIMENT = "value_readout_252"
SEEDS = "25201,25202,25203"
UPDATES = 150
# MTG-181 measured 28-36 s per two-seat update at 256 streams on an L4, so 150
# updates need at most 90 min; the 2 h window leaves 100 min after 15 min setup
# and 5 min finalization. Etude reserves window plus setup at the $0.60/h ceiling.
# The watchdog stops a slower run with its exports.
HOURS = 2.0
USD_PER_RUN = 1.35
# The first segment of each arm and seed, which later segments continue.
PARENTS: dict[tuple[ValueAggregation, int], str] = {
    ("masked_mean", 25201): "c296951dc634",
    ("value_token", 25201): "c5a1a009338c",
    ("attention_readout", 25201): "e2118d949d3a",
    ("masked_mean", 25202): "d70f68c8b612",
    ("value_token", 25202): "cf95f6d220f1",
    ("attention_readout", 25202): "4c0fc018fc8b",
    ("masked_mean", 25203): "4782034c5274",
    ("value_token", 25203): "1488ddda75ca",
    ("attention_readout", 25203): "295d9d24de78",
}
# The continuation that took each first segment to update 150. The value_token
# seed 25202 continuation c40a0c32bd05 stopped at update 124 on a capacity
# overflow; 89dd4b6d2b8e repeated it with capacity recovery.
FINALS: dict[tuple[ValueAggregation, int], str] = {
    ("masked_mean", 25201): "dfafec31a173",
    ("value_token", 25201): "6891b75cb5f8",
    ("attention_readout", 25201): "9081d387b3a1",
    ("masked_mean", 25202): "4cfb8393137f",
    ("value_token", 25202): "89dd4b6d2b8e",
    ("attention_readout", 25202): "f80ce800ed34",
    ("masked_mean", 25203): "e564b345c409",
    ("value_token", 25203): "03eab5232afd",
    ("attention_readout", 25203): "551b7e26e666",
}
# The continuation that took each update-150 learner to update 300, with
# capacity recovery in every arm.
THIRDS: dict[tuple[ValueAggregation, int], str] = {
    ("masked_mean", 25201): "9dc7467fe36f",
    ("value_token", 25201): "38ffa8f894e4",
    ("attention_readout", 25201): "18d587226949",
    ("masked_mean", 25202): "8691031b8f42",
    ("value_token", 25202): "6e5142aa73fb",
    ("attention_readout", 25202): "4aa2ace09d06",
    ("masked_mean", 25203): "f58d5f801868",
    ("value_token", 25203): "6398381e57a5",
    ("attention_readout", 25203): "63ad2b5da85c",
}
# Each segment's parents: segment 2 continues the first segments, segment 3
# the update-150 runs.
SEGMENT_PARENTS: dict[int, dict[tuple[ValueAggregation, int], str]] = {
    1: PARENTS,
    2: FINALS,
}
FIRST_SEGMENT_UPDATES = (20, 40, 75)
THIRD_SEGMENT_UPDATES = (200, 250, 300)
_SCORED = (*FIRST_SEGMENT_UPDATES, UPDATES, *THIRD_SEGMENT_UPDATES)
EVALUATION = "value-readout-252-v1"
# About 2,250 pairings (9,000 games); 07c490's pass played about one pairing a
# second on this host. The limit covers 1.25 h plus setup at the $2/h ceiling.
EVALUATION_HOURS = 1.25
EVALUATION_USD = 3.0
# A second axis, read against the depth-2 runs above: the same arms with four
# attention layers instead of two, fresh under the same seeds, recovering
# capacity from the start (a first masked-mean launch without it stopped at
# update 2 on 65 continuation relations, 64 slots, and was cancelled). It asks
# whether encoder depth, not the readout, limits play. The masked-mean runs
# stopped at update 150 and continue to 300 in `DEEPER_THIRDS`; the value-token
# runs train to 300 in one segment.
DEPTH_LAYERS = 4
DEEPER: dict[tuple[ValueAggregation, int], str] = {
    ("masked_mean", 25201): "bdcf25600bf7",
    ("masked_mean", 25202): "f9b2f54106e3",
    ("masked_mean", 25203): "aaf586b81509",
    ("value_token", 25201): "1c4aef332e5b",
    ("value_token", 25202): "8f11bcda7841",
    ("value_token", 25203): "7d5cba6b285a",
}
DEEPER_THIRDS: dict[tuple[ValueAggregation, int], str] = {
    ("masked_mean", 25201): "f1e5c84269d0",
    ("masked_mean", 25202): "072977f3704d",
    ("masked_mean", 25203): "f1664004d74e",
}
# Only masked mean has an update-150 restart; the token trains straight to 300.
DEPTH_SEGMENT_PARENTS: dict[int, dict[tuple[ValueAggregation, int], str]] = {
    1: {key: run for key, run in DEEPER.items() if key[0] == "masked_mean"}
}
# Inputs live in the bucket of the machines' storage, keyed from its root.
STAGING_BUCKET = "r2://manabot"
STAGING_PREFIX = "mtg252/continue"


def base() -> TrainingRegime:
    """MTG-183's recipe in the native world; the frozen 183 file is unchanged.

    Re-pinning resolves the saved observation at the current fresh policy
    input. The protocol pins the resulting world, shapes and source.
    """
    saved = TrainingRegime.model_validate_json(BASE_REGIME.read_bytes())
    if saved.agent.value_aggregation != ARMS[0]:
        raise ValueError("183 baseline is no longer the masked-mean recipe")
    regime = saved.model_copy(
        update={"id": "value-readout-252-base", "world": managym.WORLD_VERSION}
    )
    return validate_regime(regime)


def arm(aggregation: ValueAggregation, updates: int | None = None) -> TrainingRegime:
    """The re-pinned 183 recipe with only its critic readout replaced.

    `updates` sets the stage's absolute target for every arm alike. Damped
    learning rates follow the absolute iteration, so a shorter target leaves
    the learning schedule unchanged.
    """
    regime = base()
    if updates is not None:
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        stage.updates = updates
    derived = with_value_aggregation(
        regime, id=f"value-readout-252-{aggregation}", aggregation=aggregation
    )
    require_single_axis(regime, derived)
    return derived


def deeper(
    aggregation: ValueAggregation, layers: int, updates: int | None = None
) -> TrainingRegime:
    """An arm with `layers` attention blocks; nothing else changes."""
    baseline = arm(aggregation, updates)
    agent = baseline.agent
    derived = with_capacity(
        baseline,
        id=f"value-readout-252-{aggregation}-depth{layers}",
        width=agent.hidden_dim,
        depth=layers,
        heads=agent.num_attention_heads,
    )
    left = baseline.model_dump(mode="json")
    right = derived.model_dump(mode="json")
    for dump in (left, right):
        dump.pop("id")
        dump["agent"].pop("attention_layers")
    if left != right:
        raise ValueError("depth arm changes more than agent.attention_layers")
    return derived


@dataclass(frozen=True)
class Run:
    """One training run: an arm under one shared run seed.

    The run seed fixes initialization and the collection seed
    (`run seed + 10000`). Native game `k` on stream `s` is seeded
    `collection + s + k * streams`, and its Elemental pairing and deal are a
    function of that seed alone, so every arm under one run seed meets the same
    pairing sequence on every stream whatever its policy does. Arms differ only
    in how far through that sequence they are at a given update.
    """

    aggregation: ValueAggregation
    seed: int
    regime: TrainingRegime


def grid(seeds: Sequence[int], updates: int | None = None) -> list[Run]:
    """Every arm under every seed: one seed list, never per-arm seeds."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("readout runs need distinct seeds shared by every arm")
    regimes = {aggregation: arm(aggregation, updates) for aggregation in ARMS}
    return [
        Run(aggregation=aggregation, seed=seed, regime=regimes[aggregation])
        for seed in seeds
        for aggregation in ARMS
    ]


def runs(
    revision: str,
    seeds: str = SEEDS,
    updates: int = UPDATES,
    hours: float = HOURS,
    usd_per_run: float = USD_PER_RUN,
    continuations: str = "",
    arms: str = ",".join(ARMS),
    recover_capacity: str = "",
    after: int = 1,
    depth: int = 0,
) -> list[EtudeRun]:
    """Etude training runs: every arm under the same comma-separated seeds.

    `continuations` names a file from `stage`; each run then continues its
    parent's learner to `updates` instead of starting fresh; `after` names the
    segment whose runs are the parents (`SEGMENT_PARENTS`). A non-zero `depth`
    launches the `arms` at that depth instead, fresh or continuing
    `DEPTH_SEGMENT_PARENTS[after]`, with `recover_capacity` alike. `arms` narrows a
    continuation to some arms, and any non-empty `recover_capacity` makes a game
    that overflows observation capacity end alone instead of stopping the run.
    """
    chosen = tuple(int(seed) for seed in seeds.split(","))
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("readout runs need distinct seeds shared by every arm")
    if depth:
        return _depth_runs(
            revision,
            chosen,
            tuple(_aggregation(name) for name in arms.split(",")),
            depth,
            updates=updates,
            hours=hours,
            usd_per_run=usd_per_run,
            continuations=continuations,
            recover_capacity=bool(recover_capacity),
            after=after,
        )
    planned = grid(chosen, updates)
    if continuations:
        staged = json.loads(Path(continuations).read_text())
        return [
            run
            for planned_run in planned
            if planned_run.aggregation in arms.split(",")
            for run in segment(
                planned_run,
                LearnerContinuation.model_validate(
                    staged[
                        SEGMENT_PARENTS[after][
                            planned_run.aggregation, planned_run.seed
                        ]
                    ]
                ),
                hours,
                bool(recover_capacity),
            )
            .job(
                repo=ROOT,
                revision=revision,
                max_hours=hours,
                max_usd=Decimal(str(usd_per_run)),
            )
            .runs(planned_run.seed)
        ]
    return training_runs(
        (arm(aggregation, updates) for aggregation in ARMS),
        updates=updates,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=tuple(sorted({run.seed for run in planned})),
        hours=hours,
        usd_per_run=usd_per_run,
    )


def _aggregation(name: str) -> ValueAggregation:
    for aggregation in ARMS:
        if aggregation == name:
            return aggregation
    raise ValueError(f"unknown arm {name!r}")


def _depth_runs(
    revision: str,
    seeds: tuple[int, ...],
    aggregations: tuple[ValueAggregation, ...],
    depth: int,
    *,
    updates: int,
    hours: float,
    usd_per_run: float,
    continuations: str,
    recover_capacity: bool,
    after: int,
) -> list[EtudeRun]:
    """Depth arms, fresh or continuing their parents, scored at every update
    this study scores that falls inside the segment."""
    staged = json.loads(Path(continuations).read_text()) if continuations else {}
    found: list[EtudeRun] = []
    for seed in seeds:
        for aggregation in aggregations:
            planned = Run(aggregation, seed, deeper(aggregation, depth, updates))
            if continuations:
                parent = DEPTH_SEGMENT_PARENTS[after][aggregation, seed]
                job = segment(
                    planned,
                    LearnerContinuation.model_validate(staged[parent]),
                    hours,
                    recover_capacity,
                )
            else:
                job = _fresh(planned.regime, hours, recover_capacity)
            found += job.job(
                repo=ROOT,
                revision=revision,
                max_hours=hours,
                max_usd=Decimal(str(usd_per_run)),
            ).runs(seed)
    return found


def _fresh(regime: TrainingRegime, hours: float, recover_capacity: bool) -> TrainingJob:
    """A first segment exporting every scored update before its end."""
    wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    regime = regime.model_copy(deep=True)
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    regime.wall_seconds = wall_seconds
    stage.execution.wall_seconds = wall_seconds
    if recover_capacity:
        assert stage.game_errors is not None
        stage.game_errors.recover_capacity = True
    return TrainingJob(
        regime=TrainingRegime.model_validate(regime.model_dump(mode="json")),
        experiment=EXPERIMENT,
        milestones=tuple(u for u in _SCORED if u < stage.updates),
        finalization_seconds=FINALIZATION_SECONDS,
    )


def eval(state: ExperimentState) -> list[Job]:
    return benchmark_jobs(
        evaluation_matchups(state),
        EVALUATION,
        ELEMENTAL_PAIRS,
        hours=EVALUATION_HOURS,
        usd=EVALUATION_USD,
    )


def evaluation_matchups(state: ExperimentState) -> tuple[Matchup, ...]:
    """Each arm and seed against the heuristic, then each seed's finals pairwise.

    A checkpoint not yet stored is left out, so a later pass plays it. In a
    head-to-head the earlier arm in `ARMS` is always the first player; the
    pairing's four legs give each arm both decks on the play and on the draw.
    """
    source = [*sorted((ROOT / "manabot").rglob("*.py")), Path(__file__)]
    inference = Inference(source=source)
    heuristic = Code(name="heuristic", source=source)
    conditions = tuple(
        Conditions(design=ELEMENTAL_PAIRS.name, **pairing.model_dump()).model_dump()
        for pairing in ELEMENTAL_PAIRS.pairings
    )
    legs = ELEMENTAL_PAIRS.legs
    found: list[Matchup] = []
    for (aggregation, seed), parent in PARENTS.items():
        scheduled = [
            (update, _checkpoint(state, parent, update))
            for update in FIRST_SEGMENT_UPDATES
        ] + [(UPDATES, _checkpoint(state, FINALS[aggregation, seed], UPDATES))]
        if (aggregation, seed) in THIRDS:
            scheduled += [
                (update, _checkpoint(state, THIRDS[aggregation, seed], update))
                for update in THIRD_SEGMENT_UPDATES
            ]
        found += [
            Matchup(
                name=f"{aggregation} s{seed} u{update} vs heuristic",
                players=(checkpoint.model(inference), heuristic),
                conditions=conditions,
                games=legs,
                tags=("heuristic", aggregation),
            )
            for update, checkpoint in scheduled
            if checkpoint is not None
        ]
    deep = f"depth{DEPTH_LAYERS}"
    for aggregation, seed in DEEPER:
        found += [
            Matchup(
                name=f"{aggregation} {deep} s{seed} u{update} vs heuristic",
                players=(checkpoint.model(inference), heuristic),
                conditions=conditions,
                games=legs,
                tags=("heuristic", f"{aggregation}-{deep}"),
            )
            for update in _SCORED
            if (checkpoint := _deep(state, aggregation, seed, update)) is not None
        ]
    # Each depth-4 arm against the same arm at depth 2, then depth-4 arms
    # against each other, at 150 and 300; the depth-2 or earlier arm plays first.
    shallow_at = {UPDATES: FINALS, THIRD_SEGMENT_UPDATES[-1]: THIRDS}
    for (aggregation, seed), update in product(DEEPER, shallow_at):
        shallow = _checkpoint(state, shallow_at[update][aggregation, seed], update)
        deeper_one = _deep(state, aggregation, seed, update)
        if shallow is not None and deeper_one is not None:
            found.append(
                Matchup(
                    name=(
                        f"{ARMS[0]} v {deep} s{seed} u{update}"
                        if aggregation == ARMS[0]
                        else f"{aggregation} v {aggregation} {deep} s{seed} u{update}"
                    ),
                    players=(shallow.model(inference), deeper_one.model(inference)),
                    conditions=conditions,
                    games=legs,
                    tags=("head-to-head", "depth"),
                )
            )
    for seed, update in product(sorted({s for _, s in DEEPER}), shallow_at):
        for first, second in combinations(ARMS, 2):
            left = _deep(state, first, seed, update)
            right = _deep(state, second, seed, update)
            if left is not None and right is not None:
                found.append(
                    Matchup(
                        name=f"{first} {deep} v {second} {deep} s{seed} u{update}",
                        players=(left.model(inference), right.model(inference)),
                        conditions=conditions,
                        games=legs,
                        tags=("head-to-head", deep),
                    )
                )
    finals = [(FINALS, UPDATES), (THIRDS, THIRD_SEGMENT_UPDATES[-1])]
    for seed in sorted({seed for _, seed in PARENTS}):
        for (runs_at, update), (first, second) in product(
            finals, combinations(ARMS, 2)
        ):
            if (first, seed) not in runs_at or (second, seed) not in runs_at:
                continue
            left = _checkpoint(state, runs_at[first, seed], update)
            right = _checkpoint(state, runs_at[second, seed], update)
            if left is None or right is None:
                continue
            found.append(
                Matchup(
                    name=f"{first} v {second} s{seed} u{update}",
                    players=(left.model(inference), right.model(inference)),
                    conditions=conditions,
                    games=legs,
                    tags=("head-to-head",),
                )
            )
    return tuple(found)


def _deep(
    state: ExperimentState, aggregation: ValueAggregation, seed: int, updates: int
) -> Checkpoint | None:
    """Keep scored checkpoint identities stable across learner continuations.

    A continuation also exports its initial update-150 state. That new run's
    checkpoint must not replace the parent's already-scored update-150 export.
    """
    key = (aggregation, seed)
    runs_at = DEEPER_THIRDS if updates > UPDATES and key in DEEPER_THIRDS else DEEPER
    run = runs_at.get(key)
    return None if run is None else _checkpoint(state, run, updates)


def _checkpoint(state: ExperimentState, run: str, updates: int) -> Checkpoint | None:
    for each in state.runs:
        if each.name == run:
            return each.checkpoints.at("updates", updates).last
    return None


def segment(
    run: Run,
    continuation: LearnerContinuation,
    hours: float,
    recover_capacity: bool = False,
) -> TrainingJob:
    """One arm and seed continued from its parent's raw/EMA/Adam exports.

    Only the watchdog differs from the parent's regime, and `recover_capacity`
    when set: it acts only on a game that overflows capacity, which is discarded
    and replaced, recorded as `resource_limit`. Game streams and RNGs
    restart, so a continued run leaves the pairing sequence its first segment
    followed; every arm under a seed restarts from its own boundary.
    """
    wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    regime = run.regime.model_copy(deep=True)
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    regime.wall_seconds = wall_seconds
    stage.execution.wall_seconds = wall_seconds
    if recover_capacity:
        assert stage.game_errors is not None
        stage.game_errors.recover_capacity = True
    return TrainingJob(
        regime=TrainingRegime.model_validate(regime.model_dump(mode="json")),
        experiment=EXPERIMENT,
        # Scored updates inside this segment; its end is saved regardless.
        milestones=tuple(
            u for u in _SCORED if continuation.from_update < u < stage.updates
        ),
        finalization_seconds=FINALIZATION_SECONDS,
        continuation=continuation,
    )


def require_single_axis(baseline: TrainingRegime, candidate: TrainingRegime) -> None:
    """Reject any resolved difference other than the label and value readout."""
    left = baseline.model_dump(mode="json")
    right = candidate.model_dump(mode="json")
    for dump in (left, right):
        dump.pop("id")
        dump["agent"].pop("value_aggregation")
    if left != right:
        raise ValueError("readout arm changes more than agent.value_aggregation")


@dataclass(frozen=True)
class ArmCost:
    """Static model cost of one arm at the recipe's capacities.

    `attention_pairs` counts query-key products per observation across all
    object-attention blocks plus any readout query; it is a size measure, not
    measured time. Shared parameters are identical to the baseline's under the
    same initialization seed except where `shares_initialization` is false.
    """

    aggregation: ValueAggregation
    parameters: int
    added_parameters: int
    objects: int
    attention_pairs: int
    shares_initialization: bool


def costs(seed: int = 252) -> list[ArmCost]:
    regime = base()
    space = ObservationSpace(regime.observation)
    models: dict[ValueAggregation, ActorCritic] = {}
    for aggregation in ARMS:
        torch.manual_seed(seed)
        models[aggregation] = ActorCritic(space, arm(aggregation).agent)
    baseline = models[ARMS[0]].state_dict()
    baseline_size = sum(p.numel() for p in models[ARMS[0]].parameters())
    layers = regime.agent.attention_layers
    # Visible object sequence: both players, then each seat's cards and permanents.
    capacity = regime.observation
    objects = 2 + 2 * (
        capacity.max_cards_per_player + capacity.max_permanents_per_player
    )
    result = []
    for aggregation, model in models.items():
        state = model.state_dict()
        size = sum(p.numel() for p in model.parameters())
        sequence = objects + (aggregation == "value_token")
        result.append(
            ArmCost(
                aggregation=aggregation,
                parameters=size,
                added_parameters=size - baseline_size,
                objects=objects,
                attention_pairs=layers * sequence**2
                + (objects if aggregation == "attention_readout" else 0),
                shares_initialization=all(
                    torch.equal(state[key], value) for key, value in baseline.items()
                ),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage", help="upload parents' exports as Etude inputs")
    stage.add_argument("--parents", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.add_argument("--after", type=int, default=1, choices=SEGMENT_PARENTS)
    stage.add_argument("--depth", type=int, default=0, choices=(0, DEPTH_LAYERS))
    args = parser.parse_args()
    parents = DEPTH_SEGMENT_PARENTS if args.depth else SEGMENT_PARENTS
    storage = storage_from_uri(STAGING_BUCKET)
    staged = {
        parent: stage_parent(
            args.parents / parent, storage, f"{STAGING_PREFIX}/{parent}", seed
        ).model_dump(mode="json")
        for (_, seed), parent in parents[args.after].items()
    }
    args.out.write_text(json.dumps(staged, indent=2) + "\n")


if __name__ == "__main__":
    main()
