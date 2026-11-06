"""Frozen c5fba4 analysis over retained Etude matches and training checkpoints.

`analyze` accounts for every comparison `comparison_schedule` expects before
estimating anything, then applies the protocol's readings: learning per size,
strength against the heuristic, and scale from the equal-compute endpoint
head-to-head. A milestone past an ended run's endpoint is unreached rather than
missing, because no run can still produce it; every other absent comparison
leaves the analysis incomplete and its reading unknown. A failed run keeps its
learning and strength readings at its last stored checkpoint, with `state`
beside them, but its head-to-head score is not equal compute, so `scale` needs
both runs finished under the wall budget, neither stopped by the update cap.

A failed game stays failed: nothing replaces it and it leaves the denominator
unchanged. Its comparison has no `score`, but once every pair has a match
receipt its `result` holds `worst` and `best`, which count each unscored leg as
a loss and as a win, and readings use those bounds. Bounds rather than
exclusion, because failures are not neutral: limits and kills fall on long games.

First-64 cutoff totals sit beside the scores. An overflow is accepted behavior,
so a nonzero count adds no issue and `complete` stays about missing and failed
games. The totals span both players, so they support no per-policy rate.

Each pair's four correlated legs move together, and one set of bootstrap
indices serves every comparison. Etude keeps each checkpoint's wall time only
in its database, so `TrainingPoint` reads it here to show how far each size got
per hour. The CLI writes JSON; it neither runs games nor replaces evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
from etude.eval.query import statsdb
from etude.experiment import ExperimentState, TrainingRun, observe
import numpy as np
import numpy.typing as npt

from experiments.tla_frontier import (
    EXPERIMENT,
    PROTOCOL,
    SIZES,
    Protocol,
    ScheduledComparison,
    SizeLabel,
    comparison_schedule,
    endpoint,
    environment,
    frozen,
    size_run,
    update_cap,
)
from experiments.tla_pilot_analysis import (
    BOOTSTRAP_RESAMPLES,
    Counts,
    PairResult,
    game_columns,
    pair_result,
    total_counts,
)
from manabot.eval.results import RateInterval, bootstrap_mean_interval
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, file_sha256


class Bounded(StrictModel):
    """A played comparison's score and its range over its unscored legs.

    `score` needs every leg valid. `worst` and `best` count each unscored leg
    as 0 and as 1; with no failures all three are equal.
    """

    score: RateInterval | None
    worst: RateInterval
    best: RateInterval


class PolicyActions(StrictModel):
    """First-64 cutoff totals over `games`, the games that returned a count.

    A game killed at its limit returns none, and limits fall on long games, so
    `games` below the played count means these totals are a lower bound.
    """

    games: int = 0
    decisions: int = 0
    dropped_actions: int = 0


class Comparison(StrictModel):
    """One scheduled slot. An unreached slot expects no games, so it counts none.

    `result` needs every pair's match receipt; None means not yet played.
    """

    tag: str
    update: int | None
    name: str | None
    unreached: bool
    counts: Counts = Counts()
    pairs: list[PairResult] = []
    result: Bounded | None = None
    policy_actions: PolicyActions = PolicyActions()


class TrainingPoint(StrictModel):
    """A saved checkpoint and the hours since the run's first checkpoint."""

    update: int
    hours: float


class SizeReading(StrictModel):
    """One size's protocol reading; None means unknown, not negative."""

    label: SizeLabel
    state: str | None
    endpoint: int | None
    training: list[TrainingPoint]
    against_initial: Bounded | None
    against_heuristic: Bounded | None
    learned: bool | None


class FrontierAnalysis(StrictModel):
    protocol_sha256: str
    bootstrap_seed: int
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    counts: Counts
    policy_actions: PolicyActions
    comparisons: list[Comparison]
    sizes: list[SizeReading]
    # The large size's score at equal wall-clock compute, the primary scale
    # reading. None unless both runs finished below the update cap;
    # `comparisons` keeps the score.
    scale: Bounded | None
    complete: bool
    issues: list[str]


def _worst(pair: PairResult) -> float:
    """The pair's mean game score with every unscored leg counted as a loss."""
    return (pair.counts.wins + pair.counts.draws / 2) / pair.counts.expected


def _best(pair: PairResult) -> float:
    """The pair's mean game score with every unscored leg counted as a win."""
    unscored = pair.counts.expected - pair.counts.valid
    return _worst(pair) + unscored / pair.counts.expected


def _bounded(pairs: list[PairResult], indices: npt.NDArray[np.int64]) -> Bounded:
    """Whole-pair intervals; shared `indices` move a bound only by the legs it reassigns."""
    scores = [pair.score for pair in pairs if pair.score is not None]
    return Bounded(
        score=bootstrap_mean_interval(scores, indices)
        if len(scores) == len(pairs)
        else None,
        worst=bootstrap_mean_interval([_worst(pair) for pair in pairs], indices),
        best=bootstrap_mean_interval([_best(pair) for pair in pairs], indices),
    )


def _policy_actions_by_match(
    db: duckdb.DuckDBPyConnection, environment_id: str
) -> dict[str, PolicyActions]:
    """Each stored match's totals; Etude omits columns that are all null.

    Each seat repeats its game's count, so seat 0 counts a game once.
    """
    available = {
        row[0]
        for row in db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'games'"
        ).fetchall()
    }
    if not {"policy_decisions", "policy_dropped_actions"} <= available:
        return {}
    rows = db.execute(
        "SELECT g.match, count(g.policy_decisions), "
        "coalesce(sum(g.policy_decisions), 0), "
        "coalesce(sum(g.policy_dropped_actions), 0) "
        "FROM games g JOIN runs r USING (run) "
        "WHERE g.seat = 0 AND r.environment = ? GROUP BY g.match",
        [environment_id],
    ).fetchall()
    return {
        match: PolicyActions(
            games=games, decisions=int(decisions), dropped_actions=int(dropped)
        )
        for match, games, decisions, dropped in rows
    }


def _total_policy_actions(totals: list[PolicyActions]) -> PolicyActions:
    return PolicyActions(
        **{
            field: sum(getattr(total, field) for total in totals)
            for field in PolicyActions.model_fields
        }
    )


def _comparison(
    planned: ScheduledComparison,
    db: duckdb.DuckDBPyConnection,
    protocol: Protocol,
    environment_id: str,
    columns: tuple[str, ...],
    actions: dict[str, PolicyActions],
    indices: npt.NDArray[np.int64],
) -> Comparison:
    """Read a whole block of receipts; a partial block keeps its counts."""
    name = planned.matchup.name if planned.matchup else None
    if planned.unreached:
        return Comparison(
            tag=planned.tag,
            update=planned.update,
            name=name,
            unreached=True,
        )
    # The scheduler constructs matches in frozen pair order. Keep absent
    # checkpoints in that same order instead of selecting surviving pairs.
    keys = (
        [match.key for match in planned.matchup.matches()]
        if planned.matchup
        else [None] * len(protocol.pairs)
    )
    pairs = [
        pair_result(db, pair, key, environment_id, columns)
        for pair, key in zip(protocol.pairs, keys, strict=True)
    ]
    played = all(pair.counts.attempted_matches for pair in pairs)
    return Comparison(
        tag=planned.tag,
        update=planned.update,
        name=name,
        unreached=False,
        counts=total_counts([pair.counts for pair in pairs]),
        pairs=pairs,
        result=_bounded(pairs, indices) if played else None,
        policy_actions=_total_policy_actions(
            [actions[key] for key in keys if key in actions]
        ),
    )


def _learned(initial: Bounded | None) -> bool | None:
    """Whether the endpoint beat its initialization under every failed-leg outcome."""
    if initial is None:
        return None
    if initial.worst.lower > 0.5:
        return True
    return False if initial.best.lower <= 0.5 else None


def _training_points(
    db: duckdb.DuckDBPyConnection, run: TrainingRun | None
) -> list[TrainingPoint]:
    if run is None:
        return []
    rows = db.execute(
        "SELECT clocks, wall_ms FROM training_checkpoints WHERE run = ? "
        "ORDER BY ordinal",
        [run.name],
    ).fetchall()
    return [
        TrainingPoint(
            update=int(json.loads(clocks)["updates"]),
            hours=(wall_ms - rows[0][1]) / 3_600_000,
        )
        for clocks, wall_ms in rows
    ]


def _reading(
    label: SizeLabel,
    run: TrainingRun | None,
    comparisons: list[Comparison],
    db: duckdb.DuckDBPyConnection,
) -> SizeReading:
    """A size with no endpoint yet has endpoint slots of update None and no score."""
    end = endpoint(run)
    initial = next(c for c in comparisons if c.tag == f"initial-{label}")
    heuristic = next(
        c for c in comparisons if c.tag == f"heuristic-{label}" and c.update == end
    )
    return SizeReading(
        label=label,
        state=run.state if run else None,
        endpoint=end,
        training=_training_points(db, run),
        against_initial=initial.result,
        against_heuristic=heuristic.result,
        learned=_learned(initial.result),
    )


def analyze(state: ExperimentState, db: duckdb.DuckDBPyConnection) -> FrontierAnalysis:
    """Read the whole experiment; the two sizes are one cohort of comparisons."""
    regimes, protocol = frozen()
    cap = update_cap(regimes[SIZES[0]])
    environment_id = environment(protocol).identity
    columns = game_columns(db)
    actions = _policy_actions_by_match(db, environment_id)
    indices = np.random.default_rng(protocol.bootstrap_seed).integers(
        0, len(protocol.pairs), (BOOTSTRAP_RESAMPLES, len(protocol.pairs))
    )
    comparisons = [
        _comparison(planned, db, protocol, environment_id, columns, actions, indices)
        for planned in comparison_schedule(state, protocol)
    ]
    sizes = [
        _reading(label, size_run(state, label), comparisons, db) for label in SIZES
    ]
    issues = [
        f"{reading.label} has no endpoint; no learning or strength reading."
        for reading in sizes
        if reading.endpoint is None
    ]
    finished = all(reading.state == "finished" for reading in sizes)
    if not finished:
        issues.append("A run did not finish; no equal-compute scale reading.")
    # The cap is a second stop: a size reaching it spent less than the wall budget.
    capped = any(reading.endpoint == cap for reading in sizes)
    if capped:
        issues.append("A run hit the update cap; no equal-compute scale reading.")
    sources = {
        source for c in comparisons for pair in c.pairs for source in pair.sources
    }
    if len(sources) != 1 or None in sources:
        issues.append("Evaluation source is missing or differs across the cohort.")
    if any(c.result is None and not c.unreached for c in comparisons):
        issues.append("The frozen comparison schedule is incomplete.")
    receipted = [
        pair.counts
        for c in comparisons
        for pair in c.pairs
        if pair.counts.attempted_matches
    ]
    unscored = sum(counts.expected - counts.valid for counts in receipted)
    if unscored:
        played = sum(counts.expected for counts in receipted)
        issues.append(f"{unscored} of {played} played games have no score.")
    endpoints = next(c for c in comparisons if c.tag == "endpoint-head-to-head")
    return FrontierAnalysis(
        protocol_sha256=file_sha256(PROTOCOL),
        bootstrap_seed=protocol.bootstrap_seed,
        counts=total_counts([c.counts for c in comparisons]),
        policy_actions=_total_policy_actions([c.policy_actions for c in comparisons]),
        comparisons=comparisons,
        sizes=sizes,
        scale=endpoints.result if finished and not capped else None,
        complete=not issues,
        issues=issues,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Etude training and evaluation store"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state = observe(args.source, EXPERIMENT)
    with statsdb(EXPERIMENT, args.source) as db:
        report = analyze(state, db)
    atomic_json(args.out, report.model_dump(mode="json"))


if __name__ == "__main__":
    main()
