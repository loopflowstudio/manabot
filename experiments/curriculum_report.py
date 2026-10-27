"""Paired estimates for MTG-175's fixed evaluation grids.

`compare` consumes the same GameRow contract as MTG-183's evaluator. Both
seats of a deal stay together when resampling, within each ordered deck cell;
cells and seats have equal weight. It reports no primary contrast until every
planned game of all four arms is valid. Missing/invalid games remain counted,
with worst-case bounds over the complete target population. These intervals describe
evaluation-deal uncertainty for this one seed, not training-seed uncertainty.
`stratify` gives descriptive deck/seat coverage and full-target bounds through
the same validation and weighting path, without inventing subgroup intervals.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Literal

from etude.eval.query import statsdb
from etude.eval.records import Agent, Environment
from etude.run.storage import Storage
import numpy as np
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import Game, GameRow

Arm = Literal["U", "M", "E", "ME"]
ARMS: tuple[Arm, ...] = ("U", "M", "E", "ME")
type GameKey = tuple[str, str, int, int]


class Outcome(BaseModel):
    """The scoring fields Etude publishes; unavailable decision diagnostics are not invented."""

    model_config = ConfigDict(extra="forbid")
    model_deck: str
    heuristic_deck: str
    model_seat: Literal[0, 1]
    deal: int
    score: float | None
    valid: bool
    failure: str | None


class _StoredOutcome(Outcome):
    artifact: str | None
    model_commit: str | None
    model_source: str | None
    anchor_key: str
    anchor_commit: str | None
    anchor_source: str | None
    environment: str


def load_games(
    source: Storage,
    selections: Mapping[Arm, Agent],
    *,
    anchor: Agent,
    environment: Environment,
    tournament: str,
) -> dict[Arm, list[Outcome]]:
    """Read selected checkpoint games from Etude, refusing changed identities.

    Agents and environment must come from the frozen launch records, not be
    recomputed from a newer checkout. Absent delivered game tables mean no
    games yet. Storage/SQL failures propagate, rather than becoming empty data.
    Duplicated/off-schedule game rows remain visible for `compare` to reject.
    """
    if anchor.version.source is None:
        raise ValueError("the frozen anchor source is required")
    if any(
        agent.version.source is None
        or agent.key.checkpoint is None
        or agent.artifact is None
        or agent.artifact.sha256 is None
        for agent in selections.values()
    ):
        raise ValueError(
            "selected policies require checkpoint keys and artifact digests"
        )
    found: dict[Arm, list[Outcome]] = {arm: [] for arm in ARMS}
    db = statsdb("curriculum_adapters", source)
    try:
        tables = {row[0] for row in db.sql("show tables").fetchall()}
        if "games" not in tables:
            return found
        # Etude omits columns whose delivered values are all null. Preserve
        # unknown scores/reasons as null, including an entirely invalid cohort.
        game_columns = {row[0] for row in db.sql("describe games").fetchall()}
        score = "g.score" if "score" in game_columns else "cast(null as double)"
        failure = "g.failure" if "failure" in game_columns else "cast(null as varchar)"
        for arm, agent in selections.items():
            cursor = db.execute(
                f"""select g.deck as model_deck, g.opponent_deck as heuristic_deck,
                    g.seat as model_seat, g.deal, {score} as score, g.valid, {failure} as failure,
                    a.artifact, s.commit as model_commit, s.source as model_source,
                    o.agent as anchor_key, o.commit as anchor_commit, o.source as anchor_source,
                    r.environment
                from games g
                join seats s on s.run=g.run and s.match=g.match and s.player=0
                join agents a on a.agent=s.agent
                join seats o on o.run=g.run and o.match=g.match and o.player=1
                join runs r on r.run=g.run
                where g.player=0 and s.agent=? and r.tournament=?""",
                [agent.key.encoded, tournament],
            )
            columns = [column[0] for column in cursor.description]
            for values in cursor.fetchall():
                row = _StoredOutcome.model_validate(
                    dict(zip(columns, values, strict=True))
                )
                assert agent.artifact is not None
                if (
                    row.artifact != agent.artifact.sha256
                    or (row.model_commit, row.model_source)
                    != (agent.version.commit, agent.version.source)
                    or row.anchor_key != anchor.key.encoded
                    or (row.anchor_commit, row.anchor_source)
                    != (anchor.version.commit, anchor.version.source)
                    or row.environment != environment.identity
                ):
                    raise ValueError(f"frozen evaluation identity mismatch for {arm}")
                found[arm].append(
                    Outcome.model_validate(
                        row.model_dump(
                            exclude={
                                "artifact",
                                "model_commit",
                                "model_source",
                                "anchor_key",
                                "anchor_commit",
                                "anchor_source",
                                "environment",
                            }
                        )
                    )
                )
    finally:
        db.close()
    return found


@dataclass(frozen=True)
class Coverage:
    expected: int
    received: int
    valid: int
    invalid: int
    missing: int


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class Bounds:
    low: float
    high: float


@dataclass(frozen=True)
class Comparison:
    coverage: dict[Arm, Coverage]
    scores: dict[Arm, float]
    contrasts: dict[str, Estimate]
    complete: bool
    score_bounds: dict[Arm, Bounds]
    contrast_bounds: dict[str, Bounds]


def _key(row: Game | GameRow | Outcome) -> GameKey:
    return row.model_deck, row.heuristic_deck, row.deal, row.model_seat


@dataclass(frozen=True)
class Summary:
    coverage: dict[Arm, Coverage]
    score_bounds: dict[Arm, Bounds]


@dataclass(frozen=True)
class Stratum:
    by: Literal["model_deck", "heuristic_deck", "model_seat"]
    value: str | int
    summary: Summary


@dataclass(frozen=True)
class _Outcomes:
    planned: set[GameKey]
    received: dict[Arm, set[GameKey]]
    valid: dict[Arm, dict[GameKey, float]]


def _outcomes(
    schedule: Sequence[Game],
    results: Mapping[Arm, Sequence[GameRow | Outcome]],
) -> _Outcomes:
    if not schedule:
        raise ValueError("a nonempty schedule is required")
    planned = {_key(row) for row in schedule}
    if len(planned) != len(schedule):
        raise ValueError("duplicate scheduled game")
    cells = sorted({key[:2] for key in planned})
    deals = {
        cell: sorted({key[2] for key in planned if key[:2] == cell}) for cell in cells
    }
    if any(
        (*cell, deal, seat) not in planned
        for cell in cells
        for deal in deals[cell]
        for seat in (0, 1)
    ):
        raise ValueError("each scheduled deal must include both seats")
    if set(results) != set(ARMS):
        raise ValueError("all four arms must be present, even if their rows are empty")
    received: dict[Arm, set[GameKey]] = {}
    indexed: dict[Arm, dict[GameKey, float]] = {}
    for arm in ARMS:
        seen: set[GameKey] = set()
        valid: dict[GameKey, float] = {}
        for row in results[arm]:
            key = _key(row)
            if key not in planned or key in seen:
                raise ValueError(f"off-schedule or duplicate game in {arm}: {key}")
            seen.add(key)
            if row.valid:
                if (
                    row.score is None
                    or not math.isfinite(row.score)
                    or not 0 <= row.score <= 1
                ):
                    raise ValueError("a valid game requires a finite score in [0, 1]")
                valid[key] = row.score
        indexed[arm] = valid
        received[arm] = seen
    return _Outcomes(planned, received, indexed)


def _summarize(planned: set[GameKey], outcomes: _Outcomes) -> Summary:
    cells = sorted({key[:2] for key in planned})
    coverage: dict[Arm, Coverage] = {}
    bounds: dict[Arm, Bounds] = {}
    for arm in ARMS:
        received = planned.intersection(outcomes.received[arm])
        valid = planned.intersection(outcomes.valid[arm])
        coverage[arm] = Coverage(
            len(planned),
            len(received),
            len(valid),
            len(received) - len(valid),
            len(planned) - len(received),
        )
        low = high = 0.0
        for cell in cells:
            keys = sorted(key for key in planned if key[:2] == cell)
            low += (
                sum(outcomes.valid[arm].get(key, 0.0) for key in keys)
                / len(keys)
                / len(cells)
            )
            high += (
                sum(outcomes.valid[arm].get(key, 1.0) for key in keys)
                / len(keys)
                / len(cells)
            )
        bounds[arm] = Bounds(low, high)
    return Summary(coverage, bounds)


def stratify(
    schedule: Sequence[Game],
    results: Mapping[Arm, Sequence[GameRow | Outcome]],
) -> tuple[Stratum, ...]:
    """Describe each model deck, opponent deck and native seat, keeping all cells.

    Cells receive equal mass within each stratum, even with unequal deal counts.
    A complete stratum has a degenerate bound; missing/invalid games keep their
    full-target bounds and counts. These are not confidence intervals, subgroup
    hypothesis tests or a replacement for the paired four-arm comparison.
    """
    outcomes = _outcomes(schedule, results)
    axes: tuple[
        tuple[Literal["model_deck", "heuristic_deck", "model_seat"], int], ...
    ] = (
        ("model_deck", 0),
        ("heuristic_deck", 1),
        ("model_seat", 3),
    )
    strata: list[Stratum] = []
    for by, index in axes:
        for value in sorted({key[index] for key in outcomes.planned}):
            planned = {key for key in outcomes.planned if key[index] == value}
            strata.append(Stratum(by, value, _summarize(planned, outcomes)))
    return tuple(strata)


def compare(
    schedule: Sequence[Game],
    results: Mapping[Arm, Sequence[GameRow | Outcome]],
    *,
    seed: int,
    replicates: int = 2000,
) -> Comparison:
    """Complete-grid scores and M−U, E−U, ME−U, ME−M−E+U intervals.

    Reject duplicates and off-schedule rows instead of choosing a preferred
    attempt. An incomplete grid returns coverage and worst-case bounds, with no point
    scores or bootstrap contrasts. Bounds assign each missing/invalid score
    any value in [0, 1]; they are not confidence intervals.
    Percentile intervals resample deals independently within each cell, using
    the same sampled deals for every arm and both native seats.
    """
    if replicates < 2:
        raise ValueError("at least two replicates are required")
    outcomes = _outcomes(schedule, results)
    summary = _summarize(outcomes.planned, outcomes)
    coverage, score_bounds = summary.coverage, summary.score_bounds
    indexed = outcomes.valid
    cells = sorted({key[:2] for key in outcomes.planned})
    deals = {
        cell: sorted({key[2] for key in outcomes.planned if key[:2] == cell})
        for cell in cells
    }
    weights = {
        "M-U": (-1, 1, 0, 0),
        "E-U": (-1, 0, 1, 0),
        "ME-U": (-1, 0, 0, 1),
        "ME-M-E+U": (1, -1, -1, 1),
    }
    contrast_bounds = {
        name: Bounds(
            sum(
                w * (score_bounds[a].low if w > 0 else score_bounds[a].high)
                for a, w in zip(ARMS, coefficients, strict=True)
            ),
            sum(
                w * (score_bounds[a].high if w > 0 else score_bounds[a].low)
                for a, w in zip(ARMS, coefficients, strict=True)
            ),
        )
        for name, coefficients in weights.items()
    }
    if any(row.valid != row.expected for row in coverage.values()):
        return Comparison(coverage, {}, {}, False, score_bounds, contrast_bounds)
    # Deal values average both seats; cell means then receive equal mass.
    scores = np.zeros(4)
    draws = np.zeros((replicates, 4))
    rng = np.random.default_rng(seed)
    for cell in cells:
        values = np.asarray(
            [
                [
                    sum(indexed[arm][(*cell, deal, seat)] for seat in (0, 1)) / 2
                    for arm in ARMS
                ]
                for deal in deals[cell]
            ]
        )
        scores += values.mean(axis=0) / len(cells)
        selected = rng.integers(len(values), size=(replicates, len(values)))
        draws += values[selected].mean(axis=1) / len(cells)
    contrasts: dict[str, Estimate] = {}
    for name, coefficients in weights.items():
        vector = np.asarray(coefficients)
        low, high = np.percentile(draws @ vector, [2.5, 97.5])
        contrasts[name] = Estimate(float(scores @ vector), float(low), float(high))
    return Comparison(
        coverage,
        dict(zip(ARMS, map(float, scores), strict=True)),
        contrasts,
        True,
        score_bounds,
        contrast_bounds,
    )
