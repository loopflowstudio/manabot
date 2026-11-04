"""Frozen TLA pilot analysis over retained Etude matches and training checkpoints.

`analyze` accounts for the expected schedule before estimating anything. Each
pair's four correlated legs move together, including across checkpoints. The
CLI writes JSON beside the descriptive Etude report; neither runs games nor
replaces missing evidence. DuckDB is loaded only when the analysis is invoked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from etude.experiment import ExperimentState, observe
import numpy as np
import numpy.typing as npt
from pydantic import Field

from experiments.tla_pilot_v7 import (
    EXPERIMENT,
    GAMES_PER_PAIR,
    PROTOCOL,
    Anchor,
    Pair,
    comparison_schedule,
    environment,
    frozen,
)
from manabot.eval.results import RateInterval, bootstrap_mean_interval
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, file_sha256

if TYPE_CHECKING:
    import duckdb

BOOTSTRAP_RESAMPLES = 2000


class Counts(StrictModel):
    """Game counts: expected = valid + failed + missing; attempted is a lower bound.

    A failed Etude callback can lose all per-game outcomes. Those games remain
    missing, and the separate failed-match count retains the known failure.
    """

    expected: int = 0
    attempted: int = 0
    valid: int = 0
    failed: int = 0
    missing: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    truncated: int = 0
    expected_matches: int = 0
    attempted_matches: int = 0
    failed_matches: int = 0
    missing_matches: int = 0


class PairResult(StrictModel):
    pair: Pair
    match: str | None
    counts: Counts
    score: float | None = None
    error: str | None = None
    sources: tuple[str | None, ...] = ()


class Comparison(StrictModel):
    update: int | None
    anchor: Anchor
    endpoint: bool
    checkpoint: str | None
    clocks: dict[str, float]
    counts: Counts
    pairs: list[PairResult]
    score: RateInterval | None = None
    heuristic_gain: RateInterval | None = None


class PilotAnalysis(StrictModel):
    run: str
    state: str
    protocol_sha256: str
    bootstrap_seed: int
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    counts: Counts
    comparisons: list[Comparison]
    complete: bool
    # None means the joint criterion cannot be evaluated, not a negative result.
    success: bool | None = None
    issues: list[str]


class _Game(StrictModel):
    game: int = Field(ge=0, lt=GAMES_PER_PAIR)
    attempted: bool
    valid: bool
    # Etude may omit columns whose values are all null.
    score: Literal[0.0, 0.5, 1.0] | None = None
    terminated: bool
    truncated: bool
    failure: str | None = None


class _Receipt(StrictModel):
    run: str
    state: Literal["done", "failed"]
    error: str | None


def _total(counts: list[Counts]) -> Counts:
    return Counts(
        **{
            field: sum(getattr(count, field) for count in counts)
            for field in Counts.model_fields
        }
    )


def _pair_result(
    db: duckdb.DuckDBPyConnection,
    pair: Pair,
    key: str | None,
    environment_id: str,
    game_columns: tuple[str, ...],
) -> PairResult:
    receipts = (
        db.execute(
            "SELECT m.run, m.state, m.error FROM matches m JOIN runs r USING (run) "
            "WHERE m.match = ? AND r.environment = ?",
            [key, environment_id],
        ).fetchall()
        if key is not None
        else []
    )
    if len(receipts) > 1:
        raise ValueError(f"repeated match {key}: frozen pilot forbids retries")
    if not receipts:
        return PairResult(
            pair=pair,
            match=key,
            counts=Counts(
                expected=GAMES_PER_PAIR,
                expected_matches=1,
                missing=GAMES_PER_PAIR,
                missing_matches=1,
            ),
        )
    receipt = _Receipt.model_validate(
        dict(zip(("run", "state", "error"), receipts[0], strict=True))
    )
    source_rows = db.execute(
        "SELECT seat, source FROM seats WHERE run = ? AND match = ? ORDER BY seat",
        [receipt.run, key],
    ).fetchall()
    sources = tuple(source for _, source in source_rows)
    if [seat for seat, _ in source_rows] != [0, 1]:
        sources += (None,)
    games: list[_Game] = []
    if receipt.state == "done" and game_columns:
        rows = db.execute(
            f"SELECT {', '.join(game_columns)} FROM games "
            "WHERE run = ? AND match = ? AND seat = 0 ORDER BY game",
            [receipt.run, key],
        ).fetchall()
        games = [
            _Game.model_validate(dict(zip(game_columns, row, strict=True)))
            for row in rows
        ]
    if len({game.game for game in games}) != len(games):
        raise ValueError(f"duplicate legs in {key}")
    if any(
        not game.attempted and (game.valid or game.score is not None) for game in games
    ):
        raise ValueError(f"unattempted game has a result in {key}")
    attempted = [game for game in games if game.attempted]
    scores = [
        game.score
        for game in attempted
        if game.valid
        and game.terminated
        and not game.truncated
        and not game.failure
        and game.score is not None
    ]
    return PairResult(
        pair=pair,
        match=key,
        error=receipt.error,
        sources=sources,
        counts=Counts(
            expected=GAMES_PER_PAIR,
            attempted=len(attempted),
            valid=len(scores),
            failed=len(attempted) - len(scores),
            missing=GAMES_PER_PAIR - len(attempted),
            wins=scores.count(1),
            draws=scores.count(0.5),
            losses=scores.count(0),
            truncated=sum(game.truncated for game in attempted),
            expected_matches=1,
            attempted_matches=1,
            failed_matches=int(receipt.state == "failed"),
        ),
        score=sum(scores) / GAMES_PER_PAIR if len(scores) == GAMES_PER_PAIR else None,
    )


def analyze(
    state: ExperimentState, db: duckdb.DuckDBPyConnection
) -> list[PilotAnalysis]:
    """Read each run independently; missing checkpoints never shrink its schedule.

    All frozen milestones remain expected, even for an early endpoint. A missing
    successful endpoint has two comparison slots with unknown update coordinates.
    Match keys bind checkpoint/opponent, both deck indices and deal; source
    versions must also agree throughout a run. Scores never pool training seeds.
    """
    _, protocol = frozen()
    if state.experiment != EXPERIMENT:
        raise ValueError("analysis requires the TLA pilot experiment")
    environment_id = environment(protocol).identity
    available_columns = {
        row[0]
        for row in db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'games'"
        ).fetchall()
    }
    # One validated projection serves every pair. Keep required columns in the
    # query so malformed tables fail; only absent nullable fields get defaults.
    game_columns = (
        tuple(
            name
            for name, field in _Game.model_fields.items()
            if field.is_required() or name in available_columns
        )
        if available_columns
        else ()
    )
    indices = np.random.default_rng(protocol.bootstrap_seed).integers(
        0, len(protocol.pairs), (BOOTSTRAP_RESAMPLES, len(protocol.pairs))
    )
    reports: list[PilotAnalysis] = []
    for run in state.runs:
        comparisons: list[Comparison] = []
        for planned in comparison_schedule(run, protocol):
            checkpoint = planned.checkpoint
            # The scheduler constructs matches in frozen pair order. Keep absent
            # checkpoints in that same order instead of selecting surviving pairs.
            keys = (
                [match.key for match in planned.matchup.matches()]
                if planned.matchup
                else [None] * len(protocol.pairs)
            )
            pairs = [
                _pair_result(db, pair, key, environment_id, game_columns)
                for pair, key in zip(protocol.pairs, keys, strict=True)
            ]
            comparisons.append(
                Comparison(
                    update=planned.update,
                    anchor=planned.anchor,
                    endpoint=planned.endpoint,
                    checkpoint=str(checkpoint.key) if checkpoint else None,
                    clocks=checkpoint.clocks if checkpoint else {},
                    counts=_total([pair.counts for pair in pairs]),
                    pairs=pairs,
                )
            )
        comparisons = _estimate_scores(comparisons, indices)
        issues: list[str] = []
        if not any(c.endpoint and c.update for c in comparisons):
            issues.append("No successful trained endpoint is retained.")
        sources = {
            source for c in comparisons for pair in c.pairs for source in pair.sources
        }
        if len(sources) != 1 or None in sources:
            issues.append("Evaluation source is missing or differs across the cohort.")
        if any(comparison.score is None for comparison in comparisons):
            issues.append(
                "The frozen comparison schedule is incomplete; no qualified learning claim."
            )
        complete = not issues
        success = None
        if complete:
            final_heuristic = next(
                c for c in comparisons if c.endpoint and c.anchor == "heuristic"
            )
            final_initial = next(
                c for c in comparisons if c.endpoint and c.anchor == "initial"
            )
            gain = final_heuristic.heuristic_gain
            head_to_head = final_initial.score
            assert gain is not None and head_to_head is not None
            success = gain.mean >= 0.10 and gain.lower > 0 and head_to_head.lower > 0.5
        reports.append(
            PilotAnalysis(
                run=run.name,
                state=run.state,
                protocol_sha256=file_sha256(PROTOCOL),
                bootstrap_seed=protocol.bootstrap_seed,
                counts=_total([comparison.counts for comparison in comparisons]),
                comparisons=comparisons,
                complete=complete,
                success=success,
                issues=issues,
            )
        )
    return reports


def _estimate_scores(
    comparisons: list[Comparison], indices: npt.NDArray[np.int64]
) -> list[Comparison]:
    """Estimate complete blocks, then gains against the explicit initialization.

    Pair order comes from the frozen schedule; comparison order is irrelevant.
    Incomplete blocks never enter either the score or paired-gain estimate.
    """
    complete: dict[tuple[int | None, Anchor], npt.NDArray[np.float64]] = {}
    for comparison in comparisons:
        scores = [pair.score for pair in comparison.pairs if pair.score is not None]
        if len(scores) == len(comparison.pairs):
            complete[comparison.update, comparison.anchor] = np.asarray(
                scores, dtype=np.float64
            )
    initial = complete.get((0, "heuristic"))
    estimates: list[Comparison] = []
    for comparison in comparisons:
        scores = complete.get((comparison.update, comparison.anchor))
        score = gain = None
        if scores is not None:
            score = bootstrap_mean_interval(scores, indices)
            if comparison.anchor == "heuristic" and initial is not None:
                gain = bootstrap_mean_interval(scores - initial, indices)
        estimates.append(
            comparison.model_copy(update={"score": score, "heuristic_gain": gain})
        )
    return estimates


def main() -> None:
    from etude.eval.query import statsdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Etude training and evaluation store"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state = observe(args.source, EXPERIMENT)
    with statsdb(EXPERIMENT, args.source) as db:
        reports = analyze(state, db)
    atomic_json(
        args.out,
        {
            "runs": [report.model_dump(mode="json") for report in reports],
            "issues": []
            if reports
            else ["No pilot training run is retained; no learning claim."],
        },
    )


if __name__ == "__main__":
    main()
