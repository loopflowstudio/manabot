"""The fixed pilot preserves the evidenced learner and evaluation population."""

from collections.abc import Iterator
from decimal import Decimal
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Literal

import duckdb
from etude.eval.block import write
from etude.eval.records import MatchRecord
from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    read_experiment,
)
from etude.jobs import evaluation_spec
from etude.run.declaration import declare
from etude.run.records import ArtifactRef, Clock
from etude.train.run import track
import pytest

from experiments.tla_pilot_analysis import analyze
from experiments.tla_pilot_v3 import ROOT, eval, frozen
from manabot.eval.results import RateInterval
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.training.regime import DampedPolicyLearning, TrainingRegime, TrainSelfPlay


def _checkpoint(update: int) -> Checkpoint:
    return Checkpoint(
        run="fixture-training",
        experiment="tla_pilot_v3",
        ordinal=update + 1,
        artifact=ArtifactRef(
            uri=f"file:///fixture/u{update}.pt", sha256="a" * 64, size_bytes=1
        ),
        clocks={"updates": update},
        line="fixture-training",
    )


def test_pilot_pins_independent_population_and_elemental_learning_rule() -> None:
    regime, protocol = frozen()
    assert regime.deck_source is not None
    assert len(regime.deck_source.decks) == 134
    assert len(set(regime.deck_source.entry_ids)) == 134
    assert regime.deck_source.sampling == "uniform-independent-v1"
    regime.deck_source.check()
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    reference = json.loads(
        (ROOT / "experiments/regimes/elemental-long-183-v1.json").read_bytes()
    )
    assert stage.learning == DampedPolicyLearning.model_validate(
        reference["stages"][0]["learning"]
    )
    assert (stage.streams, stage.transitions, stage.learning_seats) == (256, 64, "both")
    assert stage.execution.device == "cuda" and stage.behavior == "current-self"
    assert regime.observation.policy_input_version == 6
    assert regime.observation.max_facts == 256
    assert ObservationConfig().max_facts == 512
    assert protocol.allocation_usd == Decimal(1) < protocol.max_paid_usd == Decimal(10)
    rng = random.Random(protocol.deck_seed)
    assert [(p.first, p.second) for p in protocol.pairs] == [
        (rng.randrange(134), rng.randrange(134)) for _ in range(8)
    ]
    assert len({p.deal for p in protocol.pairs}) == 8
    assert protocol.training_seed not in {p.deal for p in protocol.pairs}


def test_pilot_evaluates_initial_milestones_and_endpoint_against_fixed_anchors() -> (
    None
):
    state = _state((0, 25, 32))
    jobs = eval(state)
    assert len(jobs) == 1
    spec = evaluation_spec(jobs[0])
    matchups = spec.tournament.matchups
    assert len(matchups) == 5  # 0 vs heuristic; 25 and actual endpoint vs both.
    assert {m.name for m in matchups} == {
        f"fixture-training u{update} vs {anchor}"
        for update, anchors in (
            (0, ("heuristic",)),
            (25, ("heuristic", "initial")),
            (32, ("heuristic", "initial")),
        )
        for anchor in anchors
    }
    for matchup in matchups:
        assert matchup.games == 4 and len(matchup.conditions) == 8
        if "initial" in matchup.tags:
            assert matchup.players[1].checkpoint.key.name == "u0.pt"
    assert eval(ExperimentState(experiment="tla_pilot_v3")) == []


def test_pilot_cli_discovery_keeps_evaluation_callback_importable() -> None:
    experiment = read_experiment(ROOT / "experiments/tla_pilot_v3.py")
    job = experiment.eval(_state((0,)))[0]
    callback = evaluation_spec(job).play
    assert callback.module == "experiments.tla_pilot_v3"
    assert callable(callback.load(job.repo))


# Controlled outcomes exercise the analysis without running games or training.
# This schema is Etude's public statsdb projection, including both participants.
@pytest.fixture
def pilot_db() -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect() as db:
        db.execute("CREATE TABLE runs (run VARCHAR, environment VARCHAR)")
        db.execute(
            "CREATE TABLE matches (run VARCHAR, match VARCHAR, state VARCHAR, error VARCHAR)"
        )
        db.execute(
            "CREATE TABLE seats (run VARCHAR, match VARCHAR, seat BIGINT, source VARCHAR)"
        )
        db.execute(
            "CREATE TABLE games (run VARCHAR, match VARCHAR, seat BIGINT, game BIGINT, attempted BOOLEAN, valid BOOLEAN, score DOUBLE, terminated BOOLEAN, truncated BOOLEAN, failure VARCHAR)"
        )
        yield db


def _state(
    updates: tuple[int, ...] = (0, 25, 50, 75, 100),
    *,
    status: Literal["running", "finished", "failed"] = "finished",
) -> ExperimentState:
    return ExperimentState(
        experiment="tla_pilot_v3",
        runs=(
            TrainingRun(
                name="fixture-training",
                experiment="tla_pilot_v3",
                state=status,
                checkpoints=Checkpoints(_checkpoint(update) for update in updates),
            ),
        ),
    )


def _outcomes(
    db: duckdb.DuckDBPyConnection, state: ExperimentState
) -> dict[str, list[str]]:
    tournament = evaluation_spec(eval(state)[0]).tournament
    db.execute(
        "INSERT INTO runs VALUES ('fixture-eval', ?)", [tournament.environment.identity]
    )
    keys: dict[str, list[str]] = {}
    for matchup in tournament.matchups:
        keys[matchup.name] = []
        for index, match in enumerate(matchup.matches()):
            keys[matchup.name].append(match.key)
            db.execute(
                "INSERT INTO matches VALUES ('fixture-eval', ?, 'done', NULL)",
                [match.key],
            )
            # Variable initial difficulty, exactly +.25 on every pair after
            # training. Against initialization every pair scores .75.
            wins = (
                3
                if "initial" in matchup.tags
                else index % 4 + int(" u0 " not in matchup.name)
            )
            for seat in (0, 1):
                db.execute(
                    "INSERT INTO seats VALUES ('fixture-eval', ?, ?, ?)",
                    [match.key, seat, "a" * 64],
                )
                db.executemany(
                    "INSERT INTO games VALUES ('fixture-eval', ?, ?, ?, true, true, ?, true, false, NULL)",
                    [
                        [match.key, seat, leg, float((leg < wins) != bool(seat))]
                        for leg in range(4)
                    ],
                )
    return keys


def test_pilot_analysis_pairs_gain_and_checks_joint_criterion(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    report = analyze(state, pilot_db)[0]
    assert report.complete and report.success is True
    assert report.bootstrap_resamples == 2000 and report.bootstrap_seed == 21392001
    assert (
        report.counts.expected == report.counts.attempted == report.counts.valid == 288
    )
    assert report.counts.missing == report.counts.failed == 0
    assert report.counts.expected_matches == report.counts.attempted_matches == 72
    final = next(
        c for c in report.comparisons if c.endpoint and c.anchor == "heuristic"
    )
    # Independent resampling of before/after, or of individual legs, widens
    # this interval. Shared four-leg pair blocks give exactly .25 throughout.
    assert final.heuristic_gain == RateInterval(mean=0.25, lower=0.25, upper=0.25)
    assert [p.pair for p in final.pairs] == list(frozen()[1].pairs)
    assert all(p.sources == ("a" * 64, "a" * 64) for p in final.pairs)
    assert final.counts.wins == 20 and final.counts.losses == 12
    assert all(
        c.heuristic_gain is not None
        for c in report.comparisons
        if c.anchor == "heuristic"
    )


@pytest.mark.parametrize(
    "gap",
    [
        "leg",
        "pair",
        "checkpoint",
        "initial",
        "failed_match",
        "invalid",
        "unattempted",
        "truncated",
        "source",
        "environment",
        "pair_identity",
    ],
)
def test_pilot_analysis_incomplete_cohort_cannot_claim_learning(
    pilot_db: duckdb.DuckDBPyConnection, gap: str
) -> None:
    state = _state()
    keys = _outcomes(pilot_db, state)
    key = keys["fixture-training u50 vs heuristic"][0]
    if gap == "leg":
        pilot_db.execute("DELETE FROM games WHERE match = ? AND game = 0", [key])
    elif gap == "pair":
        pilot_db.execute("DELETE FROM matches WHERE match = ?", [key])
    elif gap in ("checkpoint", "initial"):
        missing = 50 if gap == "checkpoint" else 0
        state = _state(tuple(n for n in (0, 25, 50, 75, 100) if n != missing))
    elif gap == "failed_match":
        pilot_db.execute(
            "UPDATE matches SET state = 'failed', error = 'worker died' WHERE match = ?",
            [key],
        )
        pilot_db.execute("DELETE FROM games WHERE match = ?", [key])
    elif gap in ("invalid", "unattempted", "truncated"):
        pilot_db.execute(
            "UPDATE games SET valid = false, score = NULL, attempted = ?, truncated = ?, failure = 'fixture' WHERE match = ? AND game = 0",
            [gap != "unattempted", gap == "truncated", key],
        )
    elif gap == "source":
        pilot_db.execute("UPDATE seats SET source = ? WHERE match = ?", ["b" * 64, key])
    elif gap == "environment":
        pilot_db.execute("UPDATE runs SET environment = 'another-world'")
    else:
        # Same deal, different deck entry: a different pair cannot fill the hole.
        matchup = evaluation_spec(eval(state)[0]).tournament.matchups[3]
        original = matchup.matches()[0]
        replacement = original.model_copy(
            update={"conditions": {**original.conditions, "first": 0}}
        )
        assert original.key == key
        pilot_db.execute(
            "UPDATE matches SET match = ? WHERE match = ?", [replacement.key, key]
        )
    report = analyze(state, pilot_db)[0]
    assert not report.complete and report.success is None and report.issues
    assert report.counts.expected == 288  # never select the surviving games
    assert (
        report.counts.expected
        == report.counts.valid + report.counts.failed + report.counts.missing
    )
    if gap in ("leg", "unattempted"):
        assert report.counts.missing == 1 and report.counts.attempted == 287
    if gap in ("invalid", "truncated"):
        assert report.counts.failed == 1 and report.counts.attempted == 288
        assert report.counts.truncated == int(gap == "truncated")
    if gap in ("pair", "pair_identity", "failed_match"):
        assert report.counts.missing == 4
        assert report.counts.failed_matches == int(gap == "failed_match")
    if gap == "checkpoint":
        assert report.counts.missing == 64
    if gap == "initial":
        assert report.counts.missing == 160  # baseline plus all fixed-initial opponents
        assert all(c.heuristic_gain is None for c in report.comparisons)


@pytest.mark.parametrize(
    "mode", ["no_gain", "uncertain_gain", "weak_initial_opponent_score"]
)
def test_pilot_analysis_complete_negative_is_not_success(
    pilot_db: duckdb.DuckDBPyConnection, mode: str
) -> None:
    state = _state()
    keys = _outcomes(pilot_db, state)
    anchor = "initial" if mode == "weak_initial_opponent_score" else "heuristic"
    for index, key in enumerate(keys[f"fixture-training u100 vs {anchor}"]):
        wins = 2 if anchor == "initial" else index % 4
        if mode == "uncertain_gain":
            wins = 4 if index < 4 else 0  # +.125 mean; uncertainty includes zero
        pilot_db.execute(
            "UPDATE games SET score = CAST(game < ? AS DOUBLE) WHERE match = ? AND seat = 0",
            [wins, key],
        )
    report = analyze(state, pilot_db)[0]
    assert report.complete and report.success is False


@pytest.mark.parametrize("status", ["running", "failed"])
def test_pilot_analysis_retains_unknown_endpoint(
    pilot_db: duckdb.DuckDBPyConnection, status: Literal["running", "failed"]
) -> None:
    state = _state((0, 25, 50, 75), status=status)
    _outcomes(pilot_db, state)
    report = analyze(state, pilot_db)[0]
    assert report.counts.expected == 288 and report.counts.missing == 64
    assert report.success is None
    endpoint = [c for c in report.comparisons if c.endpoint]
    assert len(endpoint) == 2 and all(c.update is None for c in endpoint)


def test_pilot_analysis_early_endpoint_keeps_absent_milestones(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state((0, 25, 32))
    _outcomes(pilot_db, state)
    report = analyze(state, pilot_db)[0]
    assert report.counts.expected == 288 and report.counts.missing == 128
    assert report.success is None
    assert {c.update for c in report.comparisons if c.endpoint} == {32}


def test_pilot_analysis_rejects_duplicate_legs(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    pilot_db.execute("INSERT INTO games SELECT * FROM games LIMIT 1")
    with pytest.raises(ValueError, match="duplicate legs"):
        analyze(state, pilot_db)


def test_pilot_analysis_counts_failures_without_a_score_column(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    pilot_db.execute("UPDATE games SET valid = false, failure = 'fixture failure'")
    pilot_db.execute("ALTER TABLE games DROP COLUMN score")
    report = analyze(state, pilot_db)[0]
    assert report.counts.failed == report.counts.expected == 288
    assert report.counts.valid == report.counts.missing == 0
    assert report.success is None


@pytest.mark.parametrize("with_failure_column", [True, False])
def test_pilot_analysis_cli_reads_sealed_etude_results(
    pilot_db: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_failure_column: bool,
) -> None:
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    for key in ("ETUDE_EXPERIMENT", "ETUDE_RUN", "ETUDE_TRACK_DIR", "ETUDE_CONTINUES"):
        monkeypatch.delenv(key, raising=False)
    state = _state()
    store = tmp_path / "store"
    with track(
        experiment="tla_pilot_v3",
        run="fixture-training",
        directory=store,
        clocks=(Clock(name="updates", kind="step", meaning="Learner updates"),),
        strict=True,
    ) as training:
        for checkpoint in state.runs[0].checkpoints:
            training.checkpoint(checkpoint.artifact, checkpoint.clocks)
    _outcomes(pilot_db, state)
    if not with_failure_column:
        pilot_db.execute("ALTER TABLE games DROP COLUMN failure")
    tournament = evaluation_spec(eval(state)[0]).tournament
    declared = declare(
        experiment="tla_pilot_v3",
        run="fixture-eval",
        kind="evaluation",
        clocks=(),
        directory=store,
    )
    (declared.root / "environment.json").write_text(
        tournament.environment.model_dump_json()
    )
    tables = tmp_path / "games.parquet"
    pilot_db.execute("COPY games TO ? (FORMAT PARQUET)", [str(tables)])
    # Sealing fixture outcomes exercises the real statsdb projection and CLI,
    # with no worker, model inference, game or provider allocation.
    columns = dict(
        pilot_db.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'games' AND column_name NOT IN ('run', 'match') "
            "ORDER BY ordinal_position"
        ).fetchall()
    )
    records = [
        MatchRecord(
            match=match,
            state="done",
            wall_ms=1,
            columns={"games": columns},
        )
        for matchup in tournament.matchups
        for match in matchup.matches()
    ]
    write(declared.root, "f1", records, {"games": [tables]}, {}, pilot_db)
    output = tmp_path / "analysis.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.tla_pilot_analysis",
            "--source",
            str(store),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())["runs"][0]
    assert report["complete"] and report["success"]
    assert report["counts"]["valid"] == report["counts"]["expected"] == 288


def test_pilot_analysis_counts_draws_without_fabricating_wins(
    pilot_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state()
    _outcomes(pilot_db, state)
    pilot_db.execute("UPDATE games SET score = 0.5")
    report = analyze(state, pilot_db)[0]
    assert report.complete and report.success is False
    assert report.counts.draws == 288
    assert report.counts.wins == report.counts.losses == 0
    assert all(
        c.score == RateInterval(mean=0.5, lower=0.5, upper=0.5)
        for c in report.comparisons
    )


def test_prospective_pilot_preserves_population_and_rejects_old_initialization() -> (
    None
):
    regime, protocol = frozen()
    predecessor = json.loads(
        (ROOT / "experiments/regimes/tla-trophy-213-v1.json").read_bytes()
    )
    assert regime.deck_source is not None
    assert regime.deck_source.model_dump(mode="json") == predecessor["deck_source"]
    old_protocol = json.loads(
        (ROOT / "experiments/protocols/tla-trophy-213-v1.json").read_bytes()
    )
    assert [pair.model_dump() for pair in protocol.pairs] == old_protocol["pairs"]
    assert protocol.corpus_sha256 == old_protocol["corpus_sha256"]
    assert protocol.milestones == tuple(old_protocol["milestones"])
    previous = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/tla-trophy-213-v2.json").read_bytes()
    )
    assert regime == previous.model_copy(update={"id": regime.id})
    assert protocol.predecessor_protocol_sha256 == file_sha256(
        ROOT / "experiments/protocols/tla-trophy-213-v2.json"
    )
    for experiment in ("tla_pilot", "tla_pilot_v2"):
        with pytest.raises(ValueError, match="fresh v3"):
            eval(ExperimentState(experiment=experiment))
