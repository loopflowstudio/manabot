"""The c5fba4 frontier protocol pins two sizes and a milestone evaluation schedule."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import duckdb
from etude.eval.match import Tables, evaluate
from etude.eval.query import statsdb
from etude.eval.records import Match
from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    observe,
    read_experiment,
)
from etude.jobs import evaluation_spec
from etude.launch.job_file import read_job_file
from etude.launch.records import Run
from etude.page import build, render
from etude.run.errors import EtudeError
from etude.run.records import ArtifactRef
from etude.train.records import Step
from etude.train.run import track
import pytest

from experiments import tla_frontier
from experiments.tla_frontier import (
    EXPERIMENT,
    HEURISTIC,
    LARGE,
    REGIME,
    ROOT,
    SMALL,
    SizeLabel,
    comparison_schedule,
    eval,
    export_updates,
    frozen,
    runs,
    size_run,
)
from experiments.tla_frontier_analysis import FrontierAnalysis, analyze
from experiments.tla_frontier_calibration import (
    EXPERIMENT as CALIBRATION,
    MAX_HOURS,
    MAX_USD_PER_SIZE,
    UPDATES,
    calibration_regime,
    read_calibration,
    runs as calibration_runs,
)
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import TrainingJob
from manabot.model.actor_critic import ActorCritic
from manabot.training.execution import cuda_peak_metrics
from manabot.training.regime import TrainingRegime, TrainSelfPlay


def _checkpoint(label: SizeLabel, update: int) -> Checkpoint:
    run = f"tla-frontier-c5fba4-{label}-seed21501"
    return Checkpoint(
        run=run,
        experiment=EXPERIMENT,
        ordinal=update + 1,
        artifact=ArtifactRef(
            uri=f"file:///fixture/{label}/updates-{update}.pt",
            sha256="a" * 64,
            size_bytes=1,
        ),
        clocks={"updates": update},
        line=run,
    )


def _state(
    reached: dict[SizeLabel, int], *, finished: bool = True, export_every: int = 0
) -> ExperimentState:
    """Stored checkpoints at milestones, with `export_every` adding the cadence."""
    milestones = (0, *frozen()[1].milestones)
    return ExperimentState(
        experiment=EXPERIMENT,
        runs=tuple(
            TrainingRun(
                name=f"tla-frontier-c5fba4-{label}-seed21501",
                experiment=EXPERIMENT,
                state="finished" if finished else "running",
                checkpoints=Checkpoints(
                    _checkpoint(label, update)
                    for update in sorted(
                        {u for u in milestones if u <= end}
                        | (set(range(0, end, export_every)) if export_every else set())
                        | ({end} if finished else set())
                    )
                ),
            )
            for label, end in reached.items()
        ),
    )


def _failed(state: ExperimentState, label: SizeLabel) -> ExperimentState:
    """The state with one size's run failed after its stored checkpoints."""
    prefix = f"tla-frontier-c5fba4-{label}-"
    return replace(
        state,
        runs=tuple(
            replace(run, state="failed") if run.name.startswith(prefix) else run
            for run in state.runs
        ),
    )


def _tracked(
    launched: Run,
    source: Path,
    rows: dict[int, dict[str, float]],
    checkpoints: tuple[int, ...] = (),
) -> None:
    """Write a launched run as its worker would: its job's experiment and name."""
    job = TrainingJob.model_validate_json(launched.job.config)
    run = track(
        experiment=job.experiment,
        run=f"{job.regime.id}-seed{launched.seed}",
        seed=launched.seed,
        clocks=[Step("updates", "completed updates")],
        directory=source,
        config={"regime": job.regime.id},
        strict=True,
    )
    try:
        for update, metrics in rows.items():
            run.log(metrics, updates=update)
        for update in checkpoints:
            path = source / f"{job.regime.id}-u{update}.pt"
            path.write_text(f"synthetic export {job.regime.id} {update}")
            run.checkpoint(path, updates=update)
    finally:
        run.finish()


def _compute_parameters(regime: TrainingRegime) -> int:
    """Parameters outside the 4096-bucket hashed lookup tables."""
    agent = ActorCritic(ObservationSpace(regime.observation), regime.agent)
    return sum(
        p.numel()
        for p in agent.parameters()
        if not (p.dim() == 2 and p.shape[0] == 4096)
    )


def _recipe(regime: TrainingRegime) -> dict[str, object]:
    """The regime without its identity and compute-network size."""
    size = {"hidden_dim", "attention_layers", "num_attention_heads"}
    return regime.model_dump(mode="json", exclude={"id": True, "agent": size})


def _without_length(regime: TrainingRegime) -> dict[str, object]:
    """The regime without its identity, update target and wall."""
    stage = {"updates": True, "execution": {"wall_seconds": True}}
    return regime.model_dump(
        exclude={"id": True, "wall_seconds": True, "stages": {0: stage}}
    )


def test_sizes_differ_only_in_compute_network() -> None:
    regimes, _ = frozen()
    small, large = regimes[SMALL], regimes[LARGE]
    assert _recipe(small) == _recipe(large)
    assert (small.agent.hidden_dim, small.agent.attention_layers) == (64, 2)
    assert (large.agent.hidden_dim, large.agent.attention_layers) == (256, 4)
    ratio = _compute_parameters(large) / _compute_parameters(small)
    assert 10 <= ratio <= 100


def test_frozen_bytes_declare_the_protocol() -> None:
    regimes, protocol = frozen()
    regime = regimes[SMALL]
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert stage.learning.gradient == "damped_policy"
    assert stage.game_errors is not None and stage.game_errors.recover_capacity
    assert stage.updates > max(protocol.milestones)
    assert regime.wall_seconds == stage.execution.wall_seconds
    assert regime.world == protocol.runtime_world.world_version
    observation = regime.observation
    assert observation.policy_input_version == 9
    assert observation.action_overflow == HEURISTIC["action_overflow"]
    assert observation.max_actions == HEURISTIC["max_actions"]
    source = regime.deck_source
    assert source is not None and source.entry_ids is not None
    assert source.sampling == "uniform-independent-v1"
    assert len(protocol.pairs) == 16
    assert all(max(p.first, p.second) < len(source.decks) for p in protocol.pairs)
    assert protocol.max_paid_usd == Decimal("100")
    assert 2 * protocol.training_usd + protocol.evaluation_usd <= protocol.max_paid_usd


def test_frozen_rejects_a_changed_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "w64.json"
    changed.write_bytes(
        REGIME.read_bytes().replace(b'"hidden_dim": 64', b'"hidden_dim": 128')
    )
    monkeypatch.setattr(tla_frontier, "REGIME", changed)
    with pytest.raises(ValueError, match="regime differs"):
        frozen()


def test_milestones_play_both_sizes_against_heuristic_and_each_other() -> None:
    schedule = comparison_schedule(_state({SMALL: 1300, LARGE: 450}), frozen()[1])
    playable = [c for c in schedule if c.matchup is not None]
    names = {c.matchup.name for c in playable if c.matchup is not None}
    assert "w64 u0 vs heuristic" in names and "w256 u0 vs heuristic" in names
    assert "w256 u400 vs w64 u400" in names
    assert "w256 u600 vs w64 u600" not in names
    assert "w64 u1200 vs heuristic" in names
    assert {"w64 final u1300 vs heuristic", "w256 final u450 vs heuristic"} <= names
    assert {"w64 final u1300 vs w64 u0", "w256 final u450 vs w256 u0"} <= names
    assert "w256 final vs w64 final" in names
    # Unreached milestones stay expected for analysis.
    assert any(
        c.update == 2400 and c.tag == "head-to-head" and c.matchup is None
        for c in schedule
    )
    tags = Counter(c.tag for c in playable)
    assert tags["head-to-head"] == 8  # 25, 50, 75, 100, 150, 200, 300, 400
    assert len(names) == len(playable)


def test_running_runs_evaluate_reached_milestones_only() -> None:
    state = _state({SMALL: 100, LARGE: 50}, finished=False)
    schedule = comparison_schedule(state, frozen()[1])
    names = {c.matchup.name for c in schedule if c.matchup is not None}
    assert "w256 u50 vs w64 u50" in names
    assert not any("final" in name for name in names)
    (job,) = eval(state)
    assert job.max_usd == Decimal("2")
    assert job.max_attempts == 3


def test_exports_between_milestones_are_stored_but_not_played() -> None:
    protocol = frozen()[1]
    state = _failed(
        _state({SMALL: 1300, LARGE: 525}, export_every=protocol.export_every), LARGE
    )
    run = size_run(state, LARGE)
    assert run is not None
    assert {125, 525} <= {int(c.clocks["updates"]) for c in run.checkpoints}
    (job,) = eval(state)
    names = {matchup.name for matchup in evaluation_spec(job).tournament.matchups}
    # Update 525 is no milestone; it is played only as the failed run's endpoint.
    assert {"w256 final u525 vs heuristic", "w256 final u525 vs w256 u0"} <= names
    assert "w256 final vs w64 final" in names
    assert "w256 u400 vs w64 u400" in names
    assert "w256 u600 vs w64 u600" not in names
    played = {
        c.update for c in comparison_schedule(state, protocol) if c.matchup is not None
    }
    assert played == {0, *(u for u in protocol.milestones if u <= 1300), 525, 1300}


def test_evaluation_job_prepares_a_stock_rented_machine() -> None:
    (job,) = eval(_state({SMALL: 25, LARGE: 25}, finished=False))
    assert job.setup is not None and "--extra evaluation" in job.setup
    assert "uv run --locked --extra evaluation python" in job.command


def test_no_checkpoints_schedules_nothing() -> None:
    assert eval(ExperimentState(experiment=EXPERIMENT)) == []


def test_evaluation_rejects_a_foreign_experiment() -> None:
    with pytest.raises(ValueError, match="this experiment"):
        eval(ExperimentState(experiment="tla_pilot_v10"))


def test_etude_eval_observes_the_runs_training_records(tmp_path: Path) -> None:
    """The experiment file's name reaches the runs its jobs track, through a store."""
    experiment = read_experiment(ROOT / "experiments/tla_frontier.py")
    assert experiment.name == EXPERIMENT
    launches = read_job_file(
        ROOT / "experiments/tla_frontier.py", ["--revision", "HEAD"]
    )
    assert {launched.job.experiment for launched in launches} == {experiment.name}
    for launched in launches:
        _tracked(launched, tmp_path, {25: {"loss": 0.1}}, (0, 25))
    state = observe(tmp_path, experiment.name)
    assert size_run(state, SMALL) is not None and size_run(state, LARGE) is not None
    (job,) = experiment.eval(state)
    names = {matchup.name for matchup in evaluation_spec(job).tournament.matchups}
    assert "w256 u25 vs w64 u25" in names


def test_calibration_changes_only_length_and_identity() -> None:
    regimes, _ = frozen()
    for label, regime in regimes.items():
        changed = calibration_regime(regime, label)
        assert changed.stages[0].updates == UPDATES
        assert _without_length(changed) == _without_length(regime)


def test_calibration_fits_its_ceiling_on_the_admitted_gpus() -> None:
    launched = read_job_file(
        ROOT / "experiments/tla_frontier_calibration.py", ["--revision", "HEAD"]
    )
    assert len(launched) == 2
    # Calibration has no eval, so its worker's config alone names the experiment.
    jobs = [TrainingJob.model_validate_json(run.job.config) for run in launched]
    assert {job.experiment for job in jobs} == {CALIBRATION}
    assert {run.job.until for run in launched} == {f"updates={UPDATES}"}
    assert sum(run.job.max_usd or 0 for run in launched) == Decimal("2")
    assert Decimal(str(MAX_HOURS + 0.25)) * Decimal("0.67") <= MAX_USD_PER_SIZE


def test_calibration_reads_cost_and_memory_and_stays_out_of_the_frontier(
    tmp_path: Path,
) -> None:
    rows = {
        1: {"time/collection": 900.0, "time/learning": 100.0},
        2: {"time/collection": 180.0, "time/learning": 20.0},
        3: {
            "time/collection": 200.0,
            "time/learning": 40.0,
            "memory/peak_allocated_cuda_bytes": 4e9,
            "memory/peak_reserved_cuda_bytes": 6e9,
        },
    }
    small, large = calibration_runs("HEAD")
    _tracked(small, tmp_path, rows, (0, 3))
    _tracked(large, tmp_path, {})
    readings = {reading.run: reading for reading in read_calibration(tmp_path)}
    reading = readings["tla-frontier-c5fba4-calibration-w64-seed21501"]
    assert reading.updates == 3
    assert reading.seconds_per_update == 220  # the first update's warm-up is set aside
    assert reading.updates_in_wall_budget == 883
    assert reading.peak_allocated_cuda_bytes == 4_000_000_000
    assert reading.peak_reserved_cuda_bytes == 6_000_000_000
    silent = readings["tla-frontier-c5fba4-calibration-w256-seed21501"]
    assert silent.updates == 0 and silent.seconds_per_update is None
    assert silent.peak_reserved_cuda_bytes is None
    # A frontier pass never sees calibration, even from a shared store.
    assert observe(tmp_path, EXPERIMENT).runs == ()
    assert observe(tmp_path, CALIBRATION).runs != ()
    misnamed = replace(observe(tmp_path, CALIBRATION), experiment=EXPERIMENT)
    assert size_run(misnamed, SMALL) is None and size_run(misnamed, LARGE) is None


def test_cuda_peaks_are_absent_off_cuda_and_read_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cuda_peak_metrics("cpu") == {}
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 4)
    monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: 6)
    assert cuda_peak_metrics("cuda") == {
        "memory/peak_allocated_cuda_bytes": 4,
        "memory/peak_reserved_cuda_bytes": 6,
    }


def test_arena_journal_stays_off_disk_and_is_retained_after_failure(
    tmp_path: Path,
) -> None:
    retained = tmp_path / "arena" / "match"
    with pytest.raises(RuntimeError):
        with tla_frontier._memory_journal(retained) as out_dir:
            assert not out_dir.is_relative_to(tmp_path)
            out_dir.mkdir()
            (out_dir / "game.jsonl").write_text("record")
            raise RuntimeError("the game died")
    assert not out_dir.exists()
    assert (retained / "game.jsonl").read_text() == "record"


def test_training_runs_admit_milestones_and_wall_budget() -> None:
    launched = runs("HEAD")
    assert [run.job.name for run in launched] == [
        "tla-frontier-c5fba4-w64",
        "tla-frontier-c5fba4-w256",
    ]
    assert {run.job.until for run in launched} == {"updates=3000"}
    assert {run.job.max_attempts for run in launched} == {1}


def test_training_exports_the_cadence_joined_with_the_milestones() -> None:
    regimes, protocol = frozen()
    expected = tuple(range(25, 3000, 25))
    assert set(protocol.milestones) <= set(expected)
    for launched in runs("HEAD"):
        job = TrainingJob.model_validate_json(launched.job.config)
        assert job.milestones == expected
    # A milestone off the cadence is still exported, in order.
    offset = protocol.model_copy(update={"milestones": (30,), "export_every": 1000})
    assert export_updates(regimes[SMALL], offset) == (30, 1000, 2000)


def test_discovery_keeps_evaluation_callback_importable() -> None:
    experiment = read_experiment(ROOT / "experiments/tla_frontier.py")
    job = experiment.eval(_state({SMALL: 25, LARGE: 25}, finished=False))[0]
    callback = evaluation_spec(job).play
    assert callback.module == "experiments.tla_frontier"
    assert callable(callback.load(job.repo))


# Controlled outcomes exercise the analysis without running games or training.
# This schema is Etude's public statsdb projection.
@pytest.fixture
def frontier_db() -> Iterator[duckdb.DuckDBPyConnection]:
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
        db.execute(
            "CREATE TABLE training_checkpoints (run VARCHAR, ordinal BIGINT, wall_ms BIGINT, clocks JSON)"
        )
        yield db


def _outcomes(
    db: duckdb.DuckDBPyConnection, state: ExperimentState, wins: int = 3
) -> dict[str, list[str]]:
    """Every scheduled pair scores `wins` of four legs for its first player."""
    benchmark = evaluation_spec(eval(state)[0]).tournament
    db.execute(
        "INSERT INTO runs VALUES ('fixture-eval', ?)", [benchmark.environment.identity]
    )
    keys: dict[str, list[str]] = {}
    for matchup in benchmark.matchups:
        keys[matchup.name] = [match.key for match in matchup.matches()]
        for key in keys[matchup.name]:
            db.execute(
                "INSERT INTO matches VALUES ('fixture-eval', ?, 'done', NULL)", [key]
            )
            for seat in (0, 1):
                db.execute(
                    "INSERT INTO seats VALUES ('fixture-eval', ?, ?, ?)",
                    [key, seat, "a" * 64],
                )
                db.executemany(
                    "INSERT INTO games VALUES ('fixture-eval', ?, ?, ?, true, true, ?, true, false, NULL)",
                    [
                        [key, seat, leg, float((leg < wins) != bool(seat))]
                        for leg in range(4)
                    ],
                )
    for run in state.runs:
        db.executemany(
            "INSERT INTO training_checkpoints VALUES (?, ?, ?, ?)",
            [
                [
                    run.name,
                    saved.ordinal,
                    1_000 + int(saved.clocks["updates"]) * 3_600_000,
                    json.dumps(saved.clocks),
                ]
                for saved in run.checkpoints
            ],
        )
    return keys


def _endpoint_score(report: FrontierAnalysis) -> float:
    """The endpoint head-to-head's score, reported even without a scale reading."""
    played = next(c for c in report.comparisons if c.tag == "endpoint-head-to-head")
    assert played.result is not None and played.result.score is not None
    return played.result.score.mean


def test_analysis_reads_learning_and_scale_past_unreached_milestones(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    report = analyze(state, frontier_db)
    assert report.complete and report.issues == []
    assert report.scale is not None and report.scale.score is not None
    assert report.scale.score.mean == 0.75
    assert report.scale.worst == report.scale.best == report.scale.score
    for reading in report.sizes:
        assert reading.learned is True
        assert reading.against_heuristic is not None
    small, large = report.sizes
    assert (small.endpoint, large.endpoint) == (1300, 450)
    assert large.training[-1].update == 450 and large.training[-1].hours == 450
    unreached = {c.name or (c.tag, c.update) for c in report.comparisons if c.unreached}
    assert ("head-to-head", 600) in unreached
    assert ("heuristic-w64", 2400) in unreached
    assert report.counts.expected == report.counts.valid == 64 * len(keys)


def test_failed_size_keeps_its_reach_but_no_equal_compute_scale(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _failed(_state({SMALL: 1300, LARGE: 450}), LARGE)
    schedule = comparison_schedule(state, frozen()[1])
    names = {c.matchup.name for c in schedule if c.matchup is not None}
    assert {"w256 final u450 vs w256 u0", "w256 final vs w64 final"} <= names
    assert any(
        c.tag == "head-to-head" and c.update == 600 and c.unreached for c in schedule
    )
    _outcomes(frontier_db, state)
    report = analyze(state, frontier_db)
    small, large = report.sizes
    assert (large.state, large.endpoint) == ("failed", 450)
    assert large.learned is True and large.against_heuristic is not None
    assert small.learned is True
    assert report.scale is None and not report.complete
    assert report.issues == ["A run did not finish; no equal-compute scale reading."]
    assert _endpoint_score(report) == 0.75


def test_failed_size_without_checkpoints_has_no_endpoint() -> None:
    state = _failed(_state({SMALL: 100}), SMALL)
    bare = replace(state, runs=(replace(state.runs[0], checkpoints=Checkpoints()),))
    assert tla_frontier.endpoint(bare.runs[0]) is None
    assert eval(bare) == []


def test_analysis_keeps_a_negative_learning_reading(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 100, LARGE: 50})
    _outcomes(frontier_db, state, wins=2)
    report = analyze(state, frontier_db)
    assert report.complete
    assert [reading.learned for reading in report.sizes] == [False, False]


def test_analysis_missing_reached_comparison_leaves_reading_unknown(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 100, LARGE: 50})
    keys = _outcomes(frontier_db, state)
    frontier_db.execute(
        "DELETE FROM matches WHERE match = ?", [keys["w256 final u50 vs w256 u0"][0]]
    )
    report = analyze(state, frontier_db)
    assert not report.complete
    assert "The frozen comparison schedule is incomplete." in report.issues
    assert report.sizes[1].learned is None
    assert report.sizes[0].learned is True
    assert report.counts.missing == 4


def test_analysis_of_running_sizes_claims_nothing(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 100, LARGE: 50}, finished=False)
    _outcomes(frontier_db, state)
    report = analyze(state, frontier_db)
    assert not report.complete
    assert report.scale is None
    assert all(reading.endpoint is None for reading in report.sizes)
    assert all(reading.learned is None for reading in report.sizes)
    # While a run may still go on, later milestones are pending, never unreached.
    assert not any(c.unreached for c in report.comparisons)


def _fail_leg(db: duckdb.DuckDBPyConnection, key: str, leg: int) -> None:
    """One leg of a stored match, recorded as a game the limit killed."""
    db.execute(
        "UPDATE games SET valid = false, score = NULL, terminated = false, "
        "failure = 'game limit' WHERE match = ? AND game = ?",
        [key, leg],
    )


def _fail_match(db: duckdb.DuckDBPyConnection, key: str) -> None:
    """A match whose callback raised: a failed receipt and no game rows."""
    db.execute("DELETE FROM games WHERE match = ?", [key])
    db.execute(
        "UPDATE matches SET state = 'failed', error = 'died' WHERE match = ?", [key]
    )


def test_failed_legs_bound_readings_instead_of_erasing_them(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    _fail_leg(frontier_db, keys["w256 final vs w64 final"][0], 0)
    _fail_leg(frontier_db, keys["w64 final u1300 vs w64 u0"][0], 0)
    report = analyze(state, frontier_db)
    played = 64 * len(keys)
    assert not report.complete
    assert report.issues == [f"2 of {played} played games have no score."]
    assert report.counts.expected == played and report.counts.failed == 2
    scale = report.scale
    assert scale is not None and scale.score is None
    # One pair of 16 falls from 3/4 to 2/4, or stays 3/4 if the win had stood.
    assert scale.worst.mean == 0.75 - 0.25 / 16 and scale.best.mean == 0.75
    small = report.sizes[0]
    assert small.learned is True
    assert small.against_initial is not None and small.against_initial.score is None


def test_a_failed_match_with_no_rows_counts_all_its_legs_unscored(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    _fail_match(frontier_db, keys["w64 final u1300 vs heuristic"][0])
    report = analyze(state, frontier_db)
    assert "The frozen comparison schedule is incomplete." not in report.issues
    assert f"4 of {64 * len(keys)} played games have no score." in report.issues
    assert report.counts.failed_matches == 1
    strength = report.sizes[0].against_heuristic
    assert strength is not None and strength.score is None
    assert strength.worst.mean == 0.75 - 0.75 / 16
    assert strength.best.mean == 0.75 + 0.25 / 16


def test_a_bound_too_wide_to_decide_leaves_learning_unknown(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    for key in keys["w64 final u1300 vs w64 u0"][:12]:
        _fail_match(frontier_db, key)
    report = analyze(state, frontier_db)
    small = report.sizes[0]
    assert small.learned is None
    assert small.against_initial is not None
    assert small.against_initial.worst.lower < 0.5 < small.against_initial.best.lower


def test_failed_legs_cannot_rescue_a_clear_negative_reading(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state, wins=0)
    _fail_match(frontier_db, keys["w64 final u1300 vs w64 u0"][0])
    assert analyze(state, frontier_db).sizes[0].learned is False


def test_an_endpoint_at_the_update_cap_is_not_equal_compute(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 3000, LARGE: 450})
    _outcomes(frontier_db, state)
    report = analyze(state, frontier_db)
    assert report.scale is None and not report.complete
    assert report.issues == [
        "A run hit the update cap; no equal-compute scale reading."
    ]
    assert _endpoint_score(report) == 0.75
    assert [reading.learned for reading in report.sizes] == [True, True]


def _played(match: Match) -> Tables:
    """Three of four legs won by the first player, in `play`'s result shape."""
    return {
        "games": [
            {
                "seat": seat,
                "game": leg,
                "game_seat": (leg + seat) % 2,
                "score": float((leg < 3) != bool(seat)),
                "valid": True,
                "attempted": True,
                "terminated": True,
                "truncated": False,
                "failure": None,
                "seconds": 1.0,
            }
            for leg in range(4)
            for seat in (0, 1)
        ]
    }


def test_report_and_analysis_read_a_store_of_played_matches(tmp_path: Path) -> None:
    """`etude report` draws report(), and analysis bounds Etude's real failed match."""
    file = ROOT / "experiments/tla_frontier.py"
    small, large = read_job_file(file, ["--revision", "HEAD"])
    _tracked(small, tmp_path, {50: {"loss": 0.1}}, (0, 25, 50))
    _tracked(large, tmp_path, {25: {"loss": 0.1}}, (0, 25))
    state = observe(tmp_path, EXPERIMENT)
    benchmark = evaluation_spec(eval(state)[0]).tournament
    matchups = {matchup.name: matchup.matches() for matchup in benchmark.matchups}
    died = matchups["w256 final vs w64 final"][0]

    def play(match: Match) -> Tables:
        if match.key == died.key:
            raise RuntimeError("the worker died")
        return _played(match)

    def run(name: str, matches: list[Match]) -> None:
        evaluate(
            experiment=EXPERIMENT,
            environment=benchmark.environment,
            matches=matches,
            play=play,
            run=name,
            directory=tmp_path,
        )

    # A pass ends at a raising match, so the failure gets its own eval run here.
    run("fixture-eval", [m for ms in matchups.values() for m in ms if m != died])
    with pytest.raises(EtudeError, match="match_failed"):
        run("fixture-failed", [died])
    page = build(tmp_path, file)
    assert page.cells
    html = render(page)
    for title in (
        "Width 64 vs fixed heuristic",
        "Width 256 vs fixed heuristic",
        "Width 256 vs width 64 at equal updates",
    ):
        assert title in html
    with statsdb(EXPERIMENT, tmp_path) as db:
        report = analyze(state, db)
    played = sum(len(ms) for ms in matchups.values()) * 4
    assert report.issues == [f"4 of {played} played games have no score."]
    assert report.scale is not None and report.scale.score is None
    assert report.scale.worst.mean == 0.75 - 0.75 / 16
    assert report.scale.best.mean == 0.75 + 0.25 / 16
