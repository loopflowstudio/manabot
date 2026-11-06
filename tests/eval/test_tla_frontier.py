"""The c5fba4 frontier protocol pins two sizes and a milestone evaluation schedule."""

from collections import Counter
from collections.abc import Iterator
from decimal import Decimal
import json
from pathlib import Path

import duckdb
from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    read_experiment,
)
from etude.jobs import evaluation_spec
from etude.run.records import ArtifactRef
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
    frozen,
    runs,
)
from experiments.tla_frontier_analysis import analyze
from manabot.env.observation import ObservationSpace
from manabot.model.actor_critic import ActorCritic
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


def _state(reached: dict[SizeLabel, int], *, finished: bool = True) -> ExperimentState:
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
                        | ({end} if finished else set())
                    )
                ),
            )
            for label, end in reached.items()
        ),
    )


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
    assert tags["head-to-head"] == 6  # 25, 50, 100, 200, 300, 400
    assert len(names) == len(playable)


def test_running_runs_evaluate_reached_milestones_only() -> None:
    state = _state({SMALL: 100, LARGE: 50}, finished=False)
    schedule = comparison_schedule(state, frozen()[1])
    names = {c.matchup.name for c in schedule if c.matchup is not None}
    assert "w256 u50 vs w64 u50" in names
    assert not any("final" in name for name in names)
    (job,) = eval(state)
    assert job.max_usd == Decimal("2")
    assert job.max_attempts == 1


def test_evaluation_job_prepares_a_stock_rented_machine() -> None:
    (job,) = eval(_state({SMALL: 25, LARGE: 25}, finished=False))
    assert job.setup is not None and "--extra evaluation" in job.setup
    assert "uv run --locked --extra evaluation python" in job.command


def test_no_checkpoints_schedules_nothing() -> None:
    assert eval(ExperimentState(experiment=EXPERIMENT)) == []


def test_training_runs_admit_milestones_and_wall_budget() -> None:
    launched = runs("HEAD")
    assert [run.job.name for run in launched] == [
        "tla-frontier-c5fba4-w64",
        "tla-frontier-c5fba4-w256",
    ]
    assert {run.job.until for run in launched} == {"updates=3000"}


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


def test_analysis_reads_learning_and_scale_past_unreached_milestones(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    report = analyze(state, frontier_db)
    assert report.complete and report.issues == []
    assert report.scale is not None and report.scale.mean == 0.75
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
