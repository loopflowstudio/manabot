"""The c5fba4 frontier protocol pins two sizes and a milestone evaluation schedule."""

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

import duckdb
from etude.eval.match import Tables, evaluate
from etude.eval.query import statsdb
from etude.eval.records import Inference, Match, Matchup, Model
from etude.experiment import (
    Checkpoint,
    Checkpoints,
    ExperimentState,
    TrainingRun,
    advance,
    observe,
    read_experiment,
)
from etude.jobs import evaluation_spec
from etude.launch.errors import LauncherError
from etude.launch.job_file import read_job_file
from etude.launch.records import Run
from etude.page import build, render
from etude.run.errors import EtudeError
from etude.run.listing import as_storage
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
    HostSpeed,
    SizeLabel,
    comparison_schedule,
    eval,
    export_updates,
    frozen,
    runs,
    size_run,
)
from experiments.tla_frontier_analysis import (
    Comparison,
    FrontierAnalysis,
    PolicyActions,
    SetAside,
    analyze,
    read_hosts,
)
from experiments.tla_frontier_calibration import (
    EXPERIMENT as CALIBRATION,
    MAX_HOURS,
    MAX_USD_PER_SIZE,
    UPDATES,
    calibration_regime,
    eval as calibration_eval,
    read_calibration,
    read_rehearsal,
    read_screen,
    runs as calibration_runs,
)
from manabot.env.observation import ObservationSpace
from manabot.eval.results import EvaluationRow
from manabot.experiment.job import TrainingJob
from manabot.infra.config import RuntimeConfig
from manabot.infra.identity import file_sha256
from manabot.infra.runtime import RunContext, regime_tag
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.training.execution import cuda_peak_metrics, execute_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

CALIBRATION_FILE = ROOT / "experiments/tla_frontier_calibration.py"
# What a launcher hands a worker: `uuid4().hex[:12]`, never the job's name.
RUN_NAMES: dict[SizeLabel, str] = {SMALL: "0f3a9c1d2b4e", LARGE: "7be2d40a61c5"}


def _checkpoint(label: SizeLabel, update: int) -> Checkpoint:
    run = RUN_NAMES[label]
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
        tags=(regime_tag(f"tla-frontier-c5fba4-{label}"),),
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
                name=RUN_NAMES[label],
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
    return replace(
        state,
        runs=tuple(
            replace(run, state="failed") if run.name == RUN_NAMES[label] else run
            for run in state.runs
        ),
    )


def _tracked(
    launched: Run,
    source: Path,
    rows: dict[int, dict[str, float]],
    checkpoints: tuple[int, ...] = (),
    *,
    end: Literal["finished", "failed"] | None = "finished",
) -> str:
    """Write a launched run through the worker's path and return its stored name.

    A real `RunContext` under the launcher's environment owns the run's name and
    checkpoint tags, so no fixture states what a worker writes.
    """
    job = TrainingJob.model_validate_json(launched.job.config)
    name = uuid4().hex[:12]
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("ETUDE_RUN", name)
        environment.setenv("ETUDE_TRACK_DIR", str(source))
        environment.setenv("ETUDE_JOB_NAME", job.regime.id)
        context = RunContext(
            RuntimeConfig(name=name, runs_dir=source / "local", seed=launched.seed),
            experiment=job.experiment,
            regime=job.regime.id,
        )
        for update in sorted({*rows, *checkpoints}):
            context.log_update(
                rows.get(update, {}), updates=update, transitions=update * 100
            )
            if update in checkpoints:
                path = source / f"{name}-u{update}.pt"
                path.write_text(f"synthetic export {name} {update}")
                context.checkpoint(path, transitions=update * 100)
        context.close(end or "finished")
    if end is None:
        # A pod still training has written no end state yet.
        next(source.rglob(f"{name}/state.json")).unlink()
    return name


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
    run = size_run(state, LARGE, protocol)
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
    protocol = frozen()[1]
    assert size_run(state, SMALL, protocol) is not None
    assert size_run(state, LARGE, protocol) is not None
    (job,) = experiment.eval(state)
    names = {matchup.name for matchup in evaluation_spec(job).tournament.matchups}
    assert {
        "w64 u0 vs heuristic",
        "w256 u0 vs heuristic",
        "w64 u25 vs heuristic",
        "w256 u25 vs heuristic",
        "w256 u25 vs w64 u25",
    } <= names


def test_a_real_run_context_names_its_size_under_a_launcher_id(tmp_path: Path) -> None:
    """Stored names are random ids, so the size reads back from checkpoint tags."""
    small, large = runs("HEAD")
    names = {
        SMALL: _tracked(small, tmp_path, {}, (0, 25), end=None),
        LARGE: _tracked(large, tmp_path, {}, (0, 25), end=None),
    }
    state = observe(tmp_path, EXPERIMENT)
    protocol = frozen()[1]
    assert {run.name for run in state.runs} == set(names.values())
    for label, name in names.items():
        found = size_run(state, label, protocol)
        assert found is not None and found.name == name
        assert not found.ended
    assert len(evaluation_spec(eval(state)[0]).tournament.matchups) == 5


def test_regime_tags_stay_out_of_match_keys() -> None:
    conditions = tuple(pair.model_dump(mode="json") for pair in frozen()[1].pairs)
    inference = Inference(source=[REGIME])

    def keys(saved: Checkpoint) -> list[str]:
        played = Matchup(
            name="fixture",
            players=(saved.model(inference), saved.model(inference)),
            conditions=conditions,
            games=4,
        )
        return [match.key for match in played.matches()]

    tagged = _checkpoint(SMALL, 25)
    assert tagged.tags
    assert keys(tagged) == keys(tagged.model_copy(update={"tags": ()}))


def _protocol_blocks() -> list[str]:
    protocol = (ROOT / "experiments/protocols/tla-frontier-c5fba4-v1.md").read_text()
    return protocol.split("```")[1::2]


def test_one_size_relaunches_alone_with_its_own_ceiling() -> None:
    file = ROOT / "experiments/tla_frontier.py"
    (alone,) = read_job_file(file, ["--revision", "HEAD", "--size", LARGE])
    assert alone.job.name == "tla-frontier-c5fba4-w256"
    assert alone.job.max_usd == Decimal("38")
    assert alone.job.max_attempts == 1
    with pytest.raises(LauncherError, match="size must be"):
        read_job_file(file, ["--revision", "HEAD", "--size", "w128"])


def test_protocol_launches_set_cohort_limits_only_on_a_cohort() -> None:
    """`etude launch` refuses `--max-usd` or `--parallel` without `--cohort`."""
    launches = [block for block in _protocol_blocks() if "etude launch" in block]
    assert len(launches) == 3
    for block in launches:
        assert "--cohort" in block or not {"--max-usd", "--parallel"} & set(
            block.split()
        )


def test_protocol_commands_name_their_rentals() -> None:
    """A copied command rents a named on-demand, never a placeholder."""
    blocks = _protocol_blocks()
    # One prefix owns the control client.
    assert sum("etude-sdk" in block for block in blocks) == 1
    commands = [
        command.split()
        for block in blocks
        for command in block.split('"${control[@]}"')
        if "--rent" in command
    ]
    assert len(commands) == 5
    for command in commands:
        rented = command[command.index("--rent") + 1]
        assert rented == ("l4" if "launch" in command else "cpu32")


def test_a_replaced_pod_is_set_aside_and_listed(
    tmp_path: Path, frontier_db: duckdb.DuckDBPyConnection
) -> None:
    """Cancelled at the screen, a run leaves its size to the replacement."""
    small, large = runs("HEAD")
    retired = _tracked(large, tmp_path, {}, (0,), end="failed")
    _tracked(small, tmp_path, {}, (0, 25), end=None)
    replacement = _tracked(large, tmp_path, {}, (0, 25), end=None)
    state = observe(tmp_path, EXPERIMENT)
    protocol = frozen()[1]
    found = size_run(state, LARGE, protocol)
    assert found is not None and found.name == replacement
    assert not any(
        retired in str(comparison.matchup)
        for comparison in comparison_schedule(state, protocol)
    )
    report = analyze(state, frontier_db, {})
    assert report.set_aside == [
        SetAside(run=retired, size=LARGE, state="failed", last_update=0)
    ]


def test_a_pod_cancelled_past_the_first_milestone_is_not_replaceable(
    tmp_path: Path,
) -> None:
    """It keeps its size's run, which is why the screen reads each pod before 25."""
    small, _ = runs("HEAD")
    _tracked(small, tmp_path, {}, (0, 25), end="failed")
    _tracked(small, tmp_path, {}, (0,), end=None)
    with pytest.raises(ValueError, match="admits one w64 run"):
        size_run(observe(tmp_path, EXPERIMENT), SMALL, frozen()[1])


def test_two_live_runs_of_one_size_stay_an_error(tmp_path: Path) -> None:
    _, large = runs("HEAD")
    _tracked(large, tmp_path, {}, (0, 25), end=None)
    _tracked(large, tmp_path, {}, (0, 25), end=None)
    with pytest.raises(ValueError, match="admits one w256 run"):
        size_run(observe(tmp_path, EXPERIMENT), LARGE, frozen()[1])


def test_calibration_changes_only_length_and_identity() -> None:
    regimes, _ = frozen()
    for label, regime in regimes.items():
        changed = calibration_regime(regime, label)
        assert changed.stages[0].updates == UPDATES
        assert _without_length(changed) == _without_length(regime)


def test_calibration_fits_its_ceiling_on_the_admitted_gpus() -> None:
    launched = read_job_file(CALIBRATION_FILE, ["--revision", "HEAD"])
    assert len(launched) == 2
    jobs = [TrainingJob.model_validate_json(run.job.config) for run in launched]
    assert {job.experiment for job in jobs} == {CALIBRATION}
    assert {run.job.until for run in launched} == {f"updates={UPDATES}"}
    assert sum(run.job.max_usd or 0 for run in launched) == Decimal("2")
    assert Decimal(str(MAX_HOURS + 0.25)) * Decimal("0.67") <= MAX_USD_PER_SIZE


def _host_rows(us_per_row: dict[int, float]) -> dict[int, dict[str, float]]:
    """A pod's step phase and rows, at 1,000 rows per update."""
    return {
        update: {
            "time/collection/environment_step_and_encoding": us * 1000 / 1e6,
            "rows": 1000,
        }
        for update, us in us_per_row.items()
    }


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
    for update, (rss, count, step) in {
        1: (9e9, 40_000, 99.0),
        2: (20e9, 90_000, 18.0),
        3: (12e9, 50_000, 15.0),
    }.items():
        rows[update] |= {
            "coordinates/rss_bytes": rss,
            "rows": count,
            "time/collection/environment_step_and_encoding": step,
        }
    small, large = calibration_runs("HEAD")
    _tracked(small, tmp_path, rows, (0, 3))
    _tracked(large, tmp_path, {})
    readings = {reading.job: reading for reading in read_calibration(tmp_path)}
    reading = readings["tla-frontier-c5fba4-calibration-w64"]
    assert len(reading.run) == 12  # the launcher's id; `job` says which size
    assert reading.updates == 3
    assert reading.seconds_per_update == 220  # the first update's warm-up is set aside
    assert reading.updates_in_wall_budget == 883
    assert reading.peak_allocated_cuda_bytes == 4_000_000_000
    assert reading.peak_reserved_cuda_bytes == 6_000_000_000
    assert reading.peak_rss_bytes == 20_000_000_000  # not the last update's value
    assert reading.median_rows_per_update == 50_000
    # Per row over updates 2 and 3, the first update's warm-up set aside.
    assert reading.host.step_us_per_row == pytest.approx(250)
    assert reading.host.step_spread == pytest.approx(1.5)
    assert reading.host.cpu_model is None
    silent = readings["tla-frontier-c5fba4-calibration-w256"]
    assert silent.updates == 0 and silent.seconds_per_update is None
    assert silent.peak_reserved_cuda_bytes is None
    assert silent.peak_rss_bytes is None and silent.median_rows_per_update is None
    assert silent.host == HostSpeed(
        step_us_per_row=None, step_spread=None, cpu_model=None
    )
    # A frontier pass never sees calibration, even from a shared store.
    assert observe(tmp_path, EXPERIMENT).runs == ()
    assert observe(tmp_path, CALIBRATION).runs != ()
    misnamed = replace(observe(tmp_path, CALIBRATION), experiment=EXPERIMENT)
    protocol = frozen()[1]
    assert size_run(misnamed, SMALL, protocol) is None
    assert size_run(misnamed, LARGE, protocol) is None


def _calibrated(source: Path, last: int = 5) -> tuple[str, str]:
    """Both calibration runs, ended, stored through the worker's path."""
    small, large = calibration_runs("HEAD")
    return (
        _tracked(small, source, {last: {"loss": 0.1}}, (0, last)),
        _tracked(large, source, {last: {"loss": 0.1}}, (0, last)),
    )


def _rehearsal_matchups(source: Path) -> dict[str, Matchup]:
    (job,) = read_experiment(CALIBRATION_FILE).eval(observe(source, CALIBRATION))
    benchmark = evaluation_spec(job).tournament
    return {matchup.name: matchup for matchup in benchmark.matchups}


def test_etude_eval_reads_calibration_and_launches_the_rehearsal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`etude eval`'s real pass over a store of calibration runs, up to the rental."""
    small, large = _calibrated(tmp_path / "store")
    launched: list[tuple[Run, str | None]] = []

    def launch(run: Run, pool: str | None, *, rent: str | None) -> str:
        launched.append((run, rent))
        return run.id

    monkeypatch.setattr("etude.experiment.client.launch", launch)
    (done,) = advance(
        read_experiment(CALIBRATION_FILE),
        source=as_storage(tmp_path / "store"),
        rent="cpu32",
        directory=tmp_path / "home",
    )
    assert done.new == 32 and done.launched is not None
    ((run, rent),) = launched
    assert rent == "cpu32"
    spec = evaluation_spec(run.job)
    assert spec.experiment == CALIBRATION
    assert spec.play.module == "experiments.tla_frontier"
    assert callable(spec.play.load(run.job.repo))
    assert spec.matches is not None and len(spec.matches) == 32
    seated = {
        player.checkpoint.key.run
        for match in spec.matches
        for player in match.players
        if isinstance(player, Model)
    }
    assert seated == {small, large}


def test_rehearsal_is_a_frontier_pass_of_calibration_checkpoints(
    tmp_path: Path,
) -> None:
    small, large = _calibrated(tmp_path)
    matchups = _rehearsal_matchups(tmp_path)
    assert set(matchups) == {"w256 last vs w256 u0", "w256 last vs w64 last"}
    protocol = frozen()[1]
    pairs = [pair.model_dump(mode="json") for pair in protocol.pairs]
    for matchup in matchups.values():
        matches = matchup.matches()
        assert len(matches) == len(protocol.pairs)
        # `play` refuses anything but the frozen pairs at four legs.
        assert all(m.conditions in pairs and m.games == 4 for m in matches)
    last, first = (p.checkpoint.key for p in matchups["w256 last vs w256 u0"].players)
    assert (last.run, first.run) == (large, large)
    assert last.name != first.name
    across = matchups["w256 last vs w64 last"].players
    assert [player.checkpoint.key.run for player in across] == [large, small]

    (job,) = calibration_eval(observe(tmp_path, CALIBRATION))
    pinned = _state({SMALL: 25, LARGE: 25}, finished=False)
    (pass_job,) = eval(pinned)
    # Only what is played differs from a frontier pass.
    for field in ("command", "setup", "max_hours", "max_usd", "max_attempts"):
        assert getattr(job, field) == getattr(pass_job, field)
    assert evaluation_spec(job).workers == evaluation_spec(pass_job).workers == 28
    assert evaluation_spec(job).play == evaluation_spec(pass_job).play
    # Nothing it plays can reach a frontier slot.
    held = evaluation_spec(job).tournament.environment
    assert held != evaluation_spec(pass_job).tournament.environment


def test_rehearsal_waits_for_both_runs_to_end(tmp_path: Path) -> None:
    assert calibration_eval(ExperimentState(experiment=CALIBRATION)) == []
    small, large = calibration_runs("HEAD")
    _tracked(small, tmp_path, {5: {"loss": 0.1}}, (0, 5))
    _tracked(large, tmp_path, {3: {"loss": 0.1}}, (0, 3), end=None)
    assert calibration_eval(observe(tmp_path, CALIBRATION)) == []


def test_rehearsal_refuses_a_size_with_no_distinct_pair(tmp_path: Path) -> None:
    small, large = calibration_runs("HEAD")
    _tracked(small, tmp_path, {5: {"loss": 0.1}}, (0, 5))
    _tracked(large, tmp_path, {}, (0,), end="failed")
    with pytest.raises(ValueError, match="w256 stored no update past 0"):
        calibration_eval(observe(tmp_path, CALIBRATION))


def test_rehearsal_rejects_a_foreign_experiment() -> None:
    with pytest.raises(ValueError, match="calibration's runs"):
        calibration_eval(ExperimentState(experiment=EXPERIMENT))


def _rehearsed(
    source: Path, played: Callable[[Match], Tables], run: str = "rehearsal-pass"
) -> None:
    """Play the rehearsal's matches through Etude's real match writer."""
    matches = [
        m for each in _rehearsal_matchups(source).values() for m in each.matches()
    ]
    (job,) = calibration_eval(observe(source, CALIBRATION))
    evaluate(
        experiment=CALIBRATION,
        environment=evaluation_spec(job).tournament.environment,
        matches=matches,
        play=played,
        run=run,
        directory=source,
    )


# What `play` stores for a game killed at its limit.
_KILLED = {
    "valid": False,
    "score": None,
    "seconds": None,
    "failure": "game limit",
    "terminated": False,
    "policy_decisions": None,
    "policy_dropped_actions": None,
}


def _free(run: str) -> Decimal:
    return Decimal(0)


def _timed(seconds: float) -> Callable[[Match], Tables]:
    def played(match: Match) -> Tables:
        return {
            "games": [
                row | {"seconds": seconds + row["game"]}
                for row in _played(match)["games"]
            ]
        }

    return played


def test_rehearsal_reader_passes_a_clean_pass_and_keeps_training_readings_apart(
    tmp_path: Path,
) -> None:
    _calibrated(tmp_path)
    _rehearsed(tmp_path, _timed(100.0))
    reading = read_rehearsal(tmp_path, lambda run: Decimal("0.48"))
    assert reading.runs == ("rehearsal-pass",)
    assert (reading.matches, reading.failed_matches) == (32, 0)
    assert (reading.games, reading.invalid_games) == (128, 0)
    assert reading.median_game_seconds == 101.5
    assert reading.slowest_game_seconds == 103
    assert reading.pass_usd == Decimal("0.48")
    assert reading.passed
    # A pass is a stored run of this experiment, and no training reading.
    assert len(read_calibration(tmp_path)) == 2
    assert len(read_screen(tmp_path)) == 2


def test_rehearsal_fails_at_the_game_line(tmp_path: Path) -> None:
    _calibrated(tmp_path)
    _rehearsed(tmp_path, _timed(297.0))
    reading = read_rehearsal(tmp_path, _free)
    assert reading.slowest_game_seconds == 300 and not reading.passed


def test_rehearsal_fails_on_one_killed_game(tmp_path: Path) -> None:
    _calibrated(tmp_path)
    killed: list[str] = []

    def played(match: Match) -> Tables:
        tables = _timed(100.0)(match)
        if not killed:
            killed.append(match.key)
            for row in tables["games"]:
                if row["game"] == 3:
                    row |= _KILLED
        return tables

    _rehearsed(tmp_path, played)
    reading = read_rehearsal(tmp_path, _free)
    assert (reading.games, reading.invalid_games) == (128, 1)
    assert reading.slowest_game_seconds == 103 and not reading.passed


def test_rehearsal_fails_when_every_game_was_killed(tmp_path: Path) -> None:
    """Etude then stores no `seconds` column at all."""
    _calibrated(tmp_path)

    def played(match: Match) -> Tables:
        return {"games": [row | _KILLED for row in _played(match)["games"]]}

    _rehearsed(tmp_path, played)
    reading = read_rehearsal(tmp_path, _free)
    assert (reading.games, reading.invalid_games) == (128, 128)
    assert reading.slowest_game_seconds is None and not reading.passed


def test_rehearsal_fails_with_a_raising_match_or_nothing_played(tmp_path: Path) -> None:
    empty = read_rehearsal(tmp_path, _free)
    assert empty.games == 0 and empty.runs == () and not empty.passed
    _calibrated(tmp_path)
    keys: list[str] = []

    def played(match: Match) -> Tables:
        keys.append(match.key)
        if len(keys) == 5:
            raise RuntimeError("the worker died")
        return _timed(100.0)(match)

    with pytest.raises(EtudeError, match="match_failed"):
        _rehearsed(tmp_path, played)
    reading = read_rehearsal(tmp_path, lambda run: Decimal("0.2"))
    assert reading.failed_matches == 1 and reading.games < 128
    assert reading.pass_usd == Decimal("0.2") and not reading.passed


def test_a_finished_receipt_names_the_cpu_when_its_revision_records_it(
    tmp_path: Path,
) -> None:
    small, large = runs("HEAD")
    named = _tracked(small, tmp_path, _host_rows({2: 100}), (0, 25))
    older = _tracked(large, tmp_path, _host_rows({2: 100}), (0, 25))
    for run, receipt in (
        (named, {"host": {"cpu_model": "AMD EPYC 7763"}}),
        (older, {"status": "paused"}),
    ):
        path = tmp_path / run / "attempt" / "training" / "run.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
    hosts = read_hosts(tmp_path)
    assert hosts[named].cpu_model == "AMD EPYC 7763"
    assert hosts[older].cpu_model is None


def test_screen_reads_each_pod_against_the_fastest_in_the_store(tmp_path: Path) -> None:
    window = range(2, 7)
    small, large = runs("HEAD")
    calibrated, _ = calibration_runs("HEAD")
    fast = _host_rows(dict.fromkeys(range(2, 6), 200.0))
    _tracked(calibrated, tmp_path, fast)
    kept = _tracked(
        small, tmp_path, _host_rows(dict.fromkeys(window, 250.0)), (0,), end=None
    )
    # Both frontier pods can read slow against calibration's.
    slow = _tracked(
        large, tmp_path, _host_rows(dict.fromkeys(window, 300.0)), (0,), end=None
    )
    early = _tracked(large, tmp_path, _host_rows({2: 400.0, 3: 400.0}), (0,), end=None)
    readings = {reading.run: reading for reading in read_screen(tmp_path)}
    assert readings[kept].ratio == pytest.approx(1.25) and readings[kept].slow is False
    assert readings[slow].ratio == pytest.approx(1.5) and readings[slow].slow is True
    # Before its window closes a live run is shown but not called.
    assert readings[early].slow is None
    assert sum(r.experiment == CALIBRATION for r in readings.values()) == 1


def test_one_real_update_writes_every_key_calibration_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real `execute_regime` run, shrunk only so it finishes in a test."""
    regime = calibration_regime(frozen()[0][SMALL], SMALL)
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    short = regime.model_copy(
        update={
            "agent": regime.agent.model_copy(
                update={
                    "hidden_dim": 16,
                    "attention_layers": 1,
                    "num_attention_heads": 2,
                }
            ),
            "stages": [
                stage.model_copy(
                    update={
                        "updates": 2,
                        "streams": 2,
                        "transitions": 8,
                        "execution": stage.execution.model_copy(
                            update={"device": "cpu"}
                        ),
                    }
                )
            ],
        }
    )
    monkeypatch.setenv("ETUDE_RUN", uuid4().hex[:12])
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    monkeypatch.setenv("ETUDE_JOB_NAME", short.id)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(
            short, 21501, tmp_path / "run", store, experiment=CALIBRATION
        )
    assert run.status == "completed"
    (reading,) = read_calibration(tmp_path / "track")
    assert reading.job == short.id and reading.state == "finished"
    assert reading.updates == 2 and reading.seconds_per_update is not None
    assert reading.peak_rss_bytes is not None
    assert reading.median_rows_per_update is not None
    assert reading.host.step_us_per_row is not None
    # The real path tags its checkpoints, update 0 and the final export alike.
    (stored,) = observe(tmp_path / "track", CALIBRATION).runs
    tagged = stored.checkpoints.tagged(regime_tag(short.id))
    assert len(tagged) == len(stored.checkpoints) and len(tagged) >= 2


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


def test_play_carries_a_real_games_cutoff_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real games through the evaluation path; four slots force actual overflow."""
    regimes, protocol = frozen()
    regime = regimes[SMALL]
    narrow = regime.model_copy(
        update={"observation": regime.observation.model_copy(update={"max_actions": 4})}
    )
    space = ObservationSpace(narrow.observation)
    path = tmp_path / "u0.pt"
    save_checkpoint(ActorCritic(space, narrow.agent), space, path, player_configs=None)
    saved = _checkpoint(SMALL, 0).model_copy(
        update={
            "artifact": ArtifactRef(
                uri=path.as_uri(),
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
            )
        }
    )
    state = ExperimentState(
        experiment=EXPERIMENT,
        runs=(
            TrainingRun(
                name=saved.run,
                experiment=EXPERIMENT,
                state="running",
                checkpoints=Checkpoints([saved]),
            ),
        ),
    )
    match = evaluation_spec(eval(state)[0]).tournament.matchups[0].matches()[0]
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "etude"))
    monkeypatch.setattr(
        tla_frontier, "frozen", lambda: ({SMALL: narrow, LARGE: narrow}, protocol)
    )

    games = tla_frontier.play(match)["games"]

    retained = tmp_path / "etude" / "arena" / match.key / "rows.json"
    decisions = {
        row["leg"]: row["decisions"] for row in json.loads(retained.read_text())
    }
    assert len(decisions) == 4 and all(decisions.values())
    for row in games:
        assert row["policy_decisions"] == decisions[row["game"]]
    assert sum(row["policy_dropped_actions"] for row in games) > 0


def test_frozen_rows_keep_their_dump_without_cutoff_counts() -> None:
    """`_verify_block` digests this dump, so a frozen row must gain no key."""
    frozen_row = {
        "arena_key": {
            "world": "w17",
            "content_suite": "fixture",
            "viewer_boundary": "acting-viewer",
            "arena_version": "fixture",
            "rating_model_version": "unrated",
            "rating_prior_sha256": "a" * 64,
            "anchor_cohort_sha256": "b" * 64,
            "evaluation_compute_envelope_id": "fixture",
        },
        "deal_seed": 1,
        "leg": 0,
        "player_a": "a",
        "player_b": "b",
        "player_a_registration_sha256": "c" * 64,
        "player_b_registration_sha256": "d" * 64,
        "score_a": 1.0,
        "failure": None,
        "terminated": True,
        "truncated": False,
        "replay_passed": True,
        "trace_path": "traces/a__b.1.commands.jsonl.gz",
        "game_seconds": 1.0,
        "integrity": {},
    }
    row = EvaluationRow.model_validate(frozen_row)
    assert row.policy_action_counts is None
    assert "policy_actions" not in row.model_dump(mode="json")
    counted = EvaluationRow.model_validate(
        {**frozen_row, "policy_actions": {"decisions": 9, "dropped_actions": 2}}
    )
    assert counted.policy_action_counts is not None
    assert counted.policy_action_counts.dropped_actions == 2


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


def _endpoint_comparison(report: FrontierAnalysis) -> Comparison:
    return next(c for c in report.comparisons if c.tag == "endpoint-head-to-head")


def _endpoint_score(report: FrontierAnalysis) -> float:
    """The endpoint head-to-head's score, reported even without a scale reading."""
    played = _endpoint_comparison(report)
    assert played.result is not None and played.result.score is not None
    return played.result.score.mean


def test_analysis_reads_learning_and_scale_past_unreached_milestones(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    report = analyze(state, frontier_db, {})
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
    report = analyze(state, frontier_db, {})
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
    report = analyze(state, frontier_db, {})
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
    report = analyze(state, frontier_db, {})
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
    report = analyze(state, frontier_db, {})
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
    report = analyze(state, frontier_db, {})
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
    report = analyze(state, frontier_db, {})
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
    report = analyze(state, frontier_db, {})
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
    assert analyze(state, frontier_db, {}).sizes[0].learned is False


def test_an_endpoint_at_the_update_cap_is_not_equal_compute(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 3000, LARGE: 450})
    _outcomes(frontier_db, state)
    report = analyze(state, frontier_db, {})
    assert report.scale is None and not report.complete
    assert report.issues == [
        "A run hit the update cap; no equal-compute scale reading."
    ]
    assert _endpoint_score(report) == 0.75
    assert [reading.learned for reading in report.sizes] == [True, True]


def _add_policy_actions(db: duckdb.DuckDBPyConnection) -> None:
    """Every game, on both seats, 100 decisions and 3 dropped actions."""
    db.execute("ALTER TABLE games ADD COLUMN policy_decisions BIGINT DEFAULT 100")
    db.execute("ALTER TABLE games ADD COLUMN policy_dropped_actions BIGINT DEFAULT 3")


def test_cutoff_counts_are_totalled_per_comparison_and_add_no_issue(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    _add_policy_actions(frontier_db)
    report = analyze(state, frontier_db, {})
    assert report.complete and report.issues == []
    played = _endpoint_comparison(report)
    # 16 pairs of four games, each counted once though both seats repeat it.
    assert played.policy_actions == PolicyActions(
        games=64, decisions=6400, dropped_actions=192
    )
    assert report.policy_actions.games == 64 * len(keys)
    assert report.policy_actions.dropped_actions == 192 * len(keys)
    unreached = next(c for c in report.comparisons if c.unreached)
    assert unreached.policy_actions == PolicyActions()


def test_a_killed_game_leaves_its_count_out_of_the_totals(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 1300, LARGE: 450})
    keys = _outcomes(frontier_db, state)
    _add_policy_actions(frontier_db)
    killed = keys["w256 final vs w64 final"][0]
    _fail_leg(frontier_db, killed, 0)
    frontier_db.execute(
        "UPDATE games SET policy_decisions = NULL, policy_dropped_actions = NULL "
        "WHERE match = ? AND game = 0",
        [killed],
    )
    report = analyze(state, frontier_db, {})
    played = _endpoint_comparison(report)
    assert played.counts.expected == 64
    assert played.policy_actions == PolicyActions(
        games=63, decisions=6300, dropped_actions=189
    )
    # Only the failed game is an issue; a nonzero cutoff never is.
    assert report.issues == [f"1 of {64 * len(keys)} played games have no score."]


def test_analysis_without_cutoff_columns_reports_no_counted_games(
    frontier_db: duckdb.DuckDBPyConnection,
) -> None:
    state = _state({SMALL: 100, LARGE: 50})
    _outcomes(frontier_db, state)
    report = analyze(state, frontier_db, {})
    assert report.complete
    assert report.policy_actions == PolicyActions()


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
                # A game killed at its limit returns no count, as leg 3 here.
                "policy_decisions": 100 if leg < 3 else None,
                "policy_dropped_actions": 2 if leg < 3 else None,
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
    died, uncounted = matchups["w256 final vs w64 final"][:2]

    def play(match: Match) -> Tables:
        if match.key == died.key:
            raise RuntimeError("the worker died")
        played = _played(match)
        if match.key == uncounted.key:
            # All four legs killed at their limit: this block stores no count at all.
            for row in played["games"]:
                row |= {"policy_decisions": None, "policy_dropped_actions": None}
        return played

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
        report = analyze(state, db, read_hosts(tmp_path))
    played = sum(len(ms) for ms in matchups.values()) * 4
    assert report.issues == [f"4 of {played} played games have no score."]
    assert report.scale is not None and report.scale.score is None
    assert report.scale.worst.mean == 0.75 - 0.75 / 16
    assert report.scale.best.mean == 0.75 + 0.25 / 16
    # Fourteen counted matches of the endpoint block, three counted games each.
    assert _endpoint_comparison(report).policy_actions == PolicyActions(
        games=42, decisions=4200, dropped_actions=84
    )
