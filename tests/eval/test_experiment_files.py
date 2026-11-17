"""Manabot's experiment files are what `etude launch`, `etude eval` and `etude report` read.

A run tracked under a file's stem draws on that file's page with every metric
its panels name and a flame graph of its phases; `eval(state)` asks for the
same games the MTG-183 follower plays.
"""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
import sys

from etude.declarations import read_experiment, read_training_jobs
from etude.errors import EtudeError
from etude.eval.records import Match
from etude.experiment import Checkpoint, TrainingRun
from etude.jobs import Checkpoints, ExperimentState, evaluation_spec
from etude.page import build, experiment_files
from etude.records import ArtifactRef
from etude.report import Counters, Curve, Matches, Series
import pytest

from experiments import elemental, opponent_memory, tla_frontier
from experiments.elemental_long_eval import (
    Checkpoint as FollowedCheckpoint,
    EvalState,
    units,
)
from manabot.backends import resolve
from manabot.experiment.job import TrainingJob
from manabot.infra.config import MatchConfig
from manabot.training.execution import execute_regime
from manabot.training.regime import Learning, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_two_seat_learning import _agent

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"


@pytest.fixture
def keep_imported_modules() -> Iterator[None]:
    """Restore modules `build` drops so later tests keep their imported objects.

    `etude.page.build` re-imports repository neighbours. Restore saved modules
    and remove newly imported neighbours so later imports, pickling and
    monkeypatching agree with module-level references.
    """
    saved = dict(sys.modules)
    yield
    for name in set(sys.modules) - saved.keys():
        source = getattr(sys.modules[name], "__file__", None)
        if isinstance(source, str):
            path = Path(source).resolve()
            if path.is_relative_to(ROOT) and not any(
                part.startswith(".") for part in path.relative_to(ROOT).parts
            ):
                del sys.modules[name]
    sys.modules.update(saved)


def test_etude_finds_each_current_experiment_and_nothing_else() -> None:
    found = {path.stem for path in experiment_files(EXPERIMENTS)}
    assert found == {
        "elemental_closure_calibration",
        "llm_game",
        "policy_inference_gpu",
        "qwen_mtg183",
        "qwen_mtg183_vllm",
        "training_throughput",
        "training_throughput_packing",
        "training_throughput_sizes",
        "batch_closure",
        "bf16_behavior_logs",
        "elemental",
        "elemental_closure",
        "elemental_lambda",
        "lambda_sweep",
        "mlx_pilot",
        "mps_retest",
        "mtg183-elemental-eval",
        "one_token_layout",
        "opponent_memory",
        "precision_ladder",
        "seat_comparison",
        "tla_frontier",
        "tla_frontier_calibration",
        "tla_frontier_vocabulary",
        "two_seat",
        "value_readout_252",
    }


def test_a_tracked_self_play_run_draws_every_named_metric(
    tmp_path: Path, keep_imported_modules: None
) -> None:
    stage = TrainSelfPlay(
        id="learn",
        operation="train_self_play",
        updates=2,
        streams=2,
        transitions=1,
        learning_seats="both",
        games_per_stream=1,
        learning=Learning(epochs=1, minibatches=1, retained_fraction=1, ema=0.9),
    )
    regime = TrainingRegime(
        id="experiment-file-smoke",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=_agent().config,
        stages=[stage],
        wall_seconds=120,
        schedule_clock="iteration_fraction",
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(
            regime,
            17,
            tmp_path / "run",
            store,
            experiment="seat_comparison",
            backend=resolve("torch"),
        )
    record = run.stages[0]

    page = build(tmp_path / "run", EXPERIMENTS / "seat_comparison.py")
    panels = (*page.report.perf, *page.report.ml)
    named = {
        key
        for panel in panels
        if isinstance(panel, Series | Counters)
        for key in panel.keys
    }
    assert named - set(page.training.metrics) == set()
    # Each row holds its update's seconds, so the flame graph's phases are the
    # record's totals, with learning's subphases below them.
    (timers,) = page.training.timers
    phases = {child.name: child for child in timers.tree.children}
    assert phases["learning"].seconds == pytest.approx(record.learning_seconds)
    assert phases["collection"].seconds == pytest.approx(record.collection_seconds)
    assert "minibatches" in {child.name for child in phases["learning"].children}


def test_launched_runs_track_under_the_file_stem(tmp_path: Path) -> None:
    runs = read_training_jobs(
        EXPERIMENTS / "seat_comparison.py", ["--revision", "HEAD"]
    )
    jobs = [TrainingJob.model_validate_json(run.job.config or "") for run in runs]
    assert {job.experiment for job in jobs} == {"seat_comparison"}
    assert sorted({run.job.name for run in runs}) == [
        "seat-comparison-both-v1",
        "seat-comparison-one-v1",
    ]
    assert sorted(run.seed for run in runs) == sorted([64001, 64002, 64003, 64004] * 2)


def _finished_run(tmp_path: Path, updates: range) -> TrainingRun:
    saved = []
    for ordinal, update in enumerate(updates, start=1):
        path = tmp_path / f"updates-{update}.pt"
        path.write_bytes(str(update).encode())
        saved.append(
            Checkpoint(
                run="seed-1",
                line="seed-1",
                experiment="elemental",
                ordinal=ordinal,
                artifact=ArtifactRef(uri=path.as_uri(), sha256="0" * 64, size_bytes=1),
                at={"updates": float(update)},
            )
        )
    return TrainingRun(
        name="seed-1",
        experiment="elemental",
        job="seed-1",
        launch="seed-1",
        line="seed-1",
        state="finished",
        checkpoints=Checkpoints(saved),
    )


def _design(matches: list[Match]) -> set[tuple[str, ...]]:
    """Each match as its seated players' names and conditions, wherever it came from."""
    return {
        (
            *(player.name for player in match.players),
            str(match.conditions),
            str(match.games),
        )
        for match in matches
    }


def test_elemental_eval_asks_for_the_followers_design(tmp_path: Path) -> None:
    run = _finished_run(tmp_path, range(0, 501, 50))
    jobs = elemental.eval_jobs(ExperimentState(experiment="elemental", runs=(run,)))
    specs = {evaluation_spec(job).tournament.name: evaluation_spec(job) for job in jobs}
    assert set(specs) == {elemental.GRID, elemental.SUBSET, elemental.PAIRS}

    asked: list[Match] = []
    for spec in specs.values():
        for matchup in spec.tournament.seatings:
            if matchup.fixed:
                asked += matchup.matches()
            else:
                for checkpoint in spec.checkpoints:
                    asked += matchup.matches(checkpoint.ref)

    followed = EvalState(
        checkpoints={
            int(each.at["updates"]): FollowedCheckpoint(
                updates=int(each.at["updates"]),
                run=each.run,
                artifact=each.artifact,
                coordinates={},
            )
            for each in run.checkpoints
        },
        endpoint=500,
        segment_outcome="succeeded",
    )
    # Halving pairs extend MTG-183's follower only, not the shared design.
    expected = [
        match
        for unit in units(followed)
        if unit.role != "halving"
        for match in unit.matches
    ]

    def renamed(names: tuple[str, ...]) -> tuple[str, ...]:
        # The follower names a checkpoint `uNNNNNN`; Etude by its file.
        return tuple(
            f"u{int(name.removeprefix('updates-').removesuffix('.pt')):06d}"
            if name.startswith("updates-")
            else name
            for name in names
        )

    assert {renamed(each) for each in _design(asked)} == _design(expected)


def test_frontier_keeps_its_recorded_lost_endpoint_without_a_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(name: str) -> None:
        raise EtudeError(f"no run '{name}'")

    monkeypatch.setattr(tla_frontier.client, "status", unavailable)
    stored = _finished_run(tmp_path, range(1, 3))
    lost = replace(stored, name="5b7d8fa7b38c", state="running")
    state = ExperimentState(experiment="tla_frontier", runs=(lost, stored))
    closed = tla_frontier.with_launcher_ends(state)
    assert closed.runs[0].state == "failed"
    assert closed.runs[0].checkpoints == lost.checkpoints
    assert closed.runs[1] == stored
    assert lost.state == "running"

    unknown = replace(lost, name="unknown-run")
    with pytest.raises(EtudeError, match="unknown-run"):
        tla_frontier.with_launcher_ends(replace(state, runs=(unknown,)))


def test_opponent_memory_discovery_declares_trunks_and_empty_evaluation() -> None:

    experiment = read_experiment(EXPERIMENTS / "opponent_memory.py")
    assert experiment.training is not None and experiment.eval is not None
    runs = experiment.training(revision="HEAD")
    assert [run.seed for run in runs] == [39101, 39102, 39103]
    for run in runs:
        assert run.job.name == "decision-memory-trunk"
        spec = TrainingJob.model_validate_json(run.job.config or "")
        stage = spec.regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.target_learner_transitions == 3_300_000
        assert stage.learning.decision_schedule.transitions == 3_300_000
        assert stage.execution.collection_rows == 128
        assert spec.regime.agent.decision_memory == "none"
        assert spec.regime.observation.max_decision_records == 512
    assert experiment.eval(ExperimentState(experiment="opponent_memory")) == []


def test_opponent_memory_evaluation_draws_every_declared_seating(
    tmp_path: Path,
) -> None:

    runs = tuple(
        replace(
            _finished_run(tmp_path, range(0, final + 1, 20)),
            name=f"{arm}-{seed}",
            experiment="opponent_memory",
            seed=seed,
            job=f"decision-memory-{arm}",
            launch=f"decision-memory-{arm}",
        )
        for seed in opponent_memory.SEEDS
        for arm, final in (("control", 80), ("gru-k1", 100))
    )
    jobs = opponent_memory.eval_jobs(
        ExperimentState(experiment="opponent_memory", runs=runs)
    )
    assert len(jobs) == 1
    seatings = jobs[0].tournament.seatings
    assert len(seatings) == 30
    curves = [
        panel for panel in opponent_memory.report().evals if isinstance(panel, Curve)
    ]
    drawn = {
        selection.seating
        for panel in curves
        for selection in (
            panel.matches if isinstance(panel.matches, tuple) else (panel.matches,)
        )
        if isinstance(selection, Matches)
    }
    assert {seating.name for seating in seatings} == drawn
    games = sum(len(seating.configs) * seating.games for seating in seatings)
    assert games == 9600
    for seating in seatings:
        if "control-final" in seating.name:
            assert [player.checkpoint.at["updates"] for player in seating.players] == [
                100,
                80,
            ]
        if "reliance" in seating.tags:
            first, blanked = seating.players
            assert first.checkpoint == blanked.checkpoint
            assert first.key != blanked.key
            assert blanked.inference.config == {"blank_decision_memory": True}
    # Stored milestones remain playable while final checkpoints are unavailable.
    running = tuple(replace(run, state="running") for run in runs)
    interiors = opponent_memory.evaluation_matchups(
        ExperimentState(experiment="opponent_memory", runs=running)
    )
    assert len(interiors) == 18
    assert all("final" not in seating.name for seating in interiors)
