"""Manabot's experiment files are what `etude launch`, `etude eval` and `etude report` read.

A run tracked under a file's stem draws on that file's page with every metric
its panels name and a flame graph of its phases; `eval(state)` asks for the
same games the MTG-183 follower plays.
"""

from pathlib import Path

from etude.eval.records import Match
from etude.experiment import Checkpoint, Checkpoints, ExperimentState, TrainingRun
from etude.jobs import evaluation_spec
from etude.launch.job_file import read_job_file
from etude.page import build, experiment_files
from etude.report import Counters, Series
from etude.run.records import ArtifactRef
import pytest

from experiments import elemental
from experiments.elemental_long_eval import (
    Checkpoint as FollowedCheckpoint,
    EvalState,
    units,
)
from manabot.experiment.job import TrainingJob
from manabot.infra.config import MatchConfig
from manabot.training.execution import execute_regime
from manabot.training.regime import Learning, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_two_seat_learning import _agent

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"


def test_etude_finds_each_current_experiment_and_nothing_else() -> None:
    found = {path.stem for path in experiment_files(EXPERIMENTS)}
    assert found == {"curriculum_adapters", "elemental", "seat_comparison", "two_seat"}


def test_a_tracked_self_play_run_draws_every_named_metric(tmp_path: Path) -> None:
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
            regime, 17, tmp_path / "run", store, experiment="seat_comparison"
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
    runs = read_job_file(EXPERIMENTS / "seat_comparison.py", ["--revision", "HEAD"])
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
                experiment="elemental",
                ordinal=ordinal,
                artifact=ArtifactRef(uri=path.as_uri(), sha256="0" * 64, size_bytes=1),
                clocks={"updates": float(update)},
            )
        )
    return TrainingRun(
        name="seed-1",
        experiment="elemental",
        state="finished",
        checkpoints=Checkpoints(saved),
    )


def _design(matches: list[Match]) -> set[tuple[str, ...]]:
    """Each match as its seated agents' names and conditions, wherever it came from."""
    return {
        (
            *(agent.name for agent in match.agents),
            str(match.conditions),
            str(match.games),
        )
        for match in matches
    }


def test_elemental_eval_asks_for_the_followers_design(tmp_path: Path) -> None:
    run = _finished_run(tmp_path, range(0, 501, 50))
    jobs = elemental.eval(ExperimentState(experiment="elemental", runs=(run,)))
    specs = {evaluation_spec(job).tournament.name: evaluation_spec(job) for job in jobs}
    assert set(specs) == {elemental.GRID, elemental.SUBSET, elemental.PAIRS}

    asked: list[Match] = []
    for spec in specs.values():
        for matchup in spec.tournament.matchups:
            if matchup.fixed:
                asked += matchup.matches()
            else:
                for checkpoint in spec.checkpoints:
                    asked += matchup.matches(checkpoint.ref)

    followed = EvalState(
        checkpoints={
            int(each.clocks["updates"]): FollowedCheckpoint(
                updates=int(each.clocks["updates"]),
                run=each.run,
                artifact=each.artifact,
                coordinates={},
            )
            for each in run.checkpoints
        },
        endpoint=500,
        segment_outcome="succeeded",
    )
    expected = [match for unit in units(followed) for match in unit.matches]

    def renamed(names: tuple[str, ...]) -> tuple[str, ...]:
        # The follower names a checkpoint `uNNNNNN`; Etude by its file.
        return tuple(
            f"u{int(name.removeprefix('updates-').removesuffix('.pt')):06d}"
            if name.startswith("updates-")
            else name
            for name in names
        )

    assert {renamed(each) for each in _design(asked)} == _design(expected)
