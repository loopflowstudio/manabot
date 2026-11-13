"""Manabot's experiment files are what `etude launch`, `etude eval` and `etude report` read.

A run tracked under a file's stem draws on that file's page with every metric
its panels name and a flame graph of its phases; `eval(state)` asks for the
same games the MTG-183 follower plays.
"""

from collections.abc import Iterator
from pathlib import Path
import sys

from etude.eval.records import Match
from etude.experiment import Checkpoint, Checkpoints, ExperimentState, TrainingRun
from etude.jobs import evaluation_spec
from etude.page import experiment_files
from etude.run.records import ArtifactRef
import pytest

from experiments import elemental
from experiments.elemental_long_eval import (
    Checkpoint as FollowedCheckpoint,
    EvalState,
    units,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"


@pytest.fixture
def keep_imported_modules() -> Iterator[None]:
    """Restore modules `build` drops so later tests keep their imported objects.

    `etude.page.build` re-imports the page file's neighbours by popping every
    `experiments.*` module; without this, a later test's module-level import and
    a fresh import (pickling, monkeypatching) disagree about which object is real.
    """
    saved = dict(sys.modules)
    yield
    sys.modules.update(saved)


def test_etude_finds_each_current_experiment_and_nothing_else() -> None:
    found = {path.stem for path in experiment_files(EXPERIMENTS)}
    assert found == {
        "batch_closure",
        "elemental",
        "elemental_closure",
        "elemental_lambda",
        "lambda_sweep",
        "mlx_pilot",
        "mps_retest",
        "mtg183-elemental-eval",
        "precision_ladder",
        "seat_comparison",
        "tla_frontier",
        "tla_frontier_calibration",
        "tla_frontier_vocabulary",
        "two_seat",
        "value_readout_252",
    }


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
