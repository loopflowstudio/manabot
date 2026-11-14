"""Manabot's experiment files are what `etude launch`, `etude train` and `etude report` read.

Each kept file reads through Etude's declaration reader into one addressed run
per job and seed, and its declarations return typed jobs that lower to plain
ones: `training_jobs` only `TrainingJob`s, `eval_jobs(state)` only `EvalJob`s.
"""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import runpy
from types import ModuleType

from etude.checkpoints import CheckpointRecord
from etude.declarations import asked, read_experiment, read_job_file
from etude.errors import EtudeError
from etude.eval.records import Match
from etude.jobs import (
    Checkpoints,
    EvalJob,
    ExperimentState,
    TrainingJob,
    TrainingRunSummary,
    evaluation_spec,
)
from etude.page import experiment_files
from etude.records import ArtifactRef
import pytest

from experiments import (
    batched_baseline,
    belief,
    elemental,
    seat_comparison,
    tla_frontier,
)
from experiments.elemental_grid import DECKS
from experiments.elemental_long import heuristic_games, pair_design, rated_updates

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
FRONTIER_JOBS = {"tla-frontier-c5fba4-w64": 1, "tla-frontier-c5fba4-w160l8": 1}
# Each file, the flags its declaration needs, and its runs per job.
DECLARED: list[tuple[ModuleType, tuple[str, ...], dict[str, int]]] = [
    (batched_baseline, (), {"batched_baseline": 1}),
    (
        elemental,
        ("--revision", "HEAD", "--updates", "2", "--hours", "1", "--usd-per-run", "1"),
        {"elemental": 1},
    ),
    (
        seat_comparison,
        ("--revision", "HEAD"),
        {"seat-comparison-one": 4, "seat-comparison-both": 4},
    ),
    (tla_frontier, ("--revision", "HEAD"), FRONTIER_JOBS),
]


def _name(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


def test_etude_finds_each_current_experiment_and_nothing_else() -> None:
    found = {path.stem for path in experiment_files(EXPERIMENTS)}
    assert found == {
        "batched_baseline",
        "belief",
        "elemental",
        "seat_comparison",
        "tla_frontier",
    }


def test_no_experiment_file_keeps_a_retired_declaration() -> None:
    """Etude refuses `runs()` and `eval()`; no file may define either."""
    for path in experiment_files(EXPERIMENTS):
        assert not {"runs", "training_runs", "eval"} & set(runpy.run_path(str(path)))


@pytest.mark.parametrize(
    ("module", "given", "jobs"), DECLARED, ids=[_name(each[0]) for each in DECLARED]
)
def test_training_jobs_read_as_addressed_runs(
    module: ModuleType, given: tuple[str, ...], jobs: dict[str, int]
) -> None:
    name = _name(module)
    requests = read_job_file(EXPERIMENTS / f"{name}.py", given)
    assert Counter(request.job.name for request in requests) == jobs
    assert {request.job.experiment for request in requests} == {name}
    assert {request.id for request in requests} == {
        f"{name}/{request.job.name}@{request.seed}" for request in requests
    }
    assert all("etude.call" in request.job.command for request in requests)
    assert all(request.job.smoke is not None for request in requests)


def test_training_declarations_return_typed_jobs_that_lower() -> None:
    declared = [
        *batched_baseline.training_jobs(),
        *elemental.training_jobs("HEAD", 2, 1.0, 1.0),
        *seat_comparison.training_jobs("HEAD"),
        *tla_frontier.training_jobs("HEAD"),
        *belief.training_jobs(ExperimentState(experiment="belief")),
    ]
    assert len(declared) == 7
    for job in declared:
        assert isinstance(job, TrainingJob)
        plain = job.plain()
        assert plain.name == job.name and plain.config is not None


def _checkpoint(
    tmp_path: Path, run: str, ordinal: int, update: int
) -> CheckpointRecord:
    path = tmp_path / run / f"updates-{update}.pt"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(str(update).encode())
    return CheckpointRecord(
        run=run,
        line=run,
        experiment="fixture",
        ordinal=ordinal,
        artifact=ArtifactRef(uri=path.as_uri(), sha256="0" * 64, size_bytes=1),
        at={"updates": float(update)},
        # An hour of wall time per update, from update zero.
        wall_ms=update * 3_600_000,
    )


def _finished_run(
    tmp_path: Path, updates: range, *, name: str = "seed-1", job: str = "seed-1"
) -> TrainingRunSummary:
    return TrainingRunSummary(
        name=name,
        experiment="fixture",
        job=job,
        launch=job,
        line=name,
        state="finished",
        checkpoints=Checkpoints(
            _checkpoint(tmp_path, name, ordinal, update)
            for ordinal, update in enumerate(updates, start=1)
        ),
        seed=0,
    )


# ---------------------------------------------------------------------------
# Belief: a later stage declared from stored state


def test_belief_declares_the_sampler_once_the_policy_has_finished(
    tmp_path: Path,
) -> None:
    file = EXPERIMENTS / "belief.py"
    empty = ExperimentState(experiment="belief")
    first = read_job_file(file, local=True, state=empty)
    assert [request.id for request in first] == ["belief/policy@0"]

    policy = _finished_run(tmp_path, range(0, 2), name="policy@0", job=belief.POLICY)
    running = replace(policy, state="running")
    waiting = read_job_file(file, local=True, state=replace(empty, runs=(running,)))
    assert [request.id for request in waiting] == ["belief/policy@0"]

    both = read_job_file(file, local=True, state=replace(empty, runs=(policy,)))
    assert [request.id for request in both] == ["belief/policy@0", "belief/sampler@0"]
    (start,) = both[1].job.inputs
    final = policy.checkpoints[-1]
    assert (start.name, start.sha256) == ("start", final.artifact.sha256)
    assert ("ETUDE_START", final.key) in {
        (variable.name, variable.value) for variable in both[1].job.environment
    }


# ---------------------------------------------------------------------------
# Elemental: the rated grid and pairing graph


def test_elemental_eval_jobs_ask_for_the_rated_design(tmp_path: Path) -> None:
    run = _finished_run(tmp_path, range(0, 501, 50))
    state = ExperimentState(experiment="elemental", runs=(run,))
    declared = elemental.eval_jobs(state)
    assert all(isinstance(job, EvalJob) for job in declared)
    specs = {job.name: evaluation_spec(job) for job in declared}
    assert set(specs) == {elemental.GRID, elemental.SUBSET, elemental.PAIRS}
    file = EXPERIMENTS / "elemental.py"
    assert [job.name for job in asked(read_experiment(file).eval, state)] == list(specs)

    rated = rated_updates(500, 500)
    kinds = {update: heuristic_games(update, 500)[0] for update in rated}
    held = {
        name: sorted(int(each.at["updates"]) for each in spec.checkpoints)
        for name, spec in specs.items()
    }
    assert held == {
        elemental.GRID: [u for u in rated if kinds[u] == "full"],
        elemental.SUBSET: [u for u in rated if kinds[u] == "subset"],
        elemental.PAIRS: [],
    }
    # A grid is one match per model deck for each held checkpoint.
    (grid,) = specs[elemental.GRID].tournament.seatings
    matches = grid.matches(run.checkpoints[0].ref)
    assert [match.config["deck"] for match in matches] == list(DECKS)
    assert {match.players[1].name for match in matches} == {"heuristic"}
    # Each pair of the design is one fixed seating: newer checkpoint first.
    pairs: list[Match] = []
    for seating in specs[elemental.PAIRS].tournament.seatings:
        assert seating.fixed
        pairs += seating.matches()
    design = pair_design(rated, 500)
    assert len(pairs) == len(design) * len(DECKS)
    assert {(match.players[0].name, match.players[1].name) for match in pairs} == {
        (f"updates-{pair.newer}.pt", f"updates-{pair.older}.pt") for pair in design
    }


# ---------------------------------------------------------------------------
# Frontier: one evaluation pass, and the recorded lost endpoint


def test_frontier_eval_jobs_declare_one_typed_pass(tmp_path: Path) -> None:
    small, large = (
        _finished_run(tmp_path, range(0, 51, 25), name=f"{job}@21501", job=job)
        for job in FRONTIER_JOBS
    )
    state = ExperimentState(experiment="tla_frontier", runs=(small, large))
    (job,) = tla_frontier.eval_jobs(state)
    assert isinstance(job, EvalJob)
    plain = job.plain()
    assert plain.name == "tla-frontier-c5fba4-paired-v1"
    assert ("UV_NO_SYNC", "1") in {(v.name, v.value) for v in plain.environment}
    assert plain.setup is not None and "--extra evaluation" in plain.setup
    seatings = evaluation_spec(job).tournament.seatings
    assert {tag for seating in seatings for tag in seating.tags} >= {
        "heuristic-w64",
        "heuristic-w160l8",
    }


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
