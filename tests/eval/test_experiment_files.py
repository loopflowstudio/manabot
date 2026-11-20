"""Manabot's experiment files are what `etude launch`, `etude train` and `etude report` read.

Each kept file reads through Etude's declaration reader into one addressed run
per job and seed, and its declarations return typed jobs that lower to plain
ones: `training_jobs` only `TrainingJob`s, `eval_jobs(state)` only `EvalJob`s.
`NOT_YET_RUNNABLE` lists the files checked in ahead of the Etude support they
need; move one out of it, into the checks below, when it runs.
"""

from collections import Counter
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import ModuleType

from etude.checkpoints import CheckpointRecord
from etude.controller.declarations import asked, read_experiment, read_training_jobs
from etude.errors import EtudeError
from etude.eval.records import Match, Tournament
from etude.experiment_file import defined_functions
from etude.jobs import (
    Checkpoints,
    EvalJob,
    ExperimentState,
    TrainingJob,
    TrainingRunSummary,
    checkpoint_inputs,
)
from etude.page import experiment_files
from etude.records import ArtifactRef
from etude.report import Curve, Matches, Ranking, Series
import pytest

from experiments import (
    batched_baseline,
    belief,
    elemental,
    llm_claude,
    mana_taps,
    opponent_memory,
    seat_comparison,
    tla_frontier,
    tla_trophies,
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
    (tla_trophies, (), {"trophy-w64-raw": 3}),
    (mana_taps, (), {"mana-taps-w64-off": 3, "mana-taps-w64-usable": 3}),
]


# Eval-only files Etude cannot run yet (EXP-70): a served or hosted player and
# what its job needs (a secret, an API dollar limit, a server for the job's
# lifetime, a per-game time limit). They are found as experiments and declare
# `eval_jobs`; only Claude's declaration is read below, since Qwen's imports
# need the `llm` extra.
NOT_YET_RUNNABLE = {"llm_claude", "llm_qwen"}


def _name(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


def test_etude_finds_each_current_experiment_and_nothing_else() -> None:
    found = {path.stem for path in experiment_files(EXPERIMENTS)}
    assert found - NOT_YET_RUNNABLE == {
        "batched_baseline",
        "belief",
        "elemental",
        "etude_execution",
        "mana_taps",
        "opponent_memory",
        "seat_comparison",
        "tla_frontier",
        "tla_trophies",
    }
    assert NOT_YET_RUNNABLE <= found


@pytest.mark.parametrize(
    ("experiment", "matches"),
    [
        (elemental, Matches(tag="heuristic")),
        (seat_comparison, None),
        (tla_trophies, Matches(tournament="tla-trophies-held-out-32")),
    ],
)
def test_checkpoint_report_shows_behaviour_next_to_strength(
    experiment: ModuleType, matches: Matches | None
) -> None:
    strength, *rest = experiment.report().evals
    assert strength.title == "Score against the heuristic" and strength.score is None
    curves = rest[:11]
    assert all(
        isinstance(c, Curve) and c.matches == matches and c.score.table == "behaviour"
        for c in curves
    )
    assert {c.score.value: c.score.per for c in curves} == {
        "lands_played": "land_turns",
        "casts_made": "cast_turns",
        "mana_use_early": "mana_turns_early",
        "mana_use_mid": "mana_turns_mid",
        "mana_use_late": "mana_turns_late",
        "attackers_declared": "attackers_offered",
        "blockers_declared": "blockers_offered",
        "attackers_lost_for_nothing": "attackers_declared",
        "blockers_lost_for_nothing": "blockers_declared",
        "casts_off_turn": "casts",
        "turns": "games",
    }


def test_self_play_experiments_and_recipes_report_the_combat_probe() -> None:
    files = [
        path
        for path in experiment_files(EXPERIMENTS)
        if "training_jobs" in defined_functions(path)
    ]
    files.extend(
        ROOT / "manabot/recipes" / f"{name}.py"
        for name in ("baseline", "belief_policy", "elemental", "local")
    )
    for path in files:
        assert "report" in defined_functions(path), path
        module = import_module(".".join(path.relative_to(ROOT).with_suffix("").parts))
        report = module.report()
        ml = {
            panel.title: panel.keys for panel in report.ml if isinstance(panel, Series)
        }
        assert ml["Attack and block log-odds"] == (
            "combat/attack/gap",
            "combat/block/gap",
        )
        for kind in ("attack", "block"):
            assert set(ml[f"{kind.title()} gap shift under board edits"]) == {
                f"combat/{kind}/shift/{edit}"
                for edit in (
                    "my_creatures_0_1",
                    "my_creatures_9_9",
                    "their_creatures_0_1_tapped",
                    "their_creatures_9_9_untapped",
                )
            }
        (update_time,) = (
            p for p in report.perf if isinstance(p, Series) and p.title == "Update time"
        )
        assert "time/combat_probe" in update_time.keys
    assert belief.report().other_metrics


def test_no_experiment_file_keeps_a_retired_declaration() -> None:
    """Etude refuses `runs()` and `eval()`; no file may define either."""
    for path in experiment_files(EXPERIMENTS):
        assert not {"runs", "training_runs", "eval"} & defined_functions(path)


def test_not_yet_runnable_files_are_eval_only() -> None:
    for name in sorted(NOT_YET_RUNNABLE):
        defined = defined_functions(EXPERIMENTS / f"{name}.py")
        assert {"eval_jobs", "play", "report"} <= defined
        assert "training_jobs" not in defined


def test_claude_seats_the_outside_checkpoint_with_its_extras() -> None:
    (job,) = llm_claude.eval_jobs(ExperimentState(experiment="llm_claude"))
    assert "--extra evaluation --extra openrouter" in job.plain().command
    matches = job.recorded().tournament.seatings[0].matches()
    (seated,) = checkpoint_inputs(matches, ExperimentState(experiment="llm_claude"))
    assert seated == llm_claude.MTG183_FINAL


@pytest.mark.parametrize(
    ("module", "given", "jobs"), DECLARED, ids=[_name(each[0]) for each in DECLARED]
)
def test_training_jobs_read_as_addressed_runs(
    module: ModuleType, given: tuple[str, ...], jobs: dict[str, int]
) -> None:
    name = _name(module)
    requests = read_training_jobs(EXPERIMENTS / f"{name}.py", given)
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


def test_trophy_calibration_job_and_held_out_deals(tmp_path: Path) -> None:
    (train,) = tla_trophies.training_jobs(updates=6, threads=3, streams=64, rows=64)
    assert train.config == tla_trophies.Config(
        updates=6, threads=3, streams=64, rows=64, checkpoint_every=100
    )
    assert {v.name: v.value for v in train.plain().environment} == {
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
    }
    pilot = _finished_run(tmp_path, range(2), job="trophy-w64")
    raw = _finished_run(tmp_path, range(0, 201, 100), name="raw", job=train.name)
    second = replace(
        _finished_run(tmp_path, range(0, 201, 100), name="raw-2", job=train.name),
        seed=1,
    )
    state = ExperimentState(experiment="fixture", runs=(pilot, raw, second))
    (evaluation,) = tla_trophies.eval_jobs(state)
    assert list(evaluation.checkpoints) == [*raw.checkpoints, *second.checkpoints]
    tournament, gaps = Tournament(
        definition=evaluation.tournament,
        checkpoints=evaluation.recorded().checkpoints,
    ).resolve(state.lines)
    assert not gaps
    assert Counter(m.checkpoint for m in tournament.schedule()) == {
        checkpoint.key: 32 for checkpoint in evaluation.checkpoints
    }
    assert tla_trophies.report().evals[0].matches == Matches(tournament=evaluation.name)
    assert [
        tla_trophies.Deal.model_validate(s.config)
        for s in evaluation.tournament.seatings
    ] == [
        tla_trophies.Deal(first=2 * i, second=2 * i + 1, seed=40900 + i)
        for i in range(32)
    ]


# ---------------------------------------------------------------------------
# Belief: a later stage declared from stored state


def test_belief_declares_the_sampler_once_the_policy_has_finished(
    tmp_path: Path,
) -> None:
    file = EXPERIMENTS / "belief.py"
    empty = ExperimentState(experiment="belief")
    first = read_training_jobs(file, local=True, state=empty)
    assert [request.id for request in first] == ["belief/policy@0"]

    policy = _finished_run(tmp_path, range(0, 2), name="policy@0", job=belief.POLICY)
    running = replace(policy, state="running")
    waiting = read_training_jobs(
        file, local=True, state=replace(empty, runs=(running,))
    )
    assert [request.id for request in waiting] == ["belief/policy@0"]

    both = read_training_jobs(file, local=True, state=replace(empty, runs=(policy,)))
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
    specs = {job.name: job.recorded() for job in declared}
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
    # A new match owns one deck/seat group, never a whole grid row.
    grid = specs[elemental.GRID].tournament.seatings
    matches = [m for seating in grid for m in seating.matches(run.checkpoints[0].ref)]
    assert {m.config["deck"] for m in matches} == set(DECKS)
    assert all(len(m.config["groups"]) == 1 and m.games == 1 for m in matches)
    assert len(matches) == len(DECKS) ** 2 * 2
    assert {m.players[1].name for m in matches} == {"heuristic"}
    assert all(job.agents == elemental.AGENTS for job in declared)
    # Each pair of the design is one fixed seating: newer checkpoint first.
    pairs: list[Match] = []
    for seating in specs[elemental.PAIRS].tournament.seatings:
        assert seating.fixed
        pairs += seating.matches()
    design = pair_design(rated, 500)
    assert len(pairs) == len(design) * len(DECKS) ** 2 * 2
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
    assert plain.name == "tla-frontier-c5fba4-paired-v1-agents-v1"
    assert "UV_NO_SYNC" not in {v.name for v in plain.environment}
    assert "--extra evaluation" in plain.command
    assert plain.setup is not None and "--extra evaluation" in plain.setup
    seatings = job.recorded().tournament.seatings
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

    # A run this pass only queues is not asked about: the launcher has no record yet.
    queued = replace(stored, name="unknown-run", state="queued")
    assert tla_frontier.with_launcher_ends(replace(state, runs=(queued,))).runs == (
        queued,
    )

    unknown = replace(lost, name="unknown-run")
    with pytest.raises(EtudeError, match="unknown-run"):
        tla_frontier.with_launcher_ends(replace(state, runs=(unknown,)))


# ---------------------------------------------------------------------------
# Opponent memory: trunks, then both arms of each finished trunk


def test_opponent_memory_declares_trunks_then_both_arms_of_a_finished_trunk(
    tmp_path: Path,
) -> None:
    file = EXPERIMENTS / "opponent_memory.py"
    empty = ExperimentState(experiment="opponent_memory")
    for wide in (False, True):
        given = ("--revision", "HEAD", *(("--size", "wide") if wide else ()))
        trunks = read_training_jobs(file, given, state=empty)
        assert [(request.job.name, request.seed) for request in trunks] == [
            (f"decision-memory-{name}", seed)
            for seed, name in (
                dict.fromkeys(opponent_memory.SEEDS, "wide-trunk")
                if wide
                else opponent_memory.TRUNK_NAMES
            ).items()
        ]
        assert all(request.job.max_attempts == 3 for request in trunks)
    recipe = opponent_memory.settings(opponent_memory.PilotConfig(size="wide"))
    assert (recipe.model.hidden_dim, recipe.model.attention_layers) == (128, 3)
    assert recipe.model.decision_memory == "none"
    assert recipe.observation.max_decision_records == 1024
    assert recipe.learner_transitions == 3_300_000
    assert recipe.learning.decision_schedule.transitions == 3_300_000
    assert (recipe.collection_rows, recipe.learner_rows) == (128, 128)
    for request in trunks:
        # The declared CPU smoke must be a valid recipe of its own.
        smoke = opponent_memory.PilotConfig.model_validate_json(
            request.job.smoke.config
        )
        assert opponent_memory.settings(smoke).learner_transitions == 64
    arm = opponent_memory.settings(opponent_memory.PilotConfig(stage="gru-k1"))
    assert arm.model.decision_memory == "gru"
    assert arm.learner_transitions == 8_800_000

    seed = opponent_memory.SEEDS[0]
    trunk = replace(
        _finished_run(
            tmp_path, range(0, 81, 40), name="trunk", job="decision-memory-trunk-2"
        ),
        seed=seed,
    )
    both = read_training_jobs(
        file, ("--revision", "HEAD"), state=replace(empty, runs=(trunk,))
    )
    arms = [request for request in both if request.job.inputs]
    assert [(request.job.name, request.seed) for request in arms] == [
        (f"decision-memory-control-{seed}", seed),
        (f"decision-memory-gru-k1-{seed}", seed),
    ]
    final = trunk.checkpoints[-1]
    assert {request.job.inputs[0].sha256 for request in arms} == {final.artifact.sha256}
    assert opponent_memory.eval_jobs(empty) == []


def test_opponent_memory_evaluation_draws_every_declared_seating(
    tmp_path: Path,
) -> None:

    runs = tuple(
        replace(
            _finished_run(
                tmp_path, range(0, final + 1, 40), name=f"{prefix}{arm}-{seed}"
            ),
            experiment="opponent_memory",
            seed=seed,
            job=f"decision-memory-{prefix}{arm}",
            launch=f"decision-memory-{prefix}{arm}",
        )
        for seed in opponent_memory.SEEDS
        for prefix in ("", "wide-")
        for arm, final in (("control", 160), ("gru-k1", 200))
    )
    jobs = opponent_memory.eval_jobs(
        ExperimentState(experiment="opponent_memory", runs=runs)
    )
    assert len(jobs) == 1
    seatings = jobs[0].tournament.seatings
    assert len(seatings) == 168
    assert jobs[0].max_hours == 3 and jobs[0].max_usd == 4
    curves = [
        panel for panel in opponent_memory.report().evals if isinstance(panel, Curve)
    ]
    # Each seed must remain visible; Etude unions a panel's selections by job.
    for size in ("small", "wide"):
        for seed in opponent_memory.SEEDS:
            curve = next(c for c in curves if size in c.title and str(seed) in c.title)
            assert isinstance(curve.matches, tuple)
            assert {selection.seating for selection in curve.matches} == {
                opponent_memory.seating_name(size, seed, disclosure, "control", point)
                for disclosure in opponent_memory.DISCLOSURES
                for point in (40, 80, 120, "final")
            }
        rankings = {
            panel.matches
            for panel in opponent_memory.report().evals
            if isinstance(panel, Ranking) and size in panel.title
        }
        assert rankings == {
            Matches(tag=f"{size}-control"),
            Matches(tag=f"{size}-heuristic"),
        }
    drawn = {
        selection.seating
        for panel in curves
        for selection in (
            panel.matches if isinstance(panel.matches, tuple) else (panel.matches,)
        )
        if isinstance(selection, Matches)
    }
    heuristic = [seating for seating in seatings if "heuristic" in seating.tags]
    head_to_head = [seating for seating in seatings if seating not in heuristic]
    assert {seating.name for seating in head_to_head} == drawn
    # Against the heuristic, per size, seed and disclosure: both arms' finals
    # (50 deals), and 25-deal curves over control's 0/40/80/120 and GRU's
    # 40/80/120.
    finals = [seating for seating in heuristic if seating.name.endswith("-final")]
    assert (len(finals), len(heuristic)) == (2 * 3 * 2 * 2, 2 * 3 * 2 * 9)
    for seating in heuristic:
        opponent = seating.players[1]
        assert opponent.kind == "code" and opponent.name == "heuristic"
        assert len(seating.configs) == (50 if seating in finals else 25)
    games = sum(len(seating.configs) * seating.games for seating in seatings)
    assert games == 19200 + len(finals) * 200 + (len(heuristic) - len(finals)) * 100
    for seating in head_to_head:
        assert seating.fixed and seating.seeds == "one"
        size, _, seed, *_ = seating.name.split("-")
        prefix = "wide-" if size == "wide" else ""
        first, second = seating.players
        assert first.checkpoint.line == f"{prefix}gru-k1-{seed}"
        opponent = "gru-k1" if "reliance" in seating.tags else "control"
        assert second.checkpoint.line == f"{prefix}{opponent}-{seed}"
        if "control-final" in seating.name:
            assert [player.checkpoint.at["updates"] for player in seating.players] == [
                200,
                160,
            ]
        if "reliance" in seating.tags:
            first, blanked = seating.players
            assert first.checkpoint == blanked.checkpoint
            assert first.key != blanked.key and not first.config
            assert blanked.config == {"blank_decision_memory": True}
    # Stored milestones remain playable while final checkpoints are unavailable.
    running = tuple(replace(run, state="running") for run in runs)
    interiors = opponent_memory.evaluation_matchups(
        ExperimentState(experiment="opponent_memory", runs=running)
    )
    # 36 head-to-heads plus the 84 heuristic-curve seatings.
    assert len(interiors) == 36 + 84
    assert all("final" not in seating.name for seating in interiors)
