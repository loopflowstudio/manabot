"""Elemental, ten decks: two-seat self-play rated on the 10×10 deck grid (MTG-181/183).

Jack Heart's design (MTG-181, 2026-10-23; MTG-183, 2026-10-23): train on the
ten-deck Elemental world from both seats, and score each checkpoint as the
model playing every deck against the heuristic playing every deck, on fixed
deals (`experiments.elemental_grid`). MTG-183 also asks whether newer
checkpoints keep beating older ones once the heuristic score stops rising, so
rated checkpoints play each other on a fixed pairing graph
(`experiments.elemental_long`). The protocols are
experiments/protocols/elemental-174.md and elemental-183.md.

This file is the experiment `elemental` for Etude:

- `runs()` trains fresh seeds of the frozen recipe (`etude launch`);
- `eval(state)` asks for the grid and pairs of each run's rated checkpoints,
  played by `experiments.elemental_long_eval.play` (`etude eval`);
- `report()` draws them beside training (`etude report`).

    uv run etude launch experiments/elemental.py --revision COMMIT --updates 3200 \\
        --hours 11.9 --usd-per-run 9 --rent l4
    uv run etude eval experiments/elemental.py --source r2://manabot/jobs
    uv run etude report experiments/elemental.py --source r2://manabot/jobs --open

MTG-183's own seed is not yet this experiment's: its windows w1/w2 predate
Etude tracking and its segments track under `self-play`, so its follower
(`python -m experiments.elemental_long_eval`) still evaluates it.
"""

from decimal import Decimal
from pathlib import Path

from etude.eval.records import Benchmark, Inference, Matchup, Player
from etude.experiment import Checkpoint, ExperimentState, TrainingRun
from etude.jobs import evaluation
from etude.launch.records import Job, Run
from etude.report import (
    Anchors,
    Curve,
    HeadToHead,
    Matches,
    Matrix,
    Ranking,
    Report,
    Score,
)

from experiments.elemental_grid import Game, groups, schedule
from experiments.elemental_long import (
    CADENCE,
    h2h_schedule,
    heuristic_games,
    pair_design,
    rated_updates,
    recipe,
)
from experiments.elemental_long_eval import (
    INFERENCE_CODE,
    Conditions,
    Kind,
    eval_environment,
    h2h_groups,
    heuristic_code,
    play,
    rows,
)
from manabot.experiment.job import TrainingJob
from manabot.training.panels import ML, PERF

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
# Benchmarks eval(state) holds; the report selects their matches by tag.
GRID = "heuristic-grid"
SUBSET = "heuristic-subset"
PAIRS = "checkpoint-pairs"
# Each grid row plays on one CPU worker per (opposing deck, seat) group.
WORKERS = 10
# Wall hours one evaluation pass may take on a launched machine.
EVAL_HOURS = 12.0

# The checkpoint in the model's seat: the inference code is its version.
INFERENCE = Inference(source=list(INFERENCE_CODE))


# ---------------------------------------------------------------------------
# Training


def runs(
    revision: str,
    updates: int,
    hours: float,
    usd_per_run: float,
    seed: int = 18101,
) -> list[Run]:
    """One seed of the frozen recipe to `updates`, exporting every `CADENCE` updates."""
    job = TrainingJob(
        regime=recipe(updates, hours=hours),
        experiment=EXPERIMENT,
        milestones=tuple(range(CADENCE, updates, CADENCE)),
        finalization_seconds=600,
        validate_numerics=True,
    ).job(
        repo=ROOT,
        revision=revision,
        max_hours=hours,
        max_usd=Decimal(str(usd_per_run)),
    )
    return job.runs(seed)


# ---------------------------------------------------------------------------
# Evaluation


def _updates(checkpoint: Checkpoint) -> int:
    return int(checkpoint.clocks["updates"])


def _rated(run: TrainingRun) -> tuple[dict[int, Checkpoint], int | None]:
    """The run's rated checkpoints by update, and its endpoint once it has finished."""
    by_update = {_updates(each): each for each in run.checkpoints}
    endpoint = None if run.final is None else _updates(run.final)
    wanted = rated_updates(max(by_update, default=0), endpoint)
    return {u: by_update[u] for u in wanted if u in by_update}, endpoint


def _matchup(
    name: str,
    tags: tuple[str, ...],
    players: tuple[Player | Inference, Player | Inference],
    grid: list[Conditions],
) -> Matchup:
    """One match per grid row; every row of a grid plays the same number of games."""
    (games,) = {sum(len(group.deals) for group in row.groups) for row in grid}
    return Matchup(
        name=name,
        tags=tags,
        players=players,
        conditions=tuple(row.model_dump(mode="json") for row in grid),
        games=games,
    )


def _grid(name: str, kind: Kind, games: list[Game]) -> Benchmark:
    return Benchmark(
        name=name,
        environment=eval_environment(),
        matchups=(
            _matchup(
                f"vs heuristic, {kind} grid",
                ("heuristic",),
                (INFERENCE, heuristic_code()),
                rows(kind, groups(games)),
            ),
        ),
    )


def _pairs(
    run: TrainingRun, rated: dict[int, Checkpoint], endpoint: int | None
) -> list[Matchup]:
    """The run's pairing graph: each pair plays the 10×10 cross-checkpoint schedule."""
    grid = rows("pair", h2h_groups(h2h_schedule()))
    return [
        _matchup(
            f"{run.name} u{pair.newer} vs u{pair.older}",
            (pair.role,),
            (rated[pair.newer].model(INFERENCE), rated[pair.older].model(INFERENCE)),
            grid,
        )
        for pair in pair_design(tuple(sorted(rated)), endpoint)
    ]


def eval(state: ExperimentState) -> list[Job]:
    """The heuristic grid of every rated checkpoint, and each run's pairing graph.

    A rated checkpoint on the full cadence (and the endpoint) plays all 5,000
    grid games, any other the declared 10-game-per-cell subset.
    """
    full: list[Checkpoint] = []
    subset: list[Checkpoint] = []
    pairs: list[Matchup] = []
    for run in state.runs:
        rated, endpoint = _rated(run)
        for updates, checkpoint in rated.items():
            kind, _ = heuristic_games(updates, endpoint)
            (full if kind == "full" else subset).append(checkpoint)
        pairs += _pairs(run, rated, endpoint)
    # The full grid is every scheduled game; the subset is any subset update's.
    jobs = [
        evaluation(
            _grid(GRID, "full", schedule()),
            play=play,
            checkpoints=full,
            workers=WORKERS,
            max_hours=EVAL_HOURS,
        ),
        evaluation(
            _grid(SUBSET, "subset", heuristic_games(CADENCE, None)[1]),
            play=play,
            checkpoints=subset,
            workers=WORKERS,
            max_hours=EVAL_HOURS,
        ),
    ]
    if pairs:
        ladder = Benchmark(
            name=PAIRS, environment=eval_environment(), matchups=tuple(pairs)
        )
        jobs.append(
            evaluation(ladder, play=play, workers=WORKERS, max_hours=EVAL_HOURS)
        )
    return jobs


# ---------------------------------------------------------------------------
# Report


def report() -> Report:
    """Score against the heuristic by deck, and whether newer checkpoints still win.

    Ten decks make a hundred matchups (50 games per cell), read as a matrix and
    rankings rather than a hundred curves. `games` holds a row per game and
    player; player 0 is the checkpoint (or the newer one of a pair), and
    `deck`/`opponent_deck` are the decks each actually played.
    """
    heuristic = Matches(tag="heuristic")
    return Report(
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            Curve(
                title="Against the rated checkpoint before",
                matches=Matches(tag="consecutive"),
            ),
            Curve(
                title="Endpoint against each earlier checkpoint",
                matches=Matches(tag="endpoint"),
            ),
            Matrix(
                title="Model deck × heuristic deck",
                rows="opponent_deck",
                columns="deck",
                matches=heuristic,
            ),
            Ranking(title="By model deck", by="deck", matches=heuristic),
            Ranking(title="By heuristic deck", by="opponent_deck", matches=heuristic),
            Ranking(title="On the play or the draw", by="game_seat", matches=heuristic),
            Curve(title="By model deck", split="deck", matches=heuristic),
            HeadToHead(matches=Matches(tournament=PAIRS)),
            Anchors(),
        ),
        perf=PERF,
        ml=ML,
    )
