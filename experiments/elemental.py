"""Elemental, ten decks: two-seat self-play rated on the 10×10 deck grid.

Train on the ten-deck Elemental world from both seats, and score each
checkpoint as the model playing every deck against the heuristic playing every
deck, on fixed deals (`experiments.elemental_grid`). To find whether newer
checkpoints keep beating older ones once the heuristic score stops rising,
rated checkpoints also play each other on a fixed pairing graph
(`experiments.elemental_long`). The protocols are
experiments/protocols/elemental-174.md and elemental-183.md.

This file is the experiment `elemental` for Etude, and the template for a
recipe with a checkpoint ladder:

- `training_jobs()` declares the job `elemental`: fresh seeds on the current
  loop with `elemental_self_play()` settings, without current-game recovery;
- `eval_jobs(state)` asks for the grid and pairs of each run's rated
  checkpoints, played by `experiments.elemental_play.play`;
- `report()` draws them beside training.

    uv run etude launch experiments/elemental.py --revision COMMIT --updates 3200 \\
        --hours 11.9 --usd-per-run 9 --rent l4
    uv run etude report experiments/elemental.py --source r2://manabot/jobs --open

Typing the launch command again launches what is new: evaluation of the
checkpoints stored since, and nothing already run. `etude eval
experiments/elemental.py` plays the same matches on this computer.
"""

from decimal import Decimal
from pathlib import Path

from etude.checkpoints import CheckpointRecord
from etude.eval.records import (
    Benchmark,
    CheckpointSlot,
    Player,
    Seating,
    checkpoint_player,
)
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.report import (
    Anchors,
    Crosstable,
    Curve,
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
)
from experiments.elemental_play import (
    Conditions,
    Kind,
    eval_environment,
    h2h_groups,
    heuristic_code,
    play,
    rows,
)
from manabot.eval.agents import AGENTS
from manabot.experiment.setup import machine_setup
from manabot.recipes.elemental import ElementalConfig, train
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
# Benchmarks eval_jobs(state) holds; the report selects their matches by tag.
GRID = "heuristic-groups-agents-v1"
SUBSET = "heuristic-subset-groups-agents-v1"
PAIRS = "checkpoint-pair-groups-agents-v1"
# Wall hours one evaluation pass may take on a launched machine.
EVAL_HOURS = 12.0
# The training seed `training_jobs()` launches; `--seed` is the launcher's own option.
SEED = 18101

# The held checkpoint uses the ordinary Manabot policy.
CHECKPOINT = CheckpointSlot()


# ---------------------------------------------------------------------------
# Training


def training_jobs(
    revision: str,
    updates: int,
    hours: float,
    usd_per_run: float,
) -> list[TrainingJob]:
    """Current-world self-play; frozen regimes use their recorded revision."""
    return [
        TrainingJob(
            name=EXPERIMENT,
            train=train,
            config=ElementalConfig(
                updates=updates, device="cuda", validate_numerics=True
            ),
            smoke=ElementalConfig(updates=1, device="cpu", streams=2),
            seeds=(SEED,),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
    ]


# ---------------------------------------------------------------------------
# Evaluation


def _updates(checkpoint: CheckpointRecord) -> int:
    return int(checkpoint.at["updates"])


def _rated(
    run: TrainingRunSummary,
) -> tuple[dict[int, CheckpointRecord], int | None]:
    """The run's rated checkpoints by update, and its endpoint once it has finished."""
    by_update = {_updates(each): each for each in run.checkpoints}
    endpoint = None if run.final is None else _updates(run.final)
    wanted = rated_updates(max(by_update, default=0), endpoint)
    return {u: by_update[u] for u in wanted if u in by_update}, endpoint


def _matchup(
    name: str,
    tags: tuple[str, ...],
    players: tuple[Player | CheckpointSlot, Player | CheckpointSlot],
    grid: list[Conditions],
) -> list[Seating]:
    """One match per group; every row of a grid plays the same number of games."""
    return [
        Seating(
            name=f"{name} group {i}", tags=tags, players=players, config=config, games=1
        )
        for i, config in enumerate(grid)
    ]


def _grid(name: str, kind: Kind, games: list[Game]) -> Benchmark:
    return Benchmark(
        name=name,
        environment=eval_environment(),
        seatings=tuple(
            _matchup(
                f"vs heuristic, {kind} grid",
                ("heuristic",),
                (CHECKPOINT, heuristic_code()),
                rows(kind, groups(games)),
            )
        ),
    )


def _pairs(
    run: TrainingRunSummary, rated: dict[int, CheckpointRecord], endpoint: int | None
) -> list[Seating]:
    """The run's pairing graph: each pair plays the 10×10 cross-checkpoint schedule."""
    grid = rows("pair", h2h_groups(h2h_schedule()))
    return [
        seating
        for pair in pair_design(tuple(sorted(rated)), endpoint)
        for seating in _matchup(
            f"{run.name} u{pair.newer} vs u{pair.older}",
            (pair.role,),
            (
                checkpoint_player(rated[pair.newer].ref),
                checkpoint_player(rated[pair.older].ref),
            ),
            grid,
        )
    ]


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """The heuristic grid of every rated checkpoint, and each run's pairing graph.

    A rated checkpoint on the full cadence (and the endpoint) plays all 5,000
    grid games, any other the declared 10-game-per-cell subset.
    """
    full: list[CheckpointRecord] = []
    subset: list[CheckpointRecord] = []
    pairs: list[Seating] = []
    for run in state.runs:
        rated, endpoint = _rated(run)
        for updates, checkpoint in rated.items():
            kind, _ = heuristic_games(updates, endpoint)
            (full if kind == "full" else subset).append(checkpoint)
        pairs += _pairs(run, rated, endpoint)
    # The full grid is every scheduled game; the subset is any subset update's.
    jobs = [
        EvalJob(
            play=play,
            agents=AGENTS,
            tournament=_grid(GRID, "full", schedule()),
            checkpoints=full,
            max_hours=EVAL_HOURS,
        ),
        EvalJob(
            play=play,
            agents=AGENTS,
            tournament=_grid(SUBSET, "subset", heuristic_games(CADENCE, None)[1]),
            checkpoints=subset,
            max_hours=EVAL_HOURS,
        ),
    ]
    if pairs:
        ladder = Benchmark(
            name=PAIRS, environment=eval_environment(), seatings=tuple(pairs)
        )
        jobs.append(
            EvalJob(play=play, agents=AGENTS, tournament=ladder, max_hours=EVAL_HOURS)
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
            *behaviour_curves(matches=heuristic),
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
            Crosstable(matches=Matches(tournament=PAIRS)),
            Anchors(),
        ),
        perf=PERF,
        ml=ML,
    )
