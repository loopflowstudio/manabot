"""MTG-175: three adaptive arms and a reused uniform control on MTG-183's both-seat Elemental recipe.

The treatment is game-root selection, including adaptive coverage/age safeguards.
U reuses MTG-183's native independent-uniform first window; it has no teacher
metrics retroactively invented. M/E/ME share its exact saved initial weights. Each arm has a $10
inclusive allocation: at most $9.50 for its one Etude training attempt, with
$0.50 reserved for storage/observation uncertainty. The protocol is
`experiments/protocols/curriculum-175.md`. Construction and quoting rent nothing.

`runs` freezes three single-attempt adaptive jobs by default, `eval` asks Etude for development
checkpoint grids and a disjoint final grid, and `report` draws their scores.
Default evaluations use the owned CPU pool. The protocol records a separately
bounded U-initial evaluation fallback; this planner selects no paid rental.

    uv run etude launch experiments/curriculum_adapters.py --revision COMMIT \\
        --arm M --cohort mtg175-v1-m --rent l4 --parallel 1 --max-usd 9.50 --quote
    uv run etude eval experiments/curriculum_adapters.py \\
        --source r2://manabot/jobs --pool mini --watch 60
    uv run etude report experiments/curriculum_adapters.py \\
        --source r2://manabot/jobs --open
"""

from decimal import Decimal
import json
from pathlib import Path
from typing import Literal

from etude.eval.records import Benchmark, Inference, Matchup
from etude.experiment import Checkpoint, ExperimentState
from etude.jobs import evaluation
from etude.launch.records import Job, Run
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score, Series

from experiments.curriculum_costs import selected_checkpoints
from experiments.curriculum_uniform import UNIFORM_RUN, load_control
from experiments.elemental_grid import DECKS, PACK, Game, groups
from experiments.elemental_long import REGIME, recipe as elemental_recipe
from experiments.elemental_long_eval import (
    INFERENCE_CODE,
    eval_environment,
    heuristic_code,
    play,
    rows,
)
from manabot.experiment.job import TrainingJob
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.rollout.adaptive import AdaptiveConfig

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
ARMS: tuple[Literal["U", "M", "E", "ME"], ...] = ("U", "M", "E", "ME")
SEED = 18301
UPDATES = 1400
HOURS = 15.0
TRAINING_USD = Decimal("9.50")
RESERVE_USD = Decimal("0.50")
CADENCE = tuple(range(50, UPDATES, 50))
MONITOR_UPDATES = (0, 100, 200, 400, 800, 1200, UPDATES)
CONTROL = load_control()
INITIAL = CONTROL.initial
MONITOR = "curriculum-development"
FINAL = "curriculum-held-out"
INFERENCE = Inference(source=list(INFERENCE_CODE))
# CPU jobs queue behind existing work. No paid evaluation on-demand is selected.
EVAL_WORKERS = 1
EVAL_HOURS = 6.0
SEATS: tuple[Literal[0, 1], ...] = (0, 1)


def recipe(arm: Literal["U", "M", "E", "ME"]) -> TrainingRegime:
    """The common fresh recipe; sampler signal is the sole learning treatment."""
    regime = elemental_recipe(UPDATES, hours=HOURS)
    regime.id = f"curriculum-175-{arm.lower()}"
    regime.observation = ObservationConfig.from_checkpoint(
        json.loads(REGIME.read_bytes())["observation"]
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.matchup_curriculum = "fixed"
    stage.adaptive = AdaptiveConfig(
        signal=arm,
        seed=SEED,
        cells=tuple(MatchConfig.authored(PACK, a, b) for a in DECKS for b in DECKS),
    )
    if arm == "U":
        stage.adaptive = None
        stage.matchup_curriculum = "elemental-uniform"
    return TrainingRegime.model_validate(regime.model_dump(mode="json"))


def runs(
    revision: str,
    reuse_uniform: bool = True,
    arm: Literal["M", "E", "ME"] | None = None,
) -> list[Run]:
    """One attempt per arm; submit separate cohorts to isolate rental budgets."""
    found: list[Run] = []
    for candidate in ARMS:
        if (candidate == "U" and reuse_uniform) or (
            arm is not None and candidate != arm
        ):
            continue
        job = TrainingJob(
            regime=recipe(candidate),
            experiment=EXPERIMENT,
            milestones=CADENCE,
            finalization_seconds=600,
            validate_numerics=True,
            initial_checkpoint=INITIAL,
        ).job(repo=ROOT, revision=revision, max_hours=HOURS, max_usd=TRAINING_USD)
        found += job.runs(SEED, params={"arm": candidate})
    return found


def evaluation_schedule(held_out: bool) -> list[Game]:
    """New paired-seat deals, disjoint from MTG-183's already observed grids."""
    base, deals = (2_165_300_000, 10) if held_out else (2_165_200_000, 2)
    return [
        Game(first, second, 2 * index + seat, base + 1000 * cell + index, seat)
        for cell, (first, second) in enumerate((a, b) for a in DECKS for b in DECKS)
        for index in range(deals)
        for seat in SEATS
    ]


def _benchmark(held_out: bool) -> Benchmark:
    games = evaluation_schedule(held_out)
    conditions = rows("subset", groups(games))
    (per_match,) = {sum(len(group.deals) for group in row.groups) for row in conditions}
    return Benchmark(
        name=FINAL if held_out else MONITOR,
        environment=eval_environment(),
        matchups=(
            Matchup(
                name="held-out grid" if held_out else "development grid",
                tags=("held-out" if held_out else "development", "heuristic"),
                players=(INFERENCE, heuristic_code()),
                conditions=tuple(row.model_dump(mode="json") for row in conditions),
                games=per_match,
            ),
        ),
    )


def eval(state: ExperimentState) -> list[Job]:
    """Evaluate real exports once; a failed/unfinished endpoint stays missing."""
    monitoring: list[Checkpoint] = []
    endpoints: list[Checkpoint] = []
    cost_selected = selected_checkpoints()
    for run in state.runs:
        endpoints += [cp for cp in run.checkpoints if int(cp.clocks["updates"]) == 0]
        monitoring += [
            checkpoint
            for checkpoint in run.checkpoints
            if int(checkpoint.clocks["updates"]) in MONITOR_UPDATES
            or (
                run.name == UNIFORM_RUN
                and int(checkpoint.clocks["updates"]) == CONTROL.endpoint_updates
            )
        ]
        # A clean time-limited finish is an actual endpoint even when it falls
        # between monitoring milestones. TrainingRun.final excludes failures
        # and live runs; an initialization-only run has no trained endpoint.
        if (final := run.final) is not None and final.clocks["updates"] > 0:
            endpoints.append(final)
            if final not in monitoring:
                monitoring.append(final)
        if selected := cost_selected.get(run.name):
            # Stop changing the primary selection once this run has ended.
            # During training the observer may still see a later eligible export.
            if run.ended and selected in run.checkpoints and selected not in endpoints:
                endpoints.append(selected)
    return [
        evaluation(
            _benchmark(held_out),
            play=play,
            checkpoints=[checkpoint],
            workers=EVAL_WORKERS,
            max_hours=EVAL_HOURS,
        )
        for held_out, checkpoints in ((False, monitoring), (True, endpoints))
        for checkpoint in checkpoints
    ]


def report() -> Report:
    """Learning curves and held-out deck/seat outcomes with curriculum diagnostics."""
    development = Matches(tag="development")
    held_out = Matches(tag="held-out")
    return Report(
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        baseline="curriculum-175-u",
        evals=(
            Curve(title="Development score vs heuristic v2", matches=development),
            Curve(
                title="GPU development score vs heuristic v2",
                matches=Matches(tag="development-gpu"),
            ),
            Curve(title="Held-out endpoint", matches=held_out),
            Curve(
                title="GPU held-out endpoint",
                matches=Matches(tag="held-out-gpu"),
            ),
            Matrix(
                title="Held-out model deck × heuristic deck",
                rows="opponent_deck",
                columns="deck",
                matches=held_out,
            ),
            Ranking(title="Held-out score by model deck", by="deck", matches=held_out),
            Ranking(title="Held-out score by seat", by="seat", matches=held_out),
        ),
        perf=PERF,
        ml=ML
        + (
            Series(
                title="Curriculum game starts", keys=("adaptive/starts",), unit="starts"
            ),
            Series(
                title="Distinct completed curriculum games per update",
                keys=("adaptive/completed_games",),
                unit="games",
            ),
            Series(
                title="Unfiltered target-error total", keys=("adaptive/residual_total",)
            ),
        ),
    )
