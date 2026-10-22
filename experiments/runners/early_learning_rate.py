"""Early learning-rate screen: constant-rate arms from shared initial weights.

The Ataraxos move recipe clamps its iteration schedule to a maximum for the
first few thousand updates. This screen replaces only that clamp: each arm sets
`learning_rate_min == learning_rate_max`, so the rate is constant for the whole
short run while every other setting matches the ETU-118 mini recipe.

Principal entry points:

- `declaration` resolves the arms through the shared `Experiment` API.
- `run` executes them with `run_experiment`, which owns the host lease, the
  learner/evaluator processes, deadlines and the retained evidence.
- `report` projects retained evidence into per-arm monitoring scores (overall
  and by the deck the policy played) and per-window update diagnostics.

Pairing is by training seed. Initialization and collection streams derive from
the seed alone, so arms sharing a seed start from identical weights and identical
first rollouts, and every checkpoint is scored on the same monitoring deals.
See `experiments/early-learning-rate.md` for the protocol and reading rule.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
import statistics

from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.execution import atomic_json
from manabot.training.experiment_execution import (
    ExperimentRun,
    ExperimentSchedule,
    Hardware,
    HardwareInventory,
)
from manabot.training.experiment_runner import run_experiment
from manabot.training.experiments import Baseline, Case, Experiment, LearningRule
from manabot.training.models import (
    AtaraxosMoveLearning,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult

NAME = "early-learning-rate"
HARDWARE = "early-learning-rate-mini"
# Case name -> constant learning rate. The first arm is the control: it equals
# the rate the unmodified schedule holds for its first ~2,300 updates.
ARMS: tuple[tuple[str, float], ...] = (
    ("lr-1e-4", 1e-4),
    ("lr-3e-4", 3e-4),
    ("lr-1e-3", 1e-3),
)
# ETU-118's first two training seeds, so the control arm can be read loosely
# against that baseline's early monitoring curve.
SEEDS = (11851, 11852)
UPDATES = 1500
CHECKPOINT_UPDATES = 300
# ETU-118's scripted-greedy monitoring deals: 25 deals x 4 seat/deck legs.
MONITORING_DEALS = tuple(range(1_911_183_000, 1_911_183_025))
# ETU-118's held-out endpoint deals. Reserved here only so that the schedule
# rejects any overlap; this screen never plays them.
RESERVED_DEALS = tuple(range(1_911_184_000, 1_911_184_100))
# Seconds. One attempt is a single arm/seed learner process.
ATTEMPT_SECONDS = 10800.0
WALL_SECONDS = 16 * 3600.0
# Accounting reserve for evaluator process time; it overlaps learning.
MONITOR_SECONDS = 21600.0
MONITOR_ATTEMPT_SECONDS = 900.0
SOURCE = Path(__file__).resolve().parents[1] / (
    "regimes/early-learning-rate-mini-source.json"
)


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def recipe(
    updates: int = UPDATES, allowance: float = ATTEMPT_SECONDS
) -> TrainingRegime:
    """The ETU-118 mini recipe as one fresh self-play stage of `updates`."""
    base = TrainingRegime.model_validate_json(SOURCE.read_text())
    (stage,) = base.stages
    if not isinstance(stage, TrainSelfPlay):
        raise ValueError("source recipe must be one self-play stage")
    stage.updates = updates
    stage.execution.wall_seconds = allowance
    base.wall_seconds = allowance
    return TrainingRegime.model_validate(base.model_dump())


def constant_rate(base: TrainingRegime, rate: float) -> AtaraxosMoveLearning:
    """Copy the base rule, pinning both schedule clamps to one rate."""
    (stage,) = base.stages
    if not isinstance(stage, TrainSelfPlay) or not isinstance(
        stage.learning, AtaraxosMoveLearning
    ):
        raise ValueError("constant-rate arms require the Ataraxos move rule")
    return AtaraxosMoveLearning.model_validate(
        {
            **stage.learning.model_dump(),
            "learning_rate_min": rate,
            "learning_rate_max": rate,
        }
    )


def schedule(
    seeds: tuple[int, ...] = SEEDS,
    *,
    checkpoint_updates: int = CHECKPOINT_UPDATES,
    attempt_seconds: float = ATTEMPT_SECONDS,
    wall_seconds: float = WALL_SECONDS,
    monitor_seconds: float = MONITOR_SECONDS,
    monitor_attempt_seconds: float = MONITOR_ATTEMPT_SECONDS,
    deals: tuple[int, ...] = MONITORING_DEALS,
    game_seconds: float = 60,
) -> ExperimentSchedule:
    """Sequential arms within each seed; one evaluator overlaps the learner.

    `process_seconds` admits every attempt at its full allowance plus the
    monitoring reserve. `wall_seconds` is the binding cap on the whole run.
    """
    return ExperimentSchedule(
        seeds=seeds,
        hardware=HARDWARE,
        wall_seconds=wall_seconds,
        process_seconds=len(seeds) * len(ARMS) * attempt_seconds + monitor_seconds,
        monitoring=MonitoringBudget(
            seconds=monitor_seconds,
            attempt_seconds=monitor_attempt_seconds,
            include_initial=True,
            protocol=MonitorProtocol(deal_seeds=deals, game_seconds=game_seconds),
        ),
        checkpoint_updates=checkpoint_updates,
        scientific_deal_seeds=RESERVED_DEALS,
        # The Mini must keep 40 GiB free for other retained evidence.
        disk_reserve_bytes=40 * 1024**3,
    )


def declaration(base: TrainingRegime, plan: ExperimentSchedule) -> Experiment:
    return Experiment(
        name=NAME,
        baseline=Baseline.capture("etu118-mini-recipe", base),
        cases=tuple(
            Case(name, (LearningRule(constant_rate(base, rate)),), label=f"lr {rate:g}")
            for name, rate in ARMS
        ),
        schedule=plan,
    )


def hardware() -> HardwareInventory:
    """This host: one learner thread plus the reserved evaluator thread."""
    return HardwareInventory(
        resources=(Hardware(name=HARDWARE, host=socket.gethostname(), cpu_threads=2),)
    )


def run(out: Path) -> ExperimentRun:
    return run_experiment(declaration(recipe(), schedule()), hardware(), out)


def smoke(out: Path) -> ExperimentRun:
    """Two updates per arm, one seed and one monitoring deal: path proof only."""
    plan = schedule(
        (SEEDS[0],),
        checkpoint_updates=2,
        attempt_seconds=120,
        wall_seconds=600,
        monitor_seconds=600,
        monitor_attempt_seconds=60,
        deals=MONITORING_DEALS[:1],
        game_seconds=30,
    )
    return run_experiment(declaration(recipe(2, 120), plan), hardware(), out)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Arena deck identifier -> the name this screen reports for the policy's deck.
DECKS = {"ur_lessons": "lessons", "gw_allies": "allies"}


@dataclass(frozen=True)
class Score:
    """Monitoring score of one checkpoint; a draw counts as half a win."""

    updates: int
    games: int
    score: float
    lower: float
    upper: float
    as_lessons: float
    as_allies: float


@dataclass(frozen=True)
class Window:
    """Update diagnostics over iterations (first, last], one row per update.

    Means are over the updates in the window. `gradient_norm` is the norm before
    clipping, so `clipped_fraction` is the share of updates whose final
    optimizer step was rescaled by `max_grad_norm`.
    """

    first: int
    last: int
    collection_kl: float
    clip_fraction: float
    entropy: float
    value_loss: float
    gradient_norm_median: float
    clipped_fraction: float
    legal_logit_gap_max: float
    skipped_steps: int
    rejected_steps: int
    nonfinite_count: int


@dataclass(frozen=True)
class ArmRun:
    case: str
    seed: int
    status: str
    error: str | None
    learning_rate: float
    updates: int
    seconds: float
    scores: tuple[Score, ...]
    windows: tuple[Window, ...]


def _number(row: dict[str, object], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"update diagnostic lacks numeric {name}")
    return float(value)


def windows(
    diagnostics: list[dict[str, object]], max_grad_norm: float, size: int
) -> tuple[Window, ...]:
    """Summarize consecutive `size`-update windows; a short tail is kept."""
    found: list[Window] = []
    for start in range(0, len(diagnostics), size):
        rows = diagnostics[start : start + size]
        numerical: list[dict[str, object]] = []
        for row in rows:
            health = row.get("numerical")
            if not isinstance(health, dict):
                raise ValueError("update diagnostic lacks numerical health")
            numerical.append(health)
        norms = [_number(row, "gradient_norm") for row in rows]
        found.append(
            Window(
                first=start,
                last=start + len(rows),
                collection_kl=statistics.fmean(
                    _number(row, "collection_kl") for row in rows
                ),
                clip_fraction=statistics.fmean(
                    _number(h, "clip_fraction") for h in numerical
                ),
                entropy=statistics.fmean(_number(row, "entropy") for row in rows),
                value_loss=statistics.fmean(_number(row, "value_loss") for row in rows),
                gradient_norm_median=statistics.median(norms),
                clipped_fraction=statistics.fmean(n > max_grad_norm for n in norms),
                legal_logit_gap_max=max(
                    _number(h, "legal_logit_gap_max") for h in numerical
                ),
                skipped_steps=sum(int(_number(h, "skipped_steps")) for h in numerical),
                rejected_steps=sum(
                    int(_number(h, "rejected_steps")) for h in numerical
                ),
                nonfinite_count=sum(
                    int(_number(h, "nonfinite_count")) for h in numerical
                ),
            )
        )
    return tuple(found)


def score(result: MonitorResult) -> Score:
    """Project one completed evaluation; the policy is always arena player A."""
    if result.status != "completed" or result.score is None:
        raise ValueError("monitoring evaluation is incomplete")
    by_deck: dict[str, list[float]] = {name: [] for name in DECKS.values()}
    for row in result.rows:
        extra = row.model_extra or {}
        decks, seat = extra.get("seat_decks"), extra.get("player_a_seat")
        if not isinstance(decks, list) or not isinstance(seat, int):
            raise ValueError("arena row lacks deck/seat assignment")
        if row.score_a is None:
            raise ValueError("completed evaluation has an unscored row")
        by_deck[DECKS[decks[seat]]].append(row.score_a)
    return Score(
        updates=result.coordinates.updates,
        games=len(result.rows),
        score=result.score.mean,
        lower=result.score.lower,
        upper=result.score.upper,
        as_lessons=statistics.fmean(by_deck["lessons"]),
        as_allies=statistics.fmean(by_deck["allies"]),
    )


def report(out: Path) -> tuple[ArmRun, ...]:
    """Read retained evidence only; safe while the experiment is running."""
    record = ExperimentRun.model_validate_json((out / "experiment.json").read_text())
    evaluations: dict[str, list[Score]] = {}
    for path in sorted(out.glob("monitoring/attempt-*/evaluation/monitor.json")):
        result = MonitorResult.model_validate_json(path.read_text())
        if result.status == "completed":
            evaluations.setdefault(result.run_id, []).append(score(result))
    rates = dict(ARMS)
    runs: list[ArmRun] = []
    for attempt in record.attempts:
        path = Path(attempt.path) / "run.json"
        if not path.exists():
            continue
        training = TrainingRun.model_validate_json(path.read_text())
        (stage,) = training.regime.stages
        assert isinstance(stage, TrainSelfPlay)
        assert isinstance(stage.learning, AtaraxosMoveLearning)
        diagnostics = training.stages[0].diagnostics if training.stages else []
        runs.append(
            ArmRun(
                case=attempt.case,
                seed=attempt.seed,
                status=attempt.status,
                error=attempt.error,
                learning_rate=rates[attempt.case.removeprefix(f"{NAME}-")],
                updates=len(diagnostics),
                seconds=attempt.process_seconds,
                scores=tuple(
                    sorted(evaluations.get(training.id, []), key=lambda s: s.updates)
                ),
                windows=windows(
                    diagnostics,
                    stage.learning.max_grad_norm,
                    training.monitoring_checkpoint_updates or CHECKPOINT_UPDATES,
                ),
            )
        )
    return tuple(runs)


def render(runs: tuple[ArmRun, ...]) -> str:
    """Plain-text tables: scores at matched updates, then update diagnostics."""
    lines = ["score vs scripted greedy: overall (as Lessons / as Allies)"]
    for item in runs:
        cells = "  ".join(
            f"u{s.updates}: {s.score:.2f} ({s.as_lessons:.2f}/{s.as_allies:.2f})"
            for s in item.scores
        )
        lines.append(f"{item.case} seed {item.seed} [{item.status}]  {cells}")
    lines.append("")
    lines.append(
        "window: collection KL, clip fraction, entropy, value loss, "
        "median pre-clip gradient norm, share of updates clipped, "
        "max legal-logit gap, skipped/rejected steps, non-finite logs"
    )
    for item in runs:
        for w in item.windows:
            lines.append(
                f"{item.case} seed {item.seed} ({w.first},{w.last}]: "
                f"kl {w.collection_kl:.5f}  clip {w.clip_fraction:.4f}  "
                f"H {w.entropy:.3f}  v {w.value_loss:.4f}  "
                f"g {w.gradient_norm_median:.3f}  clipped {w.clipped_fraction:.2f}  "
                f"gap {w.legal_logit_gap_max:.1f}  "
                f"skip {w.skipped_steps} rej {w.rejected_steps} "
                f"nonfinite {w.nonfinite_count}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke", "report"):
        commands.add_parser(name).add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        runs = report(args.out.resolve())
        atomic_json(
            args.out / "early-learning-rate-report.json",
            json.loads(json.dumps([asdict(item) for item in runs])),
        )
        print(render(runs))
        return
    record = (run if args.command == "run" else smoke)(args.out)
    print(record.status)
    if record.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
