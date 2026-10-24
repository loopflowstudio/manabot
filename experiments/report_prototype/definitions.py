"""Report definitions for the PROTOTYPE: what an experiment author would write.

A report is two top-level parts: Evals, and Perf with ML details. Evals is the part an author
usually chooses; Perf and ML details default to panels over the trainer's own
metric keys, so most experiments only declare `evals`. Panels name data by
dimension and metric key and never carry rendering code; the page renders any
panel kind it knows and reports an unknown kind rather than dropping it.

Eval vocabulary (all over the same game cells: run x checkpoint x anchor x
candidate deck x opponent deck x seat):
  Curve       score by checkpoint, one line per run; `split` -> small multiples
  Matrix      model deck (x) x heuristic deck (y) at one checkpoint, latest by default
  Ranking     ranked bars by deck / opponent / seat; rows filter the page
  HeadToHead  direct run-vs-run games, 50% reference
Metric vocabulary (per-update training series):
  Series      one or more metric keys on one axis (same unit only)
  Flame       icicle of where time went, as deep as timers go, per run and resource
  Counters    keys expected to stay at zero; shown as status, not as a chart
"""

from dataclasses import dataclass, field
from typing import Literal

Dimension = Literal["matchup", "deck", "opponent", "seat"]


# ---------------------------------------------------------------- eval panels


@dataclass(frozen=True)
class Curve:
    title: str = "Score"
    split: Dimension | None = None
    kind: str = "curve"


@dataclass(frozen=True)
class Matrix:
    title: str = "Matchups"
    pool: int = 1  # checkpoints pooled per cell, ending at the shown one; >1 is labeled
    kind: str = "matrix"


@dataclass(frozen=True)
class Ranking:
    by: Dimension = "deck"
    kind: str = "ranking"


@dataclass(frozen=True)
class HeadToHead:
    title: str = "Head to head"
    kind: str = "head_to_head"


# ---------------------------------------------------------------- metric panels


@dataclass(frozen=True)
class Series:
    title: str
    keys: tuple[str, ...]
    unit: str = ""
    scale: Literal["linear", "log"] = "linear"
    kind: str = "series"


@dataclass(frozen=True)
class Flame:
    title: str = "Where the time went"
    scale: Literal["share", "total", "per_decision"] = "per_decision"
    kind: str = "flame"


@dataclass(frozen=True)
class Counters:
    title: str
    keys: tuple[str, ...]
    kind: str = "counters"


EvalPanel = Curve | Matrix | Ranking | HeadToHead
MetricPanel = Series | Flame | Counters


def default_perf() -> list[MetricPanel]:
    return [
        Flame(),
        Series(
            "Learner throughput",
            ("throughput/learner_transitions_per_second",),
            "decisions/s",
        ),
        Series("Seconds per update", ("derived/seconds_per_update",), "s"),
        Series("Games per second", ("throughput/games_per_second",), "games/s"),
        Series("Process memory", ("progress/rss_bytes",), "bytes"),
        Series("Host load", ("progress/host_load_1m",), "1-min load"),
    ]


def default_ml() -> list[MetricPanel]:
    return [
        Counters(
            "Numerical stability",
            (
                "numerical/nonfinite_count",
                "numerical/rejected_steps",
                "numerical/legal_underflow_count",
            ),
        ),
        Series("Losses", ("rl/value_loss", "rl/policy_loss", "rl/loss")),
        Series("Policy entropy", ("rl/entropy",), "nats"),
        Series("Collection KL", ("rl/collection_kl",), "KL", "log"),
        Series(
            "Gradient norm",
            ("numerical/gradient_norm_before", "numerical/gradient_norm_after"),
            "L2",
        ),
        Series("Learning rate", ("rl/learning_rate",)),
        Series("Retained fraction", ("rl/retained_fraction",), "fraction"),
    ]


@dataclass(frozen=True)
class Report:
    """Every panel shows every run of the experiment; there is no primary run.

    Run labels are not declared here. A run is labeled by its launch: the
    job's `name` (`etude.jobs.training(..., name="Both seats learn")`), which
    Etude records as `RunDeclaration.job_name`, plus its seed when a job has
    several runs. Run-vs-run numbers use updates every run evaluated.
    """

    evals: list[EvalPanel]
    clock: Literal["updates", "decisions", "hours"] = "updates"  # default x axis
    # Job drawn as the dotted reference line. None uses the experiment's own
    # declared baseline (`Experiment(baseline=...)`); name a job to override.
    baseline: str | None = None
    perf: list[MetricPanel] = field(default_factory=default_perf)
    ml: list[MetricPanel] = field(default_factory=default_ml)


# ---------------------------------------------------------------- the two experiments

# Two decks: four matchups read best as small multiples of curves, and the
# direct comparison is the question itself.
TWO_SEAT = Report(
    evals=[
        Curve("Score vs heuristic"),
        HeadToHead("Both seats vs one seat, direct games"),
        Curve("By matchup", split="matchup"),
        Ranking("seat"),
    ],
)

# Ten decks: a hundred matchups need a matrix (MTG-181: 50 games per cell) and
# rankings; per-matchup curves would be 100 panels.
ELEMENTAL = Report(
    evals=[
        Curve("Score vs heuristic"),
        Matrix("Model deck × heuristic deck"),
        Ranking("deck"),
        Ranking("opponent"),
        Curve("By model deck", split="deck"),
    ],
)

# Scored at learner-decision milestones, so that is the default clock.
SEAT_COMPARISON = Report(
    clock="decisions",
    evals=[
        Curve("Score vs heuristic"),
        Curve("By matchup", split="matchup"),
        Matrix("Model deck × heuristic deck"),
        Ranking("seat"),
    ],
)

REPORTS: dict[str, Report] = {
    "seat-comparison": SEAT_COMPARISON,
    "two-seat": TWO_SEAT,
    "elemental": ELEMENTAL,
}
