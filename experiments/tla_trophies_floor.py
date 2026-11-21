"""Compare forward-KL weights and larger tau on trophy policies.

The fresh and pullback arms vary the forward-KL floor under the trophy tau
schedule. The tau arms keep that schedule and multiply its scale. Each fresh
checkpoint plays the held-out deals against the heuristic and, in its own
benchmark, against its own line at half its updates. At updates 200 and 400
each treated fresh arm also meets every control seed at the same update.
"""

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from etude.checkpoints import CheckpointRecord
from etude.eval.match import environment
from etude.eval.records import Aligned, Benchmark, CheckpointSlot, Fraction, Seating
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.records import ArtifactRef
from etude.report import Curve, Matches, Report, Score
from etude.train.tracking import track

from experiments import tla_trophies as trophies
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves

FRESH = {"control": 0.0, "floor-0.002": 0.002, "floor-0.01": 0.01}
PULLBACK = {"pullback-0": 0.0, "pullback-0.01": 0.01}
# Arm -> tau_scale, 2x and 4x the trophy schedule's 0.05.
TAU = {"tau-x2": 0.1, "tau-x4": 0.2}
FRESH_ARMS = (*FRESH, *TAU)
TAU_SEEDS = (40901, 40902)
START_SEED = 40902
START_RUN = f"{trophies.RAW_JOB}@{START_SEED}"
START_FILE = "out/policies/00000400-raw.safetensors"
START = CheckpointRecord(
    experiment="tla_trophies",
    run=START_RUN,
    ordinal=5,
    at={"updates": 400},
    file=START_FILE,
    artifact=ArtifactRef(
        uri=f"r2://manabot/jobs/tla_trophies/{START_RUN}/{START_FILE}",
        sha256="896ee236e0227ae4b94f68e2732b8955a2a7c7c1cb9f9415e62a11296c7d3ce4",
        size_bytes=5_671_666,
    ),
)


class Config(trophies.Config):
    reference_forward_kl: float
    pullback: bool
    # The tau schedule's scale; None keeps the trophy recipe's.
    tau_scale: float | None = None


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    base = trophies.settings(config)
    with track(clocks=CLOCKS, config=config) as run:
        settings = base.model_copy(
            update={
                "pretrained": run.input("start") if config.pullback else None,
                "learning": base.learning.model_copy(
                    update={"reference_forward_kl": config.reference_forward_kl}
                    | (
                        {"tau_scale": config.tau_scale}
                        if config.tau_scale is not None
                        else {}
                    )
                ),
            }
        )
        self_play(settings, run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    pullback_updates: int = 100,
    pullback_hours: float = 2.75,
    tau_hours: float | None = None,
    tau_usd: float | None = None,
) -> list[TrainingJob]:
    """`tau_hours` and `tau_usd` size the tau arms for their own placement."""
    (base,) = trophies.training_jobs(revision=revision, usd_per_run=2.54)
    assert base.smoke is not None
    jobs = []
    for name, weight in (FRESH | PULLBACK).items():
        pullback = name in PULLBACK
        config = Config(
            **base.config.model_dump(), reference_forward_kl=weight, pullback=pullback
        )
        smoke = Config(
            **base.smoke.model_dump(), reference_forward_kl=weight, pullback=pullback
        )
        jobs.append(
            base.model_copy(
                update={
                    "name": name,
                    "train": train,
                    "config": config.model_copy(update={"updates": pullback_updates})
                    if pullback
                    else config,
                    "smoke": smoke,
                    "start": START if pullback else None,
                    "seeds": (START_SEED,) if pullback else base.seeds,
                    "max_hours": pullback_hours if pullback else base.max_hours,
                    "max_usd": Decimal("1.06") if pullback else base.max_usd,
                }
            )
        )
    for name, tau_scale in TAU.items():
        arm = {"reference_forward_kl": 0.0, "pullback": False, "tau_scale": tau_scale}
        jobs.append(
            base.model_copy(
                update={
                    "name": name,
                    "train": train,
                    "config": Config(**base.config.model_dump(), **arm),
                    "smoke": Config(**base.smoke.model_dump(), **arm),
                    "seeds": TAU_SEEDS,
                    "max_hours": tau_hours or base.max_hours,
                    "max_usd": Decimal(str(tau_usd)) if tau_usd else base.max_usd,
                }
            )
        )
    return jobs


LABELS = {
    "control": "Control",
    "floor-0.002": "Forward KL 0.002",
    "floor-0.01": "Forward KL 0.01",
    "tau-x2": "Tau x2",
    "tau-x4": "Tau x4",
    "pullback-0": "Warm start, control",
    "pullback-0.01": "Warm start, forward KL 0.01",
}
# The floor's own held-out benchmark: the shared trophy benchmark's 96
# update-zero control games were recorded under an earlier tla_trophies.py.
HELD_OUT = "tla-trophies-floor-held-out-32"
HALF_TRAINING = "tla-trophies-floor-half-training-32"
HALF_TRAINING_SEATINGS = tuple(f"Half-training deal {i + 1}" for i in range(32))
HEAD_TO_HEAD = "tla-trophies-floor-vs-control-32"
HEAD_TO_HEAD_SEATINGS = tuple(f"Matched control deal {i + 1}" for i in range(32))
HEAD_TO_HEAD_UPDATES = (200, 400)


def trophy_benchmark(name: str, seatings: tuple[Seating, ...]) -> Benchmark:
    """A benchmark on the trophy game, pinned to the files that define its play.

    This file only schedules seatings, so editing its report keeps the keys of
    games already played.
    """
    return Benchmark(
        name=name,
        environment=environment(
            "tla-trophies",
            files=[
                Path(trophies.__file__),
                trophies.ROOT / "manabot/eval/pairing.py",
            ],
            config={"pools": trophies.corpus_input().sha256},
        ),
        seatings=seatings,
    )


def deal(i: int) -> trophies.Deal:
    return trophies.Deal(first=i * 2, second=i * 2 + 1, seed=40900 + i)


def arm_evaluations(
    runs: Sequence[TrainingRunSummary], arm: str, control_seeds: int
) -> list[EvalJob]:
    """Give each seed one match per deal, including arms with fewer seeds."""
    held_out_name = f"{HELD_OUT}-{arm}"
    half_name = f"{HALF_TRAINING}-{arm}"
    versus_name = f"{HEAD_TO_HEAD}-{arm}"
    held_out = trophies.held_out_jobs(runs)
    if not held_out:
        return []
    (shared,) = held_out
    job = shared.model_copy(
        update={
            "name": held_out_name,
            "tournament": trophy_benchmark(held_out_name, shared.tournament.seatings),
        }
    )
    # Each checkpoint meets its own line at exactly half its updates on the
    # same held-out deals, seats and decks crossed (200 vs 100, 400 vs 200 at
    # checkpoints every 100). Without a trained checkpoint there it is a gap.
    seeds = len({run.seed for run in runs})
    earlier = Fraction(clock="updates", value=0.5, trained=True)
    half = job.model_copy(
        update={
            "name": half_name,
            "tournament": trophy_benchmark(
                half_name,
                tuple(
                    Seating(
                        name=name,
                        players=(CheckpointSlot(), earlier),
                        config=deal(i),
                        games=seeds,
                    )
                    for i, name in enumerate(HALF_TRAINING_SEATINGS)
                ),
            ),
        }
    )
    # Treated arms at matched updates against each control seed: a held
    # checkpoint's games split evenly over its seed pairs.
    treated = [
        c
        for run in runs
        for c in run.checkpoints
        if c.at.get("updates") in HEAD_TO_HEAD_UPDATES
    ]
    if arm == "control" or not treated or not control_seeds:
        return [job, half]
    control = Aligned(clock="updates", job="control")
    versus = job.model_copy(
        update={
            "name": versus_name,
            "checkpoints": treated,
            "tournament": trophy_benchmark(
                versus_name,
                tuple(
                    Seating(
                        name=name,
                        players=(CheckpointSlot(), control),
                        config=deal(i),
                        # One match per treated seed x control seed pair.
                        games=seeds * control_seeds,
                    )
                    for i, name in enumerate(HEAD_TO_HEAD_SEATINGS)
                ),
            ),
        }
    )
    return [job, half, versus]


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    controls = {run.seed for run in state.runs if run.launch == "control"}
    return [
        job
        for arm in FRESH_ARMS
        for job in arm_evaluations(
            [r for r in state.runs if r.launch == arm], arm, len(controls)
        )
    ]


def report() -> Report:
    held_out = tuple(Matches(tournament=f"{HELD_OUT}-{arm}") for arm in FRESH_ARMS)
    # Named seatings draw the held checkpoint's score, one line per arm.
    against_half = Curve(
        title="Against itself at half its updates",
        matches=tuple(Matches(seating=name) for name in HALF_TRAINING_SEATINGS),
    )
    against_control = Curve(
        title="Against the control at the same update",
        matches=tuple(Matches(seating=name) for name in HEAD_TO_HEAD_SEATINGS),
    )
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5),
        evals=(
            Curve(title="Score against the heuristic", matches=held_out),
            against_half,
            against_control,
            *behaviour_curves(matches=held_out),
        ),
        baseline="control",
        labels=LABELS,
        perf=PERF,
        ml=ML,
    )
