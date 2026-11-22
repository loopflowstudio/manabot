"""Compare forward-KL weights and larger tau on trophy policies.

The fresh and pullback arms vary the forward-KL floor under the trophy tau
schedule. The tau arms keep that schedule and multiply its scale. Each fresh
checkpoint plays the held-out deals against the heuristic and, in its own
benchmark, against its own line at half its updates. At updates 200 and 400
each treated fresh arm also meets every control seed at the same update.
Past update 400, where only one seed of the tau arms continues, that seed
plays the same benchmarks in its own one-seed tournaments and the extended
arms meet each other at the same update.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from etude.checkpoints import CheckpointRecord
from etude.eval.records import Aligned, Fraction
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.records import ArtifactRef
from etude.report import Curve, Matches, Report, Score
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_trophies as trophies
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves, head_to_head


@dataclass(frozen=True)
class Arm:
    label: str
    reference_forward_kl: float = 0.0
    tau_scale: float | None = None
    pullback: bool = False


ARMS = {
    "control": Arm("Control"),
    "floor-0.002": Arm("Forward KL 0.002", 0.002),
    "floor-0.01": Arm("Forward KL 0.01", 0.01),
    "pullback-0": Arm("Warm start, control", pullback=True),
    "pullback-0.01": Arm("Warm start, forward KL 0.01", 0.01, pullback=True),
    "tau-x2": Arm("Tau x2", tau_scale=0.1),
    "tau-x4": Arm("Tau x4", tau_scale=0.2),
    "tau-x2-floor-0.002": Arm("Tau x2, forward KL 0.002", 0.002, tau_scale=0.1),
}
FRESH_ARMS = tuple(name for name, arm in ARMS.items() if not arm.pullback)
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
    # Etude resumes the original trophy arms by their serialized config.
    tau_scale: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    base = trophies.settings(config)
    learning = base.learning.model_copy(
        update=config.model_dump(include={"reference_forward_kl", "tau_scale"})
    )
    with track(clocks=CLOCKS, config=config) as run:
        settings = base.model_copy(
            update={
                "pretrained": run.input("start") if config.pullback else None,
                "learning": learning,
            }
        )
        self_play(settings, run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    pullback_updates: int = 100,
    pullback_hours: float = 2.75,
    tau_hours: float | None = None,
    tau_usd: float | None = None,
    attempts: int = 1,
    hours: float | None = None,
    usd: float | None = None,
) -> list[TrainingJob]:
    """`tau_hours` and `tau_usd` size the tau arms for their own placement;
    `attempts`, `hours` and `usd` let a resume, or an extension past the
    declared updates, run again in what its job limits cover."""
    (base,) = trophies.training_jobs(revision=revision, usd_per_run=2.54)
    assert base.smoke is not None
    jobs = []
    for name, arm in ARMS.items():
        config_fields = {
            "reference_forward_kl": arm.reference_forward_kl,
            "pullback": arm.pullback,
            "tau_scale": arm.tau_scale,
        }
        placement = {} if hours is None else {"max_hours": hours}
        if usd is not None:
            placement["max_usd"] = Decimal(str(usd))
        updates = base.config.updates
        if arm.pullback:
            updates = pullback_updates
            placement.update(
                start=START,
                seeds=(START_SEED,),
                max_hours=pullback_hours,
                max_usd=Decimal("1.06"),
            )
        elif arm.tau_scale is not None:
            placement.update(
                seeds=TAU_SEEDS,
                max_hours=tau_hours or base.max_hours,
                max_usd=Decimal(str(tau_usd)) if tau_usd else base.max_usd,
            )
        jobs.append(
            base.model_copy(
                update={
                    **placement,
                    "name": name,
                    "train": train,
                    "config": Config(
                        **base.config.model_dump(exclude={"updates"}),
                        updates=updates,
                        **config_fields,
                    ),
                    "smoke": Config(**base.smoke.model_dump(), **config_fields),
                    "max_attempts": attempts,
                }
            )
        )
    return jobs


# The floor's own held-out benchmark: the shared trophy benchmark's 96
# update-zero control games were recorded under an earlier tla_trophies.py.
HELD_OUT = "tla-trophies-floor-held-out-32"
HALF_TRAINING = "tla-trophies-floor-half-training-32"
HALF_TRAINING_SEATINGS = tuple(f"Half-training deal {i + 1}" for i in range(32))
HEAD_TO_HEAD = "tla-trophies-floor-vs-control-32"
HEAD_TO_HEAD_SEATINGS = tuple(f"Matched control deal {i + 1}" for i in range(32))
HEAD_TO_HEAD_UPDATES = (200, 400)
# Past update 400 only seed 40902 of the tau arms continues. Its games are
# their own benchmarks with `seeds="one"` seatings, so the page draws the one
# seed rather than waiting for a seed that stopped at 400.
SEED_HEURISTIC = "tla-trophies-floor-seed-held-out-32"
SEED_HALF_TRAINING = "tla-trophies-floor-seed-half-training-32"
SEED_HALF_TRAINING_SEATINGS = tuple(
    f"One-seed half-training deal {i + 1}" for i in range(32)
)
SEED_HEAD_TO_HEAD = "tla-trophies-floor-seed-vs-32"
EXTENSION_UPDATES = tuple(range(500, 1001, 100))
EXTENSION_OPPONENTS = {
    "tau-x2-floor-0.002": ("tau-x2", "tau-x4"),
    "tau-x4": ("tau-x2",),
}
EXTENSION_ARMS = ("tau-x2-floor-0.002", "tau-x2", "tau-x4")


def extension_seatings(opponent: str) -> tuple[str, ...]:
    return tuple(f"One-seed {opponent} deal {i + 1}" for i in range(32))


def arm_evaluations(
    runs: Sequence[TrainingRunSummary], arm: str, control_seeds: int
) -> list[EvalJob]:
    """Reuse the trophy environment and deals so retained matches keep their identities."""
    held_out = trophies.held_out_jobs(runs)
    if not held_out:
        return []
    (shared,) = held_out
    benchmark = shared.tournament
    compared = [
        c
        for c in shared.checkpoints
        if c.at.get("updates", 0) <= HEAD_TO_HEAD_UPDATES[-1]
    ]
    extended = [
        c for c in shared.checkpoints if c.at.get("updates") in EXTENSION_UPDATES
    ]

    def against(
        name: str,
        checkpoints: Sequence[CheckpointRecord],
        seating_names: tuple[str, ...] | None = None,
        opponent: Fraction | Aligned | None = None,
        opponent_seeds: int = 1,
        one_seed: bool = False,
    ) -> EvalJob:
        names = seating_names or tuple(s.name for s in benchmark.seatings)
        seatings = tuple(
            seating.model_copy(
                update={
                    "name": label,
                    "games": seating.games * opponent_seeds,
                    **(
                        {}
                        if opponent is None
                        else {"players": (seating.players[0], opponent)}
                    ),
                    **({"seeds": "one"} if one_seed else {}),
                }
            )
            for seating, label in zip(benchmark.seatings, names, strict=True)
        )
        return shared.model_copy(
            update={
                "name": name,
                "tournament": benchmark.model_copy(
                    update={"name": name, "seatings": seatings}
                ),
                "checkpoints": checkpoints,
            }
        )

    half = Fraction(clock="updates", value=0.5, trained=True)
    jobs = [
        against(f"{HELD_OUT}-{arm}", compared),
        against(f"{HALF_TRAINING}-{arm}", compared, HALF_TRAINING_SEATINGS, half),
    ]
    treated = [c for c in compared if c.at.get("updates") in HEAD_TO_HEAD_UPDATES]
    if arm != "control" and treated and control_seeds:
        jobs.append(
            against(
                f"{HEAD_TO_HEAD}-{arm}",
                treated,
                HEAD_TO_HEAD_SEATINGS,
                Aligned(clock="updates", job="control"),
                control_seeds,
            )
        )
    if extended:
        jobs += [
            against(f"{SEED_HEURISTIC}-{arm}", extended, one_seed=True),
            against(
                f"{SEED_HALF_TRAINING}-{arm}",
                extended,
                SEED_HALF_TRAINING_SEATINGS,
                half,
                one_seed=True,
            ),
            *(
                against(
                    f"{SEED_HEAD_TO_HEAD}-{arm}-vs-{opponent}",
                    extended,
                    extension_seatings(opponent),
                    Aligned(clock="updates", job=opponent),
                    one_seed=True,
                )
                for opponent in EXTENSION_OPPONENTS.get(arm, ())
            ),
        ]
    return jobs


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
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        evals=(
            Curve(title="Score against the heuristic", matches=held_out),
            head_to_head("Against itself at half its updates", HALF_TRAINING_SEATINGS),
            head_to_head(
                "Against the control at the same update", HEAD_TO_HEAD_SEATINGS
            ),
            Curve(
                title="Seed 40902 past update 400: against the heuristic",
                matches=tuple(
                    Matches(tournament=f"{SEED_HEURISTIC}-{arm}")
                    for arm in EXTENSION_ARMS
                ),
            ),
            head_to_head(
                "Seed 40902 past update 400: against itself at half its updates",
                SEED_HALF_TRAINING_SEATINGS,
            ),
            head_to_head(
                "Seed 40902 past update 400: against tau ×2 at the same update",
                extension_seatings("tau-x2"),
            ),
            head_to_head(
                "Seed 40902 past update 400: against tau ×4 at the same update",
                extension_seatings("tau-x4"),
            ),
            *behaviour_curves(matches=held_out),
        ),
        baseline="control",
        labels={name: arm.label for name, arm in ARMS.items()},
        perf=PERF,
        ml=ML,
    )
