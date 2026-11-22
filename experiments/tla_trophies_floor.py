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
from functools import cache
from pathlib import Path
import random

from etude.checkpoints import CheckpointRecord
from etude.eval.match import environment
from etude.eval.records import Aligned, Environment, Fraction, Seating
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.records import ArtifactRef
from etude.report import Curve, Matches, Report, Score
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_trophies as trophies
from manabot.eval.card_metrics import CARD_ROLLUPS
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
# Treated arms that also meet another treated arm, every seed pair, at 200 and 400.
TREATED_HEAD_TO_HEAD = "tla-trophies-floor-vs-treated-32"
TREATED_OPPONENTS = {"tau-x2-floor-0.002": "tau-x4"}


def treated_seatings(opponent: str) -> tuple[str, ...]:
    return tuple(f"Matched {opponent} deal {i + 1}" for i in range(32))


# Past update 400 only seed 40902 of the tau arms continues. Its games are
# their own benchmarks with `seeds="one"` seatings, so the page draws the one
# seed rather than waiting for a seed that stopped at 400.
SEED_HEURISTIC = "tla-trophies-floor-seed-held-out-32"
SEED_HALF_TRAINING = "tla-trophies-floor-seed-half-training-32"
SEED_HEAD_TO_HEAD = "tla-trophies-floor-seed-vs-32"
# Past update 400 the one seed also plays 128 more held-out deals, on pools
# 64-319, in benchmarks of their own so the first 32 deals' games still count.
# Those games also keep card evidence for the card page.
MORE_DEALS = 128
SEED_MORE = "128-more-cards"
SEED_DEALS = 32 + MORE_DEALS
SEED_HALF_TRAINING_SEATINGS = tuple(
    f"One-seed half-training deal {i + 1}" for i in range(SEED_DEALS)
)
# The 128 more deals were first played by a copy of the pairing that kept card
# evidence; play_pairing now keeps it, and their environment keeps that identity.
CARD_ENVIRONMENT = Environment(
    name="tla-trophies",
    files="f8c0538ba83d6c05c642fb375c999cfcd1d3ca9f3cd51b312f548720700f40ed",
    config={"pools": trophies.corpus_input().sha256, "cards": True},
)
EXTENSION_UPDATES = tuple(range(500, 1001, 100))
EXTENSION_OPPONENTS = {
    "tau-x2-floor-0.002": ("tau-x2", "tau-x4"),
    "tau-x4": ("tau-x2",),
}
EXTENSION_ARMS = ("tau-x2-floor-0.002", "tau-x2", "tau-x4")


# The final round: each extended arm at update 1000 meets the heuristic and the
# other extended arms on 2,000 deals drawn from every held-out pool, so the
# head-to-heads are decisive and every Manabot seat feeds the card page.
FINAL_UPDATE = 1000
FINAL_DEALS = 2000
FINAL = "tla-trophies-floor-final"


@cache
def final_deals() -> tuple[trophies.Deal, ...]:
    pools = len(trophies.playable("held_out"))
    rng = random.Random(408)
    return tuple(
        trophies.Deal(first=first, second=second, seed=41000 + j)
        for j, (first, second) in enumerate(
            rng.sample(range(pools), 2) for _ in range(FINAL_DEALS)
        )
    )


def final_seatings(template: Seating, opponent: str) -> tuple[Seating, ...]:
    return tuple(
        template.model_copy(
            update={"name": f"Final {opponent} deal {j + 1}", "config": deal}
        )
        for j, deal in enumerate(final_deals())
    )


def final_environment() -> Environment:
    return environment(
        "tla-trophies",
        files=[Path(trophies.__file__), trophies.ROOT / "manabot/eval/pairing.py"],
        config={"pools": trophies.corpus_input().sha256, "cards": True},
    )


def extension_seatings(opponent: str) -> tuple[str, ...]:
    return tuple(f"One-seed {opponent} deal {i + 1}" for i in range(SEED_DEALS))


def more_held_out_seatings(template: Seating) -> tuple[Seating, ...]:
    """Deals 33-160: fresh pool pairs after the 64 pools of the first 32 deals."""
    return tuple(
        template.model_copy(
            update={
                "name": f"Held-out pair {33 + j}",
                "config": trophies.Deal(
                    first=64 + 2 * j, second=65 + 2 * j, seed=40932 + j
                ),
            }
        )
        for j in range(MORE_DEALS)
    )


def arm_evaluations(
    runs: Sequence[TrainingRunSummary], arm: str, seeds: dict[str, int]
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
        base: tuple[Seating, ...] = benchmark.seatings,
        cards: bool = False,
        environment_: Environment = CARD_ENVIRONMENT,
    ) -> EvalJob:
        names = seating_names or tuple(s.name for s in base)
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
            for seating, label in zip(base, names, strict=True)
        )
        card_fields = (
            {"environment": environment_, "rollups": CARD_ROLLUPS} if cards else {}
        )
        return shared.model_copy(
            update={
                "name": name,
                "tournament": benchmark.model_copy(
                    update={"name": name, "seatings": seatings, **card_fields}
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
    if arm != "control" and treated and seeds.get("control"):
        jobs.append(
            against(
                f"{HEAD_TO_HEAD}-{arm}",
                treated,
                HEAD_TO_HEAD_SEATINGS,
                Aligned(clock="updates", job="control"),
                seeds["control"],
            )
        )
    rival = TREATED_OPPONENTS.get(arm)
    if rival and treated and seeds.get(rival):
        jobs.append(
            against(
                f"{TREATED_HEAD_TO_HEAD}-{arm}-vs-{rival}",
                treated,
                treated_seatings(rival),
                Aligned(clock="updates", job=rival),
                seeds[rival],
            )
        )
    more = more_held_out_seatings(benchmark.seatings[0])
    for suffix, base, picked, cards in (
        ("", benchmark.seatings, slice(0, 32), False),
        (f"-{SEED_MORE}", more, slice(32, SEED_DEALS), True),
    ):
        if not extended:
            break
        jobs += [
            against(
                f"{SEED_HEURISTIC}{suffix}-{arm}",
                extended,
                one_seed=True,
                base=base,
                cards=cards,
            ),
            against(
                f"{SEED_HALF_TRAINING}{suffix}-{arm}",
                extended,
                SEED_HALF_TRAINING_SEATINGS[picked],
                half,
                one_seed=True,
                base=base,
                cards=cards,
            ),
            *(
                against(
                    f"{SEED_HEAD_TO_HEAD}{suffix}-{arm}-vs-{opponent}",
                    extended,
                    extension_seatings(opponent)[picked],
                    Aligned(clock="updates", job=opponent),
                    one_seed=True,
                    base=base,
                    cards=cards,
                )
                for opponent in EXTENSION_OPPONENTS.get(arm, ())
            ),
        ]
    final = [c for c in extended if c.at.get("updates") == FINAL_UPDATE]
    if final and arm in EXTENSION_ARMS:
        template = benchmark.seatings[0]
        jobs.append(
            against(
                f"{FINAL}-{arm}",
                final,
                one_seed=True,
                base=final_seatings(template, "heuristic"),
                cards=True,
                environment_=final_environment(),
            )
        )
        jobs += [
            against(
                f"{FINAL}-{arm}-vs-{opponent}",
                final,
                opponent=Aligned(clock="updates", job=opponent),
                one_seed=True,
                base=final_seatings(template, opponent),
                cards=True,
                environment_=final_environment(),
            )
            for opponent in EXTENSION_OPPONENTS.get(arm, ())
        ]
    return jobs


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    seeds = {
        arm: len({run.seed for run in state.runs if run.launch == arm})
        for arm in FRESH_ARMS
    }
    return [
        job
        for arm in FRESH_ARMS
        for job in arm_evaluations(
            [r for r in state.runs if r.launch == arm], arm, seeds
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
            head_to_head(
                "Against tau ×4 at the same update",
                treated_seatings("tau-x4"),
            ),
            Curve(
                title="Seed 40902 past update 400: against the heuristic",
                matches=tuple(
                    Matches(tournament=f"{SEED_HEURISTIC}{suffix}-{arm}")
                    for arm in EXTENSION_ARMS
                    for suffix in ("", f"-{SEED_MORE}")
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
