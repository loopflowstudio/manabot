"""Exploratory TLA trophy kickstarting trial: decaying forward-KL teacher loss.

Tests whether distilling from a frozen trained w64 teacher (`tla_trophies_floor/
tau-x2@40902` update 800, 0.824 vs heuristic) with a linearly decaying forward-KL
loss (`fade_updates=200`, exact zero on updates 201..400) accelerates fresh
self-play for:
- Same-architecture w64 students (`w64-ks-0.2-f200`, `w64-ks-0.5-f200`, seeds
  40901 & 40902) compared against retained R2 `tla_trophies_floor/tau-x2`
  checkpoints (no redundant fresh w64 baseline is trained).
- Cross-architecture transfer to a 17x larger fresh w160l8 student
  (`w160l8-ks-0.2-f200` vs fresh `w160l8-fresh`, seed 40902).

All arms share the `tla_trophies` + `tau_scale=0.1` recipe (7,859 train / 873
held-out TLA trophy pools, closed decklists, policy input v11, `focus_film=True`,
`stream_norm=True`, `streams=256`, `transitions=64`, `rows=512`, `workers=4`,
`threads=2`, float32 CUDA, bucketed shapes, live tokens, raw checkpoints every
50 updates).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRecord
from etude.eval.records import (
    Aligned,
    CheckpointPlayer,
    Fraction,
    Seating,
    checkpoint_player,
)
from etude.jobs import (
    EvalJob,
    ExperimentState,
    TrainingJob,
    TrainingRunSummary,
    checkpoint_inputs_for,
)
from etude.records import ArtifactRef
from etude.report import Curve, Matches, Report, Score
from etude.spec import Variable
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_frontier_snapshots as snapshots, tla_trophies as trophies
from experiments.tla_frontier import SizeLabel, model
from experiments.tla_trophies_floor import final_environment
from manabot.eval.card_metrics import CARD_ROLLUPS
from manabot.experiment.setup import machine_setup
from manabot.training.learning import Kickstart
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves, head_to_head

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


UPDATES = 400
CHECKPOINT_EVERY = 50
TAU_SCALE = 0.1
W64_SEEDS: tuple[int, ...] = (40901, 40902)
LARGE_SEEDS: tuple[int, ...] = (40902,)

# Frozen w64 teacher: `tla_trophies_floor/tau-x2@40902` update 800 (0.824 vs heuristic).
TEACHER_FILE = "out/policies/00000800-raw.safetensors"
TEACHER = CheckpointRecord(
    experiment="tla_trophies_floor",
    run="tau-x2@40902",
    ordinal=9,
    at={"updates": 800},
    file=TEACHER_FILE,
    artifact=ArtifactRef(
        uri=f"r2://manabot/jobs/tla_trophies_floor/tau-x2@40902/{TEACHER_FILE}",
        sha256="3456250f4fb6fe9ee8ef4fb0d2700b1196cceb97a44eac394b012a9c91a2e26f",
        size_bytes=5_671_658,
    ),
)
TEACHER_PLAYER: CheckpointPlayer = snapshots.WARM_BASELINE_TAU_X2[400]


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    size: SizeLabel
    seeds: tuple[int, ...]
    policy_weight: float = 0.0
    value_weight: float = 0.0
    fade_updates: int | None = None
    schedule: Literal["linear", "cosine"] = "linear"
    max_usd: Decimal = Decimal("3.0")
    max_hours: float = 14.0

    @property
    def uses_teacher(self) -> bool:
        return self.policy_weight > 0.0 or self.value_weight > 0.0


ARMS: tuple[Arm, ...] = (
    Arm(
        name="w64-ks-0.2-f200",
        label="w64 kickstart 0.2 (fade 200)",
        size="w64",
        seeds=W64_SEEDS,
        policy_weight=0.2,
        fade_updates=200,
        max_usd=Decimal("2.65"),
        max_hours=8.0,
    ),
    Arm(
        name="w64-ks-0.5-f200",
        label="w64 kickstart 0.5 (fade 200)",
        size="w64",
        seeds=W64_SEEDS,
        policy_weight=0.5,
        fade_updates=200,
        max_usd=Decimal("2.65"),
        max_hours=8.0,
    ),
    Arm(
        name="w160l8-fresh",
        label="w160l8 fresh (control)",
        size="w160l8",
        seeds=LARGE_SEEDS,
        max_usd=Decimal("5.20"),
        max_hours=16.0,
    ),
    Arm(
        name="w160l8-ks-0.2-f200",
        label="w160l8 kickstart 0.2 from w64 (fade 200)",
        size="w160l8",
        seeds=LARGE_SEEDS,
        policy_weight=0.2,
        fade_updates=200,
        max_usd=Decimal("5.20"),
        max_hours=16.0,
    ),
)
ARMS_BY_NAME: dict[str, Arm] = {arm.name: arm for arm in ARMS}

HELD_OUT = "tla-kickstart-held-out-32"
HALF_TRAINING = "tla-kickstart-half-training-32"
HALF_TRAINING_SEATINGS = tuple(
    f"Kickstart half-training deal {i + 1}" for i in range(32)
)
VS_TAU_X2 = "tla-kickstart-vs-tau-x2-32"
VS_TAU_X2_TAG = "vs-tau-x2"
VS_FRESH_LARGE = "tla-kickstart-w160l8-vs-fresh-32"
VS_FRESH_LARGE_SEATINGS = tuple(
    f"Kickstart w160l8 vs fresh deal {i + 1}" for i in range(32)
)
VS_TEACHER = "tla-kickstart-vs-teacher-32"
VS_TEACHER_TAG = "vs-teacher"
H2H_UPDATES: tuple[int, ...] = (100, 200, 300, 400)


# -- Training -----------------------------------------------------------------


class Config(trophies.Config):
    size: SizeLabel
    tau_scale: float = TAU_SCALE
    workers: Literal[1, 2, 4] = 4
    teacher_policy_weight: float = Field(
        default=0.0, exclude_if=lambda value: value == 0.0
    )
    teacher_value_weight: float = Field(
        default=0.0, exclude_if=lambda value: value == 0.0
    )
    fade_updates: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    kickstart_schedule: Literal["linear", "cosine"] = Field(
        default="linear", exclude_if=lambda value: value == "linear"
    )

    @property
    def uses_teacher(self) -> bool:
        return self.teacher_policy_weight > 0.0 or self.teacher_value_weight > 0.0


def settings(config: Config) -> SelfPlay:
    base = trophies.settings(config)
    kickstart = (
        Kickstart(
            policy_weight=config.teacher_policy_weight,
            value_weight=config.teacher_value_weight,
            schedule=config.kickstart_schedule,
            fade_updates=config.fade_updates,
        )
        if config.fade_updates is not None
        else None
    )
    return base.model_copy(
        update={
            "workers": config.workers,
            "model": model(config.size).model_copy(
                update={"focus_film": True, "stream_norm": True}
            ),
            "learning": base.learning.model_copy(
                update={
                    "tau_scale": config.tau_scale,
                    "kickstart": kickstart,
                }
            ),
        }
    )


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        run_settings = settings(config)
        if config.uses_teacher:
            run_settings = run_settings.model_copy(
                update={"teacher": run.input("teacher")}
            )
        self_play(run_settings, run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = UPDATES,
    attempts: int = 4,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = CHECKPOINT_EVERY,
    hours: float | None = None,
    usd: float | None = None,
) -> list[TrainingJob]:
    """Kickstarted w64 arms (2 seeds each) + w160l8 fresh vs kickstarted (seed 40902)."""
    env = tuple(
        Variable(name=name, value=str(threads))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    )
    teacher_inputs = checkpoint_inputs_for(TEACHER, "teacher")
    jobs: list[TrainingJob] = []
    for arm in ARMS:
        inputs = (
            (trophies.corpus_input(), *teacher_inputs)
            if arm.uses_teacher
            else (trophies.corpus_input(),)
        )
        smoke_fade = 1 if arm.uses_teacher else None
        jobs.append(
            TrainingJob(
                name=arm.name,
                train=train,
                config=Config(
                    size=arm.size,
                    updates=updates,
                    threads=threads,
                    workers=workers,
                    streams=streams,
                    rows=rows,
                    checkpoint_every=checkpoint_every,
                    teacher_policy_weight=arm.policy_weight,
                    teacher_value_weight=arm.value_weight,
                    fade_updates=arm.fade_updates,
                    kickstart_schedule=arm.schedule,
                ),
                smoke=Config(
                    size=arm.size,
                    updates=2,
                    device="cpu",
                    threads=1,
                    workers=1,
                    streams=2,
                    rows=64,
                    checkpoint_every=1,
                    teacher_policy_weight=arm.policy_weight,
                    teacher_value_weight=arm.value_weight,
                    fade_updates=smoke_fade,
                    kickstart_schedule=arm.schedule,
                ),
                environment=env,
                seeds=arm.seeds,
                repo=str(trophies.ROOT),
                revision=revision,
                setup=machine_setup(" --extra artifacts"),
                inputs=inputs,
                extras=("artifacts",),
                max_hours=hours if hours is not None else arm.max_hours,
                max_usd=Decimal(str(usd)) if usd is not None else arm.max_usd,
                max_attempts=attempts,
            )
        )
    return jobs


# -- Evaluation ---------------------------------------------------------------


def _arm_evaluations(
    runs: Sequence[TrainingRunSummary], arm: Arm, has_fresh_large: bool
) -> list[EvalJob]:
    held_out = trophies.held_out_jobs(runs)
    if not held_out:
        return []
    (shared,) = held_out
    benchmark = shared.tournament
    env = final_environment()
    one_seed = len(arm.seeds) == 1

    def against(
        name: str,
        checkpoints: Sequence[CheckpointRecord],
        seating_names: tuple[str, ...] | None = None,
        opponent: Fraction | Aligned | None = None,
    ) -> EvalJob:
        names = seating_names or tuple(s.name for s in benchmark.seatings)
        seatings = tuple(
            seating.model_copy(
                update={
                    "name": label,
                    "games": 1 if one_seed else seating.games,
                    **({"seeds": "one"} if one_seed else {}),
                    **(
                        {}
                        if opponent is None
                        else {"players": (seating.players[0], opponent)}
                    ),
                }
            )
            for seating, label in zip(benchmark.seatings, names, strict=True)
        )
        return shared.model_copy(
            update={
                "name": name,
                "tournament": benchmark.model_copy(
                    update={
                        "name": name,
                        "environment": env,
                        "seatings": seatings,
                        "rollups": CARD_ROLLUPS,
                    }
                ),
                "checkpoints": checkpoints,
            }
        )

    half = Fraction(clock="updates", value=0.5, trained=True)
    jobs = [
        against(f"{HELD_OUT}-{arm.name}", shared.checkpoints),
        against(
            f"{HALF_TRAINING}-{arm.name}",
            shared.checkpoints,
            HALF_TRAINING_SEATINGS,
            half,
        ),
    ]

    # w64 kickstarted arms vs retained R2 tau-x2 at matched updates 100..400.
    if arm.size == "w64":
        fixed_tau: list[Seating] = []
        for run in runs:
            by_update = {
                int(c.at["updates"]): c
                for c in run.checkpoints
                if "updates" in c.at
            }
            for update in H2H_UPDATES:
                candidate = by_update.get(update)
                if candidate is None:
                    continue
                for base_seed, opponent in snapshots.BASELINE_TAU_X2[update]:
                    fixed_tau.extend(
                        seating.model_copy(
                            update={
                                "name": (
                                    f"{run.name} u{update} vs tau-x2@{base_seed} "
                                    f"deal {i + 1}"
                                ),
                                "players": (
                                    checkpoint_player(candidate.ref),
                                    opponent,
                                ),
                                "games": 1,
                                "seeds": "one",
                                "tags": (VS_TAU_X2_TAG, arm.name, f"u{update}"),
                            }
                        )
                        for i, seating in enumerate(benchmark.seatings)
                    )
        if fixed_tau:
            jobs.append(
                shared.model_copy(
                    update={
                        "name": f"{VS_TAU_X2}-{arm.name}",
                        "tournament": benchmark.model_copy(
                            update={
                                "name": f"{VS_TAU_X2}-{arm.name}",
                                "environment": env,
                                "seatings": tuple(fixed_tau),
                                "rollups": CARD_ROLLUPS,
                            }
                        ),
                        "checkpoints": (),
                    }
                )
            )

    # w160l8 kickstarted vs w160l8 fresh at matched updates 100..400.
    if arm.name == "w160l8-ks-0.2-f200" and has_fresh_large:
        matched = [
            c for c in shared.checkpoints if c.at.get("updates") in H2H_UPDATES
        ]
        if matched:
            jobs.append(
                against(
                    VS_FRESH_LARGE,
                    matched,
                    VS_FRESH_LARGE_SEATINGS,
                    Aligned(clock="updates", job="w160l8-fresh"),
                )
            )

    # Endpoint vs the frozen w64 teacher (tau-x2@40902 u800).
    if arm.uses_teacher:
        teacher_seatings: list[Seating] = []
        for run in runs:
            by_update = {
                int(c.at["updates"]): c
                for c in run.checkpoints
                if "updates" in c.at
            }
            final = by_update.get(UPDATES)
            if final is None:
                continue
            teacher_seatings.extend(
                seating.model_copy(
                    update={
                        "name": f"{run.name} u{UPDATES} vs teacher deal {i + 1}",
                        "players": (checkpoint_player(final.ref), TEACHER_PLAYER),
                        "games": 1,
                        "seeds": "one",
                        "tags": (VS_TEACHER_TAG, arm.name),
                    }
                )
                for i, seating in enumerate(benchmark.seatings)
            )
        if teacher_seatings:
            jobs.append(
                shared.model_copy(
                    update={
                        "name": f"{VS_TEACHER}-{arm.name}",
                        "tournament": benchmark.model_copy(
                            update={
                                "name": f"{VS_TEACHER}-{arm.name}",
                                "environment": env,
                                "seatings": tuple(teacher_seatings),
                                "rollups": CARD_ROLLUPS,
                            }
                        ),
                        "checkpoints": (),
                    }
                )
            )
    return jobs


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    has_fresh_large = any(
        run.launch == "w160l8-fresh" and run.checkpoints for run in state.runs
    )
    return [
        job
        for arm in ARMS
        for job in _arm_evaluations(
            [run for run in state.runs if run.launch == arm.name],
            arm,
            has_fresh_large,
        )
    ]


# -- Report -------------------------------------------------------------------


def report() -> Report:
    held_out = tuple(Matches(tournament=f"{HELD_OUT}-{arm.name}") for arm in ARMS)
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        evals=(
            Curve(title="Score against the heuristic", matches=held_out),
            head_to_head("Against itself at half its updates", HALF_TRAINING_SEATINGS),
            Curve(
                title="w64 kickstarted vs stored tau-x2 (matched u100–u400)",
                matches=tuple(
                    Matches(tournament=f"{VS_TAU_X2}-{arm.name}")
                    for arm in ARMS
                    if arm.size == "w64"
                ),
            ),
            head_to_head(
                "w160l8 kickstarted vs w160l8 fresh (matched u100–u400)",
                VS_FRESH_LARGE_SEATINGS,
            ),
            Curve(
                title="Endpoint (u400) vs frozen w64 teacher (tau-x2 u800)",
                matches=tuple(
                    Matches(tournament=f"{VS_TEACHER}-{arm.name}")
                    for arm in ARMS
                    if arm.uses_teacher
                ),
            ),
            *behaviour_curves(matches=held_out),
        ),
        labels={arm.name: arm.label for arm in ARMS},
        perf=PERF,
        ml=ML,
    )
