"""Exploratory TLA trophy w160l8 learning-utilization screen.

Evaluates three single-factor increases to optimizer transition utilization
against the incumbent `tla_trophies_colab` `w160l8` recipe (which already
collects both seats via `learning_seats="both"` and retains the top 25% of
`|advantage|` rows for one damped-KL pass):

- `control`: incumbent (`advantage_quantile=0.75, min_advantage=0.01, epochs=1`)
- `keep-50`: retain top 50% (`advantage_quantile=0.50, min_advantage=0.01, epochs=1`)
- `keep-all`: retain 100% (`advantage_quantile=0.0, min_advantage=0.0, epochs=1`)
- `epochs-2`: two damped-KL passes over the top 25% (`advantage_quantile=0.75, epochs=2`)

Two paired training seeds (`40901`, `40902`) train each arm to 400 updates on
the complete 7,859-pool TLA trophy split with `schedule_steps=2000` (matching
`tla_trophies_colab`'s LR/tau trajectory over `u0..u400` and allowing seamless
resumption to `u2000` if a winner emerges). Held-out evaluations score all arms
against the first-64 heuristic, against themselves at half updates, and
head-to-head against `control` at both equal updates and equal active training
time (`training_seconds`), with 160-deal (640-game/seed) density at `u400` and
the final overlapping equal-time point.
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
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.report import Curve, Matches, Report, Score
from etude.spec import Variable
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_trophies as trophies
from experiments.tla_frontier import learning, model
from experiments.tla_trophies_floor import final_environment
from manabot.eval.card_metrics import CARD_ROLLUPS
from manabot.experiment.setup import machine_setup
from manabot.training.learning import IterationSchedule
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves, head_to_head

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


SEEDS: tuple[int, ...] = (40901, 40902)
TARGET_UPDATES = 400
SCHEDULE_STEPS = 2_000
CHECKPOINT_EVERY = 50
H2H_UPDATES: tuple[int, ...] = (100, 200, 300, 400)
EXTRA_PAIRS = range(32, 160)

HELD_OUT = "tla-util-held-out-32"
HELD_OUT_EXTRA = "tla-util-held-out-extra-128"
HALF_TRAINING = "tla-util-half-training-32"
HALF_TRAINING_SEATINGS = tuple(f"Util half-training deal {i + 1}" for i in range(32))
VS_CONTROL_UPDATES = "tla-util-vs-control-updates-32"
VS_CONTROL_UPDATES_EXTRA = "tla-util-vs-control-updates-extra-128"
VS_CONTROL_UPDATES_SEATINGS = tuple(
    f"Util vs control equal-updates deal {i + 1}" for i in range(32)
)
VS_CONTROL_UPDATES_EXTRA_SEATINGS = tuple(
    f"Util vs control equal-updates deal {i + 1}" for i in EXTRA_PAIRS
)
VS_CONTROL_TIME = "tla-util-vs-control-time-32"
VS_CONTROL_TIME_EXTRA = "tla-util-vs-control-time-extra-128"
VS_CONTROL_TIME_SEATINGS = tuple(
    f"Util vs control equal-time deal {i + 1}" for i in range(32)
)
VS_CONTROL_TIME_EXTRA_SEATINGS = tuple(
    f"Util vs control equal-time deal {i + 1}" for i in EXTRA_PAIRS
)


@dataclass(frozen=True)
class Arm:
    label: str
    advantage_quantile: float = 0.75
    min_advantage: float = 0.01
    epochs: int = 1


ARMS: dict[str, Arm] = {
    "control": Arm("Control (q75, 1ep)"),
    "keep-50": Arm("Keep 50% (q50, 1ep)", advantage_quantile=0.50),
    "keep-all": Arm("Keep 100% (q0, 1ep)", advantage_quantile=0.0, min_advantage=0.0),
    "epochs-2": Arm("2 epochs (q75, 2ep)", epochs=2),
}


class Config(trophies.Config):
    advantage_quantile: float = 0.75
    min_advantage: float = 0.01
    epochs: int = Field(default=1, ge=1, exclude_if=lambda value: value == 1)
    schedule_steps: int = SCHEDULE_STEPS
    tau_scale: float = 0.1
    workers: Literal[1, 2, 4] = Field(default=1, exclude_if=lambda value: value == 1)


def settings(config: Config) -> "SelfPlay":
    base = trophies.settings(config)
    return base.model_copy(
        update={
            "workers": config.workers,
            "model": model("w160l8").model_copy(
                update={"focus_film": True, "stream_norm": True}
            ),
            "learning": learning("w64").model_copy(
                update={
                    "advantage_quantile": config.advantage_quantile,
                    "min_advantage": config.min_advantage,
                    "epochs": config.epochs,
                    "tau_scale": config.tau_scale,
                    "iteration_schedule": IterationSchedule(
                        steps=config.schedule_steps,
                        reference_steps=10_000,
                    ),
                }
            ),
        }
    )


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = TARGET_UPDATES,
    schedule_steps: int = SCHEDULE_STEPS,
    hours: float = 22.0,
    usd_per_run: float = 3.00,
    attempts: int = 4,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name=name,
            train=train,
            config=Config(
                updates=updates,
                schedule_steps=schedule_steps,
                advantage_quantile=arm.advantage_quantile,
                min_advantage=arm.min_advantage,
                epochs=arm.epochs,
                threads=threads,
                workers=workers,
                streams=streams,
                rows=rows,
                checkpoint_every=checkpoint_every,
            ),
            smoke=Config(
                updates=1,
                schedule_steps=schedule_steps,
                advantage_quantile=arm.advantage_quantile,
                min_advantage=arm.min_advantage,
                epochs=arm.epochs,
                device="cpu",
                threads=1,
                streams=2,
                rows=64,
                checkpoint_every=1,
            ),
            environment=tuple(
                Variable(name=var, value=str(threads))
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            ),
            seeds=SEEDS,
            repo=str(trophies.ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            inputs=(trophies.corpus_input(),),
            extras=("artifacts",),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=attempts,
        )
        for name, arm in ARMS.items()
    ]


def _extra_seatings(
    template: Seating,
    names: tuple[str, ...],
    *,
    games: int,
    one_seed: bool,
    opponent: Aligned | CheckpointPlayer | None = None,
    tags: tuple[str, ...] = (),
) -> tuple[Seating, ...]:
    return tuple(
        template.model_copy(
            update={
                "name": label,
                "config": trophies.Deal(first=i * 2, second=i * 2 + 1, seed=40900 + i),
                "games": games,
                "tags": tags,
                **({"seeds": "one"} if one_seed else {}),
                **(
                    {}
                    if opponent is None
                    else {"players": (template.players[0], opponent)}
                ),
            }
        )
        for i, label in zip(EXTRA_PAIRS, names, strict=True)
    )


def equal_time_partner(
    target_seconds: float,
    candidates: Sequence[CheckpointRecord],
    ended: bool,
) -> CheckpointRecord | None:
    """Nearest trained checkpoint in `training_seconds` once the run covers `target_seconds`.

    Returns None while the candidate run is still running below `target_seconds`,
    or if it ended before reaching half of `target_seconds` (no extrapolation).
    """
    timed = [
        (float(c.at["training_seconds"]), c)
        for c in candidates
        if "training_seconds" in c.at and c.at.get("updates", 0) > 0
    ]
    if not timed:
        return None
    if not ended and all(seconds < target_seconds for seconds, _ in timed):
        return None
    if ended and max(seconds for seconds, _ in timed) < target_seconds * 0.5:
        return None
    _, partner = min(timed, key=lambda item: (abs(item[0] - target_seconds), item[0]))
    return partner


def arm_evaluations(
    runs: Sequence[TrainingRunSummary],
    arm: str,
    control_runs: Sequence[TrainingRunSummary],
) -> list[EvalJob]:
    held_out = trophies.held_out_jobs(runs)
    if not held_out:
        return []
    (shared,) = held_out
    benchmark = shared.tournament
    env = final_environment()

    def against(
        name: str,
        checkpoints: Sequence[CheckpointRecord],
        seating_names: tuple[str, ...] | None = None,
        opponent: Fraction | Aligned | CheckpointPlayer | None = None,
        opponent_seeds: int = 1,
        one_seed: bool = False,
        base: tuple[Seating, ...] = benchmark.seatings,
        tags: tuple[str, ...] = (),
    ) -> EvalJob:
        names = seating_names or tuple(s.name for s in base)
        seatings = tuple(
            seating.model_copy(
                update={
                    "name": label,
                    "games": 1 if one_seed else seating.games * opponent_seeds,
                    "tags": tags,
                    **({"seeds": "one"} if one_seed else {}),
                    **(
                        {}
                        if opponent is None
                        else {"players": (seating.players[0], opponent)}
                    ),
                }
            )
            for seating, label in zip(base, names, strict=True)
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
    half_eligible = [
        c for c in shared.checkpoints if c.at.get("updates", 0) in H2H_UPDATES
    ]
    jobs = [against(f"{HELD_OUT}-{arm}", shared.checkpoints)]
    if half_eligible:
        jobs.append(
            against(
                f"{HALF_TRAINING}-{arm}",
                half_eligible,
                HALF_TRAINING_SEATINGS,
                half,
            )
        )

    endpoint = [c for c in shared.checkpoints if c.at.get("updates") == TARGET_UPDATES]
    if endpoint:
        extra_heur = _extra_seatings(
            benchmark.seatings[0],
            tuple(f"Held-out pair {i + 1}" for i in EXTRA_PAIRS),
            games=benchmark.seatings[0].games,
            one_seed=False,
            tags=(f"{HELD_OUT_EXTRA}-{arm}",),
        )
        jobs.append(
            shared.model_copy(
                update={
                    "name": f"{HELD_OUT_EXTRA}-{arm}-u{TARGET_UPDATES}",
                    "tournament": benchmark.model_copy(
                        update={
                            "name": f"{HELD_OUT_EXTRA}-{arm}-u{TARGET_UPDATES}",
                            "environment": env,
                            "seatings": extra_heur,
                            "rollups": CARD_ROLLUPS,
                        }
                    ),
                    "checkpoints": endpoint,
                }
            )
        )

    control_seeds = len({run.seed for run in control_runs})
    if arm != "control" and control_seeds:
        compared = [
            c for c in shared.checkpoints if c.at.get("updates", 0) in H2H_UPDATES
        ]
        if compared:
            jobs.append(
                against(
                    f"{VS_CONTROL_UPDATES}-{arm}",
                    compared,
                    VS_CONTROL_UPDATES_SEATINGS,
                    Aligned(clock="updates", job="control"),
                    opponent_seeds=control_seeds,
                )
            )
        if endpoint:
            extra_h2h = _extra_seatings(
                benchmark.seatings[0],
                VS_CONTROL_UPDATES_EXTRA_SEATINGS,
                games=benchmark.seatings[0].games * control_seeds,
                one_seed=False,
                opponent=Aligned(clock="updates", job="control"),
            )
            jobs.append(
                shared.model_copy(
                    update={
                        "name": f"{VS_CONTROL_UPDATES_EXTRA}-{arm}-u{TARGET_UPDATES}",
                        "tournament": benchmark.model_copy(
                            update={
                                "name": f"{VS_CONTROL_UPDATES_EXTRA}-{arm}-u{TARGET_UPDATES}",
                                "environment": env,
                                "seatings": extra_h2h,
                                "rollups": CARD_ROLLUPS,
                            }
                        ),
                        "checkpoints": endpoint,
                    }
                )
            )

        control_by_seed = {run.seed: run for run in control_runs}
        for run in runs:
            control_run = control_by_seed.get(run.seed)
            if control_run is None:
                continue
            treated_ckpts = list(run.checkpoints)
            control_milestones = sorted(
                (
                    c
                    for c in control_run.checkpoints
                    if c.at.get("updates", 0) in H2H_UPDATES
                    and "training_seconds" in c.at
                ),
                key=lambda c: c.at["updates"],
            )
            matched_pairs: list[tuple[int, CheckpointRecord, CheckpointRecord]] = []
            seen_pairs: set[tuple[int, int]] = set()
            for ctrl_ckpt in control_milestones:
                ctrl_update = int(ctrl_ckpt.at["updates"])
                target_seconds = float(ctrl_ckpt.at["training_seconds"])
                partner = equal_time_partner(target_seconds, treated_ckpts, run.ended)
                if partner is None:
                    continue
                pair_key = (int(partner.at["updates"]), ctrl_update)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                matched_pairs.append((ctrl_update, partner, ctrl_ckpt))
                jobs.append(
                    against(
                        f"{VS_CONTROL_TIME}-{arm}-s{run.seed}-c{ctrl_update}",
                        (partner,),
                        VS_CONTROL_TIME_SEATINGS,
                        checkpoint_player(ctrl_ckpt.ref),
                        one_seed=True,
                    )
                )
            # Dense 128-deal follow-up at the final overlapping equal-time point.
            if matched_pairs and control_run.ended:
                ctrl_update, partner, ctrl_ckpt = matched_pairs[-1]
                extra_time = _extra_seatings(
                    benchmark.seatings[0],
                    VS_CONTROL_TIME_EXTRA_SEATINGS,
                    games=1,
                    one_seed=True,
                    opponent=checkpoint_player(ctrl_ckpt.ref),
                )
                jobs.append(
                    shared.model_copy(
                        update={
                            "name": f"{VS_CONTROL_TIME_EXTRA}-{arm}-s{run.seed}-c{ctrl_update}",
                            "tournament": benchmark.model_copy(
                                update={
                                    "name": f"{VS_CONTROL_TIME_EXTRA}-{arm}-s{run.seed}-c{ctrl_update}",
                                    "environment": env,
                                    "seatings": extra_time,
                                    "rollups": CARD_ROLLUPS,
                                }
                            ),
                            "checkpoints": (partner,),
                        }
                    )
                )

    return jobs


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    control_runs = [run for run in state.runs if run.launch == "control"]
    return [
        job
        for arm in ARMS
        for job in arm_evaluations(
            [run for run in state.runs if run.launch == arm],
            arm,
            control_runs,
        )
    ]


def report() -> Report:
    held_out = tuple(
        selection
        for arm in ARMS
        for selection in (
            Matches(tournament=f"{HELD_OUT}-{arm}"),
            Matches(tag=f"{HELD_OUT_EXTRA}-{arm}"),
        )
    )
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        evals=(
            Curve(title="Score against the heuristic", matches=held_out),
            head_to_head("Against itself at half its updates", HALF_TRAINING_SEATINGS),
            head_to_head(
                "Against control at equal updates",
                (*VS_CONTROL_UPDATES_SEATINGS, *VS_CONTROL_UPDATES_EXTRA_SEATINGS),
            ),
            head_to_head(
                "Against control at equal training time",
                (*VS_CONTROL_TIME_SEATINGS, *VS_CONTROL_TIME_EXTRA_SEATINGS),
            ),
            *behaviour_curves(matches=held_out),
        ),
        baseline="control",
        labels={name: arm.label for name, arm in ARMS.items()},
        perf=PERF,
        ml=ML,
    )
