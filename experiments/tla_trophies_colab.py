"""Train one w64 seed and one w160l8 seed together on Colab to ~500k games.

Applies the trophy-floor study's recommended default (`tau_scale=0.1`, no
forward-KL floor) over the complete TLA trophy split (7,859 training pools,
873 held-out pools) with both model sizes packed onto one two-slot Colab GPU.
At ~257 complete games per update (`streams=256, transitions=64`), 2,000 updates
is roughly 500k games per model.
"""

from collections.abc import Sequence
from decimal import Decimal
from itertools import zip_longest
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRecord
from etude.eval.records import (
    Aligned,
    CheckpointPlayer,
    Earlier,
    Fraction,
    checkpoint_player,
)
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.report import Curve, Matches, Report, Score
from etude.spec import Variable
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_trophies as trophies
from experiments.tla_frontier import SIZES, SizeLabel, learning, model
from experiments.tla_trophies_floor import final_environment
from manabot.eval.card_metrics import CARD_ROLLUPS
from manabot.experiment.setup import machine_setup
from manabot.training.learning import IterationSchedule
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves, head_to_head

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


SEED = 40902
# ~257 complete games per update at 256 streams × 64-decision floor ≈ 514k games.
TARGET_UPDATES = 2_000
HELD_OUT = "tla-trophies-colab-held-out-32"
HELD_OUT_EXTRA = "tla-trophies-colab-held-out-extra-128"
EXTRA_AFTER_UPDATES = 400
EXTRA_PAIRS = range(32, 160)
HALF_TRAINING = "tla-trophies-colab-half-training-32"
HALF_TRAINING_EXTRA = "tla-trophies-colab-half-training-extra-128"
HALF_TRAINING_SEATINGS = tuple(f"Colab half-training deal {i + 1}" for i in range(32))
HALF_TRAINING_EXTRA_SEATINGS = tuple(
    f"Colab half-training deal {i + 1}" for i in EXTRA_PAIRS
)
LAG_200 = "tla-trophies-colab-lag-200-32"
LAG_200_EXTRA = "tla-trophies-colab-lag-200-extra-128"
LAG_200_EXTRA_MIN_UPDATES = 1_000
LAG_200_SEATINGS = tuple(f"Colab -200 updates deal {i + 1}" for i in range(32))
LAG_200_EXTRA_SEATINGS = tuple(f"Colab -200 updates deal {i + 1}" for i in EXTRA_PAIRS)
LARGE_VS_SMALL = "tla-trophies-colab-w160l8-vs-w64-32"
LARGE_VS_SMALL_EXTRA = "tla-trophies-colab-w160l8-vs-w64-extra-128"
LARGE_VS_SMALL_SEATINGS = tuple(f"Colab w160l8 vs w64 deal {i + 1}" for i in range(32))
LARGE_VS_SMALL_EXTRA_SEATINGS = tuple(
    f"Colab w160l8 vs w64 deal {i + 1}" for i in EXTRA_PAIRS
)
LARGE_VS_SMALL_EQUAL_TIME = "tla-trophies-colab-w160l8-vs-w64-equal-time-32"
LARGE_VS_SMALL_EQUAL_TIME_EXTRA = (
    "tla-trophies-colab-w160l8-vs-w64-equal-time-extra-128"
)
LARGE_VS_SMALL_EQUAL_TIME_SEATINGS = tuple(
    f"Colab w160l8 vs w64 equal-time deal {i + 1}" for i in range(32)
)
LARGE_VS_SMALL_EQUAL_TIME_EXTRA_SEATINGS = tuple(
    f"Colab w160l8 vs w64 equal-time deal {i + 1}" for i in EXTRA_PAIRS
)
EXTRA_PREFIXES = (
    f"{HELD_OUT_EXTRA}-",
    f"{HALF_TRAINING_EXTRA}-",
    f"{LAG_200_EXTRA}-",
    f"{LARGE_VS_SMALL_EXTRA}-",
    f"{LARGE_VS_SMALL_EQUAL_TIME_EXTRA}-",
)


class Config(trophies.Config):
    size: SizeLabel
    schedule_steps: int = TARGET_UPDATES
    tau_scale: float = 0.1
    workers: Literal[1, 2, 4] = Field(default=1, exclude_if=lambda value: value == 1)


def settings(config: Config) -> "SelfPlay":
    base = trophies.settings(config)
    return base.model_copy(
        update={
            "workers": config.workers,
            "model": model(config.size).model_copy(
                update={"focus_film": True, "stream_norm": True}
            ),
            "learning": learning("w64").model_copy(
                update={
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
    hours: float = 80.0,
    usd_per_run: float = 10.0,
    attempts: int = 12,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = 100,
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name=size,
            train=train,
            config=Config(
                size=size,
                updates=updates,
                schedule_steps=updates,
                threads=threads,
                workers=workers,
                streams=streams,
                rows=rows,
                checkpoint_every=checkpoint_every,
            ),
            smoke=Config(
                size=size,
                updates=1,
                schedule_steps=TARGET_UPDATES,
                device="cpu",
                threads=1,
                streams=2,
                rows=64,
                checkpoint_every=1,
            ),
            environment=tuple(
                Variable(name=name, value=str(threads))
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            ),
            seeds=(SEED,),
            repo=str(trophies.ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            inputs=(trophies.corpus_input(),),
            extras=("artifacts",),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=attempts,
        )
        for size in SIZES
    ]


def equal_time_partner(
    target_seconds: float,
    small_checkpoints: Sequence[CheckpointRecord],
    small_ended: bool,
) -> CheckpointRecord | None:
    """Nearest trained w64 checkpoint in `training_seconds`, fixed once w64 passes `target_seconds`."""
    timed = [
        (c.at["training_seconds"], c)
        for c in small_checkpoints
        if "training_seconds" in c.at and c.at.get("updates", 0) > 0
    ]
    if not timed:
        return None
    if not small_ended and all(seconds < target_seconds for seconds, _ in timed):
        return None
    _, partner = min(
        timed, key=lambda item: (abs(item[0] - target_seconds), item[0])
    )
    return partner


def size_evaluations(
    runs: Sequence[TrainingRunSummary],
    size: SizeLabel,
    small_runs: Sequence[TrainingRunSummary] = (),
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
    ) -> EvalJob:
        names = seating_names or tuple(s.name for s in benchmark.seatings)
        seatings = tuple(
            seating.model_copy(
                update={
                    "name": label,
                    "games": 1,
                    "seeds": "one",
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

    def extra_against(
        prefix: str,
        checkpoints: Sequence[CheckpointRecord],
        seating_names: tuple[str, ...] | None = None,
        opponent: Fraction | Aligned | CheckpointPlayer | None = None,
        tags: tuple[str, ...] = (),
        min_updates: int = EXTRA_AFTER_UPDATES + 1,
    ) -> list[EvalJob]:
        eligible = sorted(
            (c for c in checkpoints if c.at.get("updates", 0) >= min_updates),
            key=lambda c: c.at["updates"],
            reverse=True,
        )
        if not eligible:
            return []
        names = seating_names or tuple(f"Held-out pair {i + 1}" for i in EXTRA_PAIRS)
        seatings = tuple(
            benchmark.seatings[0].model_copy(
                update={
                    "name": label,
                    "config": trophies.Deal(
                        first=i * 2, second=i * 2 + 1, seed=40900 + i
                    ),
                    "games": 1,
                    "seeds": "one",
                    "tags": tags,
                    **(
                        {}
                        if opponent is None
                        else {"players": (benchmark.seatings[0].players[0], opponent)}
                    ),
                }
            )
            for i, label in zip(EXTRA_PAIRS, names, strict=True)
        )
        return [
            shared.model_copy(
                update={
                    "name": f"{prefix}-u{int(ckpt.at['updates'])}",
                    "tournament": benchmark.model_copy(
                        update={
                            "name": f"{prefix}-u{int(ckpt.at['updates'])}",
                            "environment": env,
                            "seatings": seatings,
                            "rollups": CARD_ROLLUPS,
                        }
                    ),
                    "checkpoints": (ckpt,),
                    "max_hours": 3,
                }
            )
            for ckpt in eligible
        ]

    trained_updates = {
        c.at["updates"] for c in shared.checkpoints if c.at.get("updates", 0) > 0
    }
    half_checkpoints = [
        c
        for c in shared.checkpoints
        if c.at.get("updates", 0) > 0 and (c.at["updates"] / 2) in trained_updates
    ]
    lag_200_checkpoints = [
        c
        for c in shared.checkpoints
        if c.at.get("updates", 0) > 0 and (c.at["updates"] - 200) in trained_updates
    ]
    half = Fraction(clock="updates", value=0.5, trained=True)
    lag_200 = Earlier(clock="updates", by=200, trained=True)
    jobs = [
        against(f"{HELD_OUT}-{size}", shared.checkpoints),
        *extra_against(
            f"{HELD_OUT_EXTRA}-{size}",
            shared.checkpoints,
            tags=(f"{HELD_OUT_EXTRA}-{size}",),
        ),
    ]
    if half_checkpoints:
        jobs.append(
            against(
                f"{HALF_TRAINING}-{size}",
                half_checkpoints,
                HALF_TRAINING_SEATINGS,
                half,
            )
        )
        jobs.extend(
            extra_against(
                f"{HALF_TRAINING_EXTRA}-{size}",
                half_checkpoints,
                HALF_TRAINING_EXTRA_SEATINGS,
                half,
            )
        )
    if lag_200_checkpoints:
        jobs.append(
            against(
                f"{LAG_200}-{size}",
                lag_200_checkpoints,
                LAG_200_SEATINGS,
                lag_200,
            )
        )
    if size == "w160l8":
        trained = [c for c in shared.checkpoints if c.at.get("updates", 0) > 0]
        if trained:
            aligned_w64 = Aligned(clock="updates", job="w64")
            jobs.append(
                against(
                    LARGE_VS_SMALL,
                    trained,
                    LARGE_VS_SMALL_SEATINGS,
                    aligned_w64,
                )
            )
            jobs.extend(
                extra_against(
                    LARGE_VS_SMALL_EXTRA,
                    trained,
                    LARGE_VS_SMALL_EXTRA_SEATINGS,
                    aligned_w64,
                )
            )
            small_checkpoints = [
                c for run in small_runs for c in run.checkpoints
            ]
            small_ended = bool(small_runs) and all(run.ended for run in small_runs)
            equal_time_pairs: list[tuple[CheckpointRecord, CheckpointRecord]] = []
            for large_ckpt in trained:
                seconds = large_ckpt.at.get("training_seconds")
                if seconds is None:
                    continue
                partner = equal_time_partner(seconds, small_checkpoints, small_ended)
                if partner is None:
                    continue
                equal_time_pairs.append((large_ckpt, partner))
                update = int(large_ckpt.at["updates"])
                jobs.append(
                    against(
                        f"{LARGE_VS_SMALL_EQUAL_TIME}-u{update}",
                        (large_ckpt,),
                        LARGE_VS_SMALL_EQUAL_TIME_SEATINGS,
                        checkpoint_player(partner.ref),
                    )
                )
            for large_ckpt, partner in sorted(
                equal_time_pairs,
                key=lambda pair: pair[0].at["updates"],
                reverse=True,
            ):
                jobs.extend(
                    extra_against(
                        LARGE_VS_SMALL_EQUAL_TIME_EXTRA,
                        (large_ckpt,),
                        LARGE_VS_SMALL_EQUAL_TIME_EXTRA_SEATINGS,
                        checkpoint_player(partner.ref),
                    )
                )
    if lag_200_checkpoints:
        jobs.extend(
            extra_against(
                f"{LAG_200_EXTRA}-{size}",
                lag_200_checkpoints,
                LAG_200_EXTRA_SEATINGS,
                lag_200,
                min_updates=LAG_200_EXTRA_MIN_UPDATES,
            )
        )
    return jobs


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    small_runs = [run for run in state.runs if run.launch == "w64"]
    by_size = {
        size: size_evaluations(
            [run for run in state.runs if run.launch == size],
            size,
            small_runs=small_runs,
        )
        for size in SIZES
    }
    base = [
        job
        for size in SIZES
        for job in by_size[size]
        if not job.name.startswith(EXTRA_PREFIXES)
    ]
    # Interleave extra post-400 eval jobs latest-first across sizes.
    extras_by_size = [
        [job for job in by_size[size] if job.name.startswith(EXTRA_PREFIXES)]
        for size in ("w160l8", "w64")
    ]
    extras = [
        job
        for group in zip_longest(*extras_by_size)
        for job in group
        if job is not None
    ]
    return [*base, *extras]


def report() -> Report:
    held_out = tuple(
        selection
        for size in SIZES
        for selection in (
            Matches(tournament=f"{HELD_OUT}-{size}"),
            Matches(tag=f"{HELD_OUT_EXTRA}-{size}"),
        )
    )
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        evals=(
            Curve(title="Score against the heuristic", matches=held_out),
            head_to_head(
                "Against itself at half its updates",
                (*HALF_TRAINING_SEATINGS, *HALF_TRAINING_EXTRA_SEATINGS),
            ),
            head_to_head(
                "Against itself 200 updates earlier",
                (*LAG_200_SEATINGS, *LAG_200_EXTRA_SEATINGS),
            ),
            head_to_head(
                "Large (w160l8) against Small (w64) at equal training time",
                (
                    *LARGE_VS_SMALL_EQUAL_TIME_SEATINGS,
                    *LARGE_VS_SMALL_EQUAL_TIME_EXTRA_SEATINGS,
                ),
            ),
            head_to_head(
                "Large (w160l8) against Small (w64) at equal updates",
                (*LARGE_VS_SMALL_SEATINGS, *LARGE_VS_SMALL_EXTRA_SEATINGS),
            ),
            *behaviour_curves(matches=held_out),
        ),
        labels={"w64": "Small (w64)", "w160l8": "Large (w160l8)"},
        perf=PERF,
        ml=ML,
    )
