"""Exploratory L4 comparison of complete games, exact quota, and cohort overlap.

Both model sizes start from fresh weights on the TLA trophy corpus. The
`w160l8` u2000 checkpoint has a different input vocabulary.

Each arm runs 5 updates on one L4 with 256 streams and 4 native workers.
Exact-quota uses `transitions=186` (47,616 rows/update) to match complete-games'
~47.7k rows/update at `w160l8` u2000.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRecord
from etude.jobs import TrainingJob
from etude.records import ArtifactRef
from etude.report import Report
from etude.spec import Variable
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_trophies as trophies
from experiments.tla_frontier import learning, model
from manabot.experiment.setup import machine_setup
from manabot.infra.config import ModelConfig
from manabot.training.learning import BatchClosure, IterationSchedule
from manabot.training.panels import ML, PERF

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


SEED = 40902
UPDATES = 5
EXACT_TRANSITIONS = 186

# Late-training policy from `tla_trophies_colab_500k_v2/w160l8@40902` at update 2000.
W160L8_U2000 = CheckpointRecord(
    experiment="tla_trophies_colab_500k_v2",
    run="w160l8@40902",
    ordinal=21,
    at={"updates": 2000},
    file="out/policies/00002000-raw.safetensors",
    artifact=ArtifactRef(
        uri="r2://manabot/jobs/tla_trophies_colab_500k_v2/w160l8@40902/out/policies/00002000-raw.safetensors",
        sha256="26ff27b76cffd9a1076d5224b81886560262ee48a575c9cde5b9026520951f37",
        size_bytes=27_688_498,
    ),
)

type BenchSize = Literal["w160l8", "w384l8"]


@dataclass(frozen=True)
class Arm:
    size: BenchSize
    batch_closure: BatchClosure
    transitions: int
    rows: int
    warm_start: bool
    collection_pipeline: bool = False


ARMS: dict[str, Arm] = {
    "w160l8-fresh-complete": Arm("w160l8", "complete-games", 64, 512, False),
    "w160l8-fresh-exact": Arm("w160l8", "exact-quota", EXACT_TRANSITIONS, 512, False),
    "w160l8-fresh-exact-pipe": Arm(
        "w160l8", "exact-quota", EXACT_TRANSITIONS, 512, False, collection_pipeline=True
    ),
    "w384l8-complete": Arm("w384l8", "complete-games", 64, 128, False),
    "w384l8-exact": Arm("w384l8", "exact-quota", EXACT_TRANSITIONS, 128, False),
    "w384l8-exact-pipe": Arm(
        "w384l8", "exact-quota", EXACT_TRANSITIONS, 128, False, collection_pipeline=True
    ),
}


class Config(trophies.Config):
    size: BenchSize
    batch_closure: BatchClosure
    transitions: int = Field(ge=1)
    warm_start: bool
    collection_pipeline: bool = Field(
        default=False, exclude_if=lambda enabled: not enabled
    )
    workers: Literal[1, 2, 4] = 4


def _model_config(size: BenchSize) -> ModelConfig:
    base = model("w160l8").model_copy(update={"focus_film": True, "stream_norm": True})
    if size == "w384l8":
        return base.model_copy(
            update={"hidden_dim": 384, "attention_layers": 8, "num_attention_heads": 8}
        )
    return base


def settings(config: Config) -> SelfPlay:
    base = trophies.settings(config)
    return base.model_copy(
        update={
            "workers": config.workers,
            "transitions": config.transitions,
            "batch_closure": config.batch_closure,
            "collection_pipeline": config.collection_pipeline,
            "model": _model_config(config.size),
            "learning": learning("w64").model_copy(
                update={
                    "tau_scale": 0.1,
                    "iteration_schedule": IterationSchedule(
                        steps=2000,
                        reference_steps=10_000,
                    ),
                }
            ),
        }
    )


def _recipe(name: str) -> SelfPlay:
    arm = ARMS[name]
    return settings(
        Config(
            size=arm.size,
            batch_closure=arm.batch_closure,
            transitions=arm.transitions,
            warm_start=arm.warm_start,
            collection_pipeline=arm.collection_pipeline,
            updates=UPDATES,
            threads=2,
            workers=4,
            streams=256,
            rows=arm.rows,
            checkpoint_every=UPDATES,
            device="cuda",
        )
    )


def w160l8_complete() -> SelfPlay:
    return _recipe("w160l8-fresh-complete")


def w160l8_exact() -> SelfPlay:
    return _recipe("w160l8-fresh-exact")


def w384l8_complete() -> SelfPlay:
    return _recipe("w384l8-complete")


def w384l8_exact() -> SelfPlay:
    return _recipe("w384l8-exact")


def w160l8_exact_pipe() -> SelfPlay:
    return _recipe("w160l8-fresh-exact-pipe")


def w384l8_exact_pipe() -> SelfPlay:
    return _recipe("w384l8-exact-pipe")


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        built = settings(config)
        if config.warm_start:
            built = built.model_copy(update={"pretrained": run.input("start")})
        self_play(built, run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = UPDATES,
    hours: float = 1.0,
    usd_per_run: float = 1.50,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
) -> list[TrainingJob]:
    jobs: list[TrainingJob] = []
    for name, arm in ARMS.items():
        jobs.append(
            TrainingJob(
                name=name,
                train=train,
                config=Config(
                    size=arm.size,
                    batch_closure=arm.batch_closure,
                    transitions=arm.transitions,
                    warm_start=arm.warm_start,
                    collection_pipeline=arm.collection_pipeline,
                    updates=updates,
                    threads=threads,
                    workers=workers,
                    streams=streams,
                    rows=arm.rows,
                    checkpoint_every=updates,
                    device="cuda",
                ),
                smoke=Config(
                    size=arm.size,
                    batch_closure=arm.batch_closure,
                    transitions=4 if arm.batch_closure == "exact-quota" else 1,
                    warm_start=arm.warm_start,
                    collection_pipeline=arm.collection_pipeline,
                    updates=2 if arm.batch_closure == "exact-quota" else 1,
                    threads=1,
                    workers=1,
                    streams=2,
                    rows=64,
                    checkpoint_every=2,
                    device="cpu",
                ),
                environment=tuple(
                    Variable(name=var, value=str(threads))
                    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
                ),
                seeds=(SEED,),
                repo=str(trophies.ROOT),
                revision=revision,
                setup=machine_setup(" --extra artifacts"),
                inputs=(trophies.corpus_input(),),
                start=W160L8_U2000 if arm.warm_start else None,
                extras=("artifacts",),
                max_hours=hours,
                max_usd=Decimal(str(usd_per_run)),
                max_attempts=1,
            )
        )
    return jobs


def report() -> Report:
    return Report(
        clock="updates",
        labels={name: name for name in ARMS},
        perf=PERF,
        ml=ML,
    )
