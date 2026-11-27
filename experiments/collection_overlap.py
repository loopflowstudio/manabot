"""Exploratory L4 throughput comparison: complete-games vs two-seat exact-quota.

Compares `batch_closure="complete-games"` against `batch_closure="exact-quota"`
on the TLA trophy corpus at `w160l8` and `w384l8`, both from fresh weights: the
`tla_trophies_colab_500k_v2/w160l8@40902` u2000 checkpoint was written at
`v0.2.4+10` and its vocabulary no longer loads strictly on HEAD.

Each arm runs 5 updates on one L4 with 256 streams and 4 native workers.
Exact-quota uses `transitions=186` (47,616 rows/update) to match complete-games'
~47.7k rows/update at late-training game lengths. Results:
`experiments/reports/collection_overlap.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from etude.jobs import TrainingJob
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

type BenchSize = Literal["w160l8", "w384l8"]


@dataclass(frozen=True)
class Arm:
    size: BenchSize
    batch_closure: BatchClosure
    transitions: int
    rows: int


ARMS: dict[str, Arm] = {
    "w160l8-fresh-complete": Arm("w160l8", "complete-games", 64, 512),
    "w160l8-fresh-exact": Arm("w160l8", "exact-quota", EXACT_TRANSITIONS, 512),
    "w384l8-complete": Arm("w384l8", "complete-games", 64, 128),
    "w384l8-exact": Arm("w384l8", "exact-quota", EXACT_TRANSITIONS, 128),
}


class Config(trophies.Config):
    size: BenchSize
    batch_closure: BatchClosure
    transitions: int = Field(ge=1)
    workers: Literal[1, 2, 4] = 4


def _model_config(size: BenchSize) -> ModelConfig:
    base = model("w160l8").model_copy(update={"focus_film": True, "stream_norm": True})
    if size == "w384l8":
        return base.model_copy(
            update={"hidden_dim": 384, "attention_layers": 8, "num_attention_heads": 8}
        )
    return base


def _arm_config(
    arm: Arm,
    *,
    updates: int = UPDATES,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
) -> Config:
    return Config(
        size=arm.size,
        batch_closure=arm.batch_closure,
        transitions=arm.transitions,
        updates=updates,
        threads=threads,
        workers=workers,
        streams=streams,
        rows=arm.rows,
        checkpoint_every=updates,
        device="cuda",
    )


def settings(config: Config) -> SelfPlay:
    base = trophies.settings(config)
    return base.model_copy(
        update={
            "workers": config.workers,
            "transitions": config.transitions,
            "batch_closure": config.batch_closure,
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
    return settings(_arm_config(ARMS[name]))


def w160l8_complete() -> SelfPlay:
    return _recipe("w160l8-fresh-complete")


def w160l8_exact() -> SelfPlay:
    return _recipe("w160l8-fresh-exact")


def w384l8_complete() -> SelfPlay:
    return _recipe("w384l8-complete")


def w384l8_exact() -> SelfPlay:
    return _recipe("w384l8-exact")


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = UPDATES,
    hours: float = 1.0,
    usd_per_run: float = 1.50,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name=name,
            train=train,
            config=_arm_config(
                arm,
                updates=updates,
                threads=threads,
                workers=workers,
                streams=streams,
            ),
            smoke=Config(
                size=arm.size,
                batch_closure=arm.batch_closure,
                transitions=4 if arm.batch_closure == "exact-quota" else 1,
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
            extras=("artifacts",),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=1,
        )
        for name, arm in ARMS.items()
    ]


def report() -> Report:
    return Report(
        clock="updates",
        labels={name: name for name in ARMS},
        perf=PERF,
        ml=ML,
    )
