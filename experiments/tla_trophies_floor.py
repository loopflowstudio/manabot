"""Compare forward-KL weights on fresh trophy policies and a saturated raw policy."""

from decimal import Decimal
from pathlib import Path

from etude.checkpoints import CheckpointRecord
from etude.jobs import EvalJob, ExperimentState, TrainingJob
from etude.records import ArtifactRef
from etude.report import Report
from etude.train.tracking import track

from experiments import tla_trophies as trophies
from manabot.training.loops.self_play import CLOCKS, self_play

FRESH = {"control": 0.0, "floor-0.002": 0.002, "floor-0.01": 0.01}
PULLBACK = {"pullback-0": 0.0, "pullback-0.01": 0.01}
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


def train(config: Config) -> None:
    base = trophies.settings(config)
    with track(clocks=CLOCKS, config=config) as run:
        settings = base.model_copy(
            update={
                "pretrained": Path(run.input("start").path)
                if config.pullback
                else None,
                "learning": base.learning.model_copy(
                    update={"reference_forward_kl": config.reference_forward_kl}
                ),
            }
        )
        self_play(settings, run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    pullback_updates: int = 100,
    pullback_hours: float = 2.75,
) -> list[TrainingJob]:
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
    return jobs


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    return trophies.held_out_jobs([run for run in state.runs if run.launch in FRESH])


def report() -> Report:
    return trophies.report()
