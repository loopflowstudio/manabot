"""Train fresh w64 seeds on the complete TLA trophy split and rate their checkpoints.

Every checkpoint plays the same held-out deals with seats and decks crossed.
The published pool snapshot supplies both populations on the worker.
"""

from collections.abc import Sequence
from decimal import Decimal
from functools import cache
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from etude.eval.records import (
    Benchmark,
    CheckpointSlot,
    Code,
    Environment,
    Match,
    Seating,
)
from etude.eval.result import Result
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.report import Curve, Matches, Report, Score
from etude.spec import Input, Variable
from etude.train.tracking import track

from experiments.mana_taps import OBSERVATION
from experiments.tla_frontier import HEURISTIC, learning, model
from manabot.data import load_manifest
from manabot.eval.agents import AGENTS
from manabot.eval.pairing import play_pairing
from manabot.experiment.setup import machine_setup
from manabot.infra.identity import StrictModel
from manabot.pools import Pool, PoolCorpus, PoolSplit
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import DeckMatchup
from manabot.training.decks import UniformDecks
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves
from manabot.training.state.game_errors import GameErrorPolicy
import managym

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


ROOT = Path(__file__).resolve().parents[1]
POOLS = "experiments/data/mtg409/pools-v2/corpus.json"
RAW_JOB = "trophy-w64-raw"
BENCHMARK = "tla-trophies-held-out-32"
ENVIRONMENT_FILES = "d7dab787f31f7e2ff651cf9092c6c9d39a5530ad95ea0e20bf03b64b688e09e8"


def corpus_input() -> Input:
    entry = load_manifest().files[POOLS]
    return Input(
        name="pools.json",
        key=f"sha256/{entry.sha256}",
        sha256=entry.sha256,
        size_bytes=entry.bytes,
        uri=(ROOT / POOLS).as_uri(),
    )


@cache
def playable(split: PoolSplit) -> tuple[Pool, ...]:
    directory = os.environ.get("ETUDE_INPUTS_DIR")
    path = Path(directory) / "pools.json" if directory else ROOT / POOLS
    return tuple(
        p
        for p in PoolCorpus.read(path).pools
        if p.split == split
        and not managym.unsupported_cards([*p.cards, *p.human_build])
    )


def deck(pool: Pool) -> managym.Deck:
    side = {
        name: n - pool.human_build.get(name, 0)
        for name, n in pool.cards.items()
        if n > pool.human_build.get(name, 0)
    }
    return managym.Deck(pool.human_build, side)


class Config(StrictModel):
    updates: int
    threads: int
    streams: int
    rows: int
    checkpoint_every: int
    device: Literal["cuda", "cpu"] = "cuda"


def settings(config: Config) -> "SelfPlay":
    from manabot.training.loops.self_play import SelfPlay

    pools = playable("train")
    local = config.device == "cpu"
    return SelfPlay(
        deck_source=UniformDecks(
            decks=tuple(map(deck, pools)),
            entry_ids=tuple(p.id for p in pools),
            sampling="uniform-independent-v1",
            decklists="closed",
        ),
        observation=OBSERVATION,
        model=model("w64").model_copy(update={"focus_film": True, "stream_norm": True}),
        learning=learning("w64"),
        behavior="current-self",
        device=config.device,
        streams=config.streams,
        threads=config.threads,
        transitions=1 if local else 64,
        workers=1,
        precision="float32",
        bucketed_shapes=True,
        live_tokens=not local,
        bootstrap_rows=config.rows,
        learner_rows=config.rows,
        checkpoint_every=config.checkpoint_every,
        checkpoint_weights="raw",
        game_errors=GameErrorPolicy(
            threshold=0.25,
            window=100,
            warmup=100,
            max_game_actions=40_000,
            recover_capacity=True,
        ),
    )


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = 400,
    seeds: int = 3,
    hours: float = 10.75,
    usd_per_run: float = 2.60,
    threads: int = 2,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = 100,
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name=RAW_JOB,
            train=train,
            config=Config(
                updates=updates,
                threads=threads,
                streams=streams,
                rows=rows,
                checkpoint_every=checkpoint_every,
            ),
            smoke=Config(
                updates=1,
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
            seeds=tuple(range(40901, 40901 + seeds)),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            inputs=(corpus_input(),),
            extras=("artifacts",),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=1,
        )
    ]


class Deal(StrictModel):
    first: int
    second: int
    seed: int


def play(match: Match, agents: Sequence[object]) -> Result:
    deal = Deal.model_validate(match.config)
    pools = playable("held_out")
    a, b = pools[deal.first], pools[deal.second]
    return play_pairing(
        match,
        agents,
        matchup=DeckMatchup(
            population_sha256=corpus_input().sha256,
            entries=(a.id, b.id),
            lists=(deck(a), deck(b)),
            decklists="closed",
        ),
        deal=deal.seed,
        matchup_mode="cross",
        game_seconds=120,
        arena="tla-trophies",
    )


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    return held_out_jobs([run for run in state.runs if run.launch == RAW_JOB])


def held_out_jobs(runs: Sequence[TrainingRunSummary]) -> list[EvalJob]:
    checkpoints = [c for run in runs for c in run.checkpoints]
    if not checkpoints:
        return []
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config=HEURISTIC
    )
    # Etude divides games among the seeds of each job independently.
    seeds = len({run.seed for run in runs})
    seatings = tuple(
        Seating(
            name=f"Held-out pair {i + 1}",
            players=(CheckpointSlot(), heuristic),
            config=Deal(first=i * 2, second=i * 2 + 1, seed=40900 + i),
            games=seeds,
        )
        for i in range(32)
    )
    return [
        EvalJob(
            play=play,
            agents=AGENTS,
            checkpoints=checkpoints,
            tournament=Benchmark(
                name=BENCHMARK,
                environment=Environment(
                    name="tla-trophies",
                    # This file and pairing.py when the stored games were played.
                    files=ENVIRONMENT_FILES,
                    config={"pools": corpus_input().sha256},
                ),
                seatings=seatings,
            ),
            inputs=(corpus_input(),),
            setup=machine_setup(" --extra artifacts --extra evaluation"),
            extras=("artifacts", "evaluation"),
            max_hours=1,
            max_usd=Decimal("1.30"),
            max_attempts=1,
        )
    ]


def report() -> Report:
    matches = Matches(tournament=BENCHMARK)
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5),
        evals=(
            Curve(title="Score against the heuristic", matches=matches),
            *behaviour_curves(matches=matches),
        ),
        perf=PERF,
        ml=ML,
    )
