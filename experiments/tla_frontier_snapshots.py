"""Exploratory TLA trophy w64 learnability trial: PLR + Vine-RLOO snapshots vs stored tau-x2 baseline.

Trains two seeds (40901, 40902) for 400 updates on the weekend `tla_trophies` +
`tau_scale=0.1` recipe (complete 7,859-pool TLA trophy training split, closed
decklists, policy input v11, `focus_film=True`, `stream_norm=True`, `streams=256`,
`transitions=64`, `rows=512`, `workers=4`, `threads=2`, raw checkpoints), adding a
512-entry `SnapshotBank` with 4-branch Vine-RLOO (`horizon=20`, root
`critic_residual` priority, 80% replay / 20% fresh floor targeting ~50/50 row mix)
and `root_actor_weight=2.0`.

No fresh baseline is trained. Evaluation scores checkpoints on `tla_trophies`'s
32 held-out deals (4 crossed legs = 128 games per seed per checkpoint) against
the first-64 heuristic, against each line at half its updates, and head-to-head
at matched updates (100, 200, 300, 400) against the retained
`tla_trophies_floor/tau-x2` checkpoints (`40901`, `40902`) stored in R2.

    doppler run -p manabot -c prd -- .venv/bin/etude launch experiments/tla_frontier_snapshots.py --rent colab-l4-bench --max-usd 25
    doppler run -p manabot -c prd -- .venv/bin/etude launch experiments/tla_frontier_snapshots.py --rent mtg344-cpu32 --max-usd 25
    doppler run -p manabot -c prd -- .venv/bin/etude report experiments/tla_frontier_snapshots.py --source r2://manabot/jobs --open
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRecord, CheckpointRef
from etude.eval.records import (
    Benchmark,
    CheckpointPlayer,
    Fraction,
    Seating,
    checkpoint_player,
)
from etude.jobs import EvalJob, ExperimentState, TrainingJob
from etude.records import ArtifactRef
from etude.report import Curve, Matches, Report, Score, Series
from etude.spec import Variable
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_trophies as trophies
from experiments.tla_trophies_floor import final_environment
from manabot.eval.card_metrics import CARD_ROLLUPS
from manabot.experiment.setup import machine_setup
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves, head_to_head
from manabot.training.rollout.snapshots import CapturePolicy, SnapshotConfig

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay

JOB = "plr-vine-rloo"
SEEDS: tuple[int, ...] = (40901, 40902)
UPDATES = 400
CHECKPOINT_EVERY = 50
TAU_SCALE = 0.1
ROOT_ACTOR_WEIGHT = 2.0

HELD_OUT = "tla-snapshots-held-out-32"
HALF_TRAINING = "tla-snapshots-half-training-32"
HALF_TRAINING_SEATINGS = tuple(
    f"Snapshots half-training deal {i + 1}" for i in range(32)
)
HEAD_TO_HEAD = "tla-snapshots-vs-tau-x2-32"
HEAD_TO_HEAD_TAG = "vs-tau-x2"

# Targets roughly a 50/50 fresh-vs-snapshot row mix (~1 fresh game per K=4, H=20 group).
SNAPSHOT_CONFIG = SnapshotConfig(
    capacity=512,
    replay_fraction=0.8,
    fresh_floor=0.2,
    group_size=4,
    horizon=20,
    signal="critic_residual",
    signal_scope="root",
    score_half_life=8,
    fresh_weight=1.0,
    snapshot_weight=1.0,
    capture=CapturePolicy(
        min_legal_actions=3,
        probability=0.05,
        max_captures_per_batch=64,
        fresh_only=True,
    ),
)


def _baseline_player(
    seed: int, update: int, sha256: str, size_bytes: int
) -> CheckpointPlayer:
    file = f"out/policies/{update:08d}-raw.safetensors"
    uri = f"r2://manabot/jobs/tla_trophies_floor/tau-x2@{seed}/{file}"
    return CheckpointPlayer(
        name=f"tau-x2@{seed}-u{update}",
        checkpoint=CheckpointRef(
            address=uri,
            artifact=ArtifactRef(uri=uri, sha256=sha256, size_bytes=size_bytes),
        ),
    )


# Retained baseline checkpoints from `tla_trophies_floor/tau-x2` (seeds 40901, 40902).
BASELINE_TAU_X2: dict[int, tuple[tuple[int, CheckpointPlayer], ...]] = {
    100: (
        (
            40901,
            _baseline_player(
                40901,
                100,
                "7c4f8d86eb0214bf0b2dad50f1dfbef4cfdb58ae4cd0a97c70969e8ae1238c1b",
                5_671_658,
            ),
        ),
        (
            40902,
            _baseline_player(
                40902,
                100,
                "8a15c4c77fcb96d4a3955b7db0cd3384be21357c381d032fa8b90ae36d7d036c",
                5_671_658,
            ),
        ),
    ),
    200: (
        (
            40901,
            _baseline_player(
                40901,
                200,
                "9b37dd9e44c3d773154cf61622fc5afe0694f6602a3a717f929c8ed443715268",
                5_671_658,
            ),
        ),
        (
            40902,
            _baseline_player(
                40902,
                200,
                "12e25f8f2f2f277daf2e17f7fa8548345b09d00b735454131000e06618cec9c7",
                5_671_658,
            ),
        ),
    ),
    300: (
        (
            40901,
            _baseline_player(
                40901,
                300,
                "dd236e3a910539fa9391002d9b9b030e877e0b49342a44428541de742812c409",
                5_671_658,
            ),
        ),
        (
            40902,
            _baseline_player(
                40902,
                300,
                "657991a70ab61eff1aef69688e9e8dfc4158c1b6e7059e7e3d81ddd1ca2014c4",
                5_671_658,
            ),
        ),
    ),
    400: (
        (
            40901,
            _baseline_player(
                40901,
                400,
                "e40993b9a2ab8bd1126d95b20acc38f9cc056a9a39047c5dfbb1be72cf513322",
                5_671_658,
            ),
        ),
        (
            40902,
            _baseline_player(
                40902,
                400,
                "7fed58a3bbbf482898b773966163eda844cb780629bf6ce5628e024602c44499",
                5_671_658,
            ),
        ),
    ),
}


# -- Training -----------------------------------------------------------------


class Config(trophies.Config):
    tau_scale: float = TAU_SCALE
    workers: Literal[1, 2, 4] = Field(default=4)


def settings(config: Config) -> SelfPlay:
    """Weekend tla_trophies + tau_scale=0.1 stack with PLR + Vine-RLOO snapshots."""
    base = trophies.settings(config)
    return base.model_copy(
        update={
            "workers": config.workers,
            "learning": base.learning.model_copy(
                update={
                    "tau_scale": config.tau_scale,
                    "root_actor_weight": ROOT_ACTOR_WEIGHT,
                }
            ),
            "snapshot": SNAPSHOT_CONFIG,
        }
    )


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = UPDATES,
    hours: float = 18.0,
    usd_per_run: float = 5.0,
    attempts: int = 6,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> list[TrainingJob]:
    """Two snapshot seeds packed on a 2-slot Colab L4; baseline is reused from R2."""
    return [
        TrainingJob(
            name=JOB,
            train=train,
            config=Config(
                updates=updates,
                threads=threads,
                workers=workers,
                streams=streams,
                rows=rows,
                checkpoint_every=checkpoint_every,
            ),
            smoke=Config(
                updates=2,
                device="cpu",
                threads=1,
                workers=4,
                streams=4,
                rows=64,
                checkpoint_every=2,
            ),
            environment=tuple(
                Variable(name=name, value=str(threads))
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
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
    ]


# -- Evaluation ---------------------------------------------------------------


def _checkpoint_update(record: CheckpointRecord) -> int:
    return int(record.at["updates"])


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """Score checkpoints against heuristic, half-training self, and stored tau-x2."""
    runs = [run for run in state.runs if run.launch == JOB]
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
        opponent: Fraction | None = None,
    ) -> EvalJob:
        names = seating_names or tuple(s.name for s in benchmark.seatings)
        seatings = tuple(
            seating.model_copy(
                update={
                    "name": label,
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
    jobs: list[EvalJob] = [
        against(HELD_OUT, shared.checkpoints),
        against(HALF_TRAINING, shared.checkpoints, HALF_TRAINING_SEATINGS, half),
    ]

    h2h_seatings: list[Seating] = []
    h2h_checkpoints: list[CheckpointRecord] = []
    deals = tuple(s.config for s in benchmark.seatings)
    for run in runs:
        for record in run.checkpoints:
            update = _checkpoint_update(record)
            opponents = BASELINE_TAU_X2.get(update)
            if not opponents:
                continue
            h2h_checkpoints.append(record)
            cand = checkpoint_player(record.ref)
            for base_seed, base_player in opponents:
                h2h_seatings.append(
                    Seating(
                        name=f"{run.name} u{update} vs tau-x2@{base_seed}",
                        tags=(HEAD_TO_HEAD_TAG,),
                        players=(cand, base_player),
                        configs=deals,
                        games=1,
                        seeds="one",
                    )
                )
    if h2h_seatings:
        jobs.append(
            shared.model_copy(
                update={
                    "name": HEAD_TO_HEAD,
                    "tournament": Benchmark(
                        name=HEAD_TO_HEAD,
                        environment=env,
                        seatings=tuple(h2h_seatings),
                        rollups=CARD_ROLLUPS,
                    ),
                    "checkpoints": tuple(h2h_checkpoints),
                    "max_hours": 1.5,
                    "max_usd": Decimal("2.00"),
                }
            )
        )
    return jobs


# -- Report -------------------------------------------------------------------


SNAPSHOT_PANELS: tuple[Series, ...] = (
    Series(
        title="Snapshot continuations",
        keys=(
            "snapshot/starts",
            "snapshot/rows",
            "snapshot/captures",
            "snapshot/groups_completed",
            "snapshot/groups_discarded",
            "snapshot/failed_branches",
        ),
    ),
    Series(
        title="Snapshot episode endings",
        keys=("snapshot/terminals", "snapshot/horizon_cuts"),
    ),
    Series(
        title="Source advantage retention",
        keys=(
            "source_selection/fresh/retention_rate",
            "source_selection/snapshot/retention_rate",
        ),
        unit="fraction",
    ),
    Series(
        title="Source eligible and retained rows",
        keys=(
            "source_selection/fresh/eligible",
            "source_selection/fresh/retained",
            "source_selection/snapshot/eligible",
            "source_selection/snapshot/retained",
        ),
        unit="rows",
    ),
    Series(
        title="Fresh decisions per game",
        keys=("decisions_per_game",),
        unit="decisions",
    ),
    Series(
        title="Bootstrapped tail fraction",
        keys=("bootstrapped_tail_fraction",),
        unit="fraction",
    ),
)


def report() -> Report:
    heuristic = Matches(tournament=HELD_OUT)
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        labels={JOB: "PLR + Vine-RLOO (K=4, H=20)"},
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            head_to_head("Against itself at half its updates", HALF_TRAINING_SEATINGS),
            Curve(
                title="PLR + Vine-RLOO vs stored tau-x2 baseline at matched updates",
                matches=Matches(tag=HEAD_TO_HEAD_TAG),
            ),
            *behaviour_curves(matches=heuristic),
        ),
        perf=PERF,
        ml=(*ML, *SNAPSHOT_PANELS),
    )
