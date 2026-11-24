"""Exploratory TLA trophy w64 learnability trial: PLR + Vine-RLOO snapshots vs stored tau-x2 baseline.

Compares PLR + Vine-RLOO snapshot self-play on the weekend `tla_trophies` +
`tau_scale=0.1` recipe (complete 7,859-pool TLA trophy training split, closed
decklists, policy input v11, `focus_film=True`, `stream_norm=True`, `streams=256`,
`transitions=64`, `rows=512`, `workers=4`, `threads=2`, raw checkpoints,
`root_actor_weight=2.0`) against retained `tla_trophies_floor/tau-x2` checkpoints
stored in R2 — no fresh baseline is trained.

Arms:
- `plr-vine-rloo` (seeds 40901, 40902; 400 updates from scratch): `H=20`, 80/20 starts.
- `warm-h20` (seed 40902; +400 updates from `tau-x2@40902` u400): `H=20`, 80/20 starts.
- `warm-h60` (seed 40902; +400 updates from `tau-x2@40902` u400): `H=60`, 75/25 starts.
- `warm-terminal` (seed 40902; +400 updates from `tau-x2@40902` u400): `H=None`, 65/35 starts.

Evaluation scores checkpoints on `tla_trophies`'s 32 held-out deals (4 crossed legs =
128 games per seed per checkpoint) against the first-64 heuristic, against each line
at half its updates, and head-to-head at matched effective updates against retained
R2 `tau-x2` checkpoints (`u100–u400` for from-scratch; `u500–u800` for warm-start).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
WARM_SEED = 40902
UPDATES = 400
CHECKPOINT_EVERY = 50
TAU_SCALE = 0.1
ROOT_ACTOR_WEIGHT = 2.0

HELD_OUT = "tla-snapshots-held-out-32"
HALF_TRAINING = "tla-snapshots-half-training-32"
HALF_TRAINING_SEATINGS = tuple(
    f"Snapshots half-training deal {i + 1}" for i in range(32)
)
HEAD_TO_HEAD = "tla-snapshots-vs-tau-x2-32-v2"
HEAD_TO_HEAD_TAG = "vs-tau-x2"
WARM_HEAD_TO_HEAD = "tla-snapshots-warm-vs-tau-x2-32"
WARM_HEAD_TO_HEAD_TAG = "warm-vs-tau-x2"


@dataclass(frozen=True)
class WarmArm:
    name: str
    label: str
    horizon: int | None
    replay_fraction: float
    fresh_floor: float


WARM_ARMS: tuple[WarmArm, ...] = (
    WarmArm("warm-h20", "Warm u400: H=20 (80/20)", 20, 0.8, 0.2),
    WarmArm("warm-h60", "Warm u400: H=60 (75/25)", 60, 0.75, 0.25),
    WarmArm("warm-terminal", "Warm u400: H=None terminal (65/35)", None, 0.65, 0.35),
)
ALL_JOBS: tuple[str, ...] = (JOB, *(arm.name for arm in WARM_ARMS))

WARM_START_FILE = "out/policies/00000400-raw.safetensors"
WARM_START = CheckpointRecord(
    experiment="tla_trophies_floor",
    run=f"tau-x2@{WARM_SEED}",
    ordinal=5,
    at={"updates": 400},
    file=WARM_START_FILE,
    artifact=ArtifactRef(
        uri=f"r2://manabot/jobs/tla_trophies_floor/tau-x2@{WARM_SEED}/{WARM_START_FILE}",
        sha256="7fed58a3bbbf482898b773966163eda844cb780629bf6ce5628e024602c44499",
        size_bytes=5_671_658,
    ),
)

CAPTURE_POLICY = CapturePolicy(
    min_legal_actions=3,
    probability=0.05,
    max_captures_per_batch=64,
    fresh_only=True,
)


def snapshot_config(
    *, horizon: int | None = 20, replay_fraction: float = 0.8, fresh_floor: float = 0.2
) -> SnapshotConfig:
    return SnapshotConfig(
        capacity=512,
        replay_fraction=replay_fraction,
        fresh_floor=fresh_floor,
        group_size=4,
        horizon=horizon,
        signal="critic_residual",
        signal_scope="root",
        score_half_life=8,
        fresh_weight=1.0,
        snapshot_weight=1.0,
        capture=CAPTURE_POLICY,
    )


SNAPSHOT_CONFIG = snapshot_config()


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

# Retained `tau-x2@40902` checkpoints at updates 500..800 for matched warm-start H2H.
WARM_BASELINE_TAU_X2: dict[int, CheckpointPlayer] = {
    100: _baseline_player(
        WARM_SEED,
        500,
        "a6e0f75848bf212769e3b1a91da4afcf2d9b2a8d7fbcbefaf2df36d1e6cd68ca",
        5_671_658,
    ),
    200: _baseline_player(
        WARM_SEED,
        600,
        "c71942c13e98db74bcb5cf7c235f22245e390747a96710758857d4fa7e94b36f",
        5_671_658,
    ),
    300: _baseline_player(
        WARM_SEED,
        700,
        "3053705bb3d622ceafa698809b9e94e11101a5c77f5038e9fa32459a597389ed",
        5_671_658,
    ),
    400: _baseline_player(
        WARM_SEED,
        800,
        "3456250f4fb6fe9ee8ef4fb0d2700b1196cceb97a44eac394b012a9c91a2e26f",
        5_671_658,
    ),
}


# -- Training -----------------------------------------------------------------


class Config(trophies.Config):
    tau_scale: float = TAU_SCALE
    workers: Literal[1, 2, 4] = Field(default=4)
    # Exclude default values so existing `plr-vine-rloo` serialized configs stay unchanged.
    horizon: int | None = Field(default=20, exclude_if=lambda value: value == 20)
    replay_fraction: float = Field(default=0.8, exclude_if=lambda value: value == 0.8)
    fresh_floor: float = Field(default=0.2, exclude_if=lambda value: value == 0.2)
    warm_start: bool = Field(default=False, exclude_if=lambda value: not value)


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
            "snapshot": snapshot_config(
                horizon=config.horizon,
                replay_fraction=config.replay_fraction,
                fresh_floor=config.fresh_floor,
            ),
        }
    )


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        run_settings = settings(config)
        if config.warm_start:
            run_settings = run_settings.model_copy(
                update={"pretrained": run.input("start")}
            )
        self_play(run_settings, run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = UPDATES,
    hours: float = 18.0,
    usd_per_run: float = 5.0,
    warm_usd_per_run: float = 3.0,
    attempts: int = 6,
    threads: int = 2,
    workers: Literal[1, 2, 4] = 4,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> list[TrainingJob]:
    """From-scratch H=20 (2 seeds) + warm-start u400 horizon arms (seed 40902)."""
    env = tuple(
        Variable(name=name, value=str(threads))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    )
    jobs: list[TrainingJob] = [
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
            environment=env,
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
    for arm in WARM_ARMS:
        jobs.append(
            TrainingJob(
                name=arm.name,
                train=train,
                config=Config(
                    updates=updates,
                    threads=threads,
                    workers=workers,
                    streams=streams,
                    rows=rows,
                    checkpoint_every=checkpoint_every,
                    horizon=arm.horizon,
                    replay_fraction=arm.replay_fraction,
                    fresh_floor=arm.fresh_floor,
                    warm_start=True,
                ),
                smoke=Config(
                    updates=2,
                    device="cpu",
                    threads=1,
                    workers=4,
                    streams=4,
                    rows=64,
                    checkpoint_every=2,
                    horizon=arm.horizon,
                    replay_fraction=arm.replay_fraction,
                    fresh_floor=arm.fresh_floor,
                    warm_start=False,
                ),
                start=WARM_START,
                environment=env,
                seeds=(WARM_SEED,),
                repo=str(trophies.ROOT),
                revision=revision,
                setup=machine_setup(" --extra artifacts"),
                inputs=(trophies.corpus_input(),),
                extras=("artifacts",),
                max_hours=hours,
                max_usd=Decimal(str(warm_usd_per_run)),
                max_attempts=attempts,
            )
        )
    return jobs


# -- Evaluation ---------------------------------------------------------------


def _checkpoint_update(record: CheckpointRecord) -> int:
    return int(record.at["updates"])


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """Score from-scratch and warm-start arms vs heuristic, half-self, and stored tau-x2."""
    jobs: list[EvalJob] = []
    env = final_environment()

    fresh_runs = [run for run in state.runs if run.launch == JOB]
    fresh_held_out = trophies.held_out_jobs(fresh_runs)
    if fresh_held_out:
        (shared,) = fresh_held_out
        benchmark = shared.tournament

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
        jobs.extend(
            [
                against(HELD_OUT, shared.checkpoints),
                against(
                    HALF_TRAINING, shared.checkpoints, HALF_TRAINING_SEATINGS, half
                ),
            ]
        )

        deals = tuple(s.config for s in benchmark.seatings)
        h2h_seatings: list[Seating] = []
        h2h_checkpoints: list[CheckpointRecord] = []
        for run in fresh_runs:
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
                            games=4,
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

    warm_h2h_seatings: list[Seating] = []
    warm_h2h_checkpoints: list[CheckpointRecord] = []
    warm_shared: EvalJob | None = None
    for arm in WARM_ARMS:
        arm_runs = [run for run in state.runs if run.launch == arm.name]
        arm_held = trophies.held_out_jobs(arm_runs)
        if not arm_held:
            continue
        (shared,) = arm_held
        warm_shared = shared
        benchmark = shared.tournament
        seatings = tuple(
            s.model_copy(update={"games": 1, "seeds": "one"})
            for s in benchmark.seatings
        )
        jobs.append(
            shared.model_copy(
                update={
                    "name": f"{HELD_OUT}-{arm.name}",
                    "tournament": benchmark.model_copy(
                        update={
                            "name": f"{HELD_OUT}-{arm.name}",
                            "environment": env,
                            "seatings": seatings,
                            "rollups": CARD_ROLLUPS,
                        }
                    ),
                    "checkpoints": shared.checkpoints,
                }
            )
        )
        deals = tuple(s.config for s in benchmark.seatings)
        for run in arm_runs:
            for record in run.checkpoints:
                update = _checkpoint_update(record)
                base_player = WARM_BASELINE_TAU_X2.get(update)
                if base_player is None:
                    continue
                warm_h2h_checkpoints.append(record)
                warm_h2h_seatings.append(
                    Seating(
                        name=f"{run.name} +{update} vs tau-x2@40902-u{400 + update}",
                        tags=(WARM_HEAD_TO_HEAD_TAG,),
                        players=(checkpoint_player(record.ref), base_player),
                        configs=deals,
                        games=4,
                        seeds="one",
                    )
                )
    if warm_h2h_seatings and warm_shared is not None:
        jobs.append(
            warm_shared.model_copy(
                update={
                    "name": WARM_HEAD_TO_HEAD,
                    "tournament": Benchmark(
                        name=WARM_HEAD_TO_HEAD,
                        environment=env,
                        seatings=tuple(warm_h2h_seatings),
                        rollups=CARD_ROLLUPS,
                    ),
                    "checkpoints": tuple(warm_h2h_checkpoints),
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
    heuristic = (
        Matches(tournament=HELD_OUT),
        *(Matches(tournament=f"{HELD_OUT}-{arm.name}") for arm in WARM_ARMS),
    )
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        labels={
            JOB: "From scratch: PLR + Vine-RLOO (H=20)",
            **{arm.name: arm.label for arm in WARM_ARMS},
        },
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            head_to_head("Against itself at half its updates", HALF_TRAINING_SEATINGS),
            Curve(
                title="From-scratch H=20 vs stored tau-x2 at matched updates (u100–u400)",
                matches=Matches(tag=HEAD_TO_HEAD_TAG),
            ),
            Curve(
                title="Warm-start (+u) vs stored tau-x2@40902 (u500–u800)",
                matches=Matches(tag=WARM_HEAD_TO_HEAD_TAG),
            ),
            *behaviour_curves(matches=heuristic),
        ),
        perf=PERF,
        ml=(*ML, *SNAPSHOT_PANELS),
    )
