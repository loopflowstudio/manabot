"""Exploratory TLA frontier w64 learnability trial: PLR + Vine-RLOO snapshots vs baseline.

Both arms train one seed (21501) on the 134-deck TLA trophy corpus (`tla_frontier`),
policy input v11, and identical w64 damped-policy settings. Arm 0 (`baseline`) runs
ordinary fresh self-play; Arm 1 (`plr-vine-rloo`) adds a 512-entry `SnapshotBank`
with 4-branch RLOO (`horizon=20`, root `critic_residual` priority, 80% replay /
20% fresh floor targeting ~50/50 row mix) and `root_actor_weight=2.0`.

Milestones (`25, 50, 100, 150, 200, 300, 400`, plus initialization `0` against the
heuristic) play `tla_frontier`'s 16 paired trophy deals (4 crossed legs = 64 games
per comparison). Design notes and results live in
`experiments/reports/tla_frontier_snapshots.md`.

    uv run --extra artifacts etude launch experiments/tla_frontier_snapshots.py --revision COMMIT --source r2://manabot/jobs --rent l4 --max-usd 25

Once both training runs are recorded, repeat the launch at the same revision
to schedule missing checkpoint matches on CPU32 under the experiment cap:

    uv run --extra artifacts etude launch experiments/tla_frontier_snapshots.py --revision COMMIT --source r2://manabot/jobs --rent cpu32 --max-usd 25
    uv run --extra artifacts etude report experiments/tla_frontier_snapshots.py --source r2://manabot/jobs --open
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRecord
from etude.eval.match import environment
from etude.eval.records import (
    Aligned,
    Benchmark,
    CheckpointSlot,
    Code,
    Environment,
    Match,
    Seating,
)
from etude.eval.result import Result
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.report import Curve, Matches, Report, Score, Series
from etude.train.tracking import track
from pydantic import Field

from experiments import tla_frontier, tla_frontier_snapshots
from manabot.eval.agents import AGENTS
from manabot.eval.pairing import play_pairing
from manabot.experiment.setup import machine_setup
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import StrictModel
from manabot.sim.heuristic import heuristic_source_paths
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves, head_to_head
from manabot.training.rollout.snapshots import CapturePolicy, SnapshotConfig
from manabot.training.state.game_errors import GameErrorPolicy

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay

ROOT = Path(__file__).resolve().parents[1]

type ArmName = Literal["baseline", "plr-vine-rloo"]
BASELINE: ArmName = "baseline"
SNAPSHOT: ArmName = "plr-vine-rloo"
ARMS: tuple[ArmName, ArmName] = (BASELINE, SNAPSHOT)

SEEDS: tuple[int, ...] = (tla_frontier.PROTOCOL.training_seed,)
UPDATES = 400
CHECKPOINT_EVERY = 25
MILESTONES: tuple[int, ...] = (25, 50, 100, 150, 200, 300, 400)

HEURISTIC_BENCHMARK = "tla-frontier-snapshots-heuristic-16"
HEAD_TO_HEAD_BENCHMARK = "tla-frontier-snapshots-vs-baseline-16"
HEAD_TO_HEAD_SEATINGS: tuple[str, ...] = tuple(
    f"Matched baseline pair {i + 1}" for i in range(len(tla_frontier.PROTOCOL.pairs))
)

OBSERVATION = ObservationConfig(
    policy_input_version=11,
    max_facts=512,
    max_cards_per_player=128,
    max_permanents_per_player=96,
    max_actions=64,
    action_overflow="first-legal",
    max_focus_objects=2,
    max_events=32,
)

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
ROOT_ACTOR_WEIGHT = 2.0


# -- Training -----------------------------------------------------------------


class Config(StrictModel):
    arm: ArmName
    updates: int = Field(default=UPDATES, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cuda"
    workers: Literal[1, 2, 4] = 4
    streams: int = Field(default=256, ge=2)
    checkpoint_every: int = Field(default=CHECKPOINT_EVERY, ge=1)


def settings(config: Config) -> SelfPlay:
    """Shared TLA frontier w64 settings on input v11; arms differ only in snapshot/root weight."""
    from manabot.training.loops.self_play import SelfPlay

    local = config.device == "cpu"
    use_snapshots = config.arm == SNAPSHOT
    rule = tla_frontier.learning("w64")
    if use_snapshots:
        rule = rule.model_copy(update={"root_actor_weight": ROOT_ACTOR_WEIGHT})
    return SelfPlay(
        deck_source=tla_frontier.corpus(),
        observation=OBSERVATION,
        model=tla_frontier.model("w64"),
        learning=rule,
        behavior="current-self",
        streams=config.streams,
        transitions=1 if local else 64,
        learning_seats="both",
        batch_closure="complete-games",
        snapshot=SNAPSHOT_CONFIG if use_snapshots else None,
        game_errors=GameErrorPolicy(
            threshold=0.25,
            window=100,
            warmup=100,
            max_game_actions=40_000,
            recover_capacity=True,
        ),
        device=config.device,
        precision="float32",
        workers=config.workers,
        bucketed_shapes=not local,
        live_tokens=config.device == "cuda",
        checkpoint_every=config.checkpoint_every,
        # Per-update ema=0.999 stays near initialization through update 400; evaluate
        # raw weights directly while still writing EMA siblings beside them.
        checkpoint_weights="raw",
    )


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(
    revision: str | None = None,
    updates: int = UPDATES,
    hours: float = 11.0,
    usd_per_run: float = 9.5,
    workers: Literal[1, 2, 4] = 4,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> list[TrainingJob]:
    """One L4 job per arm on the shared training seed; both jobs + eval fit under USD25."""
    return [
        TrainingJob(
            name=arm,
            train=train,
            config=Config(
                arm=arm,
                updates=updates,
                workers=workers,
                checkpoint_every=checkpoint_every,
            ),
            smoke=Config(
                arm=arm,
                updates=1,
                device="cpu",
                workers=1,
                streams=2,
                checkpoint_every=1,
            ),
            seeds=SEEDS,
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            extras=("artifacts",),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=1,
        )
        for arm in ARMS
    ]


# -- Evaluation ---------------------------------------------------------------


def _checkpoint_update(record: CheckpointRecord) -> int:
    return int(record.at["updates"])


def _selected_checkpoints(
    run: TrainingRunSummary, *, include_init: bool
) -> list[CheckpointRecord]:
    """Keep declared milestones plus an ended run's final export (including smoke u1)."""
    if not run.checkpoints:
        return []
    wanted = set(MILESTONES)
    if include_init:
        wanted.add(0)
    last = _checkpoint_update(run.checkpoints.last)
    if run.ended and (include_init or last > 0):
        wanted.add(last)
    return [
        record for record in run.checkpoints if _checkpoint_update(record) in wanted
    ]


def _eval_environment() -> Environment:
    return environment(
        name="tla-frontier-snapshots",
        files=[
            Path(__file__),
            tla_frontier.CORPUS,
            ROOT / "manabot/eval/pairing.py",
        ],
        config={"corpus_sha256": tla_frontier.PROTOCOL.corpus_sha256},
    )


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """Score both arms against the first-64 heuristic and head-to-head at matched updates."""
    by_arm: dict[ArmName, list[TrainingRunSummary]] = {arm: [] for arm in ARMS}
    for run in state.runs:
        if run.launch in by_arm:
            by_arm[run.launch].append(run)
    heuristic_checkpoints = [
        record
        for arm in ARMS
        for run in by_arm[arm]
        for record in _selected_checkpoints(run, include_init=True)
    ]
    if not heuristic_checkpoints:
        return []

    env = _eval_environment()
    heuristic = Code(
        name="heuristic",
        source=heuristic_source_paths(),
        config=tla_frontier.HEURISTIC,
    )
    heuristic_seatings = tuple(
        Seating(
            name=f"Frontier pair {i + 1}",
            players=(CheckpointSlot(), heuristic),
            config=pair,
            games=1,
        )
        for i, pair in enumerate(tla_frontier.PROTOCOL.pairs)
    )
    jobs: list[EvalJob] = [
        EvalJob(
            name=HEURISTIC_BENCHMARK,
            play=tla_frontier_snapshots.play,
            agents=AGENTS,
            checkpoints=heuristic_checkpoints,
            tournament=Benchmark(
                name=HEURISTIC_BENCHMARK,
                environment=env,
                seatings=heuristic_seatings,
            ),
            setup=machine_setup(" --extra artifacts --extra evaluation"),
            extras=("artifacts", "evaluation"),
            max_hours=1.0,
            max_usd=Decimal("1.20"),
            max_attempts=1,
        )
    ]

    baseline_updates = {
        _checkpoint_update(record)
        for run in by_arm[BASELINE]
        for record in run.checkpoints
    }
    matched_snapshot = [
        record
        for run in by_arm[SNAPSHOT]
        for record in _selected_checkpoints(run, include_init=False)
        if _checkpoint_update(record) in baseline_updates
    ]
    baseline_seeds = len({run.seed for run in by_arm[BASELINE]})
    if matched_snapshot and baseline_seeds:
        h2h_seatings = tuple(
            Seating(
                name=label,
                players=(CheckpointSlot(), Aligned(clock="updates", job=BASELINE)),
                config=pair,
                games=baseline_seeds,
            )
            for label, pair in zip(
                HEAD_TO_HEAD_SEATINGS, tla_frontier.PROTOCOL.pairs, strict=True
            )
        )
        jobs.append(
            EvalJob(
                name=HEAD_TO_HEAD_BENCHMARK,
                play=tla_frontier_snapshots.play,
                agents=AGENTS,
                checkpoints=matched_snapshot,
                tournament=Benchmark(
                    name=HEAD_TO_HEAD_BENCHMARK,
                    environment=env,
                    seatings=h2h_seatings,
                ),
                setup=machine_setup(" --extra artifacts --extra evaluation"),
                extras=("artifacts", "evaluation"),
                max_hours=0.5,
                max_usd=Decimal("0.80"),
                max_attempts=1,
            )
        )
    return jobs


def play(match: Match[tla_frontier.Pair], agents: Sequence[object]) -> Result:
    """Play one frozen 17Lands deck pair across all four crossed seat/deck legs."""
    pair = tla_frontier.Pair.model_validate(match.config)
    if pair not in tla_frontier.PROTOCOL.pairs:
        raise ValueError("evaluation pair differs from tla_frontier schedule")
    matchup = tla_frontier.deck_matchup(
        tla_frontier.corpus(), tla_frontier.PROTOCOL, pair
    )
    return play_pairing(
        match,
        agents,
        matchup=matchup,
        deal=pair.deal,
        matchup_mode="cross",
        game_seconds=tla_frontier.PROTOCOL.game_seconds,
        arena="tla-frontier-snapshots",
    )


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
    heuristic = Matches(tournament=HEURISTIC_BENCHMARK)
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        baseline=BASELINE,
        labels={
            BASELINE: "Baseline (fresh self-play)",
            SNAPSHOT: "PLR + Vine-RLOO (K=4, H=20)",
        },
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            head_to_head(
                "PLR + Vine-RLOO vs baseline at matched updates",
                HEAD_TO_HEAD_SEATINGS,
            ),
            *behaviour_curves(matches=heuristic),
        ),
        perf=PERF,
        ml=(*ML, *SNAPSHOT_PANELS),
    )
