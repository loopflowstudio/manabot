"""Mini-only w6 baseline: bounded calibration, frozen counts, then three fresh fits.

Reuses the complexity campaign's ordinary TrainingRun/arena execution, not its
pool contrast. Calibration changes only the update count and time allocation;
its scores cannot select a recipe. Etude owns the one hard allocation deadline.
"""

from datetime import datetime
import os
from pathlib import Path
import socket
import time
from typing import Literal

from pydantic import Field, model_validator

from experiments.runners.complexity_campaign import execute
from experiments.runners.environment_complexity import Pool
from manabot.arena.matchup import AuthoredMatchup
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import Strict, TrainingRegime, TrainSelfPlay
from manabot.training.monitor_evaluation import (
    MonitorProtocol,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.verify.store import VerifyStore
import managym

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (12261, 12262, 12263)
CALIBRATION_SEED = 12260
MATCHUP = AuthoredMatchup(
    pack_key="ur-lessons-vs-gw-allies", role_a="ur_lessons", role_b="gw_allies"
)


def recipe(updates: int, seconds: float) -> TrainingRegime:
    """Explicit new-world derivative; original w4 recipe bytes remain untouched."""
    if managym.WORLD_VERSION != "w6":
        raise ValueError("corrected baseline requires w6 native runtime")
    base = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/early-learning-rate-mini-source.json").read_text()
    )
    base.id = "mtg122-w6-baseline"
    base.world = "w6"
    base.match = MATCHUP.match()
    base.wall_seconds = seconds
    stage = base.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = updates
    stage.execution.wall_seconds = seconds
    # Preserve cross-role self-play; balance deck/play order explicitly. No
    # mirror training, history, larger catalog or learning-rate treatment.
    stage.matchup_curriculum = "cross-balanced"
    return TrainingRegime.model_validate(base.model_dump())


class BaselinePlan(Strict):
    """All counts and measured admission costs frozen before scientific fitting."""

    updates: int = Field(ge=1000, le=10000)
    seed_order: tuple[int, int, int] = SEEDS
    training_seconds_per_run: float = Field(gt=120)
    evaluation_reserve_seconds: float = Field(gt=0)
    total_seconds: float = Field(gt=0)
    calibration_hashes: dict[str, str]

    @model_validator(mode="after")
    def admit(self) -> "BaselinePlan":
        if self.updates % 100 or self.seed_order != SEEDS:
            raise ValueError("baseline requires fixed seeds and 100-update granularity")
        if set(self.calibration_hashes) != {"training", "scripted_greedy", "random"}:
            raise ValueError("complete calibration receipts required")
        if (
            3 * self.training_seconds_per_run + self.evaluation_reserve_seconds + 600
            > self.total_seconds
        ):
            raise ValueError("baseline exceeds remaining allocation")
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    def experiment(self) -> Experiment:
        return Experiment(
            name="mtg122-w6-baseline",
            baseline=Baseline.capture(
                "sustained-masked-mean",
                recipe(self.updates, self.training_seconds_per_run),
            ),
        )

    def recipes(self) -> dict[str, TrainingRegime]:
        return {"l3": self.experiment().resolve().cases[0].regime}

    def protocol(
        self, pool: Pool, anchor: Literal["scripted_greedy", "random"], update: int
    ) -> MonitorProtocol:
        if pool != "l3":
            raise ValueError("baseline has no simple-pool arms")
        ordinal = (0, self.updates // 2, self.updates).index(update)
        start = 1_226_100_000 + ordinal * 1000
        return MonitorProtocol(
            authored_matchup=MATCHUP,
            include_mirrors=True,
            deal_seeds=tuple(range(start, start + 50)),
            opponent=anchor,
            purpose="frozen-study-evaluation",
            study_protocol_sha256=self.digest,
            bootstrap_seed=12290,
            bootstrap_replicates=10000,
        )


def resolve_counts(
    *,
    slowest_update: float,
    panel_costs: tuple[float, float],
    remaining: float,
    hashes: dict[str, str],
) -> BaselinePlan:
    """1.5x timing margins; never reduce seeds, controls or 12,000-game matrix."""
    evaluation = 6000 * 1.5 * sum(panel_costs)
    training = (remaining - evaluation - 601) / 3
    if slowest_update <= 0 or training <= 120:
        raise ValueError("insufficient measured capacity")
    updates = min(10000, int((training - 120) / (1.5 * slowest_update)) // 100 * 100)
    return BaselinePlan(
        updates=updates,
        training_seconds_per_run=training,
        evaluation_reserve_seconds=evaluation,
        total_seconds=remaining,
        calibration_hashes=hashes,
    )


def run() -> None:
    """Execute only inside the authorized Mini allocation, with no retry path."""
    env = os.environ
    if not env.get("ETUDE_RUN") or not socket.gethostname().startswith("mini-heart"):
        raise ValueError("baseline requires Etude on Mini; local execution forbidden")
    if int(env["ETUDE_SEED"]) != CALIBRATION_SEED:
        raise ValueError("incorrect calibration seed")
    deadline = datetime.fromisoformat(env["ETUDE_DEADLINE"]).timestamp()
    output = Path(env["ETUDE_OUTPUT_DIR"])
    root_run, root_track = env["ETUDE_RUN"], env["ETUDE_TRACK_DIR"]
    calibration = output / "calibration"
    calibration.mkdir(parents=True, exist_ok=False)
    env["ETUDE_RUN"] = f"{root_run}-calibration"
    env["ETUDE_TRACK_DIR"] = str(Path(root_track) / "calibration")
    env["ETUDE_OUTPUT_DIR"] = str(calibration)
    started = time.time()
    try:
        with VerifyStore(calibration / "training.sqlite") as store:
            trained = execute_regime(
                recipe(100, 1200),
                CALIBRATION_SEED,
                calibration / "training",
                store,
                checkpoint_updates=(50,),
                stop_path=Path(env["ETUDE_STOP_FILE"]),
            )
        if trained.status != "completed":
            raise RuntimeError(f"baseline calibration failed: {trained.error}")
        point = stage_checkpoint(trained, trained.stages[0].id)
        if point is None:
            raise RuntimeError("calibration export absent")
        costs: list[float] = []
        hashes = {"training": file_sha256(calibration / "training/run.json")}
        for anchor in ("scripted_greedy", "random"):
            if time.time() - started > 2400 or Path(env["ETUDE_STOP_FILE"]).exists():
                raise RuntimeError("calibration allowance exhausted")
            result = evaluate_checkpoint(
                trained,
                point.artifact,
                point.coordinates,
                calibration / anchor,
                protocol=MonitorProtocol(
                    authored_matchup=MATCHUP,
                    include_mirrors=True,
                    deal_seeds=(1_226_000_000,),
                    opponent=anchor,
                    game_seconds=120,
                    bootstrap_replicates=100,
                ),
            )
            result.validate_cohort()
            if result.status != "completed" or len(result.rows) != 8:
                raise RuntimeError("calibration matrix incomplete")
            costs.append(
                max(
                    [result.evaluation_seconds / 8]
                    + [
                        sum(r.game_seconds for r in result.rows if r.leg in legs) / 2
                        for legs in ((0, 3), (1, 2), (4, 5), (6, 7))
                    ]
                )
            )
            hashes[anchor] = file_sha256(calibration / anchor / "monitor.json")
        clocks = [0.0] + [
            float(d["coordinates"]["active_training_seconds"])
            for d in trained.stages[0].diagnostics
        ]
        if len(clocks) != 101:
            raise ValueError("calibration update evidence incomplete")
        slowest = max((clocks[n] - clocks[n - 20]) / 20 for n in range(20, 101, 20))
        # Leave a fresh 120 seconds outside the scientific plan for launch
        # overhead, final receipt publication and bounded cleanup.
        plan = resolve_counts(
            slowest_update=slowest,
            panel_costs=(costs[0], costs[1]),
            remaining=deadline - time.time() - 120,
            hashes=hashes,
        )
        atomic_json(output / "calibrated-plan.json", plan.model_dump(mode="json"))
    finally:
        env["ETUDE_RUN"], env["ETUDE_TRACK_DIR"] = root_run, root_track
        env["ETUDE_OUTPUT_DIR"] = str(output)
    execute(plan, pools=("l3",))


if __name__ == "__main__":
    run()
