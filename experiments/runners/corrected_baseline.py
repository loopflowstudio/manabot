"""Mini-only w6 baseline: bounded calibration, frozen counts, then three fresh fits.

Reuses the complexity campaign's ordinary TrainingRun/arena execution, not its
pool contrast. Calibration changes only the update count and time allocation;
its scores cannot select a recipe. Etude owns the one hard allocation deadline.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import socket
import time
from typing import Literal

from etude.train.records import Step
from etude.train.run import track
from pydantic import Field, model_validator

from experiments.recipes import ataraxos_mini
from experiments.runners.complexity_campaign import (
    Anchor,
    allocation_stop,
    evaluate_fit,
    execute,
)
from experiments.runners.environment_complexity import Pool
from manabot.arena.matchup import AuthoredMatchup
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import Strict, TrainingRegime, TrainingRun, TrainSelfPlay
from manabot.training.monitor_evaluation import (
    MonitorProtocol,
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.verify.store import VerifyStore
import managym

SEEDS = (12261, 12262, 12263)
CALIBRATION_SEED = 12260
MATCHUP = AuthoredMatchup(
    pack_key="ur-lessons-vs-gw-allies", role_a="ur_lessons", role_b="gw_allies"
)


def recipe(updates: int, seconds: float) -> TrainingRegime:
    """`ataraxos_mini` on the Allies/Lessons matchup with a cross-balanced curriculum."""
    if managym.WORLD_VERSION != "w6":
        raise ValueError("corrected baseline requires w6 native runtime")
    base = ataraxos_mini()
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

    updates: Literal[10000] = 10000
    seed_order: tuple[int, int, int] = SEEDS
    training_seconds_per_run: float = Field(gt=120)
    evaluation_reserve_seconds: float = Field(gt=0)
    total_seconds: float = Field(gt=0)
    calibration_hashes: dict[str, str]

    @model_validator(mode="after")
    def admit(self) -> "BaselinePlan":
        if self.seed_order != SEEDS:
            raise ValueError("baseline requires its fixed three seeds")
        if set(self.calibration_hashes) != {"training", "scripted_greedy", "random"}:
            raise ValueError("complete calibration receipts required")
        if (
            3 * self.training_seconds_per_run + self.evaluation_reserve_seconds + 600
            > self.total_seconds
        ):
            raise ValueError("baseline exceeds remaining allocation")
        return self

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return (0, self.updates)

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

    def protocol(self, pool: Pool, anchor: Anchor, update: int) -> MonitorProtocol:
        if pool != "l3":
            raise ValueError("baseline has no simple-pool arms")
        ordinal = self.evaluation_updates.index(update)
        start = 1_226_100_000 + ordinal * 1000
        return MonitorProtocol(
            authored_matchup=MATCHUP,
            include_mirrors=True,
            deal_seeds=tuple(range(start, start + 25)),
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
    """Admit fixed 10,000-update fits and 3,600 games with 1.5x timing margins."""
    evaluation = 1800 * 1.5 * sum(panel_costs)
    training = 10000 * 1.5 * slowest_update + 120
    if slowest_update <= 0 or training <= 120:
        raise ValueError("insufficient measured capacity")
    return BaselinePlan(
        updates=10000,
        training_seconds_per_run=training,
        evaluation_reserve_seconds=evaluation,
        total_seconds=remaining,
        calibration_hashes=hashes,
    )


def complete_retained_first(plan: BaselinePlan, source: Path, output: Path) -> None:
    """Finish seed1's fixed evaluations without training or porting its weights."""
    run = TrainingRun.model_validate_json(source.read_text())
    if (
        run.seed != SEEDS[0]
        or run.status != "completed"
        or run.regime != recipe(10000, plan.training_seconds_per_run)
    ):
        raise ValueError("retained first fit differs from the frozen completed recipe")
    if run.updates_through() != 10000:
        raise ValueError("retained first fit has not completed10000 updates")
    original = json.loads(
        (source.parents[2] / "initialization-parameters.json").read_text()
    )
    target = output / "l3-12261"
    target.mkdir(parents=True, exist_ok=False)
    atomic_json(target / "training/run.json", run.model_dump(mode="json"))
    tracker = track(
        experiment="mtg122-evaluation",
        run=os.environ["ETUDE_RUN"] + "-retained-first",
        directory=Path(os.environ["ETUDE_TRACK_DIR"]) / "retained-first",
        clocks=[
            Step(
                "completed_evaluations", "Retained first seed fixed endpoint comparison"
            )
        ],
        config={"producer": run.id, "plan_sha256": plan.digest},
        strict=True,
    )
    try:
        evaluate_fit(
            plan,
            run,
            target,
            "l3",
            original[str(run.seed)],
            tracker,
            allocation_stop(
                datetime.fromisoformat(os.environ["ETUDE_DEADLINE"]).timestamp()
            ),
        )
        tracker.log(completed_evaluations=1)
    except BaseException:
        tracker.finish("failed")
        raise
    else:
        tracker.finish("finished")


def run() -> None:
    """Execute only inside the authorized Mini allocation, with no retry path."""
    env = os.environ
    if not env.get("ETUDE_RUN") or not socket.gethostname().startswith("mini-heart"):
        raise ValueError("baseline requires Etude on Mini; local execution forbidden")
    if int(env["ETUDE_SEED"]) != CALIBRATION_SEED:
        raise ValueError("incorrect calibration seed")
    config = json.loads(Path(env["ETUDE_CONFIG"]).read_text())
    if "plan" in config:
        # An already measured, pre-frozen plan avoids repeating calibration to
        # obtain a favorable timing sample. The same execution admission applies.
        plan = BaselinePlan.model_validate(config["plan"])
        output = Path(env["ETUDE_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "calibrated-plan.json", plan.model_dump(mode="json"))
        if "retained_first" in config:
            remaining = (
                datetime.fromisoformat(env["ETUDE_DEADLINE"]).timestamp() - time.time()
            )
            if (
                remaining
                < 2 * plan.training_seconds_per_run
                + plan.evaluation_reserve_seconds
                + 19800
                + 600
            ):
                raise ValueError(
                    "remaining allocation cannot cover retained evaluation and two monitored fits"
                )
            complete_retained_first(plan, Path(config["retained_first"]), output)
            execute(plan, pools=("l3",), seeds=SEEDS[1:], frequent_diagnostics=True)
        else:
            execute(plan, pools=("l3",))
        return
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
