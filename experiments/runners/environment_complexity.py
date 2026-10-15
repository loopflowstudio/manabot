"""MTG-122 pool declarations and remote-only bounded calibration.

The SDK supplies source, attempt, output and deadline. TrainingRun and arena
receipts retain the learning and complete-game evidence. Calibration uses the
reserved fourth seed, not a scored seed; failure never becomes a smaller cohort.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Literal

from manabot.arena.matchup import AuthoredMatchup
from manabot.training.execution import execute_regime
from manabot.training.models import TrainingCoordinates, TrainingRegime, TrainSelfPlay
from manabot.training.monitor_evaluation import MonitorProtocol, evaluate_checkpoint
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]
Pool = Literal["l1", "l2", "l3"]
PACK = "mtg122-complexity-v1"


def roster(pool: Pool) -> AuthoredMatchup:
    roles = {
        "l1": ("l1_swift", "l1_heavy"),
        "l2": ("l2_swift", "l2_heavy"),
        "l3": ("ur_lessons", "gw_allies"),
    }[pool]
    return AuthoredMatchup(pack_key=PACK, role_a=roles[0], role_b=roles[1])


def regime(pool: Pool, *, updates: int, seconds: float) -> TrainingRegime:
    """Identical learner and catalog; only exact deck composition changes."""
    recipe = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/early-learning-rate-mini-source.json").read_text()
    )
    recipe.id = f"mtg122-{pool}"
    recipe.agent.semantic_pack = PACK
    recipe.match = roster(pool).match()
    recipe.wall_seconds = seconds
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = updates
    stage.matchup_curriculum = "cross-balanced"
    stage.execution.wall_seconds = seconds
    return TrainingRegime.model_validate(recipe.model_dump())


def calibrate() -> None:
    """One remote hour at most, including the launcher's setup/build time."""
    env = os.environ
    if not env.get("ETUDE_RUN") or not env.get("ETUDE_DEADLINE"):
        raise ValueError("calibration requires an Etude-owned remote allocation")
    if not socket.gethostname().startswith("mini-heart"):
        raise ValueError(
            "this admitted calibration is Mini-only; local execution forbidden"
        )
    if int(env["ETUDE_SEED"]) != 12250:
        raise ValueError("calibration requires reserved seed 12250")
    deadline = datetime.fromisoformat(env["ETUDE_DEADLINE"]).timestamp()
    output = Path(env["ETUDE_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "calibration.json"
    if summary.exists():
        raise ValueError("calibration evidence already exists")
    start = time.time()
    root_run, root_tracking = env["ETUDE_RUN"], env["ETUDE_TRACK_DIR"]
    receipt: dict[str, object] = {
        "status": "running",
        "host": socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(),
        "seed": 12250,
        "deadline": deadline,
        "pools": [],
    }
    completed: list[dict[str, object]] = []
    try:
        for index, pool in enumerate(("l1", "l2", "l3")):
            if deadline - time.time() < 900:
                raise RuntimeError(
                    "insufficient remaining allocation for next calibration pool"
                )
            # Separate child tracker identities retain a single outer SDK allocation.
            env["ETUDE_RUN"] = f"{root_run}-{pool}"
            env["ETUDE_TRACK_DIR"] = str(Path(root_tracking) / pool)
            spec = regime(pool, updates=100, seconds=600)
            directory = output / pool
            directory.mkdir()
            (directory / "regime.json").write_text(spec.model_dump_json(indent=2))
            with VerifyStore(directory / "training.sqlite") as store:
                run = execute_regime(
                    spec,
                    12250,
                    directory / "training",
                    store,
                    checkpoint_updates=(50,),
                    stop_path=Path(env["ETUDE_STOP_FILE"]),
                )
            if run.status != "completed" or run.selected_artifact is None:
                raise RuntimeError(f"{pool} calibration training failed: {run.error}")
            stage = run.stages[-1]
            coordinates = TrainingCoordinates(
                stage_id=stage.id,
                updates=run.updates_through(),
                training_seconds=run.seconds,
                active_training_seconds=stage.collection_seconds
                + stage.learning_seconds,
                environment_decisions=stage.environment_decisions,
                learner_transitions=stage.learner_transitions,
                optimizer_exposures=stage.optimizer_exposures,
                games=stage.games,
                optimizer_updates=stage.optimizer_updates,
            )
            for anchor in ("scripted_greedy", "random"):
                if (
                    deadline - time.time() < 180
                    or Path(env["ETUDE_STOP_FILE"]).exists()
                ):
                    raise RuntimeError(
                        "allocation exhausted before calibration evaluation"
                    )
                result = evaluate_checkpoint(
                    run,
                    run.selected_artifact,
                    coordinates,
                    directory / anchor,
                    protocol=MonitorProtocol(
                        authored_matchup=roster(pool),
                        include_mirrors=True,
                        deal_seeds=(1_225_000_000 + index,),
                        opponent=anchor,
                        bootstrap_replicates=100,
                        game_seconds=120,
                    ),
                )
                if result.status != "completed":
                    raise RuntimeError(f"{pool}/{anchor} incomplete calibration matrix")
            completed.append(
                {
                    "pool": pool,
                    "run_id": run.id,
                    "training_seconds": run.seconds,
                    "updates": run.updates_through(),
                }
            )
            receipt["pools"] = completed
            summary.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["status"] = "completed"
        (output / "etude-result.json").write_text(json.dumps({"clocks": {"pools": 3}}))
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["seconds"] = time.time() - start
        summary.write_text(json.dumps(receipt, indent=2) + "\n")
        env["ETUDE_RUN"], env["ETUDE_TRACK_DIR"] = root_run, root_tracking


if __name__ == "__main__":
    calibrate()
