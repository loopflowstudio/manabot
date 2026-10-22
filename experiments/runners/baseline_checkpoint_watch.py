"""Bounded supplemental evaluation and a user-requested seed-boundary hold.

Run beside, never inside or by editing, a frozen training source. Imports must
resolve to that source's environment. One existing CheckpointQueue owns evaluator
leases, watchdogs and immutable attempts; Etude owns the displayed evaluation.
The hold uses the running job's existing stop-file contract only after its first
TrainingRun is complete. It does not resume or launch replacement training.
"""

import argparse
from pathlib import Path
import socket
import time

from etude.train.records import Step
from etude.train.run import track

from experiments.runners.corrected_baseline import MATCHUP
from manabot.training.checkpoint_queue import CheckpointQueue, MonitoringBudget
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorProtocol


def ready(run: TrainingRun) -> bool:
    """Select only a fully published 5,000-update artifact of the intended fit."""
    if run.seed != 12261 or run.regime.stages[0].updates != 10000:
        raise ValueError("watcher is bound to baseline seed12261 at10000 updates")
    return any(
        c.updates == 5000 and c.artifact is not None and c.error is None
        for c in run.monitoring_checkpoints
    )


def diagnostic(snapshot: Path, output: Path) -> None:
    """208 supplemental games, up to30 evaluator minutes, with exact replay."""
    protocols = tuple(
        MonitorProtocol(
            authored_matchup=MATCHUP,
            include_mirrors=True,
            purpose="supplemental-matchup-diagnostic",
            deal_seeds=tuple(range(1_226_900_000, 1_226_900_013)),
            opponent=anchor,
            bootstrap_replicates=2000,
            bootstrap_seed=12290,
        )
        for anchor in ("scripted_greedy", "random")
    )
    config = MonitoringBudget(
        seconds=1800,
        attempt_seconds=900,
        updates=(5000,),
        protocol=protocols[0],
        additional_protocols=(protocols[1],),
    )
    tracker = track(
        experiment="mtg122-evaluation",
        run="335bd645dafe-5000-diagnostic",
        directory=output / "etude",
        clocks=[Step("evaluations", "Completed supplemental anchor cohorts")],
        config=config.model_dump(mode="json"),
        strict=True,
    )
    queue = CheckpointQueue(output / "queue", config)
    try:
        while True:
            queue.tick([snapshot])
            if queue.process is None:
                break  # exhausted budget, settled queue or failed launch; retain all
            while queue.process is not None and queue.process.poll() is None:
                time.sleep(2)
        count = len(list((output / "queue").glob("attempt-*/evaluation/monitor.json")))
        tracker.log(evaluations=count)
    finally:
        queue.close()
        tracker.finish(
            "finished"
            if len(queue.attempts) == 2
            and all(a.status == "completed" for a in queue.attempts)
            else "failed"
        )


def watch(run_path: Path, stop_path: Path, output: Path) -> None:
    """Wait at most24h; preserve seed1 learning, then hold before later seeds."""
    if not socket.gethostname().startswith("mini-heart"):
        raise ValueError("this allocation is Mini-only")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    state: dict[str, object] = {
        "status": "waiting-for-5000",
        "started_unix": started,
        "run_path": str(run_path),
        "stop_path": str(stop_path),
    }
    atomic_json(output / "watch.json", state)
    identity: str | None = None
    evaluated = False
    try:
        while time.time() - started < 24 * 3600:
            run = TrainingRun.model_validate_json(run_path.read_text())
            if identity is not None and run.id != identity:
                raise ValueError("producer run identity changed")
            identity = run.id
            available = ready(run)
            if not evaluated and available:
                snapshot = output / "producer-snapshot.json"
                # Exact receipt snapshot. This does not edit or advance its producer.
                atomic_json(snapshot, run.model_dump(mode="json"))
                state["status"] = "evaluating-5000"
                atomic_json(output / "watch.json", state)
                try:
                    diagnostic(snapshot, output)
                except Exception as error:
                    state["diagnostic_error"] = f"{type(error).__name__}: {error}"
                evaluated = True
                state["status"] = "waiting-for-seed-boundary"
                atomic_json(output / "watch.json", state)
            if run.status == "completed":
                if run.updates_through() != 10000:
                    raise ValueError("completed producer has wrong target")
                stop_path.parent.mkdir(parents=True, exist_ok=True)
                stop_path.write_text(
                    "Jack Heart requested a cadence handoff after seed12261.\n"
                )
                state.update(
                    status="held-at-seed-boundary", held_unix=time.time(), run_id=run.id
                )
                break
            if run.status not in ("running", "pending"):
                state.update(status="producer-stopped", producer_status=run.status)
                break
            time.sleep(30)
        else:
            state["status"] = "watch-deadline-exhausted"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.time() - started
        atomic_json(output / "watch.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    watch(args.run, args.stop, args.out)


if __name__ == "__main__":
    main()
