"""Score seat-comparison exports against the heuristic while the jobs train.

For each job, the export nearest each learner-decision milestone is scored once
its successor exists (so the choice is final), plus each seed's shared initial
model. Scoring reuses the live evaluator's pieces: committed job manifests,
digest-verified fetches, `checkpoint_manifest` admission and the Etude CPU
evaluation child. Up to `--parallel` children run at once. The observer never
rents or changes training; failed attempts stay in the ledger, never replayed.

    uv run python -m experiments.seat_comparison_follow --directory DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.seat_comparison import SEEDS
from manabot.infra.artifacts import verify_file
from manabot.remote.job_client import fetch_job_file, job_manifest, load_job
from manabot.remote.job_store import S3JobStore
from manabot.remote.jobs import Job, JobRecord
from manabot.training.execution import atomic_json
from manabot.training.monitor_evaluation import (
    Checkpoint,
    MonitorProtocol,
    checkpoint_manifest,
    stage_checkpoint,
)
from manabot.training.records import TrainingCoordinates, TrainingRun

MILESTONES = tuple(m * 1_000_000 for m in (1, 2, 4, 8, 16, 32, 64))
# Fresh deals, fixed before any scoring (protocols/seat-comparison-v1.md).
PROTOCOL = MonitorProtocol(
    deal_seeds=tuple(range(6_400_000_000, 6_400_000_025)),
    include_mirrors=True,
    opponent="heuristic",
    game_seconds=600,
    bootstrap_seed=17,
    bootstrap_replicates=4000,
)
JOBS = tuple(
    f"seat-comparison-v1-{seed}-{arm}" for seed in SEEDS for arm in ("one", "both")
)


def _selected(run: TrainingRun) -> dict[str, Checkpoint]:
    """Label -> checkpoint; a milestone is chosen once a checkpoint lies beyond it.

    Candidates are the periodic raw exports and the stage's final raw export
    (which a deadline pause also writes). The initial raw export is the shared
    starting point.
    """
    stage = next(s for s in run.stages if s.id == "self-play")
    found = {
        c.artifact["sha256"]: Checkpoint(artifact=c.artifact, coordinates=c)
        for c in run.monitoring_checkpoints
        if c.artifact is not None and c.updates > 0
    }
    final = stage_checkpoint(run, "self-play")
    if final is not None and final.coordinates.updates > 0:
        found.setdefault(final.artifact["sha256"], final)
    trained = sorted(found.values(), key=lambda c: c.coordinates.learner_transitions)
    chosen: dict[str, Checkpoint] = {}
    initial = stage.artifacts.get("initial_raw")
    if initial is not None:
        chosen["initial"] = Checkpoint(
            artifact=initial,
            coordinates=TrainingCoordinates(
                stage_id="self-play", updates=0, training_seconds=0
            ),
        )
    for target in MILESTONES:
        if not trained or trained[-1].coordinates.learner_transitions < target:
            break
        chosen[f"{target // 1_000_000}M"] = min(
            trained, key=lambda c: abs(c.coordinates.learner_transitions - target)
        )
    return chosen


def _record(spec: Job) -> JobRecord | None:
    raw = S3JobStore(spec.prefix).read("runtime/record.json")
    return None if raw is None else JobRecord.model_validate_json(raw.data)


def _admit(job_id: str, directory: Path, ledger: dict) -> list[tuple[Path, str]]:
    """Write admissions for newly decidable milestones; return (target, key) pairs."""
    spec = load_job(job_id)
    record = _record(spec)
    if record is None or record.manifest is None:
        return []
    ledger["jobs"][job_id] = {
        "updates": record.updates,
        "phase": record.phase,
        "terminal": record.terminal,
        "heartbeat_at": record.heartbeat_at,
    }
    cache = directory / "cache" / job_id
    manifest = job_manifest(spec, record, cache)
    entries = {e.relative_path: e for e in manifest.bundle.files}
    run_path = fetch_job_file(spec, record, "run/run.json", cache)
    if run_path is None:
        return []
    run = TrainingRun.model_validate_json(run_path.read_bytes())
    seed = job_id.split("-")[-2]
    admitted = []
    for label, chosen in _selected(run).items():
        checkpoint = chosen.coordinates
        # Both arms of a seed share identical initial weights; score it once.
        key = f"{seed}-initial" if label == "initial" else f"{job_id}-{label}"
        if key in ledger["attempts"]:
            continue
        artifact = chosen.artifact
        name = "run/" + Path(artifact["path"]).name
        entry = entries.get(name)
        if entry is None or entry.sha256 != artifact["sha256"]:
            continue  # not yet in this committed generation
        path = fetch_job_file(spec, record, name, cache)
        if path is None:
            continue
        verify_file(path, artifact["sha256"], artifact["bytes"])
        admission = checkpoint_manifest(
            run, {**artifact, "path": str(path.resolve())}, checkpoint, PROTOCOL
        )
        target = directory / "evaluations" / key
        target.mkdir(parents=True, exist_ok=True)
        atomic_json(target / "admission.json", admission.model_dump(mode="json"))
        ledger["attempts"][key] = {
            "job": job_id,
            "label": label,
            "sha256": artifact["sha256"],
            "updates": checkpoint.updates,
            "learner_decisions": checkpoint.learner_transitions,
            "environment_decisions": checkpoint.environment_decisions,
            "games": checkpoint.games,
            "training_seconds": checkpoint.training_seconds,
            "phase": "queued",
        }
        admitted.append((target, key))
    return admitted


def follow(directory: Path, parallel: int, until: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ledger.json"
    ledger = (
        json.loads(path.read_text()) if path.exists() else {"jobs": {}, "attempts": {}}
    )
    for item in ledger["attempts"].values():
        if item["phase"] in ("queued", "running"):
            item["phase"] = "interrupted"  # never silently replayed
    queue: list[tuple[Path, str]] = []
    running: dict[str, subprocess.Popen[bytes]] = {}
    polled = 0.0
    while time.time() < until:
        if time.time() - polled > 300:
            polled = time.time()
            for job_id in JOBS:
                try:
                    queue += _admit(job_id, directory, ledger)
                except Exception as error:
                    ledger["error"] = f"{job_id}: {type(error).__name__}: {error}"
            atomic_json(path, ledger)
        while queue and len(running) < parallel:
            target, key = queue.pop(0)
            console = (target / "console.log").open("ab")
            running[key] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "manabot.training.etude_evaluation",
                    "--manifest",
                    str(target / "admission.json"),
                ],
                stdout=console,
                stderr=console,
                start_new_session=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            ledger["attempts"][key].update(phase="running", started_at=time.time())
            atomic_json(path, ledger)
        for key, child in list(running.items()):
            if child.poll() is None:
                continue
            done = (directory / "evaluations" / key / "result.json").is_file()
            ledger["attempts"][key].update(
                phase="completed" if child.returncode == 0 and done else "failed",
                exit_code=child.returncode,
                finished_at=time.time(),
            )
            del running[key]
            atomic_json(path, ledger)
        finished = all(ledger["jobs"].get(j, {}).get("terminal") for j in JOBS)
        if finished and not queue and not running and time.time() - polled < 300:
            break
        time.sleep(10)
    atomic_json(path, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--hours", type=float, default=30)
    args = parser.parse_args()
    follow(args.directory.resolve(), args.parallel, time.time() + args.hours * 3600)


if __name__ == "__main__":
    main()
