"""Evaluate ETU-105's six frozen endpoints with its retained source/runtime.

This evaluation-only driver runs outside the preserved source export. A detached
supervisor owns one process group per cell and the original absolute deadline.
Arena owns Commands, replay and game rows; no fitting or endpoint selection occurs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict
import torch

from manabot.arena import players
from manabot.arena.match import play_cell
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.execution import atomic_json
from manabot.training.experiment_runner import _runtime
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import (
    ArenaRow,
    MonitorProtocol,
    _manifest,
    stage_checkpoint,
)

DEADLINE = 1791415121.724962


class CellJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_path: Path
    run_sha256: str
    opponent: Literal["scripted_greedy", "random"]
    deals: tuple[int, ...]
    output: Path
    runtime: dict[str, str]


def evaluate(job: CellJob) -> None:
    """Persist every four-leg block and stop at the first invalid game/replay."""
    torch.set_num_threads(1)
    if _runtime() != job.runtime or file_sha256(job.run_path) != job.run_sha256:
        raise ValueError("retained runtime or TrainingRun changed")
    run = TrainingRun.model_validate_json(job.run_path.read_text())
    checkpoint = stage_checkpoint(run, "policy-1")
    if (
        run.status != "completed"
        or checkpoint is None
        or checkpoint.coordinates.updates != 1240
    ):
        raise ValueError("requires completed frozen 1240-update raw endpoint")
    # Reuse ordinary checkpoint hash, world/ABI/setup admission and registrations.
    manifest = _manifest(
        run,
        checkpoint.artifact,
        checkpoint.coordinates,
        MonitorProtocol(deal_seeds=job.deals, bootstrap_seed=105),
    )
    if job.opponent == "random":
        manifest.opponent = manifest.opponent.model_copy(
            update={
                "player_id": "final-random",
                "display_name": "Uniform random",
                "player_spec": {"kind": "random"},
                "compute_class_id": "random-cpu",
                "source_sha256": file_sha256(Path(players.__file__)),
            }
        )
    manifest.key = manifest.key.model_copy(
        update={
            "arena_version": "etu105-filter-scope-final-v1",
            "rating_model_version": "unrated-exploratory-screen",
            "anchor_cohort_sha256": canonical_sha256(manifest.opponent.model_dump()),
        }
    )
    job.output.mkdir(exist_ok=False)
    atomic_json(
        job.output / "admission.json",
        {
            "artifact": checkpoint.artifact,
            "coordinates": checkpoint.coordinates.model_dump(mode="json"),
            "identities": manifest.evaluation_identities,
            "candidate": manifest.candidate.model_dump(mode="json"),
            "opponent": manifest.opponent.model_dump(mode="json"),
            "key": manifest.key.model_dump(mode="json"),
        },
    )
    started = time.time()
    result: dict[str, object] = {
        "status": "running",
        "started_unix": started,
        "rows": [],
        "replay": {"passed": False},
    }
    rows: list[dict[str, object]] = []
    try:
        for block, deal in enumerate(job.deals):
            directory = job.output / f"deal-{block:03d}"
            directory.mkdir()
            raw, trace, replay = play_cell(
                key=manifest.key,
                player_a=manifest.candidate,
                player_b=manifest.opponent,
                deal_seeds=(deal,),
                out_dir=directory,
                checkpoint_paths={
                    manifest.candidate.player_id: checkpoint.artifact["path"]
                },
                game_seconds=120,
                max_commands=10000,
                comparison_seed_aliases={
                    manifest.candidate.player_id: "candidate",
                    manifest.opponent.player_id: "reference",
                },
            )
            atomic_json(directory / "rows.json", raw)
            atomic_json(directory / "trace.json", trace)
            atomic_json(directory / "replay.json", replay)
            rows.extend(raw)
            result.update(rows=rows, seconds=time.time() - started)
            atomic_json(job.output / "result.json", result)
            if (
                len(raw) != 4
                or not replay["passed"]
                or not all(ArenaRow.model_validate(row).valid for row in raw)
            ):
                raise ValueError(
                    "invalid arena block; retain failure without replacement"
                )
        result.update(status="completed", replay={"passed": True})
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(seconds=time.time() - started, finished_unix=time.time())
        atomic_json(job.output / "result.json", result)


def supervise(source: Path, output: Path) -> None:
    """Hold existing host/evaluator leases; never overlap ETU-118 or extend budget."""
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((source / "frozen-plan.json").read_text())
    experiment = json.loads((source / "science/experiment.json").read_text())
    if (
        plan["admission"]["original_deadline_unix"] != DEADLINE
        or experiment["status"] != "completed"
    ):
        raise ValueError("wrong frozen cohort")
    started = time.time()
    stop = min(DEADLINE, started + 7200)
    state: dict[str, object] = {
        "status": "running",
        "started_unix": started,
        "deadline_unix": stop,
        "original_deadline_unix": DEADLINE,
        "attempts": [],
        "driver_sha256": file_sha256(Path(__file__)),
        "frozen_plan_sha256": file_sha256(source / "frozen-plan.json"),
    }
    attempts: list[dict[str, object]] = []
    atomic_json(output / "supervisor.json", state)
    try:
        with ExitStack() as stack:
            # The historical source was patched to use a retained campaign lease root.
            roots = [
                Path.home() / ".cache/manabot",
                source.parent / "etu105-mini-filter-scope-20261006-1-leases",
            ]
            for root in roots:
                root.mkdir(parents=True, exist_ok=True)
                for name in ("experiment-host.lock", "checkpoint-evaluator.lock"):
                    lease = stack.enter_context((root / name).open("a+b"))
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for family, opponent in (
                ("untouched_final", "scripted_greedy"),
                ("random_diagnostic", "random"),
            ):
                for attempt in experiment["attempts"]:
                    if time.time() >= stop:
                        raise TimeoutError("original remaining allocation exhausted")
                    path = Path(attempt["path"]) / "run.json"
                    job = CellJob(
                        run_path=path,
                        run_sha256=file_sha256(path),
                        opponent=opponent,
                        deals=tuple(plan[family]["deal_seeds"]),
                        output=output / f"{family}-{attempt['ordinal']:04d}",
                        runtime=experiment["runtime"],
                    )
                    job_path = output / f"job-{family}-{attempt['ordinal']:04d}.json"
                    atomic_json(job_path, job.model_dump(mode="json"))
                    record: dict[str, object] = {
                        "job": str(job_path),
                        "case": attempt["case"],
                        "seed": attempt["seed"],
                        "family": family,
                        "started_unix": time.time(),
                        "status": "running",
                    }
                    attempts.append(record)
                    state["attempts"] = attempts
                    atomic_json(output / "supervisor.json", state)
                    with (job.output.with_suffix(".log")).open("wb") as log:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--job",
                                str(job_path),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        record["pid"] = process.pid
                        atomic_json(output / "supervisor.json", state)
                        try:
                            code = process.wait(
                                timeout=min(600, max(0.001, stop - time.time()))
                            )
                        except BaseException:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            record.update(
                                status="interrupted", finished_unix=time.time()
                            )
                            raise
                    record.update(
                        status="completed" if code == 0 else "failed",
                        exit_code=code,
                        finished_unix=time.time(),
                    )
                    atomic_json(output / "supervisor.json", state)
                    if code != 0:
                        raise RuntimeError(f"cell failed: {job_path}")
            state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state.update(finished_unix=time.time(), seconds=time.time() - started)
        atomic_json(output / "supervisor.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.job:
        evaluate(CellJob.model_validate_json(args.job.read_text()))
    elif args.source and args.out:
        supervise(args.source, args.out)
    else:
        parser.error("provide --job or --source and --out")


if __name__ == "__main__":
    main()
