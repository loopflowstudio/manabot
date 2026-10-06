"""Supervise one calibrated, eight-hour pooling/floor campaign on landed code.

The parent enforces wall deadlines outside native calls, retains child failures,
and freezes a validated ResolvedStudy before any scientific training or scoring.
All artifacts are exclusive-create; there is no retry or replacement-seed path.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from experiments.runners.pooling_filter import (
    ORDER,
    ROOT,
    SEEDS,
    Calibration,
    CalibrationArm,
    recipe_digests,
    recipes,
)
from experiments.runners.pooling_filter_analysis import factorial_report
from experiments.runners.run_training_regimes import run_study
from experiments.runners.run_value_screen import screen_plan
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import file_sha256
from manabot.training.analysis import report
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import TrainingRun
from manabot.verify.store import VerifyStore


def followup_plan(calibration: Calibration) -> ResolvedStudy:
    resolved = recipes(calibration.admitted_updates(), calibration.run_seconds())
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study="pooling-filter",
            purpose="screening",
            regime_digests=recipe_digests(resolved),
            training_seeds=SEEDS,
            paired_deals=(),
            anchor_deals=tuple(range(961160, 961185)),
            anchors=("scripted-greedy",),
            process_seconds=28800,
            uncertainty="paired-seed-descriptive",
        ),
        recipes=tuple(r.model_dump(mode="json") for r in resolved),
        allocation_seconds=28800,
        prior_campaign_seconds=0,
        runtime_identities=calibration.runtime_identities,
        projected_disk_bytes=3 * 1024**3,
        calibration_evidence=calibration.model_dump_json(),
    )


def _child(
    arguments: list[str],
    out: Path,
    timeout: float,
    *,
    training_deadline: float | None = None,
) -> None:
    """Bound the whole child process group, including native/multiprocess work."""
    with (out / "child.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.runners.run_pooling_filter",
                *arguments,
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        atomic_json(
            out / "active-child.json",
            {
                "pid": process.pid,
                "arguments": arguments,
                "timeout_seconds": timeout,
                "started_unix": time.time(),
            },
        )
        try:
            total_deadline = time.monotonic() + timeout
            while True:
                deadline = total_deadline
                if training_deadline is not None:
                    phase_path = out / "study/phase.json"
                    phase = (
                        json.loads(phase_path.read_text())
                        if phase_path.exists()
                        else {}
                    )
                    phase_deadline = phase.get("evaluation_deadline_monotonic")
                    deadline = min(
                        deadline,
                        phase_deadline
                        if phase_deadline is not None
                        else training_deadline,
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("supervisor phase/total deadline exceeded")
                try:
                    code = process.wait(timeout=min(1.0, deadline - time.monotonic()))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if code:
                raise RuntimeError(f"child exited {code}; retained {out / 'child.log'}")
        except BaseException:
            # Kill descendants even if the group leader already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def campaign(out: Path, source_commit: str) -> None:
    """Calibrate all arms, admit counts, then execute exactly one frozen cohort."""
    # A fixed checkout lock prevents independent output paths launching competitors.
    with (ROOT / ".runs/pooling-filter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        )
        if actual != source_commit or dirty:
            raise ValueError("campaign requires the exact clean landed source commit")
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        state: dict[str, object] = {
            "pid": os.getpid(),
            "source_commit": actual,
            "started_unix": time.time(),
            "status": "calibrating",
            "training_order": ORDER,
        }

        def save() -> None:
            state["seconds"] = time.monotonic() - started
            atomic_json(out / "supervisor.json", state)

        save()
        try:
            if shutil.disk_usage(out).free < 7 * 1024**3:
                raise RuntimeError("requires 3 GiB output allowance plus 4 GiB reserve")
            identities = screen_plan().runtime_identities
            atomic_json(out / "runtime-identities.json", identities)
            arms: list[CalibrationArm] = []
            for index, recipe in enumerate(recipes(40, 400)):
                _child(
                    ["--calibrate-arm", str(index), "--out", str(out)],
                    out,
                    min(420, 1800 - (time.monotonic() - started)),
                )
                path = out / recipe.id / "run.json"
                run = TrainingRun.model_validate_json(path.read_text())
                if (
                    run.status != "completed"
                    or sum(len(s.diagnostics) for s in run.stages) != 40
                ):
                    raise RuntimeError("calibration failed or omitted updates")
                arms.append(
                    CalibrationArm(
                        recipe_id=recipe.id,
                        run_path=str(path),
                        run_sha256=file_sha256(path),
                        seconds=run.seconds,
                    )
                )
            calibration = Calibration(
                arms=tuple(arms),
                seconds=time.monotonic() - started,
                source_commit=actual,
                runtime_identities=identities,
            )
            atomic_json(out / "calibration.json", calibration.model_dump(mode="json"))
            plan = followup_plan(calibration)
            atomic_json(out / "resolved-plan.json", plan.model_dump(mode="json"))
            state.update(
                status="running",
                updates_per_run=calibration.admitted_updates(),
                calibration_seconds=calibration.seconds,
            )
            save()
            _child(
                [
                    "--execute-plan",
                    str(out / "resolved-plan.json"),
                    "--out",
                    str(out / "study"),
                ],
                out,
                28800 - (time.monotonic() - started),
                training_deadline=started + 21600,
            )
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--report-only", action="store_true")
    action.add_argument("--campaign", metavar="LANDED_COMMIT")
    action.add_argument("--calibrate-arm", type=int, choices=range(4))
    action.add_argument("--execute-plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = args.out.resolve()
    if args.report_only:
        report(out)
        factorial_report(out)
    elif args.campaign:
        campaign(out, args.campaign)
    elif args.calibrate_arm is not None:
        recipe = recipes(40, 400)[args.calibrate_arm]
        with VerifyStore(out / "calibration.sqlite") as store:
            execute_regime(recipe, 10620, out / recipe.id, store)
    else:
        plan = ResolvedStudy.model_validate_json(args.execute_plan.read_text())
        if plan.protocol.study != "pooling-filter":
            raise ValueError("requires pooling-filter plan")
        calibration = Calibration.model_validate_json(plan.calibration_evidence)
        for arm in calibration.arms:
            if file_sha256(Path(arm.run_path)) != arm.run_sha256:
                raise ValueError("calibration artifact changed")
        run_study("pooling-filter", out, plan)
        factorial_report(out)


if __name__ == "__main__":
    main()
