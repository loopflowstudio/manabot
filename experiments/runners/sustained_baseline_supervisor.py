"""Detached ETU-118 process owner and bounded editable-notebook refresher.

The Experiment coordinator owns learner/evaluator leases, deadlines and recovery.
This supervisor never retries learning. Its retained receipt accounts separately
for report subprocess occupancy within the plan's final/report reserve.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.runners.sustained_baseline import SustainedPlan
from manabot.arena.models import file_sha256
from manabot.training.execution import atomic_json


def supervise(plan_path: Path, out: Path) -> None:
    plan = SustainedPlan.model_validate_json(plan_path.read_text())
    plan.admit()
    receipt_path = out.with_suffix(".supervisor.json")
    if receipt_path.exists() or out.exists():
        raise FileExistsError("retain existing attempts; supervision never retries")
    started = time.time()
    reports: list[dict[str, str | float | int | None]] = []
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "experiments.runners.sustained_baseline",
            "run",
            str(plan_path),
            str(out),
        ],
        start_new_session=True,
    )

    def save() -> None:
        atomic_json(
            receipt_path,
            {
                "pid": os.getpid(),
                "child_pid": child.pid,
                "started_unix": started,
                "last_seen_unix": time.time(),
                "exit_code": child.poll(),
                "plan_sha256": file_sha256(plan_path),
                "report_process_seconds": sum(float(r["seconds"]) for r in reports),
                "report_attempts": reports,
            },
        )

    last_report = 0.0
    while child.poll() is None:
        save()
        spent = sum(float(r["seconds"]) for r in reports)
        if (
            (out / "comparison.ipynb").exists()
            and not (out / "final-result.json").exists()
            and time.monotonic() - last_report >= 300
            and spent + 120 <= plan.report_reserve_seconds
        ):
            tick = time.monotonic()
            code, error = None, None
            with out.with_suffix(".reports.log").open("ab") as log:
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "experiments.runners.sustained_baseline",
                            "report",
                            str(out),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                        check=False,
                    )
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    error = "notebook refresh exceeded 120 seconds"
            reports.append(
                {
                    "started_unix": time.time() - (time.monotonic() - tick),
                    "seconds": time.monotonic() - tick,
                    "exit_code": code,
                    "error": error,
                }
            )
            last_report = time.monotonic()
            save()
        time.sleep(5)
    save()
    if child.returncode:
        raise SystemExit(child.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    supervise(args.plan.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
