"""Bounded ETU-108 benchmark supervisor; run from the repository root.

The dependency-free parent retains failed attempts and kills its own process
group on timeout/interruption. Workloads reuse the existing training machinery.
This is a disposable research harness, not a distributed training executor.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time


def _read(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=3)
    if result.returncode:
        return f"unavailable (exit {result.returncode})"
    return result.stdout.strip()


@dataclass(frozen=True)
class Host:
    platform: str
    machine: str
    cpu: str
    memory_bytes: str
    logical_cpus: int | None
    load_average: tuple[float, float, float]


def _host() -> Host:
    mac = platform.system() == "Darwin"
    return Host(
        platform.platform(),
        platform.machine(),
        _read(["sysctl", "-n", "machdep.cpu.brand_string"])
        if mac
        else platform.processor(),
        _read(["sysctl", "-n", "hw.memsize"]) if mac else "unavailable",
        os.cpu_count(),
        os.getloadavg(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inventory", "simulator", "inference", "train", "complete"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--timeout", type=float, default=110)
    args = parser.parse_args()
    attempt_start = time.monotonic()
    if not 0 < args.seconds <= 10 or not 0 < args.timeout <= 110:
        parser.error("seconds must be in (0,10]; timeout must be in (0,110]")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [
        "uv",
        "run",
        "--no-sync",
        "-m",
        "experiments.runners.distributed_workloads",
        "--mode",
        args.mode,
        "--out",
        str(out / "workload"),
        "--seconds",
        str(args.seconds),
    ]
    manifest = {
        "schema": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": asdict(_host()),
        "mode": args.mode,
        "command": command,
        "timeout_seconds": args.timeout,
        "threads": 1,
        "device": "cpu",
        "head": _read(["git", "rev-parse", "HEAD"]),
        "tracked_diff_sha256": hashlib.sha256(
            _read(["git", "diff", "HEAD"]).encode()
        ).hexdigest(),
        "files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("experiments/runners/distributed_workloads.py"),
                Path("uv.lock"),
            )
        },
        "status": "prepared",
        "limits": [
            "No energy or thermal measurement",
            "Child rusage peak RSS is not aggregate process-tree peak",
            "No hardware comparison under contention",
            "No distributed contributions measured",
            "Timed-out child store may require reconciliation; summary is not TrainingRun authority",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.mode == "inventory":
        manifest.update(status="completed")
        (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    if not Path(".venv/bin/python").exists():
        manifest.update(
            status="blocked", error="No local .venv; dependencies were not installed"
        )
        (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit(2)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        WANDB_MODE="disabled",
    )
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    status = "failed"
    try:
        with (
            (out / "stdout.log").open("wb") as stdout,
            (out / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command, env=env, stdout=stdout, stderr=stderr, start_new_session=True
            )
            remaining = max(0.001, args.timeout - (time.monotonic() - attempt_start))
            code = process.wait(timeout=remaining)
            status = "completed" if code == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except KeyboardInterrupt:
        status = "interrupted"
    except OSError as error:
        manifest["error"] = str(error)
    finally:
        if process is not None:
            # Kill only the group this supervisor created, including arena children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        manifest.update(
            status=status,
            wall_seconds=time.monotonic() - start,
            attempt_wall_seconds=time.monotonic() - attempt_start,
            exit_code=None if process is None else process.returncode,
            child_cpu_seconds=usage.ru_utime
            + usage.ru_stime
            - before.ru_utime
            - before.ru_stime,
            child_maxrss_bytes=usage.ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            load_after=os.getloadavg(),
        )
        temporary = out / "result.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(out / "result.json")
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
