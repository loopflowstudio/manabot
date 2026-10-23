"""Credential-free CLI with a parent watchdog that preserves timeout evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

from manabot.verify.learnability.report import save
from manabot.verify.learnability.schema import Config, Result


def source_identity() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(paths) - {""}):
        path = Path(name)
        if path.is_file() and not name.startswith(("scratch/", ".lf/")):
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return commit, digest.hexdigest()


def worker(out: Path) -> int:
    result = Result.model_validate_json((out / "result.json").read_text())
    try:
        from manabot.verify.learnability.capacity import run

        run(out, result)
    except Exception as error:
        result.status = "error"
        result.errors.append(f"{type(error).__name__}: {error}")
        raise
    finally:
        save(out, result)
    return 0 if result.status == "passed" else 1


def supervise(out: Path, config: Config) -> int:
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite/retry a failed run.
    source, digest = source_identity()
    result = Result(
        source=source,
        source_digest=digest,
        config=config,
        platform=platform.platform(),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    save(out, result)
    # The supervisor owns total runtime, including startup and forced termination.
    start = time.perf_counter()
    cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (out / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "manabot.verify.learnability",
                "--worker",
                "--out",
                str(out),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        try:
            code = process.wait(timeout=config.timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    result = Result.model_validate_json((out / "result.json").read_text())
    result.wall_seconds = time.perf_counter() - start
    cpu_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.cpu_seconds = (cpu_end.ru_utime + cpu_end.ru_stime) - (
        cpu_start.ru_utime + cpu_start.ru_stime
    )
    if result.status == "running" or (code and result.status == "passed"):
        code = code or 1
        result.status = "timeout" if code == 124 else "error"
        result.errors.append(
            f"Worker exited {code}; unfinished checks are not passes. See worker.log."
        )
    result.artifacts["worker.log"] = hashlib.sha256(
        (out / "worker.log").read_bytes()
    ).hexdigest()
    save(out, result)
    print(f"{result.status}: {out / 'report.md'} ({result.wall_seconds:.1f}s)")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--pr", action="store_true", help="One capacity seed; explicitly omit PPO"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.out)
    config = Config(seeds=(17,), ppo=False) if args.pr else Config()
    return supervise(args.out, config)


if __name__ == "__main__":
    raise SystemExit(main())
