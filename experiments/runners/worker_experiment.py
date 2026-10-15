"""Run bounded host replay and shared-CUDA worker curves on one paid machine.

Etude owns source admission, rental limits and deletion. This process records
hardware/limits and runs the same resident workload in both modes, preserving
failed/partial output. It never rents, changes credentials or retries a cell.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil
import torch

from experiments.runners.cuda_performance import Cell
from experiments.runners.pipeline_benchmark import Tape
from experiments.runners.worker_scaling import write_json
from manabot.training.source_bundle import SourceBundle, source_commit


def read_optional(path: str) -> str | None:
    source = Path(path)
    return source.read_text() if source.exists() else None


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-gpu", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if config_path := os.environ.get("ETUDE_CONFIG"):
        config = json.loads(Path(config_path).read_text())
        bundle = SourceBundle.model_validate(config["source"])
        for name, expected in bundle.files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or digest(path) != expected:
                raise ValueError(f"prepared source mismatch: {name}")
        for path in Path("managym").glob("*.so"):
            bundle.files[str(path)] = digest(path)
        write_json(Path(".manabot-source.json"), bundle.model_dump(mode="json"))
        if source_commit() != bundle.commit:
            raise ValueError("source owner rejected exact archive identity")
        write_json(args.out / "source-bundle.json", bundle.model_dump(mode="json"))
    name = torch.cuda.get_device_name()
    if name != args.expected_gpu:
        raise ValueError(f"expected {args.expected_gpu}, received {name}")
    write_json(
        args.out / "hardware.json",
        {
            "gpu": name,
            "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "compiled_architectures": torch.cuda.get_arch_list(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
            "memory_max": read_optional("/sys/fs/cgroup/memory.max"),
            "cpuinfo": read_optional("/proc/cpuinfo"),
            "memory_bytes": psutil.virtual_memory().total,
            "source_commit": source_commit(),
        },
    )
    statuses = []
    try:
        for device in ("replay", "cuda"):
            command = [
                sys.executable,
                "-m",
                "experiments.runners.worker_scaling",
                "--tape",
                "experiments/data/worker-scaling/tape-w6",
                "--out",
                str(args.out / device),
                "--device",
                device,
                "--actors",
                "64",
                "--repeats",
                "3",
                "--seconds",
                "600",
                "--rss-gib",
                "24",
            ]
            started = time.monotonic()
            result = subprocess.run(command, check=False)
            statuses.append(
                {
                    "device": device,
                    "exit_code": result.returncode,
                    "seconds": time.monotonic() - started,
                    "command": command,
                }
            )
            write_json(args.out / "status.json", {"cases": statuses})
        tape = Tape.model_validate_json(
            Path("experiments/data/worker-scaling/tape-w6/tape.json").read_text()
        )
        for streams in (512, 4096):
            root = args.out / f"ordinary-s{streams}"
            root.mkdir()
            write_json(root / "baseline.json", tape.recipe.model_dump(mode="json"))
            cell = Cell(
                capacity="w64-d1", kind="loop", batch=streams * 8, streams=streams
            )
            (root / cell.name).mkdir()
            started = time.monotonic()
            command = [
                sys.executable,
                "-m",
                "experiments.runners.cuda_performance",
                "--root",
                str(root),
                "--cell",
                cell.model_dump_json(),
            ]
            with (root / "worker.log").open("w") as log:
                try:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=70
                    )
                    code = completed.returncode
                except subprocess.TimeoutExpired:
                    code = -1
            training_result = root / cell.name / "result.json"
            training_status = (
                json.loads(training_result.read_text())["status"]
                if training_result.exists()
                else "missing"
            )
            if code == 0 and training_status != "completed":
                code = 1
            statuses.append(
                {
                    "device": f"ordinary-s{streams}",
                    "exit_code": code,
                    "training_status": training_status,
                    "seconds": time.monotonic() - started,
                    "command": command,
                }
            )
            write_json(args.out / "status.json", {"cases": statuses})
    finally:
        # Live exchange slots are disposable transport, not immutable evidence;
        # include them in the manifest nonetheless rather than hiding output.
        files = [
            {
                "path": str(p.relative_to(args.out)),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
        ]
        (args.out / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    if any(case["exit_code"] for case in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
