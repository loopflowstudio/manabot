"""Run bounded collector and inference-ownership curves on one paid machine.

Etude owns source admission, rental limits and deletion. This process records
hardware/limits and runs the same resident workload across selected modes,
preserving failed/partial output. It never rents, changes credentials or retries
a cell. Ordinary TrainingRun controls anchor the single-collector training path.
"""

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Trial:
    name: str
    device: str
    repeats: int = 3
    seconds: int = 600
    profile: bool = False
    reference: bool = False


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay-only", action="store_true")
    mode.add_argument("--serving-ownership", action="store_true")
    mode.add_argument("--serving-preparation", action="store_true")
    mode.add_argument("--collection-profile", action="store_true")
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
            "selected_device": torch.cuda.current_device(),
            "visible_cuda_devices": torch.cuda.device_count(),
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
        if args.collection_profile:
            trials = []
        elif args.serving_preparation:
            trials = [
                Trial(
                    f"preparation-{index}-{'reference' if reference else 'prepared'}",
                    "cuda",
                    repeats=1,
                    seconds=180,
                    reference=reference,
                )
                for index, reference in enumerate(
                    (True, False, False, True, True, False)
                )
            ]
        elif args.serving_ownership:
            # Interleave independent complete process-pool lifetimes, retaining
            # startup in each receipt and all three fixed-work worker counts.
            trials = [
                Trial(f"ownership-{index}-{device}", device, repeats=1, seconds=180)
                for index, device in enumerate(
                    ("cuda", "cuda-local", "cuda-local", "cuda", "cuda", "cuda-local")
                )
            ]
            trials.append(
                Trial("service-profile", "cuda", repeats=1, seconds=45, profile=True)
            )
        else:
            trials = [
                Trial(device, device)
                for device in (("replay",) if args.replay_only else ("replay", "cuda"))
            ]
        for trial in trials:
            command = [
                sys.executable,
                "-m",
                "experiments.runners.worker_scaling",
                "--tape",
                "experiments/data/worker-scaling/tape-w6",
                "--out",
                str(args.out / trial.name),
                "--device",
                trial.device,
                "--actors",
                "64",
                "--repeats",
                str(trial.repeats),
                "--seconds",
                str(trial.seconds),
                "--rss-gib",
                "24",
                *(["--capture-replay"] if trial.device == "replay" else []),
                *(["--profile"] if trial.profile else []),
                *(["--reference-policy"] if trial.reference else []),
            ]
            started = time.monotonic()
            result = subprocess.run(command, check=False)
            statuses.append(
                {
                    "device": trial.name,
                    "exit_code": result.returncode,
                    "seconds": time.monotonic() - started,
                    "command": command,
                }
            )
            write_json(args.out / "status.json", {"cases": statuses})
        tape = Tape.model_validate_json(
            Path("experiments/data/worker-scaling/tape-w6/tape.json").read_text()
        )
        training_cases = (
            [
                (
                    f"training-{index}-{'reference' if reference else 'prepared'}-s{streams}",
                    streams,
                    reference,
                )
                for index, reference in enumerate(
                    (True, False, False, True, True, False)
                )
                for streams in (512, 4096)
            ]
            if args.serving_preparation
            else [
                (f"ordinary-s{streams}", streams, False)
                for streams in (() if args.replay_only else (512, 4096))
            ]
        )
        if args.collection_profile:
            training_cases = [
                (
                    f"{kind}-{'reference' if reference else 'prepared'}-s{streams}",
                    streams,
                    reference,
                )
                for kind in ("control", "torch", "python")
                for reference in (True, False)
                for streams in ((4096,) if kind == "python" else (512, 4096))
            ]
        for name, streams, reference in training_cases:
            root = args.out / name
            root.mkdir()
            write_json(root / "baseline.json", tape.recipe.model_dump(mode="json"))
            cell = Cell(
                capacity="w64-d1", kind="loop", batch=streams * 8, streams=streams
            )
            (root / cell.name).mkdir()
            started = time.monotonic()
            profile_kind = name.split("-", 1)[0] if args.collection_profile else None
            profiling = profile_kind in ("torch", "python")
            command = [
                sys.executable,
                "-m",
                (
                    "experiments.runners.collection_attribution"
                    if profiling
                    else "experiments.runners.inference_preparation"
                    if args.serving_preparation or args.collection_profile
                    else "experiments.runners.cuda_performance"
                ),
                "--root",
                str(root),
                "--cell",
                cell.model_dump_json(),
                *(["--profile-kind", profile_kind] if profiling else []),
                *(["--reference-policy"] if reference else []),
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
                    "device": name,
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
