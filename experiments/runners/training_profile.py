"""Basic host-time attribution in ordinary GPU training, with matched controls.

Each Case runs in a fresh process. The parent fixes source/hardware and bounds the
whole study; Etude owns rental limits and deletion. Profiler-off controls measure
observer overhead. Full trajectory witnesses belong only to deterministic proof
cases. No PyTorch traces, extra subphase synchronization or evaluation is used.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from pydantic import BaseModel, Field
import torch

from experiments.runners.inference_preparation import (
    checkpoint_witnesses,
    collection_witnesses,
)
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import TrainingRegime, TrainSelfPlay
from manabot.training.source_bundle import SourceBundle, source_commit
from manabot.verify.store import VerifyStore


class Case(BaseModel):
    name: str
    width: int = Field(gt=0)
    layers: int = Field(gt=0)
    streams: int = Field(gt=0)
    profile: bool
    updates: int = Field(default=16, gt=0)
    proof: bool = False


def cases() -> list[Case]:
    result = [
        Case(
            name=f"proof-s{streams}-{'on' if enabled else 'off'}",
            width=384,
            layers=8,
            streams=streams,
            profile=enabled,
            updates=3,
            proof=True,
        )
        for streams in (64, 512)
        for enabled in (False, True)
    ]
    for repeat, enabled in enumerate((False, True, True, False, False, True)):
        for width, layers in ((64, 1), (384, 8)):
            for streams in (64, 512):
                result.append(
                    Case(
                        name=f"timing-{repeat}-w{width}-d{layers}-s{streams}-{'on' if enabled else 'off'}",
                        width=width,
                        layers=layers,
                        streams=streams,
                        profile=enabled,
                    )
                )
    return result


def child(root: Path, case: Case, *, device: str = "cuda") -> None:
    """Execute the real recipe; CPU is explicit and used only by correctness tests."""
    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if case.proof:
        torch.use_deterministic_algorithms(True)
    recipe = TrainingRegime.model_validate_json((root / "baseline.json").read_text())
    recipe.agent.hidden_dim, recipe.agent.attention_layers = case.width, case.layers
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.streams, stage.transitions, stage.updates = case.streams, 8, case.updates
    stage.execution.device = device
    stage.execution.threads, stage.execution.workers = 1, 1
    stage.execution.profile_collection = case.profile
    stage.execution.wall_seconds, recipe.wall_seconds = 110, 115
    recipe = TrainingRegime.model_validate(recipe.model_dump())
    out = root / case.name
    out.mkdir()
    atomic_json(out / "case.json", case.model_dump(mode="json"))
    atomic_json(out / "regime.json", recipe.model_dump(mode="json"))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    context = collection_witnesses(out) if case.proof else nullcontext()
    with context, VerifyStore(out / "training.sqlite") as store:
        run = execute_regime(recipe, 10349, out / "run", store)
    atomic_json(out / "witnesses.json", checkpoint_witnesses(out / "run"))
    atomic_json(
        out / "result.json",
        {
            "status": run.status,
            "run_id": run.id,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if device == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if device == "cuda"
            else None,
            "timing_basis": "host wall intervals; existing collection/learning synchronization only",
        },
    )
    if run.status != "completed":
        raise ValueError("Incomplete training; retained run owns its failure")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def study(out: Path, expected_gpu: str) -> None:
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())
    bundle = SourceBundle.model_validate(config["source"])
    for name, digest in bundle.files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or _digest(path) != digest:
            raise ValueError(f"Admitted source differs: {name}")
    for path in Path("managym").glob("*.so"):
        bundle.files[str(path)] = _digest(path)
    atomic_json(".manabot-source.json", bundle.model_dump(mode="json"))
    if source_commit() != bundle.commit:
        raise ValueError("Source owner rejected archive identity")
    atomic_json(out / "source-bundle.json", bundle.model_dump(mode="json"))
    gpu = torch.cuda.get_device_name()
    if gpu != expected_gpu:
        raise ValueError(f"Expected {expected_gpu}, received {gpu}")
    atomic_json(
        out / "hardware.json",
        {
            "gpu": gpu,
            "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text(),
            "memory_max": Path("/sys/fs/cgroup/memory.max").read_text(),
            "cpuinfo": Path("/proc/cpuinfo").read_text(),
            "source_commit": source_commit(),
        },
    )
    (out / "baseline.json").write_bytes(
        Path("experiments/regimes/gpu-training-profile.json").read_bytes()
    )
    trials = cases()
    deadline = time.monotonic() + 900
    statuses: list[dict[str, str | int | float | None]] = []
    try:
        for case in trials:
            remaining = deadline - time.monotonic()
            if remaining < 125:
                statuses.append({"name": case.name, "status": "not_started_deadline"})
                continue
            started = time.monotonic()
            status: dict[str, str | int | float | None] = {
                "name": case.name,
                "status": "running",
            }
            statuses.append(status)
            try:
                environment = dict(os.environ)
                environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
                if case.proof:
                    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                with (out / f"{case.name}.log").open("w") as log:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "experiments.runners.training_profile",
                            "--out",
                            str(out),
                            "--case",
                            case.model_dump_json(),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=125,
                        env=environment,
                    )
                status.update(
                    status="completed" if result.returncode == 0 else "failed",
                    exit_code=result.returncode,
                )
            except subprocess.TimeoutExpired:
                status.update(status="timeout")
            finally:
                status["process_seconds"] = time.monotonic() - started
                atomic_json(
                    out / "status.json",
                    {
                        "cases": [c.model_dump(mode="json") for c in trials],
                        "attempts": statuses,
                    },
                )
    finally:
        atomic_json(
            out / "status.json",
            {
                "cases": [c.model_dump(mode="json") for c in trials],
                "attempts": statuses,
            },
        )
        atomic_json(
            out / "manifest.json",
            [
                {
                    "path": str(p.relative_to(out)),
                    "bytes": p.stat().st_size,
                    "sha256": _digest(p),
                }
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            ],
        )
    if any(s["status"] != "completed" for s in statuses):
        raise RuntimeError("Study incomplete; failures and unstarted cells retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-gpu")
    parser.add_argument("--case", type=Case.model_validate_json)
    args = parser.parse_args()
    if args.case:
        child(args.out, args.case)
    elif args.expected_gpu:
        study(args.out, args.expected_gpu)
    else:
        parser.error("--expected-gpu is required for a study")


if __name__ == "__main__":
    main()
