"""Bounded CUDA validation using the existing collector and capacity instruments.

Fresh child processes isolate each cell. The parent preserves failure outcomes
and packages returned evidence. This research driver changes no training rule.
"""

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch

import numpy as np
from pydantic import BaseModel
import torch

from experiments.runners.cuda_collection_profile import Trace, busy_microseconds
from experiments.runners.cuda_performance import Cell
from manabot.arena.models import file_sha256
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.model.world import checkpoint_world
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime
import managym


class Attempt(BaseModel):
    name: str
    command: list[str]
    status: str = "running"
    seconds: float = 0
    exit_code: int | None = None


def digest_batch(batch: RolloutBatch) -> str:
    digest = hashlib.sha256()
    for item in fields(batch):
        value = getattr(batch, item.name)
        arrays = value if isinstance(value, dict) else {item.name: value}
        for key, array in sorted(arrays.items()):
            if array is None:
                continue
            if not isinstance(array, np.ndarray):
                raise TypeError("unexpected rollout value")
            digest.update(
                json.dumps([item.name, key, str(array.dtype), array.shape]).encode()
            )
            digest.update(array.tobytes())
    return digest.hexdigest()


def collector(recipe: TrainingRegime, streams: int, seed: int) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        ObservationSpace(recipe.observation),
        Match(recipe.match),
        Reward(RewardHypers()),
        num_envs=streams,
        seed=seed,
        opponent_mode="self",
        device="cuda",
        recovery_max_microsteps=1_000_000,
    )


def collection(
    recipe: TrainingRegime, out: Path, streams: int, seed: int, *, profile: bool
) -> None:
    torch.manual_seed(seed)
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent).to("cuda").eval()
    batches: Counter[int] = Counter()
    original = agent.forward_distribution

    def forward(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batches[len(obs["actions_valid"])] += 1
        return original(obs)

    samples: list[dict[str, object]] = []
    previous: str | None = None
    for repeat in range(1 if profile else 2):
        env = collector(recipe, streams, seed)
        batches.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with patch.object(agent, "forward_distribution", forward):
            if profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    with torch.profiler.record_function("validation/collection"):
                        batch = env.collect(agent, 16)
                        torch.cuda.synchronize()
                profiler.export_chrome_trace(str(out / "trace.json"))
            else:
                batch = env.collect(agent, 4096 // streams)
                torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        digest = digest_batch(batch)
        if previous is not None and previous != digest:
            raise ValueError("same-device deterministic replay diverged")
        previous = digest
        samples.append(
            {
                "repeat": repeat,
                "cold": repeat == 0,
                "seconds": seconds,
                "decisions": env.stats.micro_steps,
                "learner_transitions": env.stats.learner_transitions,
                "games": env.stats.games,
                "truncations": env.stats.truncations,
                "forward_batches": dict(batches),
                "batch_sha256": digest,
                "game_seeds": env._env.current_game_seeds(),
                "journal_sha256": hashlib.sha256(
                    json.dumps(env._journal).encode()
                ).hexdigest(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        atomic_json(
            out / "result.json",
            {"streams": streams, "seed": seed, "profiled": profile, "samples": samples},
        )
    if profile:
        trace = Trace.model_validate_json((out / "trace.json").read_text())
        windows = [
            e
            for e in trace.traceEvents
            if e.name == "validation/collection" and e.ph == "X"
        ]
        atomic_json(
            out / "profile.json",
            {
                "windows": [
                    {
                        "seconds": e.dur / 1e6,
                        "device_busy_seconds": busy_microseconds(
                            trace.traceEvents, e.ts, e.ts + e.dur
                        )
                        / 1e6,
                    }
                    for e in windows
                ],
                "kernel_events": sum(e.cat == "kernel" for e in trace.traceEvents),
                "copy_events": sum(e.cat == "gpu_memcpy" for e in trace.traceEvents),
                "trace_sha256": file_sha256(out / "trace.json"),
            },
        )


def child(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU fallback")
    if torch.cuda.get_device_name() != args.expected_gpu:
        raise ValueError("assigned GPU differs from the frozen validation target")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = TrainingRegime.model_validate_json(
        (args.root / "baseline.json").read_text()
    )
    if args.child == "inputs":
        torch.manual_seed(10831)
        agent = (
            Agent(ObservationSpace(recipe.observation), recipe.agent).to("cuda").eval()
        )
        env = collector(recipe, 64, 10831)
        batch = env.collect(agent, 16)
        observations = {
            key: torch.from_numpy(value.reshape((-1,) + value.shape[2:]))
            for key, value in batch.obs.items()
        }
        observations["action_focus"] = observations["action_focus"].float()
        torch.save(observations, args.root / "observations.pt")
        atomic_json(
            args.root / "identity.json",
            {
                "device": torch.cuda.get_device_name(),
                "memory": torch.cuda.get_device_properties(0).total_memory,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
                "native_sha256": file_sha256(Path(managym._managym.__file__)),
                "driver_sha256": file_sha256(Path(__file__)),
                "world": checkpoint_world(
                    Match(recipe.match).to_rust(), ObservationSpace(recipe.observation)
                ),
                "input_sha256": file_sha256(args.root / "observations.pt"),
                "input_shapes": {
                    key: list(value.shape) for key, value in observations.items()
                },
            },
        )
    else:
        collection(
            recipe, args.out, args.streams, args.seed, profile=args.child == "profile"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--child", choices=("inputs", "collect", "profile"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--streams", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10831)
    parser.add_argument("--expected-gpu", default="NVIDIA L4")
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    recipe = TrainingRegime.model_validate_json(
        Path(os.environ["ETUDE_CONFIG"]).read_text()
    )
    atomic_json(root / "baseline.json", recipe.model_dump(mode="json"))
    attempts: list[Attempt] = []
    deadline = time.monotonic() + 600

    def run(name: str, command: list[str]) -> bool:
        out = root / name
        out.mkdir(exist_ok=False)
        attempt = Attempt(name=name, command=command)
        attempts.append(attempt)
        started = time.monotonic()
        try:
            remaining = deadline - started
            if remaining <= 0:
                attempt.status = "budget_exhausted"
                return False
            with (out / "worker.log").open("w") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(90, remaining),
                    check=False,
                )
            attempt.exit_code = result.returncode
            attempt.status = "completed" if result.returncode == 0 else "failed"
            if (
                result.returncode
                and "OutOfMemoryError" in (out / "worker.log").read_text()
            ):
                attempt.status = "oom"
        except subprocess.TimeoutExpired:
            attempt.status = "timeout"
        finally:
            attempt.seconds = time.monotonic() - started
            atomic_json(
                root / "attempts.json", [a.model_dump(mode="json") for a in attempts]
            )
            print(name, attempt.status, round(attempt.seconds, 3), flush=True)
        return attempt.status == "completed"

    python = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--expected-gpu",
        args.expected_gpu,
    ]
    try:
        if not run("inputs", [*python, "--child", "inputs"]):
            raise RuntimeError("CUDA input admission failed")
        model_cells = [
            Cell(capacity="w64-d1", kind="model", batch=b)
            for b in (1, 4, 16, 64, 256, 1024)
        ]
        model_cells += [
            Cell(capacity="w384-d8", kind="model", batch=b) for b in (64, 256)
        ]
        for cell in model_cells:
            run(
                cell.name,
                [
                    sys.executable,
                    "-m",
                    "experiments.runners.cuda_performance",
                    "--root",
                    str(root),
                    "--cell",
                    cell.model_dump_json(),
                ],
            )
        for seed in (10831, 10832, 10833):
            for streams in (16, 64, 256):
                name = f"collect-s{streams}-seed{seed}"
                run(
                    name,
                    [
                        *python,
                        "--child",
                        "collect",
                        "--out",
                        str(root / name),
                        "--streams",
                        str(streams),
                        "--seed",
                        str(seed),
                    ],
                )
        run(
            "profile-s64",
            [*python, "--child", "profile", "--out", str(root / "profile-s64")],
        )
        for streams in (16, 64):
            cell = Cell(capacity="w64-d1", kind="loop", batch=512, streams=streams)
            run(
                cell.name,
                [
                    sys.executable,
                    "-m",
                    "experiments.runners.cuda_performance",
                    "--root",
                    str(root),
                    "--cell",
                    cell.model_dump_json(),
                ],
            )
    finally:
        files = sorted(p for p in root.rglob("*") if p.is_file())
        atomic_json(
            root / "manifest.json",
            [
                {
                    "path": str(p.relative_to(root)),
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
                for p in files
            ],
        )
        with tarfile.open(root.parent / "evidence.tar.gz", "w:gz") as archive:
            archive.add(root, arcname="evidence")


if __name__ == "__main__":
    main()
