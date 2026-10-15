"""Bounded CUDA validation using the existing collector and capacity instruments.

Fresh child processes isolate each cell. The parent preserves failure outcomes
and packages returned evidence. This research driver changes no training rule.
"""

import argparse
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import fields
import gzip
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
from experiments.runners.cuda_performance import (
    Cell,
    child as capacity_child,
    load_observations,
)
from manabot.arena.models import file_sha256
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.model.world import checkpoint_world
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training import execution
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime
from manabot.training.source_bundle import SourceBundle, source_commit
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


def collector(
    recipe: TrainingRegime, streams: int, seed: int, device: str = "cuda"
) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        ObservationSpace(recipe.observation),
        Match(recipe.match),
        Reward(RewardHypers()),
        num_envs=streams,
        seed=seed,
        opponent_mode="self",
        device=device,
        recovery_max_microsteps=1_000_000,
    )


@contextmanager
def catalog_session(agent: Agent) -> Iterator[None]:
    """Research-only cache lifetime: one collection, no gradients or weight changes."""
    encoder = agent.semantic_cards
    if encoder is None:
        raise ValueError("catalog experiment requires semantic inputs")
    original = encoder.definitions
    versions = tuple(p._version for p in agent.parameters())
    training = agent.training
    value: torch.Tensor | None = None

    def definitions() -> torch.Tensor:
        nonlocal value
        if torch.is_grad_enabled():
            raise ValueError("cached inference reached a gradient-enabled forward")
        if agent.training != training or versions != tuple(
            p._version for p in agent.parameters()
        ):
            raise ValueError("policy changed during frozen collection")
        if value is None:
            value = original()
        return value

    with patch.object(encoder, "definitions", definitions):
        yield


def paired_collection(args: argparse.Namespace, recipe: TrainingRegime) -> None:
    torch.manual_seed(args.seed)
    agent = (
        Agent(ObservationSpace(recipe.observation), recipe.agent).to(args.device).eval()
    )
    collector(recipe, args.streams, args.seed, args.device).collect(agent, 2)
    order = (False, True, True, False) if args.seed % 2 else (True, False, False, True)
    reference: tuple[str, str, str] | None = None
    samples: list[dict[str, object]] = []
    for cached in order:
        env = collector(recipe, args.streams, args.seed, args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with catalog_session(agent) if cached else nullcontext():
            batch = env.collect(agent, args.transitions // args.streams)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        identity = (
            digest_batch(batch),
            hashlib.sha256(json.dumps(env._journal).encode()).hexdigest(),
            hashlib.sha256(
                json.dumps(env._env.current_game_seeds()).encode()
            ).hexdigest(),
        )
        if reference is not None and reference != identity:
            raise ValueError("reference/cache trajectory parity failed")
        reference = identity
        samples.append(
            {
                "cached": cached,
                "seconds": seconds,
                "decisions": env.stats.micro_steps,
                "learner_transitions": env.stats.learner_transitions,
                "games": env.stats.games,
                "truncations": env.stats.truncations,
                "batch_sha256": identity[0],
                "journal_sha256": identity[1],
                "game_seeds_sha256": identity[2],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if args.device == "cuda"
                else None,
            }
        )
        atomic_json(
            args.out / "result.json",
            {
                "streams": args.streams,
                "seed": args.seed,
                "device": args.device,
                "warmup_learner_steps": 2,
                "samples": samples,
            },
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
            if e.name == "validation/collection"
            and e.ph == "X"
            and e.cat == "user_annotation"
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU fallback")
    if args.device == "cuda" and torch.cuda.get_device_name() != args.expected_gpu:
        raise ValueError("assigned GPU differs from the frozen validation target")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = TrainingRegime.model_validate_json(
        (args.root / "baseline.json").read_text()
    )
    if args.child == "paired":
        paired_collection(args, recipe)
    elif args.child == "loop":
        cell = Cell(
            capacity="w64-d1", kind="loop", batch=args.batch, streams=args.streams
        )
        (args.root / cell.name).mkdir()
        timings: list[dict[str, str | float]] = []

        def measured[**P, R](name: str, function: Callable[P, R]) -> Callable[P, R]:
            def call(*values: P.args, **keywords: P.kwargs) -> R:
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    return function(*values, **keywords)
                finally:
                    torch.cuda.synchronize()
                    timings.append(
                        {"phase": name, "seconds": time.perf_counter() - started}
                    )

            return call

        original = SeatRoutedCollector.collect

        def collect(
            self: SeatRoutedCollector,
            agent: Agent,
            num_steps: int,
            *,
            deadline_monotonic: float | None = None,
            check: Callable[[], None] | None = None,
        ) -> RolloutBatch:
            with catalog_session(agent) if args.cached else nullcontext():
                return original(
                    self,
                    agent,
                    num_steps,
                    deadline_monotonic=deadline_monotonic,
                    check=check,
                )

        try:
            with (
                patch.object(
                    SeatRoutedCollector, "collect", measured("collect", collect)
                ),
                patch.object(
                    execution,
                    "update_iteration",
                    measured("update_iteration", execution.update_iteration),
                ),
            ):
                capacity_child(args.root, cell)
        finally:
            atomic_json(
                args.root / cell.name / "phase-timings.json",
                {
                    "cached": args.cached,
                    "shape": [args.batch // args.streams, args.streams],
                    "phases": timings,
                    "limits": "Synchronized method boundaries; update_iteration excludes EMA and executor bookkeeping. Ordinary TrainingRun retains complete learning time.",
                },
            )
    elif args.child == "inputs":
        if args.frozen_inputs:
            compressed = Path("scratch/gpu-probes/frozen-l4-inputs.pt.gz")
            receipt = json.loads(compressed.with_suffix(".json").read_text())
            if file_sha256(compressed) != receipt["compressed_sha256"]:
                raise ValueError("frozen input archive hash differs")
            with gzip.open(compressed, "rb") as source:
                payload = source.read()
            if (
                len(payload) != receipt["uncompressed_bytes"]
                or hashlib.sha256(payload).hexdigest() != receipt["uncompressed_sha256"]
            ):
                raise ValueError("frozen input bytes differ")
            (args.root / "observations.pt").write_bytes(payload)
            observations = load_observations(args.root / "observations.pt")
            atomic_json(args.root / "frozen-input-receipt.json", receipt)
        else:
            torch.manual_seed(10831)
            agent = (
                Agent(ObservationSpace(recipe.observation), recipe.agent)
                .to("cuda")
                .eval()
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
    parser.add_argument(
        "--child", choices=("inputs", "collect", "profile", "paired", "loop")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--streams", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10831)
    parser.add_argument("--expected-gpu", default="NVIDIA L4")
    parser.add_argument(
        "--experiment",
        choices=("baseline", "catalog", "scale", "highend"),
        default="baseline",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--transitions", type=int, default=4096)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--frozen-inputs", action="store_true")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text())
    if args.experiment in ("catalog", "scale", "highend"):
        bundle = SourceBundle.model_validate(config["source"])
        for name, digest in bundle.files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or file_sha256(path) != digest:
                raise ValueError(f"prepared source mismatch: {name}")
        for path in Path("managym").glob("*.so"):
            bundle.files[str(path)] = file_sha256(path)
        atomic_json(Path(".manabot-source.json"), bundle.model_dump(mode="json"))
        if source_commit() != bundle.commit:
            raise ValueError("source owner rejected exact archive identity")
        atomic_json(root / "source-bundle.json", bundle.model_dump(mode="json"))
        recipe = TrainingRegime.model_validate(config["recipe"])
    else:
        recipe = TrainingRegime.model_validate(config)
    atomic_json(root / "baseline.json", recipe.model_dump(mode="json"))
    attempts: list[Attempt] = []
    deadline = time.monotonic() + 600

    def run(name: str, command: list[str]) -> bool:
        out = root / name
        out.mkdir(exist_ok=True)
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
        if args.experiment == "highend":
            if not run("inputs", [*python, "--child", "inputs", "--frozen-inputs"]):
                raise RuntimeError("frozen CUDA input admission failed")
            models = [
                Cell(capacity="w64-d1", kind="model", batch=b)
                for b in (1, 4, 16, 64, 256, 1024)
            ]
            models += [
                Cell(capacity="w384-d8", kind="model", batch=b) for b in (64, 256)
            ]
            for cell in models:
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
        if args.experiment in ("scale", "highend"):
            cells = [
                (streams, cached, repeat)
                for streams in (64, 512, 4096)
                for repeat in (
                    ("a", "b")
                    if streams == 64
                    or (args.experiment == "highend" and streams == 512)
                    else ("a",)
                )
                for cached in (False, True)
            ]
            for streams, cached, repeat in cells:
                name = f"shape-t8-s{streams}-{'cached' if cached else 'reference'}-{repeat}"
                nested = root / name
                nested.mkdir()
                atomic_json(nested / "baseline.json", recipe.model_dump(mode="json"))
                run(
                    name,
                    [
                        *python,
                        "--root",
                        str(nested),
                        "--child",
                        "loop",
                        "--streams",
                        str(streams),
                        "--batch",
                        str(streams * 8),
                        *(["--cached"] if cached else []),
                    ],
                )
            return
        if args.experiment == "catalog":
            for seed in (10831, 10832, 10833):
                for streams in (64, 256):
                    name = f"paired-s{streams}-seed{seed}"
                    run(
                        name,
                        [
                            *python,
                            "--child",
                            "paired",
                            "--out",
                            str(root / name),
                            "--streams",
                            str(streams),
                            "--seed",
                            str(seed),
                        ],
                    )
            for streams in (16, 64):
                for cached in (False, True):
                    name = f"loop-s{streams}-{'cached' if cached else 'reference'}"
                    nested = root / name
                    nested.mkdir()
                    atomic_json(
                        nested / "baseline.json", recipe.model_dump(mode="json")
                    )
                    run(
                        name,
                        [
                            *python,
                            "--root",
                            str(nested),
                            "--child",
                            "loop",
                            "--streams",
                            str(streams),
                            *(["--cached"] if cached else []),
                        ],
                    )
            return
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
