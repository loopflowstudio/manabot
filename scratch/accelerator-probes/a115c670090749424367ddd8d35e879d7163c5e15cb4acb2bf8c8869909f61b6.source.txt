"""Trace ordinary 4096-stream collection without replacing its rules or storage.

The external supervisor owns hard limits. This worker flushes phase boundaries,
CPU/page-fault counters and exact trajectory identities, including partial work.
Synchronized diagnostic phases intentionally do not supply throughput estimates.
"""

import argparse
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import ModuleType
from typing import Any, Protocol
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime


def staging_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "diagnostic_staging", "scratch/collector-staging-probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def optional_text(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except (FileNotFoundError, PermissionError):
        return None


def host() -> dict[str, object]:
    info = optional_text("/proc/cpuinfo") or ""
    return {
        "cpu_models": sorted(
            {
                line.split(":", 1)[1].strip()
                for line in info.splitlines()
                if line.startswith("model name")
            }
        ),
        "affinity": sorted(os.sched_getaffinity(0)),
        "memory": optional_text("/proc/meminfo"),
        "process_status": optional_text("/proc/self/status"),
        "cgroup": optional_text("/proc/self/cgroup"),
        "limits": {
            name: optional_text("/sys/fs/cgroup/" + name)
            for name in (
                "cpu.max",
                "cpu.stat",
                "memory.max",
                "memory.current",
                "memory.peak",
                "memory.events",
                "memory.pressure",
            )
        },
    }


class Trace:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.events = (out / "events.jsonl").open("w", buffering=1)
        self.stack: list[str] = []
        self.calls: list[dict[str, str | float | None]] = []

    def event(self, kind: str, name: str) -> None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.events.write(
            json.dumps(
                {
                    "kind": kind,
                    "name": name,
                    "stack": self.stack,
                    "monotonic": time.monotonic(),
                    "user_cpu": usage.ru_utime,
                    "system_cpu": usage.ru_stime,
                    "minor_faults": usage.ru_minflt,
                    "major_faults": usage.ru_majflt,
                    "peak_rss_bytes": usage.ru_maxrss * 1024,
                }
            )
            + "\n"
        )

    @contextmanager
    def span(self, name: str, *, synchronize: bool = False) -> Iterator[None]:
        self.stack.append(name)
        self.event("begin", name)
        try:
            yield
            if synchronize:
                torch.cuda.synchronize()
        finally:
            self.event("end", name)
            self.stack.pop()

    @contextmanager
    def bounded(self, name: str) -> Iterator[None]:
        entry: dict[str, str | float | None] = {
            "name": name,
            "start": time.monotonic(),
            "end": None,
        }
        self.calls.append(entry)
        atomic_json(self.out / "bounds.json", self.calls)
        try:
            with self.span(name):
                yield
        finally:
            entry["end"] = time.monotonic()
            atomic_json(self.out / "bounds.json", self.calls)

    def measured[**P, R](
        self, name: str, function: Callable[P, R], *, synchronize: bool = False
    ) -> Callable[P, R]:
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            with self.span(name, synchronize=synchronize):
                return function(*args, **kwargs)

        return call


class NativeProxy:
    """Delegate the untyped PyO3 interface; wrap only native vector stepping."""

    def __init__(self, target: object, trace: Trace) -> None:
        self.target = target
        self.trace = trace

    def __getattr__(self, name: str) -> Any:
        # The extension exposes additional methods; delegation preserves them.
        return getattr(self.target, name)

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        with self.trace.span("native_step"):
            getattr(self.target, "step_into_buffers")(actions, active)


class Staging(Protocol):
    pinned_bytes: int
    device_bytes: int

    def slice(self, rows: NDArray[np.int64]) -> dict[str, torch.Tensor]: ...
    def verify(self) -> dict[str, object]: ...
    def verify_ownership(self, batch: RolloutBatch) -> bool: ...


@contextmanager
def instrument(
    trace: Trace, env: SeatRoutedCollector, agent: Agent, backend: Staging | None
) -> Iterator[None]:
    slicer = backend.slice if backend is not None else env._slice_obs_tensors
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                env,
                "_slice_obs_tensors",
                trace.measured("gather_upload", slicer, synchronize=True),
            )
        )
        stack.enter_context(
            patch.object(
                agent,
                "forward",
                trace.measured("forward", agent.forward, synchronize=True),
            )
        )
        stack.enter_context(
            patch.object(
                env, "_build_batch", trace.measured("build_batch", env._build_batch)
            )
        )
        stack.enter_context(patch.object(env, "_env", NativeProxy(env._env, trace)))
        yield


def collect(
    trace: Trace,
    env: SeatRoutedCollector,
    agent: Agent,
    backend: Staging | None,
    name: str,
    steps: int,
) -> RolloutBatch:
    # Bounds cover context entry, all ordinary collection work, and final sync.
    # Parent enforcement is independent of Python/native progress in this call.
    with trace.bounded(name):
        with instrument(trace, env, agent, backend):
            batch = env.collect(
                agent, steps, deadline_monotonic=time.perf_counter() + 25
            )
        with trace.span("post_collect_sync", synchronize=True):
            pass
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(10349, 10350), required=True)
    parser.add_argument("--variant", choices=("reference", "staged"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert (
        torch.cuda.is_available()
        and torch.cuda.get_device_name() == "NVIDIA H100 80GB HBM3"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(exist_ok=True)
    atomic_json(args.out / "host-before.json", host())
    trace = Trace(args.out)
    module = staging_module()
    validation = module.validation_module()
    recipe = TrainingRegime.model_validate_json(
        (args.root / "baseline.json").read_text()
    )
    torch.manual_seed(args.seed)
    with trace.span("model_construction", synchronize=True):
        agent = (
            Agent(ObservationSpace(recipe.observation), recipe.agent).to("cuda").eval()
        )
    samples: list[dict[str, object]] = []
    try:
        for staged in (False, True):
            with trace.span("warmup_construction"):
                env = validation.collector(recipe, 4096, args.seed)
                assert isinstance(env, SeatRoutedCollector)
            with trace.span("warmup_allocation", synchronize=True):
                backend: Staging = module.CollectorStaging(env)
            if staged:
                with trace.span("fixed_audit"):
                    atomic_json(args.out / "fixed-witnesses.json", backend.verify())
            batch = collect(
                trace,
                env,
                agent,
                backend if staged else None,
                "warm-staged" if staged else "warm-reference",
                2,
            )
            with trace.span("warmup_release"):
                del batch, backend, env
        for repeat in range(2):
            with trace.span("construction"):
                env = validation.collector(recipe, 4096, args.seed)
                assert isinstance(env, SeatRoutedCollector)
            with trace.span("allocation", synchronize=True):
                selected: Staging | None = (
                    module.CollectorStaging(env) if args.variant == "staged" else None
                )
            torch.cuda.reset_peak_memory_stats()
            batch = collect(trace, env, agent, selected, f"measured-{repeat}", 8)
            with trace.span("hashing"):
                identity = asdict(
                    module.Identity(
                        validation.digest_batch(batch),
                        module.array_digest(env._buffers),
                        hashlib.sha256(json.dumps(env._journal).encode()).hexdigest(),
                        hashlib.sha256(
                            json.dumps(env._env.current_game_seeds()).encode()
                        ).hexdigest(),
                        hashlib.sha256(
                            env._self_rng.get_state().cpu().numpy().tobytes()
                        ).hexdigest(),
                    )
                )
            with trace.span("ownership"):
                independent = selected.verify_ownership(batch) if selected else True
            exact = not samples or samples[0]["identity"] == identity
            samples.append(
                {
                    "repeat": repeat,
                    "variant": args.variant,
                    "identity": identity,
                    "exact_within_cell": exact,
                    "storage_independent": independent,
                    "decisions": env.stats.micro_steps,
                    "learner_transitions": env.stats.learner_transitions,
                    "games": env.stats.games,
                    "truncations": env.stats.truncations,
                    "pinned_bytes": selected.pinned_bytes if selected else 0,
                    "device_bytes": selected.device_bytes if selected else 0,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            atomic_json(
                args.out / "result.json",
                {
                    "seed": args.seed,
                    "streams": 4096,
                    "variant": args.variant,
                    "samples": samples,
                    "limits": "Synchronized instrumented attribution; no throughput speedup claim.",
                },
            )
            assert (
                exact
                and independent
                and env.stats.learner_transitions == 32768
                and env.stats.truncations == 0
            )
            with trace.span("release"):
                del batch, selected, env
    finally:
        atomic_json(args.out / "host-after.json", host())
        trace.events.close()


if __name__ == "__main__":
    main()
