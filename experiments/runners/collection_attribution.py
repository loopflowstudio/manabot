"""Attribute the second collection of an ordinary three-update TrainingRun.

This instrument calls cuda_performance.child unchanged. Torch traces identify
host phase windows and aligned device work; a separate cProfile pass exposes
Python/array costs without Torch-profiler overhead. Only the selected collection
is patched, and every patch/native wrapper is restored even on failure. Neither
profile is a throughput measurement or a change to training semantics.
"""

import argparse
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
import cProfile
from functools import wraps
from pathlib import Path
import pstats
import time
from typing import Literal
from unittest.mock import patch

import torch

from experiments.runners.cuda_collection_profile import Trace, busy_microseconds
from experiments.runners.cuda_performance import Cell, child
from experiments.runners.inference_preparation import (
    checkpoint_witnesses,
    reference_inference,
)
from experiments.runners.pipeline_benchmark import _TimedEnvironment
from manabot.arena.models import file_sha256
from manabot.model.agent import Agent
from manabot.model.inference import PolicyInference
from manabot.model.semantic_cards import SemanticCardEncoder
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training.execution import atomic_json

ProfileKind = Literal["torch", "python"]


def _span[**P, R](name: str, method: Callable[P, R]) -> Callable[P, R]:
    @wraps(method)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        with torch.profiler.record_function(f"collection/{name}"):
            return method(*args, **kwargs)

    return call


class _TracedEnvironment(_TimedEnvironment):
    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        with torch.profiler.record_function("collection/native_step_and_encode"):
            super().step_into_buffers(actions, active)


@contextmanager
def attribute_collection(root: Path, kind: ProfileKind) -> Iterator[None]:
    """Observe collection number two; leave initialization/learning unpatched."""
    original = SeatRoutedCollector.collect
    count = 0

    @wraps(original)
    def collect(
        collector: SeatRoutedCollector,
        agent: Agent,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> RolloutBatch:
        nonlocal count
        count += 1

        def invoke() -> RolloutBatch:
            return original(
                collector,
                agent,
                num_steps,
                deadline_monotonic=deadline_monotonic,
                check=check,
            )

        if count != 2:
            return invoke()
        started = time.perf_counter()
        if kind == "python":
            profiler = cProfile.Profile()
            try:
                with profiler:
                    return invoke()
            finally:
                profiler.dump_stats(str(root / "collection.pstats"))
                with (root / "python-profile.txt").open("w") as output:
                    stats = pstats.Stats(profiler, stream=output)
                    stats.strip_dirs().sort_stats("tottime").print_stats(100)
                    stats.sort_stats("cumulative").print_stats(100)
                atomic_json(
                    root / "profile.json",
                    {
                        "kind": kind,
                        "collection_index": 1,
                        "seconds_including_profile_export": time.perf_counter()
                        - started,
                        "limits": "Separate cProfile diagnostic; inclusive times overlap and include Python profiler overhead.",
                    },
                )
        batches: Counter[int] = Counter()
        forward = Agent.forward_distribution

        def traced_forward(
            model: Agent, obs: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batches[len(obs["actions_valid"])] += 1
            with torch.profiler.record_function("collection/forward_distribution"):
                return forward(model, obs)

        native = collector._env
        cuda = collector.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
        with ExitStack() as stack:
            # Patch only during collection, never while the learner differentiates.
            for name in ("_slice_obs_tensors", "_build_batch"):
                stack.enter_context(
                    patch.object(
                        SeatRoutedCollector,
                        name,
                        _span(name, getattr(SeatRoutedCollector, name)),
                    )
                )
            stack.enter_context(
                patch.object(
                    SeatRoutedCollector,
                    "_policy_logs",
                    staticmethod(
                        _span("policy_logs", SeatRoutedCollector._policy_logs)
                    ),
                )
            )
            stack.enter_context(
                patch.object(Agent, "forward_distribution", traced_forward)
            )
            stack.enter_context(
                patch.object(
                    SemanticCardEncoder,
                    "definitions",
                    _span("semantic_catalog", SemanticCardEncoder.definitions),
                )
            )
            stack.enter_context(
                patch.object(
                    PolicyInference,
                    "prepare",
                    _span("prepare", PolicyInference.prepare),
                )
            )
            stack.enter_context(
                patch.object(
                    torch,
                    "multinomial",
                    _span("sample", torch.multinomial),
                )
            )
            collector._env = _TracedEnvironment(native)
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
                ) as trace:
                    with torch.profiler.record_function("collection/complete"):
                        result = invoke()
                        if cuda:
                            torch.cuda.synchronize()
            finally:
                collector._env = native
        trace.export_chrome_trace(str(root / "trace.json"))
        events = Trace.model_validate_json(
            (root / "trace.json").read_text()
        ).traceEvents
        # CUDA may duplicate annotations. Only CPU annotations define host phases.
        windows = [
            e
            for e in events
            if e.cat == "user_annotation"
            and e.ph == "X"
            and e.name.startswith("collection/")
        ]
        complete = [e for e in windows if e.name == "collection/complete"]
        if len(complete) != 1 or (cuda and not any(e.cat == "kernel" for e in events)):
            raise ValueError("collection trace missing its CPU window or CUDA kernels")
        atomic_json(
            root / "profile.json",
            {
                "kind": kind,
                "collection_index": 1,
                "forward_batches": dict(sorted(batches.items())),
                "host_windows": [e.model_dump() for e in windows],
                "collection_device_busy_seconds": busy_microseconds(
                    events,
                    complete[0].ts,
                    complete[0].ts + complete[0].dur,
                )
                / 1e6
                if cuda
                else None,
                "trace_sha256": file_sha256(root / "trace.json"),
                "limits": "Profiled diagnostic, not throughput. Host windows overlap when nested and include launch/synchronization. Device busy is interval union, not SM utilization. Native stepping includes observation encoding. Unmarked host gaps have no assigned cause.",
            },
        )
        return result

    with patch.object(SeatRoutedCollector, "collect", collect):
        yield
    if count < 2:
        raise ValueError("training did not reach the profiled second collection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", type=Cell.model_validate_json, required=True)
    parser.add_argument("--profile-kind", choices=("torch", "python"), required=True)
    parser.add_argument("--reference-policy", action="store_true")
    args = parser.parse_args()
    if args.cell.kind != "loop":
        parser.error("collection attribution requires ordinary training")
    out = args.root / args.cell.name
    with (
        reference_inference(args.reference_policy),
        attribute_collection(out, args.profile_kind),
    ):
        child(args.root, args.cell)
    atomic_json(out / "policy-witness.json", checkpoint_witnesses(out / "run"))


if __name__ == "__main__":
    main()
