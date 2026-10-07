"""Attribute ordinary CUDA collection wall time without assuming GPU starvation.

Torch's aligned CPU/CUDA trace bounds device busy intervals inside collection.
This instrument adds profiler overhead and is separate from throughput timings.
It preserves the ordinary collector, estimator, update and persistence path.
"""

import argparse
from collections import Counter
from functools import wraps
from pathlib import Path
import time
from typing import Callable

from pydantic import BaseModel, Field
import torch

from experiments.runners.model_capacity import regimes
from manabot.arena.models import file_sha256
from manabot.model.agent import Agent
from manabot.sim.net_opponent import RolloutBatch, SeatRoutedCollector
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore


class Event(BaseModel):
    name: str = ""
    cat: str = ""
    ph: str = ""
    ts: float = 0
    dur: float = 0


class Trace(BaseModel):
    traceEvents: list[Event] = Field(default_factory=list)


def busy_microseconds(events: list[Event], begin: float, end: float) -> float:
    """Union device intervals so overlapping kernels/copies never double count."""
    spans = sorted(
        (max(begin, e.ts), min(end, e.ts + e.dur))
        for e in events
        if e.ph == "X"
        and e.cat in {"kernel", "gpu_memcpy", "gpu_memset"}
        and e.dur > 0
        and e.ts < end
        and e.ts + e.dur > begin
    )
    busy = 0.0
    cursor = begin
    for left, right in spans:
        busy += max(0, right - max(cursor, left))
        cursor = max(cursor, right)
    return busy


def profile(recipe: TrainingRegime, root: Path, seconds: float) -> None:
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deadline = time.monotonic() + seconds
    original_collect = SeatRoutedCollector.collect
    original_forward = Agent.forward
    batch_sizes: Counter[int] = Counter()

    @wraps(original_collect)
    def collect(
        self: SeatRoutedCollector,
        agent: Agent,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> RolloutBatch:
        with torch.profiler.record_function("manabot/collection"):
            return original_collect(
                self,
                agent,
                num_steps,
                deadline_monotonic=deadline_monotonic,
                check=check,
            )

    @wraps(original_forward)
    def forward(
        self: Agent, observation: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_sizes[observation["actions_valid"].shape[0]] += 1
        with torch.profiler.record_function("manabot/forward"):
            return original_forward(self, observation)

    SeatRoutedCollector.collect = collect
    Agent.forward = forward
    try:
        for capacity in ("w64-d2", "w384-d8"):
            for streams in (4, 64):
                if deadline - time.monotonic() < 35:
                    raise TimeoutError("profile allocation exhausted")
                cell = regimes(recipe, include_ataraxos=True)[capacity]
                cell.stages = cell.stages[:1]
                stage = cell.stages[0]
                assert isinstance(stage, TrainSelfPlay)
                stage.streams, stage.transitions, stage.updates = (
                    streams,
                    512 // streams,
                    3,
                )
                stage.execution.device, stage.execution.threads = "cuda", 1
                stage.execution.wall_seconds = 30
                cell.wall_seconds = 32
                cell = TrainingRegime.model_validate(cell.model_dump())
                out = root / f"{capacity}-s{streams}"
                out.mkdir(parents=True)
                batch_sizes.clear()
                started = time.time()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    with VerifyStore(out / "training.sqlite") as store:
                        run = execute_regime(cell, 10349, out / "run", store)
                profiler.export_chrome_trace(str(out / "trace.json"))
                trace = Trace.model_validate_json((out / "trace.json").read_text())
                collections = sorted(
                    (
                        e
                        for e in trace.traceEvents
                        if e.name == "manabot/collection" and e.ph == "X"
                    ),
                    key=lambda e: e.ts,
                )
                samples = [
                    dict(
                        wall_seconds=e.dur / 1e6,
                        device_busy_seconds=(
                            busy_microseconds(trace.traceEvents, e.ts, e.ts + e.dur)
                            / 1e6
                            if any(event.cat == "kernel" for event in trace.traceEvents)
                            else None
                        ),
                        cold=i == 0,
                    )
                    for i, e in enumerate(collections)
                ]
                atomic_json(
                    out / "profile.json",
                    dict(
                        status=run.status,
                        capacity=capacity,
                        streams=streams,
                        total_batch=512,
                        forward_batches=dict(batch_sizes),
                        collection_windows=samples,
                        seconds=time.time() - started,
                        trace_sha256=file_sha256(out / "trace.json"),
                        limits="Profiler-instrumented diagnostic. Device busy means union of traced kernel/copy intervals, not SM utilization. Gaps include Python, engine, transfers and launch scheduling; no single cause is assigned. First collection is cold.",
                    ),
                )
                if run.status != "completed":
                    raise ValueError("profile run failed; retained")
    finally:
        SeatRoutedCollector.collect = original_collect
        Agent.forward = original_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    profile(
        TrainingRegime.model_validate_json(args.recipe.read_text()),
        args.out,
        args.seconds,
    )


if __name__ == "__main__":
    main()
