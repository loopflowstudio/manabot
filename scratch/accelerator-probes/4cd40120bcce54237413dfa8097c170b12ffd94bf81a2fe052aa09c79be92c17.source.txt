"""Real-clock, one-GPU serving replay for validating the queue model's boundary.

Logical clients wait through measured host intervals and real GPU responses. Host
work is simulated elapsed time, not game execution. A single worker owns CUDA;
FIFO batching is shared with the existing event model. There is no network, IPC,
policy sampling, learning or claim of unchanged trajectories.
"""

import argparse
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from threading import Event
import time
from types import ModuleType
from typing import cast

import torch

from experiments.runners.cuda_performance import load_observations
from experiments.runners.pipeline_benchmark import Replay, Tape
from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime


@dataclass(frozen=True)
class LiveRequest:
    actor: int
    position: int
    rows: int
    arrived: float


@dataclass(frozen=True)
class Delivery:
    actor: int
    position: int
    rows: int
    input_index_sha256: str
    output_sha256: str


@dataclass(frozen=True)
class ServiceReceipt:
    started: float
    ended: float
    phases: dict[str, float]
    deliveries: tuple[Delivery, ...]
    padded_rows: int
    input_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class LiveOutcome:
    seconds: float
    decisions: int
    requests: int
    cpu_reserved_seconds: float
    nominal_host_release_lateness_mean: float
    nominal_host_release_lateness_max: float
    mean_queue_seconds: float
    max_queue_seconds: float
    mean_request_seconds: float
    max_request_seconds: float
    max_delivery_seconds: float
    peak_queued_requests: int
    padded_batch_counts: dict[int, int]
    service_phase_seconds: dict[str, float]
    service_seconds: float
    useful_row_fraction: float
    deliveries: tuple[Delivery, ...]
    batches: tuple[ServiceReceipt, ...]


type Backend = Callable[[tuple[LiveRequest, ...], int], ServiceReceipt]
type Selector = Callable[[deque[LiveRequest], int], tuple[LiveRequest, ...]]


def queue_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "serving_queue", "scratch/accelerator-queue-probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def live(
    rows: list[int],
    host_seconds: list[float],
    tail_seconds: float,
    actors: int,
    workers: int,
    maximum: int,
    sizes: list[int],
    wait_seconds: float,
    backend: Backend,
    selector: Selector,
    executor: ThreadPoolExecutor,
) -> LiveOutcome:
    if not rows or len(rows) != len(host_seconds) or max(rows) > maximum:
        raise ValueError("invalid request tape")
    if min(actors, workers) < 1 or maximum not in sizes:
        raise ValueError("invalid resources or missing exact service size")
    events: list[tuple[float, int, int, bool]] = []
    cpu_free = [0.0] * workers
    positions = [0] * actors
    ready: deque[LiveRequest] = deque()
    future: Future[ServiceReceipt] | None = None
    active: tuple[LiveRequest, ...] = ()
    wake = Event()
    serial = finished = peak = 0
    cpu_reserved = 0.0
    releases: list[float] = []
    queues: list[float] = []
    latencies: list[float] = []
    delivery_delays: list[float] = []
    deliveries: list[Delivery] = []
    batches: list[ServiceReceipt] = []
    origin = time.perf_counter()

    def schedule(actor: int, at: float, tail: bool = False) -> None:
        nonlocal serial, cpu_reserved
        duration = tail_seconds if tail else host_seconds[positions[actor]]
        if tail and duration == 0:
            finish = at
        else:
            available = heapq.heappop(cpu_free)
            finish = max(at, available) + duration
            heapq.heappush(cpu_free, finish)
        cpu_reserved += duration
        serial += 1
        heapq.heappush(events, (finish, serial, actor, tail))

    def completed(_: Future[ServiceReceipt]) -> None:
        wake.set()

    for actor in range(actors):
        schedule(actor, 0)
    while finished < actors:
        wake.clear()  # Clear before inspecting the future; completion cannot be lost.
        now = time.perf_counter() - origin
        if now > 25:
            raise TimeoutError("live repetition exceeded25seconds")
        while events and events[0][0] <= now:
            planned, _, actor, tail = heapq.heappop(events)
            releases.append(now - planned)
            if tail:
                finished += 1
            else:
                ready.append(
                    LiveRequest(actor, positions[actor], rows[positions[actor]], now)
                )
        peak = max(peak, len(ready))
        if future is not None and future.done():
            result = future.result()
            received = time.perf_counter() - origin
            assert [(d.actor, d.position, d.rows) for d in result.deliveries] == [
                (r.actor, r.position, r.rows) for r in active
            ]
            for request in active:
                queues.append(result.started - origin - request.arrived)
                latencies.append(received - request.arrived)
                positions[request.actor] += 1
                schedule(request.actor, received, positions[request.actor] == len(rows))
            delivery_delays.append(received - (result.ended - origin))
            deliveries.extend(result.deliveries)
            batches.append(
                ServiceReceipt(
                    result.started - origin,
                    result.ended - origin,
                    result.phases,
                    result.deliveries,
                    result.padded_rows,
                    result.input_bytes,
                    result.output_bytes,
                )
            )
            future = None
        now = time.perf_counter() - origin
        flush = ready[0].arrived + wait_seconds if ready and future is None else None
        if (
            future is None
            and ready
            and (sum(r.rows for r in ready) >= maximum or now >= cast(float, flush))
        ):
            active = selector(ready, maximum)
            assert active
            used = sum(r.rows for r in active)
            padded = next(size for size in sizes if size >= used)
            future = executor.submit(backend, active, padded)
            future.add_done_callback(completed)
            continue
        if finished == actors:
            break
        deadlines = [events[0][0]] if events else []
        if flush is not None:
            deadlines.append(flush)
        timeout = max(0, min(deadlines) - now) if deadlines else 0.05
        wake.wait(min(timeout, 0.05))
    seconds = time.perf_counter() - origin
    assert future is None and not ready and not events
    assert len(deliveries) == actors * len(rows)
    assert {(d.actor, d.position) for d in deliveries} == {
        (a, p) for a in range(actors) for p in range(len(rows))
    }
    decisions = sum(d.rows for d in deliveries)
    assert decisions == actors * sum(rows)
    histogram = Counter(b.padded_rows for b in batches)
    phases: Counter[str] = Counter()
    for b in batches:
        phases.update(b.phases)
    return LiveOutcome(
        seconds,
        decisions,
        len(deliveries),
        cpu_reserved,
        sum(releases) / len(releases),
        max(releases),
        sum(queues) / len(queues),
        max(queues),
        sum(latencies) / len(latencies),
        max(latencies),
        max(delivery_delays),
        peak,
        dict(sorted(histogram.items())),
        dict(phases),
        sum(b.ended - b.started for b in batches),
        decisions / sum(b.padded_rows for b in batches),
        tuple(deliveries),
        tuple(batches),
    )


class CudaBackend:
    def __init__(self, agent: Agent, observations: dict[str, torch.Tensor]) -> None:
        self.agent = agent
        self.observations = observations
        self.available = len(observations["actions_valid"])

    def __call__(
        self, requests: tuple[LiveRequest, ...], padded: int
    ) -> ServiceReceipt:
        started = time.perf_counter()
        indices = [
            (torch.arange(r.rows) + r.actor * 7919 + r.position * 104729)
            % self.available
            for r in requests
        ]
        used = sum(r.rows for r in requests)
        if padded > used:
            indices.append(torch.arange(padded - used) % self.available)
        joined = torch.cat(indices)
        host = {k: v[joined].contiguous() for k, v in self.observations.items()}
        prepared = time.perf_counter()
        device = {k: v.to("cuda") for k, v in host.items()}
        torch.cuda.synchronize()
        uploaded = time.perf_counter()
        with torch.inference_mode():
            logits, values = self.agent(device)
        torch.cuda.synchronize()
        forwarded = time.perf_counter()
        logits, values = logits.cpu(), values.cpu()
        torch.cuda.synchronize()
        downloaded = time.perf_counter()
        assert logits.shape[0] == values.shape[0] == padded
        if not bool(torch.isfinite(logits).all() and torch.isfinite(values).all()):
            raise ValueError("nonfinite inference response")
        deliveries = []
        offset = 0
        for request, index in zip(requests, indices, strict=False):
            digest = hashlib.sha256(
                logits[offset : offset + request.rows].numpy().tobytes()
            )
            digest.update(values[offset : offset + request.rows].numpy().tobytes())
            deliveries.append(
                Delivery(
                    request.actor,
                    request.position,
                    request.rows,
                    hashlib.sha256(index.numpy().tobytes()).hexdigest(),
                    digest.hexdigest(),
                )
            )
            offset += request.rows
        ended = time.perf_counter()
        return ServiceReceipt(
            started,
            ended,
            {
                "prepare": prepared - started,
                "h2d": uploaded - prepared,
                "forward": forwarded - uploaded,
                "d2h": downloaded - forwarded,
                "validate": ended - downloaded,
            },
            tuple(deliveries),
            padded,
            sum(v.numel() * v.element_size() for v in host.values()),
            logits.numel() * logits.element_size()
            + values.numel() * values.element_size(),
        )


def check() -> None:
    module = queue_module()
    module.check()
    # Dynamic scratch-module import, narrowed to the shared FIFO selector contract.
    select = cast(Selector, module.pop_batch)
    ready = deque(
        [LiveRequest(0, 0, 5, 0), LiveRequest(1, 0, 4, 0), LiveRequest(2, 0, 3, 0)]
    )
    assert [r.rows for r in select(ready, 8)] == [5]
    assert [r.rows for r in select(ready, 8)] == [4, 3]

    def fake(requests: tuple[LiveRequest, ...], padded: int) -> ServiceReceipt:
        at = time.perf_counter()
        return ServiceReceipt(
            at,
            at,
            {},
            tuple(
                Delivery(r.actor, r.position, r.rows, "fixed", "finite")
                for r in requests
            ),
            padded,
            0,
            0,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = live(
            [3, 5],
            [0.001, 0.002],
            0.001,
            3,
            2,
            8,
            [4, 8],
            0.00025,
            fake,
            select,
            executor,
        )
    assert result.requests == 6 and result.decisions == 24
    assert abs(result.cpu_reserved_seconds - 0.012) < 1e-12
    assert result.peak_queued_requests <= 3 and 0 < result.useful_row_fraction <= 1
    print(
        "Shared FIFO, fake-backend completion, request identity and host-work conservation passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=256)
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.root is None or args.out is None or (args.out / "result.json").exists():
        raise ValueError("fresh output and evidence root required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != "NVIDIA H100 80GB HBM3"
    ):
        raise ValueError("required H100 unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = TrainingRegime.model_validate_json(
        (args.root / "baseline.json").read_text()
    )
    torch.manual_seed(10349)
    agent = Agent(ObservationSpace(recipe.observation), recipe.agent).to("cuda").eval()
    observations = load_observations(args.root / "observations.pt")
    tape_path = Path("scratch/accelerator-probes/cohort-t8-s64/tape.json")
    replay_path = tape_path.parent / "replay.json"
    tape = Tape.model_validate_json(tape_path.read_text())
    replay_data = json.loads(replay_path.read_text())
    assert (
        replay_data["tape_sha256"] == hashlib.sha256(tape_path.read_bytes()).hexdigest()
    )
    sample = Replay.model_validate(replay_data["samples"][1])
    assert (
        sample.batch_sha256 == tape.batch_sha256
        and sample.request_host_seconds is not None
        and sample.tail_host_seconds is not None
    )
    service = {}
    model_sources = {}
    for size in (1, 4, 16, 64, 256, 1024):
        path = args.root / f"w64-d1-model-b{size}-s0" / "result.json"
        r = json.loads(path.read_text())
        assert (
            r["device"] == torch.cuda.get_device_name()
            and r["input_sha256"]
            == hashlib.sha256((args.root / "observations.pt").read_bytes()).hexdigest()
        )
        service[size] = size / median(
            w["observations"] / w["seconds"] for w in r["inference"]["windows"]
        )
        model_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    module = queue_module()
    selector = cast(Selector, module.pop_batch)
    backend = CudaBackend(agent, observations)
    sizes = sorted(s for s in service if s <= args.max_batch)
    payload = (
        sum(v.numel() * v.element_size() for v in observations.values())
        / backend.available
        + (recipe.observation.max_actions + 1) * 4
    )
    settings = module.Settings(
        actors=args.actors,
        cpu_workers=min(args.actors, 8),
        gpus=1,
        max_batch=args.max_batch,
        wait_seconds=0.00025,
    )
    prediction = module.simulate(
        [f.rows for f in tape.frames],
        0,
        service,
        payload,
        settings,
        request_host_seconds=sample.request_host_seconds,
        tail_host_seconds=sample.tail_host_seconds,
    )
    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(2):
            executor.submit(
                backend, (LiveRequest(0, 0, args.max_batch, 0),), args.max_batch
            ).result()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            result = live(
                [f.rows for f in tape.frames],
                sample.request_host_seconds,
                sample.tail_host_seconds,
                args.actors,
                min(args.actors, 8),
                args.max_batch,
                sizes,
                0.00025,
                backend,
                selector,
                executor,
            )
            assert (
                abs(result.cpu_reserved_seconds - args.actors * sample.seconds) < 1e-10
            )
            results.append(result)
    mappings = [
        {
            (d.actor, d.position): (d.input_index_sha256, d.output_sha256)
            for d in r.deliveries
        }
        for r in results
    ]
    assert all(
        {k: v[0] for k, v in m.items()} == {k: v[0] for k, v in mappings[0].items()}
        for m in mappings
    )
    output_changes = [sum(m[k][1] != mappings[0][k][1] for k in m) for m in mappings]
    report = {
        "settings": asdict(settings),
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("scratch/accelerator-queue-probe.py"),
                tape_path,
                replay_path,
                args.root / "observations.pt",
            )
        },
        "model_sources": model_sources,
        "payload_bytes_per_row": payload,
        "prediction": asdict(prediction),
        "samples": [asdict(r) for r in results],
        "median_actual_to_predicted_seconds": median(r.seconds for r in results)
        / prediction.seconds,
        "different_output_requests_vs_first": output_changes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "limits": [
            "Logical clients and simulated host delay, not executed collection, network or IPC.",
            "Real frozen observations are selected by deterministic row indices; not exact original game inputs.",
            "Pageable per-field H2D and CPU response hashing are included; no pinned/flat transfer, cache or stream overlap.",
            "Reported output hash differences have no implicit tolerance or semantic/strength claim.",
            "Prediction assumes12GB/s links and20us per direction; all physical phases and timer lateness remain explicit.",
        ],
    }
    args.out.mkdir(exist_ok=True)
    atomic_json(args.out / "result.json", report)
    print(
        "Completed physical serving",
        args.actors,
        args.max_batch,
        report["median_actual_to_predicted_seconds"],
    )


if __name__ == "__main__":
    main()
