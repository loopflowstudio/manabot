"""Finite-actor research model of the captured policy-request sequence.

Actors perform measured host intervals or proportional average work, enqueue one
request, wait for its result, then continue their tape. CPU workers and GPU servers
are finite; measured final host tails must complete before an actor finishes.
FIFO batches flush on size or timeout; partial batches use the next measured
CUDA batch point with explicit padding. This models a prospective coalesced
serving boundary, not the current CUDA collector or actual distributed training.
"""

import argparse
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
from typing import Literal, Protocol

from experiments.runners.pipeline_benchmark import Replay, Tape, load_services


@dataclass(frozen=True)
class Settings:
    actors: int = 1
    cpu_workers: int = 1  # collector workers, excluding the inference server per GPU
    gpus: int = 1
    max_batch: int = 64
    wait_seconds: float = 0.0
    host_speed: float = 1.0
    actor_phase: Literal["aligned", "rotated"] = "aligned"
    transfer_gb_per_second: float = 12.0
    transfer_latency_seconds: float = 20e-6
    # Fractional diagnostic optimizer service reserves GPU time. This is not
    # an actual learner queue, policy update, sample reuse or policy-lag model.
    optimizer_observations_per_decision: float = 0.0


@dataclass(frozen=True)
class Request:
    actor: int
    rows: int
    arrived: float


class RowRequest(Protocol):
    @property
    def rows(self) -> int: ...


def pop_batch[T: RowRequest](ready: deque[T], maximum: int) -> tuple[T, ...]:
    """Consume the FIFO prefix fitting one batch; shared by modeled/live replay."""
    if maximum < 1 or (ready and ready[0].rows > maximum):
        raise ValueError("head request cannot fit the maximum batch")
    selected: list[T] = []
    used = 0
    while ready and used + ready[0].rows <= maximum:
        request = ready.popleft()
        selected.append(request)
        used += request.rows
    return tuple(selected)


@dataclass(frozen=True)
class Batch:
    requests: tuple[Request, ...]
    padded_rows: int
    seconds: float


@dataclass(order=True)
class Event:
    time: float
    serial: int
    kind: Literal["host", "device", "flush", "tail"] = field(compare=False)
    actor: int = field(default=-1, compare=False)
    gpu: int = field(default=-1, compare=False)
    batch: Batch | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Outcome:
    settings: Settings
    seconds: float
    decisions: int
    requests: int
    decisions_per_second: float
    cpu_busy_fraction: float
    cpu_busy_seconds: float
    inference_server_and_transfer_busy_fraction: float
    padded_batch_counts: dict[int, int]
    useful_row_fraction: float
    mean_queue_seconds: float
    max_queue_seconds: float
    mean_request_seconds: float
    max_request_seconds: float
    peak_queued_rows: int
    peak_queued_requests: int


@dataclass(frozen=True)
class BatchDemand:
    """Rows transferred/processed and logical requests delivered by one service."""

    padded_rows: int
    useful_rows: int
    requests: int


type BatchService = Callable[[BatchDemand], float]


def simulate(
    rows: list[int],
    cpu_seconds_per_row: float,
    service_seconds: dict[int, float],
    payload_bytes_per_row: float,
    settings: Settings,
    *,
    request_host_seconds: list[float] | None = None,
    tail_host_seconds: float = 0.0,
    batch_service: BatchService | None = None,
) -> Outcome:
    """Replay finite tapes with conserved requests and serial actor dependencies.

    CPU tasks use FIFO reservations on earliest-free workers. Actors either begin
    at the same trace position or cyclically rotate their starting positions.
    Rotation is a sensitivity test, not independent game or policy evidence.
    Device and link service serialize per GPU; GPUs have independent links.
    An explicit batch service replaces both model and transfer cost. It must
    include every serial serving phase and return finite nonnegative seconds.
    """
    if not rows or min(rows) <= 0 or max(rows) > settings.max_batch:
        raise ValueError("requests must fit the selected maximum batch")
    if min(settings.actors, settings.cpu_workers, settings.gpus) < 1:
        raise ValueError("resource counts must be positive")
    if settings.wait_seconds < 0 or settings.host_speed <= 0:
        raise ValueError("invalid wait or host speed")
    if cpu_seconds_per_row < 0 or payload_bytes_per_row < 0:
        raise ValueError("negative demand")
    if not math.isfinite(tail_host_seconds) or tail_host_seconds < 0:
        raise ValueError("invalid final host tail")
    if request_host_seconds is not None and (
        len(request_host_seconds) != len(rows)
        or any(not math.isfinite(t) or t < 0 for t in request_host_seconds)
    ):
        raise ValueError("request host intervals must match the request tape")
    if settings.transfer_gb_per_second <= 0 or settings.transfer_latency_seconds < 0:
        raise ValueError("invalid link")
    sizes = sorted(b for b in service_seconds if b <= settings.max_batch)
    if settings.max_batch not in sizes or min(service_seconds.values()) < 0:
        raise ValueError("maximum batch needs a measured, nonnegative service")

    events: list[Event] = []
    cpu_free = [0.0] * settings.cpu_workers
    free_gpus = list(range(settings.gpus))
    ready: deque[Request] = deque()
    positions = [0] * settings.actors
    serial = 0
    queued_rows = 0
    peak_rows = peak_requests = 0
    flush_at: float | None = None
    cpu_busy = device_busy = 0.0
    padded_total = decisions = completed_requests = 0
    queue_sum = queue_max = latency_sum = latency_max = 0.0
    batches: Counter[int] = Counter()
    completed_actors = 0
    now = 0.0

    def request_position(actor: int) -> int:
        offset = actor * 137 if settings.actor_phase == "rotated" else 0
        return (positions[actor] + offset) % len(rows)

    def request_rows(actor: int) -> int:
        return rows[request_position(actor)]

    def event(
        at: float,
        kind: Literal["host", "device", "flush", "tail"],
        *,
        actor: int = -1,
        gpu: int = -1,
        batch: Batch | None = None,
    ) -> None:
        nonlocal serial
        serial += 1
        heapq.heappush(events, Event(at, serial, kind, actor, gpu, batch))

    def reserve_cpu(
        actor: int, at: float, seconds: float, *, tail: bool = False
    ) -> None:
        nonlocal cpu_busy
        demand = seconds / settings.host_speed
        if tail and demand == 0:
            event(at, "tail", actor=actor)
            return
        available = heapq.heappop(cpu_free)
        finished = max(at, available) + demand
        heapq.heappush(cpu_free, finished)
        cpu_busy += demand
        event(finished, "tail" if tail else "host", actor=actor)

    def schedule_host(actor: int, at: float) -> None:
        demand = (
            request_host_seconds[request_position(actor)]
            if request_host_seconds is not None
            else request_rows(actor) * cpu_seconds_per_row
        )
        reserve_cpu(actor, at, demand)

    def dispatch(at: float) -> None:
        nonlocal queued_rows, flush_at, device_busy, padded_total, queue_sum, queue_max
        while free_gpus and ready:
            deadline = ready[0].arrived + settings.wait_seconds
            if queued_rows < settings.max_batch and at < deadline:
                if flush_at != deadline:
                    flush_at = deadline
                    event(deadline, "flush")
                return
            selected = pop_batch(ready, settings.max_batch)
            used = sum(request.rows for request in selected)
            for request in selected:
                delay = at - request.arrived
                queue_sum += delay
                queue_max = max(queue_max, delay)
            assert selected
            queued_rows -= used
            padded = next(size for size in sizes if size >= used)
            # Transferring padded rows is conservative for a server which could
            # materialize padding on device. No unmeasured CUDA interpolation.
            transfer = (
                padded * payload_bytes_per_row / (settings.transfer_gb_per_second * 1e9)
                + 2 * settings.transfer_latency_seconds
            )
            duration = (
                service_seconds[padded] + transfer
                if batch_service is None
                else batch_service(BatchDemand(padded, used, len(selected)))
            )
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid full-batch service duration")
            batch = Batch(tuple(selected), padded, duration)
            device_busy += duration
            padded_total += padded
            batches[padded] += 1
            event(at + duration, "device", gpu=free_gpus.pop(), batch=batch)
            flush_at = None

    for actor in range(settings.actors):
        schedule_host(actor, 0.0)
    while events:
        item = heapq.heappop(events)
        now = item.time
        if item.kind == "host":
            request = Request(item.actor, request_rows(item.actor), now)
            ready.append(request)
            queued_rows += request.rows
            peak_rows = max(peak_rows, queued_rows)
            peak_requests = max(peak_requests, len(ready))
        elif item.kind == "device":
            assert item.batch is not None
            free_gpus.append(item.gpu)
            for request in item.batch.requests:
                decisions += request.rows
                completed_requests += 1
                latency = now - request.arrived
                latency_sum += latency
                latency_max = max(latency_max, latency)
                positions[request.actor] += 1
                if positions[request.actor] < len(rows):
                    schedule_host(request.actor, now)
                else:
                    reserve_cpu(request.actor, now, tail_host_seconds, tail=True)
        elif item.kind == "tail":
            completed_actors += 1
        elif item.time != flush_at:
            continue  # An earlier full batch made this timeout obsolete.
        else:
            flush_at = None
        dispatch(now)
        if completed_actors == settings.actors:
            break  # Stale timeout events are not part of the completion time.
    assert not ready and queued_rows == 0
    assert completed_requests == settings.actors * len(rows)
    assert decisions == settings.actors * sum(rows)
    assert len(free_gpus) == settings.gpus
    if now <= 0:
        raise ValueError("zero-duration fixture has no finite rate")
    return Outcome(
        settings=settings,
        seconds=now,
        decisions=decisions,
        requests=completed_requests,
        decisions_per_second=decisions / now,
        cpu_busy_fraction=cpu_busy / (now * settings.cpu_workers),
        cpu_busy_seconds=cpu_busy,
        inference_server_and_transfer_busy_fraction=device_busy / (now * settings.gpus),
        padded_batch_counts=dict(sorted(batches.items())),
        useful_row_fraction=decisions / padded_total,
        mean_queue_seconds=queue_sum / completed_requests,
        max_queue_seconds=queue_max,
        mean_request_seconds=latency_sum / completed_requests,
        max_request_seconds=latency_max,
        peak_queued_rows=peak_rows,
        peak_queued_requests=peak_requests,
    )


def check() -> None:
    """Closed-form limits, finite completion, padding and resource conservation."""
    service = {4: 0.02, 8: 0.025}
    base = Settings(max_batch=8, transfer_latency_seconds=0)
    one = simulate([4, 4], 0.01, service, 0, base)
    assert abs(one.seconds - 0.12) < 1e-12
    assert one.padded_batch_counts == {4: 2}
    many_gpu = simulate(
        [4, 4], 0.01, service, 0, Settings(**{**asdict(base), "gpus": 8})
    )
    assert many_gpu.seconds == one.seconds
    waiting = simulate(
        [4, 4], 0.01, service, 0, Settings(**{**asdict(base), "wait_seconds": 0.03})
    )
    assert abs(waiting.seconds - 0.18) < 1e-12
    # Two actors fill a batch before timeout and finish before an obsolete timer.
    batched = simulate(
        [4], 0, service, 0, Settings(**{**asdict(base), "actors": 2, "wait_seconds": 1})
    )
    assert batched.seconds == 0.025 and batched.padded_batch_counts == {8: 1}
    assert batched.useful_row_fraction == 1
    host = simulate(
        [4, 4], 0.01, {4: 0, 8: 0}, 0, Settings(**{**asdict(base), "actors": 2})
    )
    assert abs(host.seconds - 0.16) < 1e-12
    assert abs(host.cpu_busy_fraction - 1) < 1e-12
    padded = simulate([3], 0, service, 0, base)
    assert padded.padded_batch_counts == {4: 1} and padded.useful_row_fraction == 0.75
    # An odd request cannot combine with its neighbor at max size; neither is lost.
    awkward = simulate(
        [5, 3],
        0.001,
        service,
        0,
        Settings(**{**asdict(base), "actors": 3, "wait_seconds": 0.01}),
    )
    assert awkward.decisions == 24 and awkward.requests == 6
    assert 0 < awkward.inference_server_and_transfer_busy_fraction <= 1 + 1e-12
    rotated = simulate(
        [5, 3],
        0.001,
        service,
        0,
        Settings(**{**asdict(base), "actors": 3, "actor_phase": "rotated"}),
    )
    assert rotated.decisions == 24 and rotated.requests == 6
    # Equal row counts may have very different host demand; a final tail cannot
    # disappear merely because every GPU request has completed.
    timed = simulate(
        [4, 4],
        999,
        service,
        0,
        base,
        request_host_seconds=[0.01, 0.07],
        tail_host_seconds=0.03,
    )
    assert abs(timed.seconds - 0.15) < 1e-12
    assert abs(timed.cpu_busy_seconds - 0.11) < 1e-12
    timed_many = simulate(
        [4, 4],
        999,
        service,
        0,
        Settings(**{**asdict(base), "actors": 3, "actor_phase": "rotated"}),
        request_host_seconds=[0.01, 0.07],
        tail_host_seconds=0.03,
    )
    assert abs(timed_many.cpu_busy_seconds - 0.33) < 1e-12
    assert timed_many.cpu_busy_fraction <= 1 + 1e-12
    print("Ten closed-form and conservation cases passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("check", "sweep", "scale", "sensitivity", "phase", "host")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--tape",
        type=Path,
        default=Path("scratch/accelerator-probes/capture-s64/tape.json"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("scratch/accelerator-probes/replay-s64/replay.json"),
    )
    parser.add_argument("--request-timings", action="store_true")
    parser.add_argument(
        "--card", default="l4", help="Exact card label in the supplied hardware report"
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path("experiments/data/cuda-capacity/capacity-report.json"),
        help="Measured service report; retained historical data is the default",
    )
    args = parser.parse_args()
    check()
    if args.mode == "check":
        return
    if args.out is None or args.out.exists():
        raise ValueError("a fresh --out path is required")
    tape_path = args.tape
    replay_path = args.replay
    hardware_path = args.hardware
    tape = Tape.model_validate_json(tape_path.read_text())
    # Keep a declared repeat, not the fastest sample; sensitivity can vary it.
    sample = Replay.model_validate(json.loads(replay_path.read_text())["samples"][1])
    if (
        sample.batch_sha256 != tape.batch_sha256
        or sample.buffers_sha256 != tape.buffers_sha256
        or sample.journal_sha256 != tape.journal_sha256
        or sample.game_seeds != tape.game_seeds
        or sample.demand.decisions != sum(f.rows for f in tape.frames)
    ):
        raise ValueError("replay and policy tape identities differ")
    if args.request_timings and (
        sample.request_host_seconds is None or sample.tail_host_seconds is None
    ):
        raise ValueError("selected replay has no measured request intervals/tail")
    if args.request_timings and not math.isclose(
        sum(sample.request_host_seconds or []) + (sample.tail_host_seconds or 0),
        sample.seconds,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "request intervals and tail do not reconstruct collection time"
        )
    services = [
        s
        for s in load_services(hardware_path)
        if s.card == args.card and s.capacity == "w64-d1"
    ]
    frames = [f.rows for f in tape.frames]
    if args.mode == "host":
        settings = [
            Settings(
                actors=4096,
                cpu_workers=512,
                gpus=64,
                max_batch=batch,
                wait_seconds=0.00025,
                actor_phase=phase,
            )
            for phase in ("aligned", "rotated")
            for batch in (256, 1024)
        ]
    elif args.mode == "sweep":
        settings = [
            Settings(
                actors=actors,
                cpu_workers=min(actors, 16),
                max_batch=batch,
                wait_seconds=wait,
            )
            for actors in (1, 8, 64)
            for batch in (64, 256, 1024)
            for wait in (0, 0.00025, 0.002)
        ]
    elif args.mode == "scale":
        settings = [
            Settings(
                actors=actors,
                cpu_workers=workers,
                gpus=gpus,
                max_batch=batch,
                wait_seconds=0.00025,
            )
            for actors, workers, gpus in ((512, 128, 8), (4096, 512, 64))
            for batch in (64, 256, 1024)
        ]
    elif args.mode == "phase":
        settings = [
            Settings(
                actors=actors,
                cpu_workers=workers,
                gpus=gpus,
                max_batch=batch,
                wait_seconds=0.00025,
                actor_phase="rotated",
            )
            for actors, workers, gpus in ((64, 16, 1), (4096, 512, 64))
            for batch in (64, 256, 1024)
        ]
    else:
        settings = [
            Settings(
                actors=64,
                cpu_workers=16,
                max_batch=batch,
                wait_seconds=0.00025,
                host_speed=speed,
                optimizer_observations_per_decision=learning,
            )
            for batch in (64, 256, 1024)
            for speed in (1, 2)
            for learning in (0, 1)
        ]
    results: list[Outcome] = []
    started = time.perf_counter()
    for setting in settings:
        service = {
            s.batch: s.inference_seconds
            + setting.optimizer_observations_per_decision * s.optimizer_seconds
            for s in services
        }
        outcome = simulate(
            frames,
            sample.seconds / sample.demand.decisions,
            service,
            (sample.demand.input_bytes + sample.demand.output_bytes)
            / sample.demand.decisions,
            setting,
            request_host_seconds=sample.request_host_seconds
            if args.request_timings
            else None,
            tail_host_seconds=(sample.tail_host_seconds or 0.0)
            if args.request_timings
            else 0.0,
        )
        results.append(outcome)
        print(
            json.dumps(
                {
                    "settings": asdict(setting),
                    "rate": outcome.decisions_per_second,
                    "busy": outcome.inference_server_and_transfer_busy_fraction,
                    "padding_efficiency": outcome.useful_row_fraction,
                }
            ),
            flush=True,
        )
    report = {
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (tape_path, replay_path, hardware_path, Path(__file__))
        },
        "host_repeat": 1,
        "host_timing": "measured request intervals and final tail"
        if args.request_timings
        else "proportional average",
        "inference_server_reservation": "One additional CPU serving thread per GPU; measured synchronized model calls include its dispatch/validation overhead",
        "card": args.card,
        "capacity": "w64-d1",
        "execution_seconds": time.perf_counter() - started,
        "outcomes": [asdict(r) for r in results],
        "limits": [
            "Hypothetical coalesced serving path, not production CUDA or distributed throughput",
            "Host intervals include laptop scheduling/measurement overhead; no target CPU contention curve",
            "Model service is synchronized device-resident call time, not pure kernel time; its host/GPU pair must be available per server",
            "All actors repeat one untrained-policy trace; phase rotation is synthetic sensitivity, not independent workloads",
            "One outstanding request per whole actor; no independent per-environment scheduling",
            "Finite trace includes startup and drain; not a steady-state capacity measurement",
            "Device and transfers serialize; independent GPU links; no network or all-reduce",
            "No learner queue or policy lag; optional diagnostic optimizer service is only reserved time",
            "Named device service is measured separately; CPU postprocessing and transfer assumptions need end-to-end calibration",
        ],
    }
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
