"""Admission stress model for one learner; no training or stale-policy estimator.

Packets are complete batches of a fixed measured shape. Producers snapshot a
policy version at production start. Only version-current packets reach the
learner. Queue capacity excludes packets held by blocked producers and the
active learner; all three occupancies are counted. Constant service and arrival
rates are explicit sensitivity inputs, not measured distributed throughput.
"""

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Calibration:
    source: str
    sha256: str
    rows: int
    streams: int
    steps: int
    iterations: int
    collection_seconds: float
    learning_seconds: float
    optimizer_exposures: int
    optimizer_steps: int


@dataclass(frozen=True)
class Settings:
    producers: int
    queue_capacity: int
    mode: Literal["serial", "block", "drop"]
    production_seconds: float
    learning_seconds: float
    duration_seconds: float
    rows: int = 512


@dataclass(frozen=True)
class Packet:
    id: int
    actor: int
    version: int
    started: float


@dataclass(order=True)
class Event:
    time: float
    serial: int
    kind: Literal["start", "ready", "learned"] = field(compare=False)
    actor: int = field(default=0, compare=False)
    packet: Packet | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Outcome:
    settings: Settings
    produced_batches: int
    learned_batches: int
    stale_discarded_batches: int
    full_discarded_batches: int
    queued_batches: int
    held_batches: int
    active_batches: int
    in_production_batches: int
    peak_queued_batches: int
    peak_held_batches: int
    peak_completed_resident_batches: int
    learner_busy_fraction: float
    unblocked_offered_rows_per_second: float
    produced_rows_per_second: float
    nominal_learner_rows_per_second: float
    admitted_rows_per_second: float
    observed_version_lag_max: int
    admitted_version_lag_max: int
    admitted_age_seconds_mean: float | None
    admitted_age_seconds_max: float | None
    event_count: int


def simulate(settings: Settings) -> Outcome:
    """Stop at a fixed horizon; retain unconsumed work instead of draining for free."""
    s = settings
    if min(s.producers, s.queue_capacity, s.rows) < 1:
        raise ValueError("positive producer, queue and row counts required")
    if any(
        not math.isfinite(t) or t <= 0
        for t in (s.production_seconds, s.learning_seconds, s.duration_seconds)
    ):
        raise ValueError("finite positive durations required")
    if s.mode == "serial" and s.producers != 1:
        raise ValueError("serial control has exactly one complete-batch producer")
    events: list[Event] = []
    queue: deque[Packet] = deque()
    held: deque[Packet] = deque()
    active: Packet | None = None
    active_age = 0.0
    serial = version = packet_id = 0
    produced = completed = stale = full = in_production = 0
    peak_queue = peak_held = peak_resident = max_observed_lag = 0
    max_admitted_lag = 0
    busy = 0.0
    ages: list[float] = []

    def event(
        at: float,
        kind: Literal["start", "ready", "learned"],
        actor: int,
        packet: Packet | None = None,
    ) -> None:
        nonlocal serial
        serial += 1
        heapq.heappush(events, Event(at, serial, kind, actor, packet))

    def kick(now: float) -> None:
        nonlocal active, active_age, stale, busy, max_observed_lag, max_admitted_lag
        while active is None and queue:
            packet = queue.popleft()
            # Unblock one completed producer when the queue gains a slot. Its
            # next production snapshots the current version via a start event.
            if held:
                waiting = held.popleft()
                queue.append(waiting)
                event(now, "start", waiting.actor)
            lag = version - packet.version
            assert lag >= 0
            max_observed_lag = max(max_observed_lag, lag)
            if lag:
                stale += 1
                continue
            max_admitted_lag = max(max_admitted_lag, lag)
            active = packet
            active_age = now - packet.started
            busy += min(s.learning_seconds, s.duration_seconds - now)
            event(now + s.learning_seconds, "learned", packet.actor, packet)

    for actor in range(s.producers):
        event(actor * s.production_seconds / s.producers, "start", actor)
    event_count = 0
    while events:
        e = heapq.heappop(events)
        if e.time > s.duration_seconds:
            # Repeated floating additions may put an exact-horizon completion
            # a few ulps late. This is clock roundoff, never a policy-lag allowance.
            if e.time - s.duration_seconds > 1e-12:
                break
            e.time = s.duration_seconds
        event_count += 1
        if event_count > 200_000:
            raise ValueError("declared per-case event cap exceeded")
        if e.kind == "start":
            packet_id += 1
            packet = Packet(packet_id, e.actor, version, e.time)
            in_production += 1
            event(e.time + s.production_seconds, "ready", e.actor, packet)
        elif e.kind == "ready":
            assert e.packet is not None
            produced += 1
            in_production -= 1
            max_observed_lag = max(max_observed_lag, version - e.packet.version)
            if len(queue) == s.queue_capacity:
                if s.mode == "block":
                    held.append(e.packet)
                else:
                    full += 1
                    event(e.time, "start", e.actor)
            else:
                queue.append(e.packet)
                if s.mode != "serial":
                    event(e.time, "start", e.actor)
        else:
            assert active == e.packet and active is not None
            assert active.version == version  # one optimizer owner, zero lag
            completed += 1
            ages.append(active_age)
            version += 1
            active = None
            if s.mode == "serial":
                event(e.time, "start", e.actor)
        peak_queue = max(peak_queue, len(queue))
        peak_held = max(peak_held, len(held))
        peak_resident = max(
            peak_resident, len(queue) + len(held) + int(active is not None)
        )
        kick(e.time)
        assert 0 <= len(queue) <= s.queue_capacity
        assert 0 <= len(held) <= s.producers
        assert 0 <= in_production <= s.producers
        assert produced == completed + stale + full + len(queue) + len(held) + int(
            active is not None
        )
    return Outcome(
        settings=s,
        produced_batches=produced,
        learned_batches=completed,
        stale_discarded_batches=stale,
        full_discarded_batches=full,
        queued_batches=len(queue),
        held_batches=len(held),
        active_batches=int(active is not None),
        in_production_batches=in_production,
        peak_queued_batches=peak_queue,
        peak_held_batches=peak_held,
        peak_completed_resident_batches=peak_resident,
        learner_busy_fraction=busy / s.duration_seconds,
        unblocked_offered_rows_per_second=s.producers * s.rows / s.production_seconds,
        produced_rows_per_second=produced * s.rows / s.duration_seconds,
        nominal_learner_rows_per_second=s.rows / s.learning_seconds,
        admitted_rows_per_second=completed * s.rows / s.duration_seconds,
        observed_version_lag_max=max_observed_lag,
        admitted_version_lag_max=max_admitted_lag,
        admitted_age_seconds_mean=sum(ages) / len(ages) if ages else None,
        admitted_age_seconds_max=max(ages) if ages else None,
        event_count=event_count,
    )


def calibration(path: Path) -> Calibration:
    """Narrow the retained ordinary TrainingRun receipt once at its JSON boundary."""
    data = json.loads(path.read_text())
    stage = data["stages"][0]
    recipe = data["regime"]["stages"][0]
    assert data["status"] == stage["status"] == "completed"
    assert recipe["streams"] == 64 and recipe["transitions"] == 8
    assert recipe["learning"]["gradient"] == "ataraxos_move"
    iterations = len(stage["diagnostics"])
    assert iterations == 3 and stage["learner_transitions"] == 1536
    assert stage["optimizer_exposures"] == 384 and stage["optimizer_updates"] == 24
    return Calibration(
        str(path),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        512,
        64,
        8,
        iterations,
        float(stage["collection_seconds"]) / iterations,
        float(stage["learning_seconds"]) / iterations,
        stage["optimizer_exposures"],
        stage["optimizer_updates"],
    )


def check() -> None:
    serial = simulate(Settings(1, 1, "serial", 2, 1, 9))
    assert serial.learned_batches == 3 and serial.produced_batches == 3
    assert serial.stale_discarded_batches == serial.full_discarded_batches == 0
    assert serial.learner_busy_fraction == 1 / 3
    assert serial.admitted_age_seconds_mean == 2
    incomplete = simulate(Settings(1, 1, "serial", 2, 1, 2.5))
    assert incomplete.learned_batches == 0 and incomplete.active_batches == 1
    assert incomplete.learner_busy_fraction == 0.2
    fractional = simulate(Settings(1, 1, "serial", 0.1, 0.2, 0.9))
    assert fractional.learned_batches == 3 and fractional.active_batches == 0
    settings = Settings(64, 1, "block", 1, 1, 10)
    blocking = simulate(settings)
    dropping = simulate(Settings(64, 1, "drop", 1, 1, 10))
    assert blocking.peak_queued_batches == dropping.peak_queued_batches == 1
    assert blocking.peak_held_batches > 1 and blocking.full_discarded_batches == 0
    assert dropping.peak_held_batches == 0 and dropping.full_discarded_batches > 0
    assert blocking.stale_discarded_batches > 0
    assert simulate(settings) == blocking
    assert blocking.admitted_version_lag_max == dropping.admitted_version_lag_max == 0
    assert blocking.peak_completed_resident_batches > settings.queue_capacity + 1
    assert 0 <= blocking.learner_busy_fraction <= 1
    print(
        "Serial timing, partial service, zero-lag, block/drop, conservation and replay checks passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh output path required")
    check()
    source = Path(
        "scratch/gpu-probes/l4-003/loop-s64-reference/w64-d1-loop-b512-s64/run/run.json"
    )
    measured = calibration(source)
    service = measured.learning_seconds
    cases = [
        Settings(
            1,
            1,
            "serial",
            measured.collection_seconds,
            service,
            20 * (measured.collection_seconds + service),
        )
    ]
    for producers, offered_multiple in ((8, 0.5), (8, 2.0), (64, 16.0), (4096, 256.0)):
        production = producers * service / offered_multiple
        for capacity in (1, 8):
            for mode in ("block", "drop"):
                cases.append(
                    Settings(
                        producers,
                        capacity,
                        mode,
                        production,
                        service,
                        max(production * 3, production + 20 * service),
                    )
                )
    outcomes = [simulate(case) for case in cases]
    report = {
        "calibration": asdict(measured),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outcomes": [asdict(outcome) for outcome in outcomes],
        "limits": [
            "Horizon comparisons allow 1e-12second timestamp roundoff; policy versions remain exact integers",
            "Scheduling model only; no sampled values, gradients or policy-quality prediction",
            "Strict zero-lag admission is a conservative compatibility control, not a selected distributed design",
            "512row 8x64 batch shape is fixed; do not extrapolate service to a different global batch",
            "Short-run average service; no calibration of latency variance, mature policies or global batch filtering",
            "Synthetic offered rates; not coupled to inference/network/shared-GPU resource accounting",
            "Resident packet count excludes in-production partial trajectories, model copies and allocator overhead",
        ],
    }
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"Recorded {len(outcomes)} cases at {args.out}")


if __name__ == "__main__":
    main()
