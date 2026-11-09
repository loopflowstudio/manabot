"""Bounded host measurements for evaluation, with explicit clocks and coverage.

Counters cover the whole observed population. Timeline and latency samples are
bounded prefixes, labeled as such; they never consume policy random numbers.
Overlapping process intervals are separate resources, never a wall-time pie.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import threading
import time
from typing import Iterator, Literal

from manabot.infra.files import atomic_json


@dataclass(frozen=True)
class PerformanceSpan:
    name: str
    start: float
    end: float
    process: int
    thread: int
    request: str | None = None
    batch: str | None = None
    resource: str = "host elapsed"
    clock: str = "monotonic seconds; same host"
    stream: str | None = None


@dataclass
class Measurement:
    seconds: float = 0.0
    population: int = 0
    maximum_seconds: float = 0.0
    unit: str = "seconds"
    clock: str = "monotonic"
    status: str = "recorded"
    coverage: str = "all observed calls"
    partitions_elapsed: bool = True

    def observe(self, seconds: float) -> None:
        """Accumulate every call, independently of retained span/sample limits."""
        self.seconds += seconds
        self.population += 1
        self.maximum_seconds = max(self.maximum_seconds, seconds)


@dataclass
class EvalPerformance:
    identity: str
    enabled: bool = True
    sample_limit: int = 2048
    latency_boundary: str = "caller interval; boundary unspecified"
    first_request_seconds: float | None = None
    first_request_outcome: Literal["completed", "failed"] | None = None
    started: float = field(default_factory=time.monotonic)
    cpu_started: float = field(default_factory=time.process_time)
    counters: Counter[str] = field(default_factory=Counter)
    measurements: dict[str, Measurement] = field(default_factory=dict)
    spans: list[PerformanceSpan] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    failed_latencies: list[float] = field(default_factory=list)
    traces: list[str] = field(default_factory=list)
    device_events: list[dict[str, object]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def count(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] += value

    def record(
        self,
        name: str,
        start: float,
        *,
        request: str | None = None,
        batch: str | None = None,
        end: float | None = None,
        partitions_elapsed: bool = True,
    ) -> None:
        if not self.enabled:
            return
        end = time.monotonic() if end is None else end
        if end < start:
            raise ValueError("measurement ends before it starts")
        with self._lock:
            measurement = self.measurements.setdefault(
                name, Measurement(partitions_elapsed=partitions_elapsed)
            )
            if measurement.partitions_elapsed != partitions_elapsed:
                raise ValueError("measurement boundary changed within one receipt")
            measurement.observe(end - start)
            if len(self.spans) < self.sample_limit:
                self.spans.append(
                    PerformanceSpan(
                        name,
                        start,
                        end,
                        os.getpid(),
                        threading.get_ident(),
                        request,
                        batch,
                    )
                )

    def measure(self, name: str, seconds: float) -> None:
        """Accumulate a high-frequency, non-partitioning interval without a span."""
        if not self.enabled:
            return
        with self._lock:
            self.measurements.setdefault(
                name, Measurement(partitions_elapsed=False)
            ).observe(seconds)

    @contextmanager
    def span(
        self, name: str, *, request: str | None = None, batch: str | None = None
    ) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.record(name, start, request=request, batch=batch)

    def latency(
        self, elapsed: float, *, outcome: Literal["completed", "failed"] = "completed"
    ) -> None:
        """Retain separate bounded populations, with all-call counts and maxima."""
        with self._lock:
            self.counters["requests"] += 1
            self.counters[f"{outcome}_requests"] += 1
            samples = (
                self.latencies if outcome == "completed" else self.failed_latencies
            )
            if self.enabled:
                if self.first_request_seconds is None:
                    self.first_request_seconds = elapsed
                    self.first_request_outcome = outcome
                measure = self.measurements.setdefault(
                    f"{outcome}_request_latency", Measurement(partitions_elapsed=False)
                )
                measure.observe(elapsed)
                if len(samples) < self.sample_limit:
                    samples.append(elapsed)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            samples = sorted(self.latencies)
            failed = sorted(self.failed_latencies)
            return {
                "schema_version": 2,
                "identity": self.identity,
                "enabled": self.enabled,
                "started_monotonic": self.started,
                "elapsed_seconds": time.monotonic() - self.started,
                "process_cpu_seconds": time.process_time() - self.cpu_started,
                "counters": dict(self.counters),
                "measurements": {k: asdict(v) for k, v in self.measurements.items()},
                "spans": [asdict(s) for s in self.spans],
                "sample_coverage": "bounded first observations, not random sampling",
                "sample_limit": self.sample_limit,
                "latency_sample_count": len(samples),
                "latency_boundary": self.latency_boundary,
                "first_request_seconds": self.first_request_seconds,
                "first_request_outcome": self.first_request_outcome,
                "first_request_note": "first observed completion or failure; included in its population, not an excluded warmup",
                "latency_method": "sorted samples[floor((n-1)*q)]; completed and failed requests separate",
                "failed_latency_sample_count": len(failed),
                "failed_request_p50_seconds": failed[int((len(failed) - 1) * 0.5)]
                if failed
                else None,
                "failed_request_p95_seconds": failed[int((len(failed) - 1) * 0.95)]
                if failed
                else None,
                "request_p50_seconds": samples[int((len(samples) - 1) * 0.5)]
                if samples
                else None,
                "request_p95_seconds": samples[int((len(samples) - 1) * 0.95)]
                if samples
                else None,
                "traces": list(self.traces),
                "device_events": list(self.device_events),
                "unavailable": [
                    "pure Rust/Python binding overhead",
                    "kernel launches outside diagnostic trace",
                    "GPU event duration and host/device clock alignment",
                    "measured allocation dollars",
                ],
                "timing_note": "host launch timers include required waits; concurrent resources overlap",
            }

    def write(self, path: Path) -> None:
        # Readiness and admission may publish from different parent threads.
        # Serialize the whole replacement, including atomic_json's temporary file.
        with self._lock:
            atomic_json(path, self.snapshot())


def allocation_cost(
    seconds: float,
    *,
    measured_usd: float | None = None,
    hourly_usd: float | None = None,
) -> dict[str, object]:
    """Keep the whole allocation's cost once; jobs receive no duplicate bills."""
    return {
        "measured_usd": measured_usd,
        "estimated_usd": seconds * hourly_usd / 3600
        if hourly_usd is not None
        else None,
        "allocation_method": "unallocated; no per-job billing attribution",
        "unallocated_fraction": 1.0,
    }
