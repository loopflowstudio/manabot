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
from typing import Iterator

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
    started: float = field(default_factory=time.monotonic)
    cpu_started: float = field(default_factory=time.process_time)
    counters: Counter[str] = field(default_factory=Counter)
    measurements: dict[str, Measurement] = field(default_factory=dict)
    spans: list[PerformanceSpan] = field(default_factory=list)
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

    @contextmanager
    def span(
        self, name: str, *, request: str | None = None, batch: str | None = None
    ) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.record(name, start, request=request, batch=batch)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
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
                "unavailable": [
                    "pure Rust/Python binding overhead",
                    "kernel launches outside diagnostic trace",
                    "GPU event duration and host/device clock alignment",
                    "measured allocation dollars",
                ],
                "timing_note": "host launch timers include required waits; concurrent resources overlap",
            }

    def write(self, path: Path) -> None:
        # Serialize the whole replacement, including its temporary file.
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
