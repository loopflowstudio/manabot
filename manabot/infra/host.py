"""The machine a run actually got, for reading its speed.

Rented machines sharing a GPU label differ by 1.6-1.8x in update time, because
with one producer training's critical path is one CPU thread. A receipt
therefore names the CPU and the share of it the container may use; the GPU is
already a run identity. These facts describe a run; they are not identities and
take no part in recovery admission.

A rented pod also sees its whole node's memory (540 GB on a 48 GB pod), so
`memory_limit` is the one reader of what the pod may actually use.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from manabot.infra.identity import StrictModel

_CGROUP_QUOTA = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V2_MEMORY = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEMORY = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CPUINFO = Path("/proc/cpuinfo")
# Etude's job environment drops RUNPOD_*, so the pod's figure is read from PID 1.
_INIT_ENVIRON = Path("/proc/1/environ")
_RUNPOD_MEMORY = b"RUNPOD_MEM_GB"
# Cgroup v1 spells "no limit" as a page-aligned value near 2**63.
_UNLIMITED_BYTES = 2**60

# "host" is the whole node's memory, for a reader that needs some figure when no
# allowance is readable; `memory_limit` itself never reports it.
MemorySource = Literal["cgroup-v2", "cgroup-v1", "runpod-env", "host"]


class MemoryLimit(StrictModel):
    """The most this container may use, and the reading that says so."""

    bytes: int
    source: MemorySource


class HostFacts(StrictModel):
    cpu_model: str | None
    logical_cpus: int | None
    # CPUs this process may be scheduled on; a container often sees all of a node.
    schedulable_cpus: int | None
    # Cgroup CPU limit in CPUs; None when the container has no limit.
    cpu_quota: float | None
    # The container's memory allowance; absent when nothing readable states one.
    memory_limit: MemoryLimit | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def _cpu_model(cpuinfo: Path = _CPUINFO) -> str | None:
    """The first `model name`; None off Linux, where no rented run happens."""
    try:
        lines = cpuinfo.read_text().splitlines()
    except OSError:
        return None
    names = (line.split(":", 1)[1].strip() for line in lines if "model name" in line)
    return next(names, None)


def _cpu_quota(cpu_max: Path = _CGROUP_QUOTA) -> float | None:
    """Limit from cgroup v2 `cpu.max` ("<quota> <period>" or "max <period>")."""
    try:
        quota, period = cpu_max.read_text().split()
    except (OSError, ValueError):
        return None
    return None if quota == "max" else int(quota) / int(period)


def _cgroup_memory(path: Path) -> int | None:
    """A cgroup memory limit in bytes; None when absent, malformed or unlimited."""
    try:
        limit = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return limit if 0 < limit < _UNLIMITED_BYTES else None


def _runpod_memory(init_environ: Path) -> int | None:
    """`RUNPOD_MEM_GB` from PID 1's environment, read as decimal gigabytes."""
    try:
        variables = init_environ.read_bytes().split(b"\0")
        environment = dict(v.split(b"=", 1) for v in variables if b"=" in v)
        gigabytes = float(environment.get(_RUNPOD_MEMORY, b"0"))
    except (OSError, ValueError):
        return None
    return int(gigabytes * 10**9) if gigabytes > 0 else None


def memory_limit(
    cgroup_v2: Path = _CGROUP_V2_MEMORY,
    cgroup_v1: Path = _CGROUP_V1_MEMORY,
    init_environ: Path = _INIT_ENVIRON,
) -> MemoryLimit | None:
    """The smallest allowance any source states; None when none states one."""
    readings: tuple[tuple[int | None, MemorySource], ...] = (
        (_cgroup_memory(cgroup_v2), "cgroup-v2"),
        (_cgroup_memory(cgroup_v1), "cgroup-v1"),
        (_runpod_memory(init_environ), "runpod-env"),
    )
    found = [
        MemoryLimit(bytes=limit, source=source)
        for limit, source in readings
        if limit is not None
    ]
    return min(found, key=lambda reading: reading.bytes, default=None)


def host_facts() -> HostFacts:
    schedulable = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return HostFacts(
        cpu_model=_cpu_model(),
        logical_cpus=os.cpu_count(),
        schedulable_cpus=schedulable,
        cpu_quota=_cpu_quota(),
        memory_limit=memory_limit(),
    )
