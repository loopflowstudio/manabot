"""The machine a run actually got, for reading its speed.

Rented machines sharing a GPU label differ by about 1.7x in update time, because
training's critical path is one CPU thread. A receipt therefore names the CPU and
the share of it the container may use; the GPU is already a run identity. These
facts describe a run; they are not identities and never gate recovery onto another
machine.
"""

import os
from pathlib import Path

from manabot.infra.identity import StrictModel

_CGROUP_QUOTA = Path("/sys/fs/cgroup/cpu.max")
_CPUINFO = Path("/proc/cpuinfo")


class HostFacts(StrictModel):
    cpu_model: str | None
    logical_cpus: int | None
    # CPUs this process may be scheduled on; a container often sees all of a node.
    schedulable_cpus: int | None
    # Cgroup CPU limit in CPUs; None when the container has no limit.
    cpu_quota: float | None


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


def host_facts() -> HostFacts:
    schedulable = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return HostFacts(
        cpu_model=_cpu_model(),
        logical_cpus=os.cpu_count(),
        schedulable_cpus=schedulable,
        cpu_quota=_cpu_quota(),
    )
