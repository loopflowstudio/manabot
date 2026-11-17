"""Machine-fact schemas for retained training receipts.

Etude owns current execution; these records preserve the fields written by
the retired Manabot executor.
"""

from typing import Literal

from pydantic import Field

from manabot.infra.identity import StrictModel

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
