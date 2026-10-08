"""Experiment execution intent and receipts, separate from regime learning rules.

ExperimentSchedule declares seeds, placement and monitoring allocation. Hardware
is explicitly configured, never provisioned. ExperimentRun records every actual
process attempt; VerifyStore is its persistence owner.
"""

import os
from pathlib import Path
import socket
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from manabot.remote.plan import JobSpec
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.models import Strict


class PlannedRun(Strict):
    """Bind one resolved case and training seed to allocation and evaluation intent."""

    case: str
    seed: int = Field(ge=0)
    spec: JobSpec
    monitoring: MonitoringBudget
    checkpoint_seconds: float = Field(default=3600, gt=0)
    validate_numerics: bool = Field(default=False, exclude_if=lambda value: not value)


class Hardware(Strict):
    name: str
    host: str
    backend: Literal["local-cpu"] = "local-cpu"
    cpu_threads: int = Field(ge=2)
    # Dollars per occupied host hour, optional and explicitly supplied.
    dollars_per_hour: float | None = Field(default=None, ge=0)

    def admit(self) -> None:
        if self.cpu_threads > (os.cpu_count() or 1):
            raise ValueError("configured CPU capacity exceeds this host")
        if self.host != socket.gethostname():
            raise ValueError(
                "configured hardware is not this host; remote execution is unsupported"
            )


class HardwareInventory(Strict):
    resources: tuple[Hardware, ...]

    @model_validator(mode="after")
    def unique(self) -> "HardwareInventory":
        if len({r.name for r in self.resources}) != len(self.resources):
            raise ValueError("hardware names must be unique")
        return self

    def select(self, name: str) -> Hardware:
        for resource in self.resources:
            if resource.name == name:
                resource.admit()
                return resource
        raise ValueError(f"hardware is not configured: {name}")


class ExperimentSchedule(Strict):
    """Frozen launch intent. Monitoring is exploratory and has its own deals."""

    seeds: tuple[int, ...]
    hardware: str
    wall_seconds: float = Field(gt=0)
    # Additive learner + evaluator process seconds, including overlap.
    process_seconds: float = Field(gt=0)
    monitoring: MonitoringBudget
    checkpoint_seconds: float = Field(default=3600, gt=0)
    # Reserve existing scientific deal families without redefining their protocol.
    scientific_deal_seeds: tuple[int, ...] = ()
    # Case indexes per seed; empty means declaration order for every seed.
    order: tuple[tuple[int, ...], ...] = ()
    disk_reserve_bytes: int = Field(default=4 * 1024**3, ge=0)

    @model_validator(mode="after")
    def valid(self) -> "ExperimentSchedule":
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(s < 0 or s >= 100000 for s in self.seeds):
            raise ValueError("training seeds must be in [0, 100000)")
        if set(self.scientific_deal_seeds) & set(self.monitoring.protocol.deal_seeds):
            raise ValueError("monitoring and scientific deals must be disjoint")
        if self.monitoring.seconds >= self.process_seconds:
            raise ValueError("monitoring must leave a positive learning allocation")
        if self.monitoring.attempt_seconds > self.monitoring.seconds:
            raise ValueError("monitoring attempt exceeds its allocation")
        return self


class RegimeAttempt(Strict):
    ordinal: int
    case: str
    seed: int
    path: str
    allowance_seconds: float
    status: Literal["pending", "running", "completed", "failed", "interrupted"] = (
        "pending"
    )
    started_unix: float | None = None
    finished_unix: float | None = None
    process_seconds: float = 0
    run_id: str | None = None
    error: str | None = None
    recovery_parent: str | None = None


class ExperimentRun(Strict):
    id: str
    intent: dict[str, JsonValue]
    intent_sha256: str
    hardware: Hardware
    runtime: dict[str, str]
    status: Literal["running", "completed", "incomplete"] = "running"
    attempts: list[RegimeAttempt] = []
    last_seen_unix: float | None = None
    elapsed_seconds: float = 0
    coordinator_cpu_seconds: float = 0
    host_load: tuple[float, float, float] | None = None
    evaluator_seconds: float = 0
    process_seconds: float = 0
    host_dollars: float | None = None
    error: str | None = None
    notebook: str

    def report(self, directory: Path) -> Path:
        from manabot.training.comparison_notebook import write_comparison_notebook

        return write_comparison_notebook(directory, Path(self.notebook))
