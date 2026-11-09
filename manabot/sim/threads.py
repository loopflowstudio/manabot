"""Declared thread budgets for evaluation processes, proven by census.

Containers often report the host's CPU count (`os.cpu_count()` returned 384 on
a 32-vCPU RunPod pod), so native and numeric libraries size their pools for the
host. Every evaluation child therefore starts from `child_environment()`, which
pins the known pool variables before its interpreter loads any library, and
counts its own OS threads against a declared `ThreadBudget` before reporting.
A count above budget is an infrastructure failure, never a player forfeit.
The same `ProcessCensus` records resident memory and whether Torch was loaded,
so receipts show which children stayed light.

`effective_cpus()` is the CPU capacity a process may actually use: the minimum
of the logical count, its affinity mask and its cgroup CPU quota.

Torch-free: game and replay children import this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import psutil

# Pools sized from these variables at library load: OpenMP (Torch intra-op,
# libgomp), MKL, OpenBLAS (numpy wheels), Accelerate, numexpr and Rayon.
POOL_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

_CGROUP_ROOT = Path("/sys/fs/cgroup")


class ThreadBudgetExceeded(RuntimeError):
    """Infrastructure failure: a process exceeded its declared thread count."""


@dataclass(frozen=True)
class ThreadBudget:
    role: str
    threads: int

    def account(self, counters: dict[str, int]) -> None:
        """Record this process's census under `role`, then enforce the bound."""
        census = ProcessCensus.take()
        census.record(counters, self)
        census.enforce(self)


# Single-threaded pools leave the interpreter's main thread plus a small
# allowance for runtime helpers (Torch's lazily created interop worker when a
# search player loads its checkpoint in-child, CUDA driver and profiler threads
# in serving).
GAME_BUDGET = ThreadBudget("game", 4)
REPLAY_BUDGET = ThreadBudget("replay", 4)
SERVING_BUDGET = ThreadBudget("serving", 16)

# The group worker runs `play_cell`. Without a shared runtime (code and search
# players only) it keeps the main thread plus a small helper allowance. Hosting
# an inference runtime adds one game thread per active game; its main thread,
# control-channel reader and sender, plus four runtime helpers make the fixed
# allowance. Recorded for calibration, not enforced: the worker coordinates
# games rather than playing them.
_GROUP_WORKER_UNSERVED_THREADS = 8
_GROUP_WORKER_ALLOWANCE = 7


def group_worker_budget(max_active_games: int | None) -> ThreadBudget:
    """Declared threads for the process running `play_cell`.

    `max_active_games` is the shared runtime's bound, or None without one.
    """
    if max_active_games is None:
        return ThreadBudget("group_worker", _GROUP_WORKER_UNSERVED_THREADS)
    return ThreadBudget("group_worker", max_active_games + _GROUP_WORKER_ALLOWANCE)


def _cgroup_cpus(root: Path) -> float | None:
    v2 = root / "cpu.max"
    if v2.exists():
        quota, _, period = v2.read_text().strip().partition(" ")
        if quota == "max":
            return None
        return int(quota) / int(period or "100000")
    quota_path, period_path = (
        root / "cpu" / "cpu.cfs_quota_us",
        root / "cpu" / "cpu.cfs_period_us",
    )
    if quota_path.exists() and period_path.exists():
        quota = int(quota_path.read_text())
        return None if quota <= 0 else quota / int(period_path.read_text())
    return None


def effective_cpus(
    *, affinity: set[int] | None = None, cgroup_root: Path = _CGROUP_ROOT
) -> int:
    """Usable CPUs; `affinity` defaults to this process's mask where supported."""
    candidates = [os.cpu_count() or 1]
    if affinity is None and hasattr(os, "sched_getaffinity"):
        affinity = os.sched_getaffinity(0)
    if affinity:
        candidates.append(len(affinity))
    quota = _cgroup_cpus(cgroup_root)
    if quota is not None:
        candidates.append(max(1, math.ceil(quota)))
    return max(1, min(candidates))


def child_environment() -> dict[str, str]:
    """This process's environment with every known pool pinned to one thread."""
    return {**os.environ, **dict.fromkeys(POOL_VARIABLES, "1")}


@dataclass(frozen=True)
class ProcessCensus:
    """This process's OS threads (names on Linux only), memory and Torch use."""

    threads: int
    names: Mapping[str, int]
    rss_bytes: int
    torch_loaded: bool

    @classmethod
    def take(cls) -> ProcessCensus:
        names: dict[str, int] = {}
        if sys.platform == "linux":
            for task in Path("/proc/self/task").iterdir():
                try:
                    name = (task / "comm").read_text().strip()
                except OSError:
                    continue
                names[name] = names.get(name, 0) + 1
        process = psutil.Process()
        return cls(
            process.num_threads(),
            names,
            process.memory_info().rss,
            "torch" in sys.modules,
        )

    def record(self, counters: dict[str, int], budget: ThreadBudget) -> None:
        """Keep maxima by role beside the declared thread budget."""
        observed = {
            f"threads/{budget.role}": self.threads,
            f"rss_bytes/{budget.role}": self.rss_bytes,
            f"torch_loaded/{budget.role}": int(self.torch_loaded),
            **{
                f"thread_names/{budget.role}/{name}": count
                for name, count in self.names.items()
            },
        }
        for key, value in observed.items():
            counters[key] = max(counters.get(key, 0), value)
        counters[f"threads_budget/{budget.role}"] = budget.threads

    def enforce(self, budget: ThreadBudget) -> None:
        if self.threads > budget.threads:
            raise ThreadBudgetExceeded(
                f"{budget.role} process ran {self.threads} threads, budget "
                f"{budget.threads}: {dict(self.names) or 'names unavailable'}"
            )
