"""Sleep-inclusive monotonic watchdog clock, separate from scientific schedules.

macOS perf_counter excludes system sleep. mach_continuous_time and Linux
CLOCK_BOOTTIME include it without trusting adjustable calendar time. Unsupported
platforms fail explicitly rather than silently changing budget semantics.
"""

import ctypes
from functools import cache
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


if sys.platform == "darwin":
    _lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    _lib.mach_continuous_time.restype = ctypes.c_uint64
    _lib.mach_continuous_time.argtypes = []
    _lib.mach_timebase_info.argtypes = [ctypes.POINTER(_Timebase)]
    _base = _Timebase()
    if _lib.mach_timebase_info(ctypes.byref(_base)) != 0 or not _base.denom:
        raise RuntimeError("cannot resolve macOS continuous clock")


def watchdog_seconds() -> float:
    """Monotonic seconds including sleep; epoch has no cross-process meaning."""
    if sys.platform == "darwin":
        return _lib.mach_continuous_time() * _base.numer / _base.denom / 1e9
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    raise RuntimeError("sleep-inclusive watchdog unavailable on this platform")


@cache
def boot_identity() -> str:
    """Bind persisted awake-clock readings to one OS boot, not wall-clock time."""
    if sys.platform == "darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "kern.bootsessionuuid"], text=True
        ).strip()
    if sys.platform.startswith("linux"):
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    raise RuntimeError("active-runtime recovery supports macOS and Linux only")


def arm_active_deadline(seconds: float) -> None:
    """Bound an isolated worker even if its supervisor dies; system sleep is free.

    macOS/Linux monotonic clocks exclude suspend. This thread does not prevent
    sleep and consumes no learning RNG or model state.
    """
    if seconds <= 0 or os.getpid() != os.getpgrp():
        raise ValueError("active watchdog requires a positive isolated-worker budget")
    boot_identity()
    deadline = time.monotonic() + seconds

    def watch() -> None:
        while time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.001, deadline - time.monotonic())))
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True, name="active-runtime-watchdog").start()


def arm_worker_deadline(
    *, active_runtime: bool, allowance_seconds: float, deadline_unix: float
) -> None:
    """Arm the selected allocation clock for an isolated learner or evaluator."""
    if active_runtime:
        arm_active_deadline(allowance_seconds)
        return

    def expire(signum: int, frame: object) -> None:
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGALRM, expire)
    remaining = deadline_unix - time.time()
    if remaining <= 0:
        expire(signal.SIGALRM, None)
    signal.setitimer(signal.ITIMER_REAL, remaining)
