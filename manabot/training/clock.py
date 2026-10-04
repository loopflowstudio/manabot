"""Sleep-inclusive monotonic watchdog clock, separate from scientific schedules.

macOS perf_counter excludes system sleep. mach_continuous_time and Linux
CLOCK_BOOTTIME include it without trusting adjustable calendar time. Unsupported
platforms fail explicitly rather than silently changing budget semantics.
"""

import ctypes
import sys
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
