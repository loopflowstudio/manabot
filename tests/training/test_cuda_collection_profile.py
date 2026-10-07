"""Device busy time is a clipped union, not a sum of overlapping activities."""

from experiments.runners.cuda_collection_profile import Event, busy_microseconds


def test_device_busy_union_clips_and_ignores_cpu_intervals() -> None:
    events = [
        Event(cat="kernel", ph="X", ts=0, dur=6),
        Event(cat="kernel", ph="X", ts=4, dur=4),
        Event(cat="gpu_memcpy", ph="X", ts=9, dur=5),
        Event(cat="cpu_op", ph="X", ts=0, dur=20),
    ]
    assert busy_microseconds(events, 2, 10) == 7
    assert busy_microseconds(events, 15, 20) == 0
