"""Hierarchical profiler semantics: labels, nesting, percentages, accumulation and samples.

Time comes from a controlled clock, so each expected duration and percentage is exact.
"""

import random
import time

import numpy as np
import pytest

from manabot.infra.profiler import Profiler


class Clock:
    """Stands in for `time.perf_counter`; `advance` is the work being timed."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(time, "perf_counter", clock)
    return clock


@pytest.mark.parametrize("enabled", [False, True])
def test_profiling_preserves_training_random_stream(enabled: bool) -> None:
    saved = random.getstate()
    try:
        random.seed(10349)
        before = random.getstate()
        profiler = Profiler(enabled=enabled, max_samples=2)
        for _ in range(20):
            with profiler.track("collection"):
                with profiler.track("policy"):
                    pass
        assert random.getstate() == before
        if enabled:
            stats = profiler.get_stats()
            assert stats["collection"]["count"] == 20
            assert len(profiler.node_cache["collection"].durations) == 2
    finally:
        random.setstate(saved)


def test_labels_nest_relative_to_the_running_node(clock: Clock) -> None:
    profiler = Profiler(enabled=True)
    with profiler.track("a"):
        with profiler.track("b"):
            clock.advance(1)
    with profiler.track("z"):
        clock.advance(1)
    assert set(profiler.get_stats()) == {"a", "a/b", "z"}

    with profiler.track("a"):
        with pytest.raises(ValueError, match="Empty label is not allowed"):
            with profiler.track(""):
                pass


def test_hierarchy_percentages(clock: Clock) -> None:
    profiler = Profiler(enabled=True)
    with profiler.track("group1"):
        with profiler.track("a"):
            with profiler.track("alpha"):
                clock.advance(0.01)
            with profiler.track("beta"):
                clock.advance(0.01)
    with profiler.track("group2"):
        with profiler.track("x"):
            clock.advance(0.018)
        with profiler.track("y"):
            clock.advance(0.002)

    stats = profiler.get_stats()
    percent = {
        path: (s["pct_of_parent"], s["pct_of_total"]) for path, s in stats.items()
    }
    assert percent["group1"] == pytest.approx((50, 50))
    assert percent["group2"] == pytest.approx((50, 50))
    assert percent["group1/a"][0] == pytest.approx(100)
    assert percent["group1/a/alpha"][0] == pytest.approx(50)
    assert percent["group1/a/beta"][0] == pytest.approx(50)
    assert percent["group2/x"][0] == pytest.approx(90)
    assert percent["group2/y"][0] == pytest.approx(10)


def test_repeated_calls_accumulate_and_a_running_node_counts_its_time(
    clock: Clock,
) -> None:
    profiler = Profiler(enabled=True)
    rollout, gradient = 0.0, 0.0
    for rollout_seconds, gradient_seconds in [
        (0.002, 0.0005),
        (0.001, 0.0003),
        (0.0015, 0.0004),
    ]:
        with profiler.track("rollout"):
            clock.advance(rollout_seconds)
        with profiler.track("gradient"):
            clock.advance(gradient_seconds)
        rollout += rollout_seconds
        gradient += gradient_seconds
        stats = profiler.get_stats()
        assert stats["rollout"]["total_time"] == pytest.approx(rollout)
        assert stats["gradient"]["total_time"] == pytest.approx(gradient)

    with profiler.track("rollout"):
        clock.advance(0.002)
        assert profiler.get_stats()["rollout"]["total_time"] == pytest.approx(
            rollout + 0.002
        )


def test_statistical_metrics(clock: Clock) -> None:
    profiler = Profiler(enabled=True, max_samples=100)
    durations = [0.002 + (i % 5) * 0.0005 for i in range(20)]
    for seconds in durations:
        with profiler.track("test_node"):
            clock.advance(seconds)

    stats = profiler.get_stats()["test_node"]
    assert stats["min"] == pytest.approx(min(durations))
    assert stats["max"] == pytest.approx(max(durations))
    assert stats["mean"] == pytest.approx(float(np.mean(durations)))
    assert stats["p5"] == pytest.approx(float(np.percentile(durations, 5)))
    assert stats["p95"] == pytest.approx(float(np.percentile(durations, 95)))


def test_samples_are_capped_while_every_call_is_counted(clock: Clock) -> None:
    profiler = Profiler(enabled=True, max_samples=10)
    for i in range(20):
        with profiler.track("sampled_node"):
            clock.advance(0.001 * (i + 1))

    stats = profiler.get_stats()["sampled_node"]
    assert stats["count"] == 20
    assert len(profiler.node_cache["sampled_node"].durations) == 10
    assert stats["min"] < stats["mean"] < stats["max"]
    assert stats["p5"] <= stats["p95"]


def test_nested_relative_labels(clock: Clock) -> None:
    profiler = Profiler(enabled=True)
    with profiler.track("level1"):
        with profiler.track("level2"):
            with profiler.track("level3"):
                with profiler.track("level4"):
                    with profiler.track("level5"):
                        clock.advance(0.01)

    stats = profiler.get_stats()
    paths = [
        "level1",
        "level1/level2",
        "level1/level2/level3",
        "level1/level2/level3/level4",
        "level1/level2/level3/level4/level5",
    ]
    assert list(stats) == paths
    for path in paths[1:]:
        assert stats[path]["pct_of_parent"] == pytest.approx(100)
    assert stats["level1"]["pct_of_total"] == pytest.approx(100)
