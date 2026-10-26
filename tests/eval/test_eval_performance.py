"""Timing coverage and cost stay honest under concurrency and incomplete work."""

import json
from pathlib import Path
import time

import pytest

from manabot.reports.evaluation import _tree, amdahl_bound, project_evaluations
from manabot.sim.performance import EvalPerformance, allocation_cost


def test_bounded_instrumentation_keeps_full_counters_and_no_zero_missing() -> None:
    perf = EvalPerformance("allocation", sample_limit=2)
    for _ in range(10):
        perf.count("decisions")
        perf.record("forward_host", time.monotonic())
        perf.latency(0.01)
    receipt = perf.snapshot()
    assert receipt["counters"]["decisions"] == 10
    assert len(receipt["spans"]) == 2
    assert receipt["latency_sample_count"] == 2
    assert receipt["measurements"]["forward_host"]["population"] == 10
    assert "native_step" not in receipt["measurements"]
    assert "bounded" in receipt["sample_coverage"]


def test_overlapping_intervals_are_not_a_wall_time_partition() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        _tree("allocation", 1, {"worker_a": 0.8, "worker_b": 0.8}, "test")
    tree = _tree("worker", 1, {"native_and_encoding": 0.4, "inference": 0.2}, "test")
    assert sum(c["seconds"] for c in tree["children"]) == pytest.approx(1)
    assert tree["children"][-1]["status"] == "derived"
    assert amdahl_bound(0.1, float("inf")) == pytest.approx(1 / 0.9)


def test_allocation_cost_is_charged_once_and_estimate_is_not_a_bill() -> None:
    cost = allocation_cost(3600, hourly_usd=2)
    assert cost["measured_usd"] is None
    assert cost["estimated_usd"] == 2
    assert cost["unallocated_fraction"] == 1
    assert "unallocated" in cost["allocation_method"]


def test_real_report_projection_retains_incomplete_population(tmp_path: Path) -> None:
    perf = EvalPerformance("cell")
    perf.count("legs_per_block", 4)
    perf.count("unattempted_games", 3)
    perf.write(tmp_path / "performance.json")
    (tmp_path / "rows.json").write_text(
        json.dumps(
            [
                {
                    "failure": "deadline",
                    "terminated": False,
                    "replay_passed": True,
                    "integrity": {"execution_failures": 1},
                    "deal_seed": 1,
                    "termination_reason": "timeout",
                }
            ]
        )
    )
    result = project_evaluations([tmp_path])
    details = result["eval_performance"]["runs"][0]
    assert details["valid_blocks"] == details["valid_games"] == 0
    assert details["unattempted_games"] == 3
    assert details["interrupted_games"] == 1
    assert details["coverage"] == {}
    assert details["games_per_dollar"] is None
    assert result["decision_unit"] == "evaluation"
    assert not result["synthetic"]
    assert "hypothesis" in details["optimization"]["status"]
