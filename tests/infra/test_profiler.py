"""
test_profiler.py
Tests for hierarchical performance tracking semantics and percentage calculations.

This suite verifies:
(1) Label Enforcement:
    - A valid nested sequence using the context manager with relative labels works as expected.
    - An invalid nested sequence (using an empty label) raises a ValueError.
(2) Percentage Calculations:
    - Build a multi-level hierarchy with simulated sleep durations and verify that each node's percentages are roughly as expected.
(3) Accumulated Timing:
    - Repeated context-managed calls for the same cached node accumulate time.
    - While a node is still running, its effective total time is higher than the stored total.
(4) Statistical Metrics:
    - The profiler computes detailed statistics (min, max, mean, p5, p95) for each node.
(5) Max Samples:
    - The durations list respects the max_samples limit.
(6) Nested Relative Labels:
    - Deeply nested calls using relative labels produce the correct hierarchical paths.
"""

import time
import pytest
import numpy as np
from manabot.infra.profiler import Profiler  # Adjust the import as needed

# --- Label Enforcement Tests ---
def test_label_enforcement():
    profiler = Profiler(enabled=True)
    profiler.reset()  # Reset the profiler
    
    # Valid sequence using relative labels.
    with profiler.track("a"):
        with profiler.track("b"):  # This is relative to "a" → full path "a/b"
            pass

    stats = profiler.get_stats()
    assert "a/b" in stats, "Missing stats for node 'a/b'"

    # Valid sequence at the root.
    with profiler.track("z"):
        pass

    # Now test an invalid sequence:
    profiler.reset()
    with profiler.track("a"):
        with pytest.raises(ValueError) as exc_info:
            # Empty label is not allowed
            with profiler.track(""):
                pass
        assert "Empty label is not allowed" in str(exc_info.value)

# --- Hierarchy Percentage Calculations ---
def test_hierarchy_percentages():
    """
    Create a multi-level hierarchy and verify percentage calculations.
    Structure:
      - group1
          - a (relative to group1)
              - alpha (relative to group1/a)
              - beta (relative to group1/a)
      - group2
          - x (relative to group2)
          - y (relative to group2)
    """
    profiler = Profiler(enabled=True)
    profiler.reset()
    
    # Group 1
    with profiler.track("group1"):
        with profiler.track("a"):
            with profiler.track("alpha"):
                time.sleep(0.1)
            with profiler.track("beta"):
                time.sleep(0.1)
    
    # Group 2
    with profiler.track("group2"):
        with profiler.track("x"):
            time.sleep(0.18)
        with profiler.track("y"):
            time.sleep(0.02)
    
    # Get stats and print for debugging
    stats = profiler.get_stats()
    for path, data in stats.items():
        print(f"{path}: {data}")

    # Check overall split: group1 and group2 should be roughly 50% each.
    group1_stats = stats.get("group1")
    group2_stats = stats.get("group2")
    assert group1_stats is not None, "Missing stats for 'group1'"
    assert group2_stats is not None, "Missing stats for 'group2'"
    tol_overall = 20.0  # tolerance percentage points
    assert abs(group1_stats["pct_of_total"] - 50) < tol_overall, f"'group1' pct_of_total not ~50: {group1_stats['pct_of_total']}"
    assert abs(group2_stats["pct_of_total"] - 50) < tol_overall, f"'group2' pct_of_total not ~50: {group2_stats['pct_of_total']}"
    
    # Within group1, "a" should be ~100% of group1.
    a_stats = stats.get("group1/a")
    assert a_stats is not None, "Missing stats for 'group1/a'"
    tol_group1 = 5.0
    assert abs(a_stats["pct_of_parent"] - 100) < tol_group1, f"'group1/a' pct_of_parent not ~100: {a_stats['pct_of_parent']}"
    
    # Within group1/a, "alpha" and "beta" should split time roughly equally.
    alpha_stats = stats.get("group1/a/alpha")
    beta_stats = stats.get("group1/a/beta")
    assert alpha_stats is not None, "Missing stats for 'group1/a/alpha'"
    assert beta_stats is not None, "Missing stats for 'group1/a/beta'"
    tol_sub = 25.0
    assert abs(alpha_stats["pct_of_parent"] - 50) < tol_sub, f"'group1/a/alpha' pct_of_parent not ~50: {alpha_stats['pct_of_parent']}"
    assert abs(beta_stats["pct_of_parent"] - 50) < tol_sub, f"'group1/a/beta' pct_of_parent not ~50: {beta_stats['pct_of_parent']}"
    
    # In group2, "x" should be about 90% and "y" about 10% of group2's time.
    x_stats = stats.get("group2/x")
    y_stats = stats.get("group2/y")
    assert x_stats is not None, "Missing stats for 'group2/x'"
    assert y_stats is not None, "Missing stats for 'group2/y'"
    tol_group2 = 25.0
    assert abs(x_stats["pct_of_parent"] - 90) < tol_group2, f"'group2/x' pct_of_parent not ~90: {x_stats['pct_of_parent']}"
    assert abs(y_stats["pct_of_parent"] - 10) < tol_group2, f"'group2/y' pct_of_parent not ~10: {y_stats['pct_of_parent']}"

# --- Accumulated Timing Tests ---
def test_accumulated_timing():
    """
    Verify that repeated context-managed calls for the same cached node accumulate time,
    and that get_stats() correctly reports effective times when nodes are no longer active.
    """
    profiler = Profiler(enabled=True)
    profiler.reset()

    rollout_times = [0.02, 0.01, 0.015]
    gradient_times = [0.005, 0.003, 0.004]
    accumulated_rollout = 0.0
    accumulated_gradient = 0.0

    for i in range(len(rollout_times)):
        with profiler.track("rollout"):
            time.sleep(rollout_times[i])
        accumulated_rollout += rollout_times[i]

        with profiler.track("gradient"):
            time.sleep(gradient_times[i])
        accumulated_gradient += gradient_times[i]

        stats = profiler.get_stats()
        rollout_stats = stats.get("rollout")
        gradient_stats = stats.get("gradient")
        print(f"Iteration {i} - rollout_stats: {rollout_stats}")
        print(f"Iteration {i} - gradient_stats: {gradient_stats}")
        assert abs(rollout_stats["total_time"] - accumulated_rollout) < 0.02, f"Iteration {i}: rollout time incorrect"
        assert abs(gradient_stats["total_time"] - accumulated_gradient) < 0.02, f"Iteration {i}: gradient time incorrect"

    # Test live stats: start "rollout" without finishing immediately.
    with profiler.track("rollout"):
        time.sleep(0.02)
        stats_running = profiler.get_stats()
        running_rollout = stats_running.get("rollout", {}).get("total_time", 0)
        # The effective time (running) should be greater than the previously accumulated time.
        assert running_rollout > accumulated_rollout, "Effective time for running node not greater than accumulated time"

# --- Statistical Metrics Tests ---
def test_statistical_metrics():
    """
    Verify that the profiler correctly computes detailed statistics (min, max, mean, p5, p95)
    for a node's call durations.
    """
    profiler = Profiler(enabled=True, max_samples=100)
    profiler.reset()
    durations = []
    
    # Increase sleep durations for better resolution.
    # Use d = 0.02 + (i % 5) * 0.005; so values cycle: 0.02, 0.025, 0.03, 0.035, 0.04 sec.
    for i in range(20):
        d = 0.02 + (i % 5) * 0.005
        durations.append(d)
        with profiler.track("test_node"):
            time.sleep(d)
    
    stats = profiler.get_stats()
    test_stats = stats.get("test_node")
    
    durations_array = np.array(durations)
    expected_min = float(np.min(durations_array))
    expected_max = float(np.max(durations_array))
    expected_mean = float(np.mean(durations_array))
    expected_p5 = float(np.percentile(durations_array, 5))
    expected_p95 = float(np.percentile(durations_array, 95))
    
    print("Expected min:", expected_min)
    print("Expected max:", expected_max)
    print("Expected mean:", expected_mean)
    print("Expected p5:", expected_p5)
    print("Expected p95:", expected_p95)
    print("Profiler reported for 'test_node':", test_stats)
    
    tol = 0.01  # Tolerance in seconds.
    assert abs(test_stats["min"] - expected_min) < tol, f"min mismatch: expected {expected_min}, got {test_stats['min']}"
    assert abs(test_stats["max"] - expected_max) < tol, f"max mismatch: expected {expected_max}, got {test_stats['max']}"
    assert abs(test_stats["mean"] - expected_mean) < tol, f"mean mismatch: expected {expected_mean}, got {test_stats['mean']}"
    assert abs(test_stats["p5"] - expected_p5) < tol, f"p5 mismatch: expected {expected_p5}, got {test_stats['p5']}"
    assert abs(test_stats["p95"] - expected_p95) < tol, f"p95 mismatch: expected {expected_p95}, got {test_stats['p95']}"

# --- Max Samples Test ---
def test_max_samples():
    """
    Verify that the profiler correctly respects the max_samples limit.
    """
    max_samples = 10
    profiler = Profiler(enabled=True, max_samples=max_samples)
    profiler.reset()
    
    iterations = max_samples * 2  # More iterations than max_samples.
    for i in range(iterations):
        # Use a slightly longer sleep to capture nonzero durations.
        with profiler.track("sampled_node"):
            time.sleep(0.005)
    
    stats = profiler.get_stats()
    node_stats = stats.get("sampled_node")
    
    # Verify node was called the correct number of times.
    assert node_stats["count"] == iterations, f"Count mismatch: expected {iterations}, got {node_stats['count']}"
    # Check that computed statistics are reasonable.
    assert node_stats["min"] < node_stats["mean"] < node_stats["max"], "Statistics ordering (min < mean < max) is incorrect"
    assert node_stats["p5"] <= node_stats["p95"], "p5 should be less than or equal to p95"

# --- Nested Relative Labels Test ---
def test_nested_relative_labels():
    """Test deeply nested relative labels to ensure proper hierarchy."""
    profiler = Profiler(enabled=True)
    profiler.reset()
    
    with profiler.track("level1"):
        with profiler.track("level2"):
            with profiler.track("level3"):
                with profiler.track("level4"):
                    with profiler.track("level5"):
                        time.sleep(0.01)
    
    stats = profiler.get_stats()
    
    # Check that all levels exist.
    assert "level1" in stats, "Missing level1"
    assert "level1/level2" in stats, "Missing level1/level2"
    assert "level1/level2/level3" in stats, "Missing level1/level2/level3"
    assert "level1/level2/level3/level4" in stats, "Missing level1/level2/level3/level4"
    assert "level1/level2/level3/level4/level5" in stats, "Missing level1/level2/level3/level4/level5"
    
    # Check that the hierarchy percentages make sense.
    level5 = stats["level1/level2/level3/level4/level5"]
    level4 = stats["level1/level2/level3/level4"]
    level3 = stats["level1/level2/level3"]
    level2 = stats["level1/level2"]
    level1 = stats["level1"]
    
    tol = 0.5  # Tolerance in percentage points.
    assert abs(level5["pct_of_parent"] - 100.0) < tol, f"level5 pct_of_parent not ~100%: {level5['pct_of_parent']}"
    assert abs(level4["pct_of_parent"] - 100.0) < tol, f"level4 pct_of_parent not ~100%: {level4['pct_of_parent']}"
    assert abs(level3["pct_of_parent"] - 100.0) < tol, f"level3 pct_of_parent not ~100%: {level3['pct_of_parent']}"
    assert abs(level2["pct_of_parent"] - 100.0) < tol, f"level2 pct_of_parent not ~100%: {level2['pct_of_parent']}"
    assert level1["pct_of_total"] > 0, "level1 should have a nonzero pct_of_total"

if __name__ == "__main__":
    pytest.main([__file__])