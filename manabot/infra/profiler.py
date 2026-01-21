"""
profiler.py
Hierarchical performance tracker with context-manager API and automatic nesting.

This module provides a Profiler class that enables tracking execution time across
various parts of your code in a hierarchical manner using relative labels. Key features include:

1. Context-manager API for clean, exception-safe timing
2. Automatic nesting of timers based on execution context
3. Node caching and time accumulation across repeated operations
4. Support for retrieving statistics including percentages of parent/total time,
   call counts, and detailed percentile statistics (min, max, mean, p5, p95, etc.)
5. Low overhead when disabled

Usage example:
    profiler = Profiler(enabled=True)
    with profiler.track("rollout"):
        # time rollout code
        with profiler.track("step"):  # Note: this is relative to "rollout"
            # time environment step call
            with profiler.track("env"):  # Note: this is relative to "rollout/step"
                # time environment operation
                ...
    stats = profiler.get_stats()  # returns a nested dict of timings
"""

import time
import random
from contextlib import contextmanager
from typing import Optional, Dict, List
import numpy as np

class TimingNode:
    def __init__(self, label: str, parent: Optional["TimingNode"] = None, max_samples: int = 100):
        self.label = label              # local label only
        self.parent = parent
        self.previous_total = 0.0       # accumulated time from previous runs
        self.start_time = None          # current start time (None if not running)
        self.children: Dict[str, "TimingNode"] = {}  # keyed by local label
        self.count = 0                  # number of times this node was entered
        self.durations: List[float] = []  # list to store durations of each call
        self.max_samples = max_samples  # maximum number of duration samples to keep

    def start(self):
        if self.start_time is not None:
            raise RuntimeError(f"Timer '{self.label}' is already running.")
        self.start_time = time.perf_counter()
        self.count += 1

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError(f"Timer '{self.label}' was not started!")
        elapsed = time.perf_counter() - self.start_time
        self.previous_total += elapsed
        
        # Add duration using reservoir sampling to maintain max_samples limit
        if len(self.durations) >= self.max_samples:
            idx = random.randint(0, self.count - 1)  # Use count for better distribution
            if idx < self.max_samples:  # Only replace if idx is within our sample array
                self.durations[idx] = elapsed
        else:
            self.durations.append(elapsed)
            
        self.start_time = None
        return elapsed

    def running_total(self) -> float:
        """Return total accumulated time plus running time if still active."""
        if self.start_time is not None:
            return self.previous_total + (time.perf_counter() - self.start_time)
        return self.previous_total

class Profiler:
    """
    Hierarchical profiler with context-manager API, auto nesting, and comprehensive statistics.
    Uses relative labels for simpler usage without a built-in root node.

    Usage example:
        profiler = Profiler(enabled=True)
        with profiler.track("main"):  # Create your own root
            # time main code
            with profiler.track("step"):  # Relative to "main"
                # time step call
                with profiler.track("env"):  # Relative to "main/step"
                    # time environment operation
                    ...
        stats = profiler.get_stats()  # returns a nested dict of timings

    The profiler caches nodes by full path so that repeated uses accumulate times.
    """
    def __init__(self, enabled: bool = True, max_samples: int = 100):
        self.enabled = enabled
        self.max_samples = max_samples
        self.stack = []  # No default root node
        self.node_cache: Dict[str, TimingNode] = {}

    @contextmanager
    def track(self, label: str):
        """
        Use as a context manager to time a code block. The provided label is automatically
        nested under the current active node if one exists. Labels are relative to the current context.
        If the same label is used again in the same position, its time is accumulated.
        """
        if not self.enabled:
            yield
            return

        if not label:
            raise ValueError("Empty label is not allowed")
            
        # Handle the case where this might be a root-level node
        if not self.stack:
            # This is a root level node
            full_label = label
            parent = None
        else:
            # This is a child node, create proper path
            parent = self.stack[-1]
            parent_path = self._get_full_path(parent)
            full_label = f"{parent_path}/{label}" if parent_path else label
        
        # Get or create the node
        if full_label in self.node_cache:
            node = self.node_cache[full_label]
            if node.start_time is not None:
                raise RuntimeError(f"Node '{full_label}' is already running.")
        else:
            node = TimingNode(label, parent, max_samples=self.max_samples)
            self.node_cache[full_label] = node
            if parent:
                parent.children[label] = node
        
        # Start timing and add to stack
        node.start()
        self.stack.append(node)
        
        try:
            yield
        finally:
            # Ensure we're stopping the correct node
            if self.stack and self.stack[-1] == node:
                node.stop()
                self.stack.pop()
            else:
                # This shouldn't happen with proper context manager usage
                raise RuntimeError(f"Profiler stack corruption detected while stopping '{full_label}'")

    def _get_full_path(self, node: TimingNode) -> str:
        """Return the full hierarchical path for a node."""
        if node is None:
            return ""
            
        parts = []
        current = node
        while current.parent is not None:
            parts.append(current.label)
            current = current.parent
        
        # Add the root node's label if it exists
        if current:  # This should be the root node
            parts.append(current.label)
            
        return "/".join(reversed(parts))

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dictionary mapping each node's full hierarchical path
        to its timing statistics:
            - total_time: The running total time.
            - pct_of_parent: Percentage of parent's time.
            - pct_of_total: Percentage of overall total time.
            - count: Number of times this node was entered.
            - min: Minimum call duration among samples.
            - max: Maximum call duration among samples.
            - mean: Mean call duration among samples.
            - p5: 5th percentile of call duration.
            - p95: 95th percentile of call duration.
        """
        stats = {}
        
        # Find the total time across all root nodes
        total_time = sum(node.running_total() for node in self.node_cache.values() 
                        if node.parent is None)
        
        # If no timing has been done, return empty stats
        if total_time <= 0:
            return stats
            
        # Process each node in the cache to build stats dictionary
        for full_path, node in self.node_cache.items():
            node_total = node.running_total()
            
            # Calculate parent total time for percentage calculation
            parent_total = 0.0
            if node.parent:
                parent_path = self._get_full_path(node.parent)
                parent_node = self.node_cache.get(parent_path)
                if parent_node:
                    parent_total = parent_node.running_total()
            else:
                # For root nodes, parent total is the overall total
                parent_total = total_time
                
            # Build samples array for statistics
            samples = []
            if node.count > 0:
                # Add stored durations
                if node.durations:
                    samples.extend(node.durations)
                
                # If node is currently running, add current duration
                if node.start_time is not None:
                    current_duration = time.perf_counter() - node.start_time
                    samples.append(current_duration)
                
            # Calculate statistics if we have samples
            if samples:
                samples_array = np.array(samples)
                min_val = float(np.min(samples_array))
                max_val = float(np.max(samples_array))
                mean_val = float(np.mean(samples_array))
                p5 = float(np.percentile(samples_array, 5)) if len(samples) > 1 else min_val
                p95 = float(np.percentile(samples_array, 95)) if len(samples) > 1 else max_val
            else:
                min_val = max_val = mean_val = p5 = p95 = 0.0

            # Calculate percentages
            pct_of_parent = 0.0
            if parent_total > 0:
                pct_of_parent = (node_total / parent_total * 100)
                
            pct_of_total = 0.0
            if total_time > 0:
                pct_of_total = (node_total / total_time * 100)
            
            # Store statistics
            stats[full_path] = {
                "total_time": node_total,
                "pct_of_parent": pct_of_parent,
                "pct_of_total": pct_of_total,
                "count": node.count,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "p5": p5,
                "p95": p95,
            }
            
        return stats
        
    def reset(self):
        """Reset the profiler to its initial state."""
        self.stack = []
        self.node_cache = {}